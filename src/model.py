# Copyright 2026 Slyatski Ilya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.



"""
Model definition for HLA-v4-safe-scale experiments.

This file intentionally preserves the logic of the provided HLA-v4 model:
  - positional encoding: RoPE (`use_rope=True, use_wpe=False`, recommended) or
    legacy learned absolute embeddings `wpe` (`use_wpe=True`); exactly one
    mechanism should be active — enforced in GPTConfig.__post_init__;
  - RMSNorm pre-norm blocks;
  - SwiGLU MLP with hidden_dim = round_up(8*n_embd/3, 64);
  - manual causal attention with a fixed causal mask (or SDPA fast path);
  - tied token embedding / LM head.

HLA mechanisms (all identity-initialized, always created for parameter
matching, activated by their alphas):
  - content-conditioned pairwise phase rotation of Q/K (`phase_mult`);
  - residual exponential K/V gating (`use_laplace` + `laplace_alpha`);
  - additive per-key salience bias (`use_salience_bias` + `salience_alpha`);
  - distance-aware Laplace bias with its OWN gate W_gate_d
    (`use_distance_laplace` + `distance_laplace_alpha`);
  - FoX-family cumulative forget gate — baseline arm, OFF in HLA configs
    (`use_forget_gate` + `forget_alpha`);
  - per-query softmax temperature, SSA-family (`use_qtemp` + `qtemp_alpha`);
  - adaptivity axes: `per_head_phase`, `layer_dependent_gate`/`_phase`,
    `learnable_layer_temp`, learned per-head ranges W_range_* (`range_flex`).

The main improvements are engineering/sterility only:
  - valid Python, type hints, clearer errors;
  - HLA identity reset after generic module initialization, so W_gate_k/v remain zero;
  - diagnostics fields for entropy/phase/gate statistics;
  - helper methods used by make_init.py and train_xla.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (self.weight * x_norm).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        multiple = int(getattr(config, "ffn_hidden_multiple_of", 64))
        hidden_dim = ((hidden_dim + multiple - 1) // multiple) * multiple  # TPU-friendly
        self.hidden_dim = hidden_dim
        self.fused = bool(getattr(config, "fused_swiglu", True))
        if self.fused:
            # Same math as two independent projections, but one larger matmul.
            self.w_gate_hidden = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        else:
            self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3.NANOGPT_SCALE_INIT = 1
        self.capture_diagnostics: bool = False
        self.last_hidden: Optional[torch.Tensor] = None
        self.last_mlp_out_norm: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            gate_hidden = self.w_gate_hidden(x)
            gate, hidden = torch.split(gate_hidden, self.hidden_dim, dim=-1)
            h = F.silu(gate) * hidden
        else:
            h = F.silu(self.w1(x)) * self.w2(x)
        out = self.w3(h)
        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_hidden = h.detach()
                self.last_mlp_out_norm = out.detach().float().pow(2).mean(dim=-1).sqrt().mean()
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig", layer_idx: int = 0):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (config.n_embd // config.n_head) % 2 != 0:
            raise ValueError("head_dim must be even for pairwise rotation")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = int(config.block_size)
        self.layer_idx = int(layer_idx)
        self.n_layer = int(config.n_layer)
        self.phase_mult = config.phase_mult
        self.use_rope = bool(getattr(config, "use_rope", False))
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.attention_backend = getattr(config, "attention_backend", "manual")
        self.layer_dependent_gate = bool(getattr(config, "layer_dependent_gate", False))
        # Static heuristic multiplier: deeper layers get wider gate envelopes.
        self.layer_gate_multiplier = (
            1.0 + float(self.layer_idx) / float(max(1, self.n_layer))
            if self.layer_dependent_gate
            else 1.0
        )
        # Learnable per-layer temperature (ablation of the fixed heuristic).
        # Effective multiplier: 1 + (l/L) * softplus(theta) / softplus(0).
        # At theta = 0 this equals the static heuristic EXACTLY, so enabling the
        # flag does not change the identity-init property or the init behavior;
        # the model then learns its own depth profile. One scalar per layer.
        self.learnable_layer_temp = bool(getattr(config, "learnable_layer_temp", False))
        # Always created for exact base/HLA parameter matching (inactive unless
        # both layer_dependent_gate and learnable_layer_temp are set).
        self.W_layer_temp = nn.Parameter(torch.zeros(1))
        self.k_log_clip = float(getattr(config, "k_log_clip", 1.5))
        self.v_log_clip = float(getattr(config, "v_log_clip", 1.0))
        self.use_distance_laplace = bool(getattr(config, "use_distance_laplace", False))
        self.distance_laplace_alpha = float(getattr(config, "distance_laplace_alpha", 0.0))
        self.distance_laplace_range = float(getattr(config, "distance_laplace_range", 1.0))
        self.distance_laplace_clip = float(getattr(config, "distance_laplace_clip", 1.0))
        # Cumulative forget-gate bias (FoX-family, identity-initialized).
        # FoX (Lin et al., ICLR 2025) down-weights score_ij by the sum of
        # log-forget values between key j and query i. We implement the same
        # cumulative structure but bidirectional and exactly identity at init:
        #   g_t   = tanh(W_gate_f(x_t))                (zero at init)
        #   S_i   = cumsum_t( alpha_f * range_f * g_t )
        #   bias_ij = clamp(S_i - S_j, +-clip_f)       (zero at init)
        # Negative g_t = token t "closes the gate" behind it (recency bias,
        # FoX-style forgetting); positive g_t = token t keeps history alive.
        # This makes the FoX-hypothesis (cumulative decay helps) directly
        # testable inside the sterile harness as one more ablation arm.
        self.use_qtemp = bool(getattr(config, "use_qtemp", False))
        self.qtemp_alpha = float(getattr(config, "qtemp_alpha", 0.0))
        self.qtemp_range = float(getattr(config, "qtemp_range", 0.5))
        self.qtemp_clip = float(getattr(config, "qtemp_clip", 1.5))
        self.range_flex = float(getattr(config, "range_flex", 0.25))
        self.use_forget_gate = bool(getattr(config, "use_forget_gate", False))
        self.forget_alpha = float(getattr(config, "forget_alpha", 0.0))
        self.forget_range = float(getattr(config, "forget_range", 0.1))
        self.forget_clip = float(getattr(config, "forget_clip", 4.0))
        # Content-based salience bias: additive log-space bias per KEY position,
        # b_j = alpha * range * tanh(W_gate_sal(x_j)). Unlike the multiplicative
        # K-mix (which has a floor of 1-beta_k), an additive bias can suppress a
        # token's attention weight arbitrarily strongly (e.g. -2 nats = x0.135)
        # or boost it, independent of distance. Identity at init: W_gate_sal = 0.
        self.use_salience_bias = bool(getattr(config, "use_salience_bias", False))
        self.salience_alpha = float(getattr(config, "salience_alpha", 0.0))
        self.salience_range = float(getattr(config, "salience_range", 1.0))
        self.salience_clip = float(getattr(config, "salience_clip", 2.0))

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.W_phase_q = nn.Parameter(torch.zeros(self.n_head, config.n_embd, self.head_dim // 2))
        self.W_phase_k = nn.Parameter(torch.zeros(self.n_head, config.n_embd, self.head_dim // 2))
        # Per-head phase budget (interpretable, H params per layer):
        # phase_mult_eff[h] = phase_mult * (1 + tanh(W_phase_scale[h])) in
        # [0, 2*phase_mult]. At init (zeros) every head gets exactly phase_mult.
        # Heads can learn to opt out of rotation (-> 0) or double their budget;
        # the learned values are directly readable as "how much phase each head
        # wants" and correlate with per-head interference metrics.
        self.per_head_phase = bool(getattr(config, "per_head_phase", False))
        self.W_phase_scale = nn.Parameter(torch.zeros(self.n_head))
        # Depth-adaptive phase: multiply the phase budget by the SAME layer
        # multiplier used for gates (incl. the learnable temperature when on).
        # Rationale: deeper layers carry more semantic content and may benefit
        # from larger retrieval rotations, mirroring the gate design. Max angle
        # with all knobs on: pi*phase_mult*2(head)*2(layer) = 0.6*pi at 0.15.
        self.layer_dependent_phase = bool(getattr(config, "layer_dependent_phase", False))

        self.use_laplace = getattr(config, "use_laplace", True)
        self.laplace_alpha = getattr(config, "laplace_alpha", 0.0)
        self.laplace_range_k = getattr(config, "laplace_range_k", 0.35)
        self.laplace_range_v = getattr(config, "laplace_range_v", 0.2)
        self.beta_k = getattr(config, "beta_k", 0.50)
        self.beta_v = getattr(config, "beta_v", 0.25)

        self.W_range_k = nn.Parameter(torch.zeros(self.n_head))
        self.W_range_v = nn.Parameter(torch.zeros(self.n_head))

        self.W_gate_k = nn.Linear(config.n_embd, self.n_head, bias=False)
        self.W_gate_v = nn.Linear(config.n_embd, self.n_head, bias=False)
        # Always created (regardless of use_salience_bias) so base/HLA stay
        # parameter-matched; inactive branch is disabled via salience_alpha=0.
        self.W_gate_sal = nn.Linear(config.n_embd, self.n_head, bias=False)
        # Forget gate params: same always-created pattern for param matching.
        self.W_gate_f = nn.Linear(config.n_embd, self.n_head, bias=False)
        self.W_range_f = nn.Parameter(torch.zeros(self.n_head))
        # Distance gate: its OWN projection (deconfound fix). Previously the
        # distance bias reused gate_k, entangling "suppress this key" with
        # "shorten this key's reach" - one weight served two mechanisms and
        # confounded single-factor ablations (a 'distance' arm trained W_gate_k).
        # K/V gates correlate with distance behavior but are not it.
        self.W_gate_d = nn.Linear(config.n_embd, self.n_head, bias=False)
        # Q-side temperature (SSA-style, Selective Self-Attention NeurIPS'24):
        # per-query learned sharpness. Scaling q_i by s>0 multiplies the WHOLE
        # score row by s == softmax temperature 1/s. Additive per-query bias
        # would be a no-op (softmax is shift-invariant per row) - multiplicative
        # is the only correct form. Applied to q directly => SDPA-compatible.
        self.W_qtemp = nn.Linear(config.n_embd, self.n_head, bias=False)

        nn.init.constant_(self.W_gate_k.weight, 0.0)
        nn.init.constant_(self.W_gate_v.weight, 0.0)
        nn.init.constant_(self.W_gate_sal.weight, 0.0)
        nn.init.constant_(self.W_gate_f.weight, 0.0)
        nn.init.constant_(self.W_gate_d.weight, 0.0)
        nn.init.constant_(self.W_qtemp.weight, 0.0)

        # Memory note (reviewer attack K3): this (block,block) bool buffer is
        # allocated per layer even on the sdpa path (which ignores it) - e.g.
        # 24 layers x 2048^2 = ~101 MB. Acceptable for our manual-path-primary
        # design; revisit with a shared/global mask if sdpa becomes primary.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        self.last_entropy: Optional[torch.Tensor] = None
        self.last_angle_q_abs_mean: Optional[torch.Tensor] = None
        self.last_angle_k_abs_mean: Optional[torch.Tensor] = None
        # Saturation fractions: how often |tanh(.)| > 0.99, i.e. the model is
        # pushing against the expressivity bound set by phase_mult / ranges.
        # Persistently high values = the bound is too tight (raise it);
        # near-zero values = the bound is not the limiting factor.
        self.last_angle_q_sat_frac: Optional[torch.Tensor] = None
        self.last_angle_q_std: Optional[torch.Tensor] = None
        self.last_angle_k_std: Optional[torch.Tensor] = None
        self.last_angle_k_sat_frac: Optional[torch.Tensor] = None
        self.last_gate_k_sat_frac: Optional[torch.Tensor] = None
        self.last_gate_k_abs_mean: Optional[torch.Tensor] = None
        self.last_gate_v_abs_mean: Optional[torch.Tensor] = None
        self.last_gate_v_sat_frac: Optional[torch.Tensor] = None
        self.last_gate_k_mean: Optional[torch.Tensor] = None
        self.last_gate_v_mean: Optional[torch.Tensor] = None
        self.last_mix_k_mean: Optional[torch.Tensor] = None
        self.last_mix_v_mean: Optional[torch.Tensor] = None
        self.capture_diagnostics: bool = False
        self.capture_attention: bool = False
        self.last_attn: Optional[torch.Tensor] = None
        self.last_gate_k: Optional[torch.Tensor] = None
        self.last_gate_v: Optional[torch.Tensor] = None
        self.last_mix_k: Optional[torch.Tensor] = None
        self.last_mix_v: Optional[torch.Tensor] = None
        self.last_attn_out_norm: Optional[torch.Tensor] = None
        self.last_distance_bias_mean: Optional[torch.Tensor] = None
        self.last_distance_bias_abs_mean: Optional[torch.Tensor] = None
        self.last_salience_bias_abs_mean: Optional[torch.Tensor] = None
        self.last_salience_sat_frac: Optional[torch.Tensor] = None
        self.last_gate_sal: Optional[torch.Tensor] = None
        self.last_gate_d: Optional[torch.Tensor] = None
        self.last_forget_bias_abs_mean: Optional[torch.Tensor] = None
        self.last_forget_sat_frac: Optional[torch.Tensor] = None
        self.last_qtemp_mean: Optional[torch.Tensor] = None
        self.last_qtemp_sat_frac: Optional[torch.Tensor] = None
        # Post-backward snapshot of mechanism gradient norms (Theorem 5
        # verification during training). Filled by capture_mechanism_grad_norms.
        self.last_mechanism_grad_norms: Optional[Dict[str, torch.Tensor]] = None

    def reset_hla_identity(self) -> None:
        """Reset HLA-specific parameters to the exact identity state."""
        with torch.no_grad():
            self.W_phase_q.zero_()
            self.W_phase_k.zero_()
            self.W_range_k.zero_()
            self.W_range_v.zero_()
            self.W_gate_k.weight.zero_()
            self.W_gate_v.weight.zero_()
            self.W_gate_sal.weight.zero_()
            self.W_gate_f.weight.zero_()
            self.W_gate_d.weight.zero_()
            self.W_qtemp.weight.zero_()
            self.W_range_f.zero_()
            self.W_layer_temp.zero_()
            self.W_phase_scale.zero_()

    def capture_mechanism_grad_norms(self) -> None:
        """Snapshot L2 gradient norms of ALL mechanism parameters.

        Call AFTER gradients are reduced (post optimizer_step_with_reduced_clip's
        reduce) and BEFORE optimizer.zero_grad(). Stores detached device tensors
        (no .item() here - keeps XLA host-sync free); convert on master at
        logging cadence via eval.mechanism_gradient_statistics.
        Inactive mechanisms have grad None -> recorded as exact 0.
        """
        with torch.no_grad():
            norms: Dict[str, torch.Tensor] = {}
            for name, p in [
                ("phase_q", self.W_phase_q),
                ("phase_k", self.W_phase_k),
                ("phase_scale", self.W_phase_scale),
                ("range_k", self.W_range_k),
                ("range_v", self.W_range_v),
                ("range_f", self.W_range_f),
                ("gate_k", self.W_gate_k.weight),
                ("gate_v", self.W_gate_v.weight),
                ("gate_sal", self.W_gate_sal.weight),
                ("gate_f", self.W_gate_f.weight),
                ("gate_d", self.W_gate_d.weight),
                ("qtemp", self.W_qtemp.weight),
                ("layer_temp", self.W_layer_temp),
            ]:
                g = p.grad
                norms[name] = (
                    g.detach().float().norm()
                    if g is not None
                    else torch.zeros((), device=p.device)
                )
            self.last_mechanism_grad_norms = norms

    @staticmethod
    def _rotate_pairwise(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.chunk(2, dim=-1)
        return torch.cat(
            [
                x_real * cos - x_imag * sin,
                x_real * sin + x_imag * cos,
            ],
            dim=-1,
        )

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard RoPE applied to Q/K before content-conditioned phase rotation.

        NOTE: this method must live on CausalSelfAttention (it uses self.use_rope
        and self.rope_inv_freq registered here). In the original file it was
        accidentally defined inside RMSNorm, which made every forward pass crash.
        """
        if not self.use_rope:
            return q, k
        pos = torch.arange(T, device=q.device, dtype=self.rope_inv_freq.dtype)
        angles = torch.einsum("t,d->td", pos, self.rope_inv_freq)
        cos = torch.cos(angles)[None, None, :, :].to(q.dtype)
        sin = torch.sin(angles)[None, None, :, :].to(q.dtype)
        return self._rotate_pairwise(q, cos, sin), self._rotate_pairwise(k, cos, sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        hs = self.head_dim

        if self.mask.size(-1) < T:
            raise ValueError(f"block_size {self.mask.size(-1)}")

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        q, k = self._apply_rope(q, k, T)

        # Effective per-layer gate multiplier: static heuristic, optionally
        # modulated by the learnable temperature (exact heuristic at theta=0).
        layer_mult = self.layer_gate_multiplier
        if self.layer_dependent_gate and self.learnable_layer_temp:
            depth = float(self.layer_idx) / float(max(1, self.n_layer))
            temp = F.softplus(self.W_layer_temp.float()) / math.log(2.0)  # softplus(0)=log 2
            layer_mult = 1.0 + depth * temp.squeeze(0)

        if self.phase_mult != 0.0:
            x_float = x.float()
            raw_angles_q = torch.einsum("btd,hdk->bhtk", x_float, self.W_phase_q.float())
            raw_angles_k = torch.einsum("btd,hdk->bhtk", x_float, self.W_phase_k.float())

            if self.per_head_phase:
                # phase_mult_eff[h] in [0, 2*phase_mult]; exactly phase_mult at init.
                head_budget = 1.0 + torch.tanh(self.W_phase_scale.float())  # (H,)
                phase_scale = (math.pi * self.phase_mult) * head_budget.view(1, self.n_head, 1, 1)
            else:
                phase_scale = math.pi * self.phase_mult
            if self.layer_dependent_phase:
                # Same depth profile as the gates (learnable when temp is on).
                phase_scale = phase_scale * layer_mult

            angles_q = phase_scale * torch.tanh(raw_angles_q)
            angles_k = phase_scale * torch.tanh(raw_angles_k)

            cos_q = torch.cos(angles_q).to(q.dtype)
            sin_q = torch.sin(angles_q).to(q.dtype)
            cos_k = torch.cos(angles_k).to(k.dtype)
            sin_k = torch.sin(angles_k).to(k.dtype)

            q = self._rotate_pairwise(q, cos_q, sin_q)
            k_rot = self._rotate_pairwise(k, cos_k, sin_k)

            with torch.no_grad():
                self.last_angle_q_abs_mean = angles_q.detach().abs().mean()
                self.last_angle_k_abs_mean = angles_k.detach().abs().mean()
                tq = torch.tanh(raw_angles_q.detach()).abs()
                tk = torch.tanh(raw_angles_k.detach()).abs()
                self.last_angle_q_sat_frac = (tq > 0.99).float().mean()
                self.last_angle_k_sat_frac = (tk > 0.99).float().mean()
                # Nanda attack: a mean |angle| of 0.1 can be "all angles 0.1"
                # or "half 0, half 0.2" - std distinguishes uniform use of the
                # budget from a bimodal on/off pattern.
                self.last_angle_q_std = angles_q.detach().float().std()
                self.last_angle_k_std = angles_k.detach().float().std()
        else:
            k_rot = k
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                self.last_angle_q_abs_mean = zero
                self.last_angle_k_abs_mean = zero
                self.last_angle_q_sat_frac = zero
                self.last_angle_k_sat_frac = zero
                self.last_angle_q_std = zero
                self.last_angle_k_std = zero

        k = k_rot

        if self.use_qtemp and self.qtemp_alpha != 0.0:
            # Per-query learned temperature (SSA-family, NeurIPS'24): scale the
            # query vector => scales its whole score row => per-row softmax
            # temperature. Bounded like every HLA mechanism; exact identity at
            # init. Query-side counterpart of the K-gate: K-gate sets how loud
            # a key IS, q-temp sets how selectively a query LISTENS. Applied to
            # q itself, so the SDPA fast path remains numerically exact.
            gate_q = torch.tanh(self.W_qtemp(x)).float()               # (B, T, H)
            log_temp = torch.clamp(self.qtemp_alpha * self.qtemp_range * gate_q,
                                   min=-self.qtemp_clip, max=self.qtemp_clip)
            temp_q = torch.exp(log_temp).to(q.dtype).transpose(1, 2).unsqueeze(-1)
            q = q * temp_q                                              # (B,H,T,1)
            with torch.no_grad():
                self.last_qtemp_mean = temp_q.detach().float().mean()
                self.last_qtemp_sat_frac = (gate_q.detach().abs() > 0.99).float().mean()
        else:
            with torch.no_grad():
                one = torch.ones((), device=x.device)
                zero = torch.zeros((), device=x.device)
                self.last_qtemp_mean = one
                self.last_qtemp_sat_frac = zero


        if self.use_laplace and self.laplace_alpha != 0.0:

            gate_k = torch.tanh(self.W_gate_k(x)).float()  # (B, T, H)
            range_k = (self.laplace_range_k * layer_mult) * (
                1.0 + self.range_flex * torch.tanh(self.W_range_k.float())
            )  # (H,)
            range_k = range_k.view(1, 1, self.n_head)

            clip_k = self.k_log_clip * layer_mult
            log_scale_k = torch.clamp(
                self.laplace_alpha * gate_k * range_k,
                min=-clip_k,
                max=clip_k,
            )
            scale_k = torch.exp(log_scale_k)
            mix_k = (1.0 - self.beta_k) + self.beta_k * scale_k  # (B, T, H)
            mix_k = mix_k.to(k.dtype).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
            k = k_rot * mix_k

            gate_v = torch.tanh(self.W_gate_v(x)).float()  # (B, T, H)
            range_v = (self.laplace_range_v * layer_mult) * (
                1.0 + self.range_flex * torch.tanh(self.W_range_v.float())
            )  # (H,)
            range_v = range_v.view(1, 1, self.n_head)

            clip_v = self.v_log_clip * layer_mult
            log_scale_v = torch.clamp(
                self.laplace_alpha * gate_v * range_v,
                min=-clip_v,
                max=clip_v,
            )
            scale_v = torch.exp(log_scale_v)
            mix_v = (1.0 - self.beta_v) + self.beta_v * scale_v  # (B, T, H)
            mix_v = mix_v.to(v.dtype).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
            v = v * mix_v

            with torch.no_grad():
                self.last_gate_k_mean = gate_k.detach().mean()
                self.last_gate_v_mean = gate_v.detach().mean()
                self.last_mix_k_mean = mix_k.detach().float().mean()
                self.last_mix_v_mean = mix_v.detach().float().mean()
                self.last_gate_k_sat_frac = (gate_k.detach().abs() > 0.99).float().mean()
                self.last_gate_v_sat_frac = (gate_v.detach().abs() > 0.99).float().mean()
                # |gate| means: signed means can sit near 0 while the gate is
                # highly active (symmetric up/down usage). Underuse detector:
                # abs_mean ~ 0 => mechanism is OFF; sat_frac high => envelope
                # too TIGHT. Together they bracket "did we size ranges right?".
                self.last_gate_k_abs_mean = gate_k.detach().abs().mean()
                self.last_gate_v_abs_mean = gate_v.detach().abs().mean()
                if self.capture_diagnostics:
                    self.last_gate_k = gate_k.detach()
                    self.last_gate_v = gate_v.detach()
                    self.last_mix_k = mix_k.detach()
                    self.last_mix_v = mix_v.detach()
        else:
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                one = torch.ones((), device=x.device)
                self.last_gate_k_mean = zero
                self.last_gate_v_mean = zero
                self.last_mix_k_mean = one
                self.last_mix_v_mean = one
                self.last_gate_k_sat_frac = zero
                self.last_gate_v_sat_frac = zero
                self.last_gate_k_abs_mean = zero
                self.last_gate_v_abs_mean = zero
                if self.capture_diagnostics:
                    self.last_gate_k = None
                    self.last_gate_v = None
                    self.last_mix_k = None
                    self.last_mix_v = None

        # =====================================================
        # 3) Attention
        # =====================================================
        # NOTE: the fast SDPA path cannot apply the distance-Laplace log-bias.
        # Guard against silently changing model semantics when it is active.
        sdpa_ok = (
            self.attention_backend == "sdpa"
            and not self.capture_diagnostics
            and not self.capture_attention
            and not (self.use_distance_laplace and self.distance_laplace_alpha != 0.0)
            and not (self.use_salience_bias and self.salience_alpha != 0.0)
            and not (self.use_forget_gate and self.forget_alpha != 0.0)
        )
        if sdpa_ok:

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))


        if self.use_distance_laplace and self.distance_laplace_alpha != 0.0:
            # Deconfound fix: distance now has its OWN gate (W_gate_d). The old
            # code reused gate_k, so "suppress this key" and "shorten this
            # key's reach" shared one weight matrix - cheap, but it meant a
            # 'distance'-only ablation arm still trained W_gate_k, and the two
            # mechanisms could not disagree (e.g. keep a key LOUD nearby but
            # SHORT-range). Identity-init preserved: W_gate_d = 0.
            gate_d = torch.tanh(self.W_gate_d(x)).float()  # (B, T, H)
            pos = torch.arange(T, device=x.device)
            # SCIENTIFIC FIX (round 6): normalize by the model constant
            # block_size-1, NOT by the current batch length T-1. The old
            # T-normalization made the bias for the SAME (i,j) pair depend on
            # how much context happened to be in the batch - breaking
            # prefix-stability (KV-cache correctness) the moment W_gate_d is
            # trained: full-vs-prefix logits diverged by ~4e-3. Training is
            # unaffected bit-for-bit (FixedDataset always yields T==block_size),
            # so existing checkpoints stay valid; inference at shorter T is now
            # consistent with training semantics.
            dist = (pos[:, None] - pos[None, :]).clamp_min(0).float() / float(max(1, self.block_size - 1))
            key_gate = gate_d.transpose(1, 2).unsqueeze(2)  # (B,H,1,T_key)
            dist_bias = (
                self.distance_laplace_alpha
                * self.distance_laplace_range
                * layer_mult
                * dist[None, None, :, :]
                * key_gate
            )
            clip_d = self.distance_laplace_clip * layer_mult
            dist_bias = torch.clamp(
                dist_bias,
                min=-clip_d,
                max=clip_d,
            )
            att = att + dist_bias.to(att.dtype)
            with torch.no_grad():
                self.last_distance_bias_mean = dist_bias.detach().float().mean()
                self.last_distance_bias_abs_mean = dist_bias.detach().float().abs().mean()
                if self.capture_diagnostics:
                    self.last_gate_d = gate_d.detach()
        elif self.capture_diagnostics:
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                self.last_distance_bias_mean = zero
                self.last_distance_bias_abs_mean = zero
                self.last_gate_d = None

        if self.use_salience_bias and self.salience_alpha != 0.0:
            # Additive per-key salience: strong suppression/boost independent of
            # distance. Complements the multiplicative K-mix, whose suppression
            # floor is (1 - beta_k); log-space bias has no such floor.
            gate_sal = torch.tanh(self.W_gate_sal(x)).float()  # (B, T, H)
            sal_bias = self.salience_alpha * self.salience_range * gate_sal
            sal_bias = torch.clamp(sal_bias, min=-self.salience_clip, max=self.salience_clip)
            sal_bias = sal_bias.transpose(1, 2).unsqueeze(2)  # (B, H, 1, T_key)
            att = att + sal_bias.to(att.dtype)
            with torch.no_grad():
                self.last_salience_bias_abs_mean = sal_bias.detach().float().abs().mean()
                self.last_salience_sat_frac = (gate_sal.detach().abs() > 0.99).float().mean()
                if self.capture_diagnostics:
                    # Full per-token tensor for the gate-redundancy probe
                    # (round 7): do the four content gates learn the same
                    # function or four different ones?
                    self.last_gate_sal = gate_sal.detach()
        elif self.capture_diagnostics:
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                self.last_salience_bias_abs_mean = zero
                self.last_salience_sat_frac = zero
                self.last_gate_sal = None

        if self.use_forget_gate and self.forget_alpha != 0.0:
            # FoX-style cumulative gate (identity at init: W_gate_f = 0).
            # S_t = cumsum of per-token log-forget; bias_ij = S_i - S_j is the
            # total gate "mass" between key j and query i. Causal mask below
            # removes j > i, so only past-directed biases survive.
            # NUMERICAL NOTE (reviewer attack N2): S_t grows as O(T*range).
            # At T=2048, range=0.1 the running sum reaches ~200; in bf16 the
            # ulp near 200 is ~0.78, so individual steps of 0.1 would be LOST
            # (204.8 + 0.1 == 204.8 in bf16). The .float() below is therefore
            # load-bearing, not cosmetic. WARNING for TPU: XLA_USE_BF16=1
            # downcasts float32 to bf16 globally and silently breaks this;
            # use XLA_DOWNCAST_BF16 or keep fp32 enabled for forget-gate runs.
            gate_f = torch.tanh(self.W_gate_f(x)).float()          # (B, T, H)
            range_f = (self.forget_range * (1.0 + self.range_flex * torch.tanh(self.W_range_f.float()))).view(1, 1, self.n_head)
            log_f = self.forget_alpha * gate_f * range_f            # (B, T, H)
            S = torch.cumsum(log_f, dim=1).transpose(1, 2)          # (B, H, T)
            forget_bias = S.unsqueeze(-1) - S.unsqueeze(-2)         # (B,H,T_q,T_k) = S_i - S_j
            forget_bias = torch.clamp(forget_bias, min=-self.forget_clip, max=self.forget_clip)
            att = att + forget_bias.to(att.dtype)
            with torch.no_grad():
                self.last_forget_bias_abs_mean = forget_bias.detach().float().abs().mean()
                self.last_forget_sat_frac = (gate_f.detach().abs() > 0.99).float().mean()
        elif self.capture_diagnostics:
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                self.last_forget_bias_abs_mean = zero
                self.last_forget_sat_frac = zero

        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(~mask, float("-inf"))

        # TPU day-1 memory fix: the old code did att.float() BEFORE softmax,
        # materializing an fp32 (B,H,T,T) copy per layer that XLA kept alive
        # for backward - 12 layers x 4 GB = 48 GB at B=16,T=2048 (measured in
        # the OOM dump). F.softmax(dtype=torch.float32) computes softmax in
        # fp32 internally (bit-identical in fp32 runs; fp32-accurate
        # normalization under bf16) but lets the fused kernel avoid the
        # stored fp32 activation.
        att = F.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)

        # Entropy is an eval-time diagnostic. Computing it on EVERY training
        # forward wastes O(B*H*T^2) work per layer and adds host-sync pressure
        # on XLA. Gate it behind eval/diagnostics; measure_attention_entropy()
        # runs in eval mode, so it still works unchanged.
        # Round 14 review find: this used to run on EVERY eval forward
        # ((not self.training) or ...), i.e. on all 50 validation batches per
        # eval - materializing p.float() + p.log() over (B,H,T,T) adds
        # ~9-13 GB peak at B=16,H=16,T=2048: a real OOM risk on v3-8's
        # 16 GB/core. Entropy is a PROBE - probes opt in via
        # set_diagnostics(True) (all eval.py probes already do).
        if self.capture_diagnostics:
            with torch.no_grad():
                p = att.detach().float().clamp(min=1e-9)
                entropy = -(p * p.log()).sum(dim=-1).mean(dim=(0, 2))  # (H,)
                self.last_entropy = entropy.detach()
                if self.capture_attention:
                    self.last_attn = att.detach().float()
                elif self.capture_diagnostics:
                    self.last_attn = None

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_attn_out_norm = out.detach().float().pow(2).mean(dim=-1).sqrt().mean()
        return out


class Block(nn.Module):
    def __init__(self, config: "GPTConfig", layer_idx: int):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln_1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # true tokenizer vocabulary size
    padded_vocab_size: int | None = None  # optional embedding/output dim rounded for TPU MXU
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    dropout: float = 0.0  # accepted for config compat but NOT implemented; enforced 0.0 in __post_init__
    bias: bool = False    # kept for config compatibility; original model uses bias=False

    phase_mult: float = 0.0
    use_rope: bool = False
    rope_theta: float = 10000.0
    # Learned absolute positional embeddings. Keep True for legacy non-RoPE
    # configs; set False in RoPE configs so the model has exactly one
    # positional mechanism (reviewers will ask why both otherwise).
    use_wpe: bool = True

    use_laplace: bool = True
    laplace_alpha: float = 0.0
    laplace_range_k: float = 0.35
    laplace_range_v: float = 0.20
    beta_k: float = 0.50
    beta_v: float = 0.25
    k_log_clip: float = 1.5
    v_log_clip: float = 1.0
    layer_dependent_gate: bool = False

    # Optional true distance-aware Laplace log-bias. This is parameter-free and
    # identity-initialized because it is proportional to gate activations, which
    # are exactly zero at init. Negative gate values decay distant tokens; positive
    # gate values preserve/boost distant important tokens.
    use_distance_laplace: bool = False
    distance_laplace_alpha: float = 0.0
    distance_laplace_range: float = 1.0
    distance_laplace_clip: float = 1.0

    # Additive per-key content salience bias (see CausalSelfAttention docstring).
    # Identity-initialized (W_gate_sal = 0). Suppression floor: none (log-space).
    use_salience_bias: bool = False
    salience_alpha: float = 0.0
    salience_range: float = 1.0
    salience_clip: float = 2.0

    # FoX-style cumulative forget gate (bidirectional, identity-init).
    # Enables testing the Forgetting-Transformer hypothesis inside the same
    # sterile harness: use_forget_gate=true + forget_alpha=1.0 as an ablation
    # arm, or as THE mechanism with all others off (= FoX-like baseline).
    use_forget_gate: bool = False
    forget_alpha: float = 0.0
    forget_range: float = 0.1
    forget_clip: float = 4.0

    # Q-side temperature (SSA-family). Identity at init (W_qtemp = 0).
    use_qtemp: bool = False
    qtemp_alpha: float = 0.0
    qtemp_range: float = 0.5
    qtemp_clip: float = 1.5

    # Range flexibility: learned per-head range modifier spans
    # base_range * (1 +- range_flex). 0.25 was the historical constant; now a
    # config knob. Guidance: watch range_k_flex_use in depth profiles - if it
    # pins near 1.0 the model wants more flexibility, raise to 0.5.
    range_flex: float = 0.25

    # Learnable per-layer gate temperature (requires layer_dependent_gate).
    # Identity-compatible: at theta=0 the multiplier equals the static heuristic.
    learnable_layer_temp: bool = False
    # Per-head phase budget: phase_mult_eff[h] = phase_mult*(1+tanh(s_h)) in
    # [0, 2*phase_mult]; exactly phase_mult at init. H params/layer, readable
    # as "how much rotation each head chose".
    per_head_phase: bool = False
    # Depth-adaptive phase: scale the phase budget by the same layer multiplier
    # as the gates (static heuristic, or learned when learnable_layer_temp=true).
    layer_dependent_phase: bool = False

    gradient_checkpointing: bool = True
    fused_swiglu: bool = True
    ffn_hidden_multiple_of: int = 64
    attention_backend: str = "manual"  # "manual" or optional "sdpa" speed-probe backend

    baseline_type: str = "parameter_matched_ablated"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.dropout != 0.0:
            # R26 (review round 4): dropout is accepted in configs for
            # compatibility but is NOT wired into any layer. Silently ignoring
            # a non-zero value would be a lying config - refuse instead.
            raise ValueError("dropout is not implemented in this model; set dropout=0.0")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.padded_vocab_size is not None and self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size must be >= vocab_size")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("n_layer, n_head and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim must be even for pairwise rotation")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if not self.use_wpe and not self.use_rope:
            raise ValueError("model needs positional information: use_wpe and use_rope are both False")
        if self.use_wpe and self.use_rope:
            # B4 fix (external review): dual positional encoding silently changes
            # semantics (learned absolute + rotary applied together shift logits
            # by ~0.25 and confound every position-related ablation). Exactly one
            # scheme must be active; this was documented as intent but never
            # enforced - and the lack of enforcement let even our own test
            # fixtures run dual-PE unnoticed.
            raise ValueError(
                "exactly one positional scheme must be active: "
                "use_wpe and use_rope are both True (set use_wpe=false for RoPE runs)"
            )
        if self.beta_k < 0 or self.beta_v < 0:
            raise ValueError("beta_k and beta_v must be non-negative")
        if self.k_log_clip <= 0 or self.v_log_clip <= 0:
            raise ValueError("k_log_clip and v_log_clip must be positive")
        if self.distance_laplace_range < 0:
            raise ValueError("distance_laplace_range must be non-negative")
        if self.distance_laplace_clip <= 0:
            raise ValueError("distance_laplace_clip must be positive")
        if self.salience_range < 0:
            raise ValueError("salience_range must be non-negative")
        if self.salience_clip <= 0:
            raise ValueError("salience_clip must be positive")
        if self.forget_range < 0:
            raise ValueError("forget_range must be non-negative")
        if self.qtemp_range < 0:
            raise ValueError("qtemp_range must be non-negative")
        if self.qtemp_clip <= 0:
            raise ValueError("qtemp_clip must be positive")
        if not (0.0 < self.range_flex <= 1.0):
            raise ValueError("range_flex must be in (0, 1]")
        if self.forget_clip <= 0:
            raise ValueError("forget_clip must be positive")
        if self.learnable_layer_temp and not self.layer_dependent_gate:
            raise ValueError("learnable_layer_temp requires layer_dependent_gate=true")
        if self.layer_dependent_phase and not self.layer_dependent_gate:
            raise ValueError("layer_dependent_phase requires layer_dependent_gate=true (shared depth profile)")
        if self.ffn_hidden_multiple_of <= 0:
            raise ValueError("ffn_hidden_multiple_of must be positive")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be 'manual' or 'sdpa'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.padded_vocab_size = int(config.padded_vocab_size or config.vocab_size)

        modules = dict(
            wte=nn.Embedding(self.padded_vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        )
        if config.use_wpe:
            # Legacy learned absolute positions. Disabled in RoPE configs so the
            # model has exactly one positional mechanism.
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)

        self.lm_head = nn.Linear(config.n_embd, self.padded_vocab_size, bias=False)

        self.apply(self._init_weights)

        # Critical: generic _init_weights reinitializes W_gate_k/v because they are
        # Linear modules. Reset HLA branches *after* apply() to preserve exact
        # identity initialization intended by the architecture.
        self.reset_hla_identity()

        # tie weights AFTER init, so shared weight is not initialized twice
        self.lm_head.weight = self.transformer.wte.weight

        self.gradient_checkpointing = config.gradient_checkpointing

    def reset_hla_identity(self) -> None:
        for block in self.transformer.h:
            block.attn.reset_hla_identity()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if self.config.block_size < T:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")

        tok = self.transformer.wte(idx)  # (B, T, C)
        if self.config.use_wpe:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)[None, :, :]  # (1, T, C)
            x = tok + pos_emb
        else:
            x = tok

        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                # TPU day-1 fix: new torch's checkpoint(use_reentrant=False)
                # resolves the device module via getattr(torch, 'xla'), which
                # does not exist (XLA lives in the separate torch_xla package)
                # -> AttributeError on the first forward. Fall back to a plain
                # call once, loudly, instead of crashing the run.
                try:
                    x = checkpoint.checkpoint(block, x, use_reentrant=False)
                except AttributeError as e:
                    if "has no attribute 'xla'" not in str(e):
                        raise
                    if not getattr(self, "_ckpt_fallback_warned", False):
                        self._ckpt_fallback_warned = True
                        import warnings as _w
                        _w.warn(
                            "gradient_checkpointing disabled at runtime: torch "
                            "checkpoint is incompatible with this torch_xla "
                            "(torch.xla device module missing). Running without "
                            "activation checkpointing - watch memory on large models.",
                            stacklevel=1,
                        )
                    self.gradient_checkpointing = False
                    x = block(x)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_for_loss = logits.float()
            if self.padded_vocab_size != self.config.vocab_size:
                # Exclude padding-only output classes from the likelihood. This keeps
                # the objective identical to vocab_size while allowing MXU-friendly
                # padded embedding/output matrices.
                logits_for_loss = logits_for_loss.clone()
                logits_for_loss[..., self.config.vocab_size :] = torch.finfo(logits_for_loss.dtype).min
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Autoregressive decoding for evals and qualitative samples.

        Minimal by design (no KV-cache): correctness over speed, suitable for
        lm-eval-harness-style scoring and small demos. Crops context to
        block_size; never samples padded-vocab ids (they are masked out).

        NOTE (forget gate): when the prompt exceeds block_size, the context is
        cropped and the forget-gate cumulative sum S_t restarts from the window
        start - the gate then measures accumulated forgetting WITHIN the
        visible window, not since sequence begin. This matches training (which
        never sees beyond block_size) but differs from a hypothetical
        infinite-memory reading; documented to avoid surprise in long
        generation loops.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(int(max_new_tokens)):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :].float()
                if self.padded_vocab_size != self.config.vocab_size:
                    logits[..., self.config.vocab_size:] = float("-inf")
                if greedy:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(float(temperature), 1e-6)
                    if top_k is not None:
                        k = min(int(top_k), logits.size(-1))
                        v, _ = torch.topk(logits, k)
                        logits[logits < v[:, [-1]]] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
            return idx
        finally:
            self.train(was_training)

    def set_diagnostics(self, *, enabled: bool, capture_attention: bool = False) -> None:
        for block in self.transformer.h:
            block.attn.capture_diagnostics = bool(enabled)
            block.attn.capture_attention = bool(enabled and capture_attention)
            block.mlp.capture_diagnostics = bool(enabled)

    def hla_identity_error(self) -> float:
        """Max absolute value of HLA identity-initialized tensors."""
        err = 0.0
        for block in self.transformer.h:
            attn = block.attn
            for tensor in [
                attn.W_phase_q,
                attn.W_phase_k,
                attn.W_range_k,
                attn.W_range_v,
                attn.W_gate_k.weight,
                attn.W_gate_v.weight,
                attn.W_gate_sal.weight,
                attn.W_gate_f.weight,
                attn.W_gate_d.weight,
                attn.W_qtemp.weight,
                attn.W_range_f,
                attn.W_layer_temp,
                attn.W_phase_scale,
            ]:
                err = max(err, float(tensor.detach().float().abs().max().cpu().item()))
        return err

    def capture_mechanism_grad_norms(self) -> None:
        """Snapshot mechanism gradient norms in every attention block."""
        for block in self.transformer.h:
            block.attn.capture_mechanism_grad_norms()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["RMSNorm", "SwiGLU", "CausalSelfAttention", "Block", "GPTConfig", "GPT"]
