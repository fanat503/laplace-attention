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



"""Evaluation probes and diagnostics for base/HLA GPT models."""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

# Token id offsets used to build synthetic A->B induction pairs. They must be
# below the smallest GPT-2 BPE ids used for random filler (100..20000) minus
# batch_size, so that induction tokens never collide with filler tokens.
INDUCTION_TOK_A_OFFSET = 10
INDUCTION_TOK_B_OFFSET = 50


def _mask_padded_logits(model, logits: torch.Tensor) -> torch.Tensor:
    """Finding #19: with padded_vocab_size > vocab_size the raw lm_head
    logits carry pad-classes that the MODEL's own loss masks out, but eval
    code did not - softmax denominators included up to 47 phantom classes
    (measured +9.4e-4 CE inflation on random init) and P(target) probes
    were correspondingly deflated. Symmetric for both twins but biased vs
    the true model distribution; every eval must mask before softmax/CE."""
    v = int(getattr(model.config, "vocab_size", logits.size(-1)))
    if logits.size(-1) > v:
        logits = logits.clone()
        logits[..., v:] = torch.finfo(logits.dtype).min
    return logits


@torch.no_grad()
def evaluate_induction(model, device, seed: int = 42, batch_size: int = 32) -> float:
    was_training = model.training
    model.eval()
    try:
        T = min(256, model.config.block_size)
        if T < 8:
            return float("nan")

        # Use the constants defined at module top.
        required_vocab = INDUCTION_TOK_B_OFFSET + batch_size
        if model.config.vocab_size < required_vocab:
            return float("nan")

        pos_a1 = T // 3
        pos_b1 = pos_a1 + 1
        pos_a2 = (2 * T) // 3
        if pos_a2 >= T - 1:
            return float("nan")

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(
            low=100,
            high=min(20000, model.config.vocab_size - 1),
            size=(batch_size, T),
            generator=g,
        ).to(device)

        # Vectorized induction-token placement.
        i = torch.arange(batch_size, device=device)
        tokens[:, pos_a1] = INDUCTION_TOK_A_OFFSET + i
        tokens[:, pos_b1] = INDUCTION_TOK_B_OFFSET + i
        tokens[:, pos_a2] = INDUCTION_TOK_A_OFFSET + i
        target_ids = INDUCTION_TOK_B_OFFSET + i

        logits, _ = model(tokens)
        if not bool(torch.isfinite(logits).all().detach().cpu().item()):
            return float("nan")
        probs = torch.softmax(_mask_padded_logits(model, logits[:, pos_a2, :].float()), dim=-1)
        return float(probs.gather(1, target_ids[:, None]).mean().detach().cpu().item())
    finally:
        model.train(was_training)


@torch.no_grad()
def measure_attention_entropy(model, device, seed: int = 42, batch_size: int = 4) -> float:
    was_training = model.training
    model.eval()
    try:
        T = min(256, model.config.block_size)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T), generator=g).to(device)
        prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
        prev_attn = bool(model.transformer.h[0].attn.capture_attention)
        model.set_diagnostics(enabled=True)
        try:
            _ = model(tokens)
        finally:
            model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        entropies = []
        for block in model.transformer.h:
            e = getattr(block.attn, "last_entropy", None)
            if e is not None:
                entropies.append(e.float().mean())
        if not entropies:
            return float("nan")
        return float(torch.stack(entropies).mean().detach().cpu().item())
    finally:
        model.train(was_training)


@torch.no_grad()
def depth_profile_statistics(model) -> Dict[str, float]:
    """Readouts of the learned depth/head profiles (interpretability for free).

    - layer_temp_l{i}: effective learned multiplier at layer i (only when
      learnable_layer_temp is on); equals the static heuristic at init.
    - phase_budget_mean/min/max: distribution of per-head phase budgets
      (1 + tanh(W_phase_scale)) in [0, 2]; 1.0 = init value for every head.
    """
    import math as _m

    out: Dict[str, float] = {}
    temps = []
    budgets = []
    for block in model.transformer.h:
        attn = block.attn
        if getattr(attn, "layer_dependent_gate", False) and getattr(attn, "learnable_layer_temp", False):
            depth = float(attn.layer_idx) / float(max(1, attn.n_layer))
            t = float(torch.nn.functional.softplus(attn.W_layer_temp.detach().float()) / _m.log(2.0))
            temps.append(1.0 + depth * t)
        if getattr(attn, "per_head_phase", False):
            budgets.append(1.0 + torch.tanh(attn.W_phase_scale.detach().float()))
    if temps:
        out["layer_temp_first"] = temps[0]
        out["layer_temp_last"] = temps[-1]
        out["layer_temp_mean"] = float(sum(temps) / len(temps))
    if budgets:
        all_b = torch.cat(budgets)
        out["phase_budget_mean"] = float(all_b.mean())
        out["phase_budget_min"] = float(all_b.min())
        out["phase_budget_max"] = float(all_b.max())
    return out


@torch.no_grad()
def phase_statistics(model) -> Tuple[Optional[torch.Tensor], float]:
    all_norms = []
    for block in model.transformer.h:
        attn = block.attn
        if not hasattr(attn, "W_phase_q") or attn.W_phase_q is None:
            return None, float("nan")
        Wq = attn.W_phase_q.detach().float()
        Wk = attn.W_phase_k.detach().float()
        norm_q = torch.linalg.norm(Wq, dim=(1, 2))
        norm_k = torch.linalg.norm(Wk, dim=(1, 2))
        all_norms.append(0.5 * (norm_q + norm_k))
    if not all_norms:
        return None, float("nan")
    stacked = torch.stack(all_norms, dim=0)
    mean_per_head = stacked.mean(dim=0).detach().cpu()
    return mean_per_head, float(stacked.mean().detach().cpu().item())


@torch.no_grad()
def hla_statistics(model) -> Dict[str, float]:
    vals: Dict[str, list] = {
        "layer_gate_multiplier": [],
        "angle_q_abs_mean": [],
        "angle_k_abs_mean": [],
        "gate_k_mean": [],
        "gate_v_mean": [],
        "mix_k_mean": [],
        "mix_v_mean": [],
        "distance_bias_mean": [],
        "distance_bias_abs_mean": [],
        "angle_q_sat_frac": [],
        "angle_q_std": [],
        "angle_k_std": [],
        "angle_k_sat_frac": [],
        "gate_k_sat_frac": [],
        "gate_k_abs_mean": [],
        "gate_v_abs_mean": [],
        "gate_v_sat_frac": [],
        "salience_bias_abs_mean": [],
        "salience_sat_frac": [],
        "forget_bias_abs_mean": [],
        "forget_sat_frac": [],
        "qtemp_mean": [],
        "qtemp_sat_frac": [],
    }
    for block in model.transformer.h:
        attn = block.attn
        mapping = {
            "layer_gate_multiplier": torch.tensor(float(getattr(attn, "layer_gate_multiplier", 1.0))),
            "angle_q_abs_mean": getattr(attn, "last_angle_q_abs_mean", None),
            "angle_k_abs_mean": getattr(attn, "last_angle_k_abs_mean", None),
            "gate_k_mean": getattr(attn, "last_gate_k_mean", None),
            "gate_v_mean": getattr(attn, "last_gate_v_mean", None),
            "mix_k_mean": getattr(attn, "last_mix_k_mean", None),
            "mix_v_mean": getattr(attn, "last_mix_v_mean", None),
            "distance_bias_mean": getattr(attn, "last_distance_bias_mean", None),
            "distance_bias_abs_mean": getattr(attn, "last_distance_bias_abs_mean", None),
            "angle_q_sat_frac": getattr(attn, "last_angle_q_sat_frac", None),
            "angle_q_std": getattr(attn, "last_angle_q_std", None),
            "angle_k_std": getattr(attn, "last_angle_k_std", None),
            "angle_k_sat_frac": getattr(attn, "last_angle_k_sat_frac", None),
            "gate_k_sat_frac": getattr(attn, "last_gate_k_sat_frac", None),
            "gate_k_abs_mean": getattr(attn, "last_gate_k_abs_mean", None),
            "gate_v_abs_mean": getattr(attn, "last_gate_v_abs_mean", None),
            "gate_v_sat_frac": getattr(attn, "last_gate_v_sat_frac", None),
            "salience_bias_abs_mean": getattr(attn, "last_salience_bias_abs_mean", None),
            "salience_sat_frac": getattr(attn, "last_salience_sat_frac", None),
            "forget_bias_abs_mean": getattr(attn, "last_forget_bias_abs_mean", None),
            "forget_sat_frac": getattr(attn, "last_forget_sat_frac", None),
            "qtemp_mean": getattr(attn, "last_qtemp_mean", None),
            "qtemp_sat_frac": getattr(attn, "last_qtemp_sat_frac", None),
        }
        for k, v in mapping.items():
            if v is not None:
                vals[k].append(v.detach().float().mean())
    out: Dict[str, float] = {}
    for k, xs in vals.items():
        if xs:
            out[k] = float(torch.stack(xs).mean().detach().cpu().item())
    return out


@torch.no_grad()
def svd_statistics(model, max_layers: Optional[int] = None) -> Dict[str, float]:
    """Spectral (SVD) diagnostics of attention weights. CPU-only, master-only.

    Metrics (averaged over layers/heads):
      - phase_erank:     effective rank of W_phase_q/k per head
                         (erank = exp(entropy of normalized singular values);
                          1.0 = rank-1 collapse, head_dim/2 = full use of rotation subspace)
      - phase_top1:      share of top singular value in W_phase (concentration; 1.0 = rank-1)
      - qk_stable_rank:  stable rank ||A||_F^2 / s_max^2 of the Q and K blocks of c_attn
      - v_stable_rank:   same for the V block
      - gate_erank:      effective rank of W_gate_k/v (H x C matrices)

    Why: HLA claims to *decouple* retrieval from content transmission. If true,
    the spectra of Q/K blocks in the HLA run should diverge from base over
    training (retrieval subspace specializes) while V spectra stay comparable.
    Tracking eranks during training gives that evidence for free.
    """

    def _svals(mat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(mat.detach().float().cpu())

    def _erank(s: torch.Tensor, eps: float = 1e-12) -> float:
        total = float(s.sum())
        if not (total > 0):
            return 0.0
        p = (s / total).clamp_min(eps)
        return float(torch.exp(-(p * p.log()).sum()))

    def _stable_rank(s: torch.Tensor) -> float:
        if s.numel() == 0 or float(s[0]) == 0.0:
            return 0.0
        return float((s * s).sum() / (s[0] * s[0]))

    phase_eranks: list = []
    phase_top1: list = []
    qk_stable: list = []
    v_stable: list = []
    gate_eranks: list = []

    blocks = list(model.transformer.h)
    if max_layers is not None:
        blocks = blocks[: int(max_layers)]

    for block in blocks:
        attn = block.attn
        C = attn.n_embd

        # Phase matrices: (H, C, head_dim//2), per-head spectra.
        for W in (attn.W_phase_q, attn.W_phase_k):
            for h in range(attn.n_head):
                s = _svals(W[h])
                if float(s.sum()) > 0:
                    phase_eranks.append(_erank(s))
                    phase_top1.append(float(s[0] / s.sum()))

        # c_attn blocks: rows [0:C] = Q, [C:2C] = K, [2C:3C] = V.
        w = attn.c_attn.weight  # (3C, C)
        s_q = _svals(w[:C, :])
        s_k = _svals(w[C : 2 * C, :])
        s_v = _svals(w[2 * C :, :])
        qk_stable.append(0.5 * (_stable_rank(s_q) + _stable_rank(s_k)))
        v_stable.append(_stable_rank(s_v))

        # Gate matrices: (H, C) -> rank <= H.
        for G in (attn.W_gate_k.weight, attn.W_gate_v.weight):
            s = _svals(G)
            if float(s.sum()) > 0:
                gate_eranks.append(_erank(s))

    out: Dict[str, float] = {
        "qk_stable_rank": float(torch.tensor(qk_stable).mean()) if qk_stable else float("nan"),
        "v_stable_rank": float(torch.tensor(v_stable).mean()) if v_stable else float("nan"),
    }
    # Phase/gate spectra are all-zero at identity init: report NaN (not 0) so
    # plots clearly show "not yet active" instead of a fake rank-0 reading.
    out["phase_erank"] = float(torch.tensor(phase_eranks).mean()) if phase_eranks else float("nan")
    out["phase_top1"] = float(torch.tensor(phase_top1).mean()) if phase_top1 else float("nan")
    out["gate_erank"] = float(torch.tensor(gate_eranks).mean()) if gate_eranks else float("nan")
    return out


@torch.no_grad()
def evaluate_distractor_induction(
    model,
    device,
    seed: int = 42,
    batch_size: int = 32,
    n_distractors: int = 24,
) -> Dict[str, float]:
    """Induction under noise: A->B recall with repeated distractor pairs.

    Standard induction ([A][B] ... [A] -> [B]?) measures retrieval. This probe
    measures retrieval *under interference*: between the first [A][B] and the
    second [A] we insert `n_distractors` REPEATED spurious pairs [C_i][D_i]
    (repetition makes them attractive to induction heads - realistic noise,
    not just random filler).

    Returns:
      distractor_induction : P(correct B | query A) with distractors present
      distractor_margin    : P(B) - max_i P(D_i)  (positive = the true target
                             beats the strongest distractor)

    Why it exists: val loss can hide selective-attention gains. The salience
    and Laplace gating mechanisms specifically claim to SUPPRESS irrelevant
    keys; this is the direct behavioral readout of that claim. Compare
    base-vs-HLA trajectories of distractor_margin during training.
    """
    was_training = model.training
    model.eval()
    try:
        T = min(512, model.config.block_size)
        if model.config.vocab_size < 20000 or 8 * (n_distractors + 2) > T:
            return {"distractor_induction": float("nan"), "distractor_margin": float("nan")}

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(100, min(20000, model.config.vocab_size - 1),
                               (batch_size, T), generator=g)

        i = torch.arange(batch_size)
        tok_a = INDUCTION_TOK_A_OFFSET + i          # (B,)
        tok_b = INDUCTION_TOK_B_OFFSET + i

        # Layout: ... [A][B] ... {[C_j][D_j] x n, each pair repeated twice} ... [A] -> ?
        pos_a1 = T // 8
        tokens[:, pos_a1] = tok_a
        tokens[:, pos_a1 + 1] = tok_b

        # Distractor pairs: unique per batch row, repeated twice to form
        # competing induction patterns.
        span = T - pos_a1 - 4
        d_tokens = torch.randint(20000, min(40000, model.config.vocab_size - 1),
                                 (batch_size, n_distractors, 2), generator=g)
        d_positions = []
        for j in range(n_distractors):
            p1 = pos_a1 + 2 + (span * (2 * j) // (2 * n_distractors + 1))
            p2 = pos_a1 + 2 + (span * (2 * j + 1) // (2 * n_distractors + 1))
            for p in (p1, p2):
                if p + 1 < T - 2:
                    tokens[:, p] = d_tokens[:, j, 0]
                    tokens[:, p + 1] = d_tokens[:, j, 1]
            d_positions.append(j)

        pos_a2 = T - 2
        tokens[:, pos_a2] = tok_a

        logits, _ = model(tokens.to(device))
        if not bool(torch.isfinite(logits).all().detach().cpu().item()):
            return {"distractor_induction": float("nan"), "distractor_margin": float("nan")}
        probs = torch.softmax(_mask_padded_logits(model, logits[:, pos_a2, :].float()), dim=-1).cpu()  # (B, V)

        p_b = probs.gather(1, tok_b[:, None]).squeeze(1)                  # (B,)
        p_d = probs.gather(1, d_tokens[:, :, 1])                          # (B, n_distr): P(D_j)
        margin = (p_b - p_d.max(dim=1).values).mean()
        return {
            "distractor_induction": float(p_b.mean()),
            "distractor_margin": float(margin),
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def head_interference_statistics(
    model,
    max_layers: Optional[int] = None,
    topk: int = 8,
) -> Dict[str, float]:
    """Cross-head subspace interference, after Anthropic's Transformer Circuits
    (Elhage et al. 2021: composition/interference via W_OV and W_QK products).

    For every layer we build per-head subspaces (rank<=head_dim, top-`topk`
    singular directions in the residual stream R^C):

      write_i = column space of W_O[:, head i]           (what head i WRITES)
      qk_read_j = row space of [W_Q; W_K][head j]        (what head j's QK READS)
      v_read_j  = row space of W_V[head j]               (what head j's OV READS)

    Metrics (mean over layers; i != j pairs within a layer):

      qk_interference : mean overlap(write_i, qk_read_j)
          "how much what one head writes lands in another head's RETRIEVAL input"
          -> the noise floor for Q/K matching. HLA's claim predicts THIS drops
             vs base: Laplace gating suppresses irrelevant key content, phase
             rotation moves matching into rotated subspaces.
      ov_interference : mean overlap(write_i, v_read_j)
          "how much one head's output feeds another head's CONTENT input"
          -> legitimate composition channel (virtual attention heads / induction
             circuits). Should NOT collapse; if it did, we'd be destroying
             composition, not removing noise.
      qk_self_overlap / ov_self_overlap : i == j diagonal, for reference.
      qk_ov_separation : ov_interference - qk_interference (higher = cleaner
          decoupling of retrieval from content transmission; the paper's headline
          mechanistic number).

    Overlap metric: ||A^T B||_F^2 / min(dim_A, dim_B) in [0, 1] for orthonormal
    bases A, B (average squared cosine of principal angles).

    Cost: one SVD per head-matrix, O(L*H) small SVDs; run at svd_every cadence.
    """

    def _col_basis(M: torch.Tensor, k: int) -> torch.Tensor:
        # Basis for column space of M (C x hd), directions live in R^C.
        U, S, _ = torch.linalg.svd(M.detach().float().cpu(), full_matrices=False)
        r = int((1e-6 * float(S.max().clamp_min(1e-12)) < S).sum())
        return U[:, : min(k, r)].contiguous()

    def _row_basis(M: torch.Tensor, k: int) -> torch.Tensor:
        # Basis for row space of M (hd x C), directions live in R^C.
        _, S, Vh = torch.linalg.svd(M.detach().float().cpu(), full_matrices=False)
        r = int((1e-6 * float(S.max().clamp_min(1e-12)) < S).sum())
        return Vh[: min(k, r), :].T.contiguous()

    def _overlap(A: torch.Tensor, B: torch.Tensor) -> float:
        if A.numel() == 0 or B.numel() == 0:
            return float("nan")
        d = float(min(A.shape[1], B.shape[1]))
        if d <= 0:
            return float("nan")
        return float((A.T @ B).pow(2).sum() / d)

    blocks = list(model.transformer.h)
    if max_layers is not None:
        blocks = blocks[: int(max_layers)]

    qk_cross: list = []
    ov_cross: list = []
    qk_self: list = []
    ov_self: list = []

    for block in blocks:
        attn = block.attn
        C, H, hd = attn.n_embd, attn.n_head, attn.head_dim
        W = attn.c_attn.weight  # (3C, C): rows [0:C]=Q, [C:2C]=K, [2C:3C]=V
        Wo = attn.c_proj.weight  # (C, C): columns grouped by head

        writes, qk_reads, v_reads = [], [], []
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            writes.append(_col_basis(Wo[:, sl], topk))
            # QK read: what the residual stream must contain for this head to match.
            qk_reads.append(_row_basis(torch.cat([W[sl, :], W[C + h * hd : C + (h + 1) * hd, :]], dim=0), topk))
            v_reads.append(_row_basis(W[2 * C + h * hd : 2 * C + (h + 1) * hd, :], topk))

        for i in range(H):
            for j in range(H):
                oqk = _overlap(writes[i], qk_reads[j])
                oov = _overlap(writes[i], v_reads[j])
                if i == j:
                    qk_self.append(oqk)
                    ov_self.append(oov)
                else:
                    qk_cross.append(oqk)
                    ov_cross.append(oov)

    def _mean(xs: list) -> float:
        xs = [x for x in xs if x == x]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    qk_i = _mean(qk_cross)
    ov_i = _mean(ov_cross)
    return {
        "qk_interference": qk_i,
        "ov_interference": ov_i,
        "qk_self_overlap": _mean(qk_self),
        "ov_self_overlap": _mean(ov_self),
        "qk_ov_separation": (ov_i - qk_i) if (qk_i == qk_i and ov_i == ov_i) else float("nan"),
    }



@torch.no_grad()
def perturbation_bounds(model, device="cpu", seed: int = 42, batch_size: int = 2) -> Dict[str, float]:
    """Actual gate modulation vs the Theorem-4 envelope, per layer.

    Self-contained: runs its own diagnostics forward on deterministic random
    tokens (same pattern as measure_attention_entropy), then compares captured
    mix tensors against the analytic envelope.

    CORRECT envelope (matches model forward and test_theory):
        eff  = lam * min(alpha * range_base * (1 + range_flex), log_clip)
        theo = (1 - beta) + beta * exp(+-eff)
    range_flex is read from the model (config knob; 0.25 default -> the
    historical 1.25 factor). Hardcoding 1.25 here was a bug: with
    range_flex != 0.25 the probe would report a silently wrong envelope.
    (The clip alone is NOT the bound: in all shipped configs the pre-clip term
    alpha*range*(1+range_flex) binds first - see audit_config_values E3.)

    utilization ~ 0 => bounds loose / mechanism dormant;
    utilization -> 1 => model pressing the envelope (widen ranges).
    """
    import math as _m
    import torch.nn.functional as _F

    was_training = model.training
    # P1 fix (reviewer attack): remember the caller's diagnostics state and
    # RESTORE it afterwards instead of blindly disabling - probes must be
    # side-effect-free even w.r.t. instrumentation flags.
    prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
    prev_attn = bool(model.transformer.h[0].attn.capture_attention)
    model.eval()
    model.set_diagnostics(enabled=True)
    try:
        T = min(256, model.config.block_size)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T), generator=g).to(device)
        model(tokens)

        out: Dict[str, float] = {}
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            lam = float(attn.layer_gate_multiplier)
            if getattr(attn, "layer_dependent_gate", False) and getattr(attn, "learnable_layer_temp", False):
                depth = float(attn.layer_idx) / float(max(1, attn.n_layer))
                lam = 1.0 + depth * float(
                    _F.softplus(attn.W_layer_temp.detach().float())
                ) / _m.log(2.0)

            for side in ("k", "v"):
                mix = getattr(attn, f"last_mix_{side}", None)
                if mix is None:
                    continue
                mix = mix.detach().float()
                alpha = float(attn.laplace_alpha)
                rng = float(getattr(attn, f"laplace_range_{side}"))
                clip = float(getattr(attn, f"{side}_log_clip"))
                beta = float(getattr(attn, f"beta_{side}"))
                flex = 1.0 + float(getattr(attn, "range_flex", 0.25))
                eff = lam * min(alpha * rng * flex, clip)
                theo_hi = (1 - beta) + beta * _m.exp(eff)
                theo_lo = (1 - beta) + beta * _m.exp(-eff)
                mx, mn = float(mix.max()), float(mix.min())
                out[f"L{i:02d}_mix_{side}_max"] = mx
                out[f"L{i:02d}_mix_{side}_min"] = mn
                out[f"L{i:02d}_theo_mix_{side}_max"] = theo_hi
                out[f"L{i:02d}_theo_mix_{side}_min"] = theo_lo
                if theo_hi > 1.0 + 1e-12:
                    out[f"L{i:02d}_util_{side}_up"] = (mx - 1.0) / (theo_hi - 1.0)
                    out[f"L{i:02d}_util_{side}_down"] = (1.0 - mn) / (1.0 - theo_lo)
        return out
    finally:
        model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        model.train(was_training)


@torch.no_grad()
def per_layer_gate_analysis(model, device="cpu", seed: int = 42, batch_size: int = 2) -> Dict[str, float]:
    """Per-layer (not averaged) mechanism activity for the depth figure.

    Self-contained diagnostics forward, then one row of scalars per layer:
    does the learned depth profile match the 1 + l/L heuristic?
    """
    was_training = model.training
    prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
    prev_attn = bool(model.transformer.h[0].attn.capture_attention)
    model.eval()
    model.set_diagnostics(enabled=True)
    try:
        T = min(256, model.config.block_size)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T), generator=g).to(device)
        model(tokens)

        out: Dict[str, float] = {}
        fields = [
            ("last_gate_k_abs_mean", "gate_k_abs_mean"),
            ("last_gate_v_abs_mean", "gate_v_abs_mean"),
            ("last_gate_k_sat_frac", "gate_k_sat_frac"),
            ("last_gate_v_sat_frac", "gate_v_sat_frac"),
            ("last_angle_q_abs_mean", "angle_q_abs_mean"),
            ("last_angle_k_abs_mean", "angle_k_abs_mean"),
            ("last_mix_k_mean", "mix_k_mean"),
            ("last_mix_v_mean", "mix_v_mean"),
            ("last_salience_bias_abs_mean", "salience_bias_abs_mean"),
            ("last_forget_bias_abs_mean", "forget_bias_abs_mean"),
        ]
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            for src, dst in fields:
                v = getattr(attn, src, None)
                if v is not None:
                    out[f"L{i:02d}_{dst}"] = float(v.detach().float().mean())
            e = getattr(attn, "last_entropy", None)
            if e is not None:
                out[f"L{i:02d}_attn_entropy"] = float(e.detach().float().mean())
        return out
    finally:
        model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        model.train(was_training)


def mechanism_gradient_statistics(model) -> Dict[str, float]:
    """Read gradient norms captured by GPT.capture_mechanism_grad_norms().

    Theorem-5 verification during training: mechanism gradients must not
    vanish. Returns per-layer per-param norms plus summary mean/min over
    ACTIVE entries (inactive mechanisms record exact 0 and are excluded
    from the summary so a disabled arm does not fake a vanishing gradient).
    """
    out: Dict[str, float] = {}
    active: list = []
    for i, block in enumerate(model.transformer.h):
        norms = getattr(block.attn, "last_mechanism_grad_norms", None)
        if not norms:
            continue
        for name, t in norms.items():
            v = float(t.detach().cpu().item())
            out[f"L{i:02d}_grad_{name}"] = v
            if v > 0.0:
                active.append(v)
    if active:
        out["mech_grad_mean"] = float(sum(active) / len(active))
        out["mech_grad_min"] = float(min(active))
    return out



@torch.no_grad()
def attention_head_similarity(model, device="cpu", seed: int = 42,
                              batch_size: int = 2) -> Dict[str, float]:
    """ACTIVATION-space head redundancy (Nanda attack: weight-space overlap
    only bounds POSSIBLE interference; circuits live in activations).

    Runs a diagnostics forward with attention capture on deterministic random
    tokens, then for every layer computes the mean pairwise Jensen-Shannon
    divergence between the attention DISTRIBUTIONS of different heads on the
    same queries.

      head_js_mean ~ 0   => heads attend identically (redundant heads);
      head_js_mean large => heads specialize.

    The decoupling claim predicts HLA heads specialize MORE than base heads
    (phase gives each head its own matching geometry) - this probe is the
    activation-level counterpart of qk_interference and Figure material.
    JS is used instead of KL for symmetry and boundedness (<= ln 2).
    """
    was_training = model.training
    prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
    prev_attn = bool(model.transformer.h[0].attn.capture_attention)
    model.eval()
    model.set_diagnostics(enabled=True, capture_attention=True)
    try:
        T = min(128, model.config.block_size)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T), generator=g).to(device)
        model(tokens)

        out: Dict[str, float] = {}
        eps = 1e-9
        per_layer: list = []
        for i, block in enumerate(model.transformer.h):
            att = getattr(block.attn, "last_attn", None)
            if att is None:
                continue
            att = att.detach().float().clamp_min(eps)   # (B, H, T, T)
            B, H, Tq, Tk = att.shape
            js_vals = []
            for a in range(H):
                for b in range(a + 1, H):
                    p, q = att[:, a], att[:, b]
                    m = 0.5 * (p + q)
                    js = 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)
                    js_vals.append(js.mean())
            if js_vals:
                v = float(torch.stack(js_vals).mean())
                out[f"L{i:02d}_head_js"] = v
                per_layer.append(v)
        if per_layer:
            out["head_js_mean"] = float(sum(per_layer) / len(per_layer))
        return out
    finally:
        model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        model.train(was_training)



@torch.no_grad()
def mechanism_knockout(model, batch, targets, mechanisms=("phase", "gates", "salience", "distance", "forget", "qtemp")) -> Dict[str, float]:
    """Causal attribution by inference-time knockout (Nanda-style ablation).

    MEASUREMENT TOOL ONLY - never a deployment/generation mode. The mechanisms
    exist precisely to help at inference (e.g. long-context quality); this
    probe temporarily silences one on a HELD-OUT batch to MEASURE how much
    trained load it carries, then restores it bit-exactly (verified by test).
    Expected paper finding: ko_*_delta GROWS with context length for the
    long-range mechanisms (distance/forget/salience) - that growth curve IS
    the quantitative evidence for the long-context claim, which is exactly
    why one measures by silencing. delta > 0 => mechanism carries load;
    delta ~ 0 => decorative or redundantly encoded.
    Complements training-time ablation arms: arms answer "what if never
    trained with X"; knockout answers "what does X do in THIS trained model".
    """
    knobs = {
        "phase": "phase_mult",
        "gates": "laplace_alpha",
        "salience": "salience_alpha",
        "distance": "distance_laplace_alpha",
        "forget": "forget_alpha",
        "qtemp": "qtemp_alpha",
    }
    was_training = model.training
    model.eval()
    try:
        _, base_loss = model(batch, targets)
        out: Dict[str, float] = {"loss_full": float(base_loss)}
        for mech in mechanisms:
            attr = knobs[mech]
            saved = [getattr(blk.attn, attr) for blk in model.transformer.h]
            if all(v == 0.0 for v in saved):
                out[f"ko_{mech}_delta"] = 0.0  # mechanism not present
                continue
            for blk in model.transformer.h:
                setattr(blk.attn, attr, 0.0)
            _, ko_loss = model(batch, targets)
            for blk, v in zip(model.transformer.h, saved, strict=True):
                setattr(blk.attn, attr, v)
            out[f"ko_{mech}_delta"] = float(ko_loss) - float(base_loss)
        return out
    finally:
        model.train(was_training)


@torch.no_grad()
def positional_recall_curve(model, device="cpu", seed: int = 42, batch_size: int = 8,
                            positions=(0.1, 0.3, 0.5, 0.7, 0.9)) -> Dict[str, float]:
    """Direct Lost-in-the-Middle measurement (Liu et al. 2023, adapted to our
    induction framework): plant a needle pair [A][B] at a controlled DEPTH
    FRACTION of the context, query with the second [A] at the very end, and
    read P(B). Sweeping the depth yields the positional recall curve whose
    U-shape (edges > middle) is the LITM signature.

    Reports P(B) per depth (pos_10 ... pos_90) plus two scalars:
      - litm_middle_drop = mean(edge positions) - min(middle positions)
        (0 for a flat curve; positive = classic LITM sag);
      - litm_worst_frac  = worst-middle / best-edge ratio (1.0 = no sag).
    The paper claim this feeds: HLA's content-conditioned distance/salience
    mechanisms should FLATTEN the curve vs the sterile base twin - measured,
    not asserted. NaN for tiny vocabs (needs the induction token block).
    """
    was_training = model.training
    model.eval()
    try:
        T = min(512, model.config.block_size)
        if T < 32 or model.config.vocab_size < 46000 + batch_size:
            return {"litm_middle_drop": float("nan"), "litm_worst_frac": float("nan")}
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        out: Dict[str, float] = {}
        vals = []
        for frac in positions:
            tokens = torch.randint(100, min(20000, model.config.vocab_size - 1),
                                   (batch_size, T), generator=g).to(device)
            pos_a1 = max(1, min(int(frac * (T - 4)), T - 4))
            targets = []
            for i in range(batch_size):
                a, b = 45000 + i, 46000 + i
                tokens[i, pos_a1] = a
                tokens[i, pos_a1 + 1] = b
                tokens[i, T - 2] = a           # query near the end
                targets.append(b)
            logits, _ = model(tokens)
            probs = torch.softmax(_mask_padded_logits(model, logits[:, T - 2, :].float()), dim=-1)
            t = torch.tensor(targets, device=device)
            pb = float(probs.gather(1, t[:, None]).mean().detach().cpu())
            out[f"pos_{int(frac * 100):02d}"] = pb
            vals.append((frac, pb))
        edges = [v for f, v in vals if f <= 0.15 or f >= 0.85]
        middle = [v for f, v in vals if 0.25 <= f <= 0.75]
        if edges and middle:
            out["litm_middle_drop"] = float(sum(edges) / len(edges) - min(middle))
            best_edge = max(edges)
            out["litm_worst_frac"] = float(min(middle) / best_edge) if best_edge > 0 else float("nan")
        return out
    finally:
        model.train(was_training)


@torch.no_grad()
def gate_redundancy_statistics(model, device="cpu", seed: int = 42, batch_size: int = 2) -> Dict[str, float]:
    """H3/M6 (round 7): do the four content gates (K, V, salience, distance)
    learn the same function of x, or four different ones?

    MEASUREMENT, deliberately not a regularizer: adding an orthogonality/
    entropy auxiliary loss would (a) break the sterility invariant "base and
    HLA optimize the SAME objective", (b) add tunable loss weights - an
    un-preregistered hyperparameter axis, and (c) presume the answer. If
    trained gates correlate near +-1, the honest conclusion is to MERGE
    mechanisms in v6, not to force them apart with a penalty.

    Reports, per layer (averaged over heads), pairwise Pearson correlation of
    per-token gate activations across all token positions, plus the mean
    absolute off-diagonal correlation as a single redundancy scalar:
      gate_corr_kv, gate_corr_ksal, gate_corr_kd, gate_corr_vsal,
      gate_corr_vd, gate_corr_sald, gate_redundancy_mean_abs.
    ~0 => gates specialize; |corr| -> 1 => redundant parameterization.
    At identity init every gate is exactly 0 => correlations are undefined;
    reported as NaN (deliberately, like the SVD metrics)."""
    was_training = model.training
    prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
    prev_attn = bool(model.transformer.h[0].attn.capture_attention)
    model.eval()
    model.set_diagnostics(enabled=True)
    try:
        T = min(256, model.config.block_size)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T), generator=g).to(device)
        model(tokens)

        pair_sums: Dict[str, list] = {}
        for block in model.transformer.h:
            attn = block.attn
            gates = {
                "k": getattr(attn, "last_gate_k", None),
                "v": getattr(attn, "last_gate_v", None),
                "sal": getattr(attn, "last_gate_sal", None),
                "d": getattr(attn, "last_gate_d", None),
            }
            flat = {}
            for name, t in gates.items():
                if t is None:
                    continue
                v = t.detach().float().reshape(-1)
                if float(v.std()) > 1e-8:  # constant/zero gate -> corr undefined
                    flat[name] = v
            names = sorted(flat)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = flat[names[i]], flat[names[j]]
                    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
                    pair_sums.setdefault(f"gate_corr_{names[i]}{names[j]}", []).append(corr)

        out: Dict[str, float] = {}
        all_abs: list = []
        for key, vals in pair_sums.items():
            out[key] = float(sum(vals) / len(vals))
            all_abs.extend(abs(v) for v in vals)
        out["gate_redundancy_mean_abs"] = (
            float(sum(all_abs) / len(all_abs)) if all_abs else float("nan")
        )
        return out
    finally:
        model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        model.train(was_training)


@torch.no_grad()
def prefix_matching_score(model, device="cpu", seed: int = 42, batch_size: int = 4) -> Dict[str, float]:
    """Canonical induction-head detector (Olsson et al. 2022): attention from
    the second [A] back to the token AFTER the first [A].

    Our evaluate_induction measures BEHAVIOR (P(B) at the output); this
    measures the MECHANISM (attention pattern), per head. A head with high
    prefix-matching is an induction head structurally, regardless of whether
    the logit lands. Returns per-layer max/mean over heads + global max -
    the standard way to report emergence of induction circuitry.
    """
    was_training = model.training
    prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
    prev_attn = bool(model.transformer.h[0].attn.capture_attention)
    model.eval()
    model.set_diagnostics(enabled=True, capture_attention=True)
    try:
        T = min(256, model.config.block_size)
        if T < 16 or model.config.vocab_size < 20000:
            return {}
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(100, min(20000, model.config.vocab_size - 1),
                               (batch_size, T), generator=g)
        i = torch.arange(batch_size)
        pos_a1, pos_a2 = T // 3, (2 * T) // 3
        tokens[:, pos_a1] = INDUCTION_TOK_A_OFFSET + i
        tokens[:, pos_a2] = INDUCTION_TOK_A_OFFSET + i
        model(tokens.to(device))

        out: Dict[str, float] = {}
        gmax = 0.0
        for li, block in enumerate(model.transformer.h):
            att = getattr(block.attn, "last_attn", None)
            if att is None:
                continue
            # attention from query pos_a2 to key pos_a1+1 ("token after first A")
            score = att[:, :, pos_a2, pos_a1 + 1].float().mean(dim=0)  # (H,)
            out[f"L{li:02d}_prefix_match_max"] = float(score.max())
            out[f"L{li:02d}_prefix_match_mean"] = float(score.mean())
            # Per-head census (mech-interp reviewer ask): WHICH heads become
            # induction heads, not just how strong the strongest one is.
            # Feeds the base-vs-HLA head-census figure.
            for hi in range(score.shape[0]):
                out[f"L{li:02d}_H{hi:02d}_prefix_match"] = float(score[hi])
            gmax = max(gmax, float(score.max()))
        out["prefix_match_global_max"] = gmax
        return out
    finally:
        model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        model.train(was_training)


@torch.no_grad()
def attention_needle_snr(model, device="cpu", seed: int = 42,
                         batch_size: int = 4) -> Dict[str, float]:
    """Attention signal-to-noise on the induction task (Diff-Transformer-style
    Table-3 readout, adapted to our probe framework).

    Plant [A][B] in random filler, query with the second [A] at T-2, capture
    attention (diagnostics forward), and read WHERE the final-layer heads look
    from the query position:

      snr_needle = attn_mass(needle pair) / attn_mass(mean filler token)

    ~1 at random init (no preference, calibration-tested); >>1 after training
    means retrieval concentrates on the target - the direct activation-level
    quantification of the "suppress irrelevant keys" claim (salience/gates).
    Reported per last layer (retrieval heads live late) plus best over layers,
    so the paper figure does not depend on a single-layer choice.
    NaN for tiny vocabs (needs the induction token block), like the probes.
    """
    was_training = model.training
    prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
    prev_attn = bool(model.transformer.h[0].attn.capture_attention)
    model.eval()
    model.set_diagnostics(enabled=True, capture_attention=True)
    try:
        T = min(256, model.config.block_size)
        if T < 16 or model.config.vocab_size < INDUCTION_TOK_B_OFFSET + batch_size:
            return {"snr_needle_last": float("nan"), "snr_needle_best": float("nan")}
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        lo = INDUCTION_TOK_B_OFFSET + batch_size
        tokens = torch.randint(lo, model.config.vocab_size,
                               (batch_size, T), generator=g)
        i = torch.arange(batch_size)
        pos_ab = T // 3
        tokens[:, pos_ab] = INDUCTION_TOK_A_OFFSET + i
        tokens[:, pos_ab + 1] = INDUCTION_TOK_B_OFFSET + i
        q_pos = T - 2
        tokens[:, q_pos] = INDUCTION_TOK_A_OFFSET + i
        model(tokens.to(device))

        eps = 1e-12
        per_layer = []
        for block in model.transformer.h:
            att = getattr(block.attn, "last_attn", None)
            if att is None:
                continue
            att = att.detach().float()               # (B, H, Tq, Tk)
            row = att[:, :, q_pos, :]                # (B, H, Tk)
            needle = row[:, :, pos_ab:pos_ab + 2].sum(-1) / 2.0
            mask = torch.ones(row.shape[-1], dtype=torch.bool)
            mask[pos_ab:pos_ab + 2] = False
            mask[q_pos:] = False                     # exclude self/future
            filler = row[:, :, mask].mean(-1)
            snr = (needle / filler.clamp_min(eps)).mean()
            per_layer.append(float(snr))
        if not per_layer:
            return {"snr_needle_last": float("nan"), "snr_needle_best": float("nan")}
        return {"snr_needle_last": per_layer[-1],
                "snr_needle_best": float(max(per_layer))}
    finally:
        model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        model.train(was_training)


@torch.no_grad()
def per_position_loss_curve(model, tokens: torch.Tensor, device="cpu",
                            n_bins: int = 8, batch_size: int = 4) -> Dict[str, float]:
    """Per-position loss on REAL data (the standard long-context metric).

    The synthetic probes (induction/distractor/LITM) are author-created;
    the field's standard evidence (FoX Fig. 1, Liu et al.) is the loss as
    a function of POSITION in the context on natural text: a model that
    truly uses long context keeps improving at late positions; positional
    pathologies show up as bumps mid-context.

    Args:
      tokens: (N, T+1) int tensor of REAL validation sequences (inputs are
        tokens[:, :-1], targets tokens[:, 1:]) - pass a slice of the val
        set; the function does NOT generate synthetic data by design.
    Returns:
      posloss_bin_00..: mean loss per position bin (bin 0 = earliest);
      posloss_early_late_ratio: bin0_loss / lastbin_loss (>1 = late
        positions genuinely benefit from context; ~1 = context unused);
      posloss_mid_bump: max(middle bins) - interp(early, late) - positive
        values flag a mid-context pathology on real data (the honest,
        non-synthetic counterpart of litm_middle_drop).
    """
    was_training = model.training
    model.eval()
    try:
        N, T1 = tokens.shape
        T = min(T1 - 1, model.config.block_size)
        if N < 1 or n_bins * 4 > T:
            return {"posloss_early_late_ratio": float("nan"),
                    "posloss_mid_bump": float("nan")}
        x = tokens[:, :T].to(device)
        y = tokens[:, 1:T + 1].to(device)
        loss_sum = torch.zeros(T)
        n_rows = 0
        for s0 in range(0, N, batch_size):
            xb, yb = x[s0:s0 + batch_size], y[s0:s0 + batch_size]
            logits, _ = model(xb)
            logits = _mask_padded_logits(model, logits.float())
            ls = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1), reduction="none").reshape(yb.shape)
            # Finding #15: summing per-batch MEANS overweights a short final
            # batch (its rows counted at 1/len(last) instead of 1/N; measured
            # bin drift up to 1.2e-2 at N=5,bs=4). Row-sum + row-count keeps
            # every sequence at exactly weight 1/N.
            loss_sum += ls.sum(0).detach().cpu()
            n_rows += int(yb.shape[0])
        per_pos = loss_sum / max(n_rows, 1)
        edges = torch.linspace(0, T, n_bins + 1, dtype=torch.long)
        bins = [float(per_pos[edges[i]:edges[i + 1]].mean())
                for i in range(n_bins)]
        out = {f"posloss_bin_{i:02d}": b for i, b in enumerate(bins)}
        early, late = bins[0], bins[-1]
        out["posloss_early_late_ratio"] = early / late if late > 0 else float("nan")
        mid = bins[1:-1]
        interp = [(early + (late - early) * (i + 1) / (n_bins - 1))
                  for i in range(len(mid))]
        out["posloss_mid_bump"] = float(max(m - e for m, e in zip(mid, interp, strict=True)))
        return out
    finally:
        model.train(was_training)


@torch.no_grad()
def attention_sink_stats(model, device="cpu", seed: int = 42,
                         batch_size: int = 2) -> Dict[str, float]:
    """Attention-sink mass (Xiao et al., StreamingLLM) - the field-standard
    positional-pathology readout: what fraction of attention lands on the
    FIRST tokens regardless of content.

      sink_mass_first  : mean attention to position 0 from queries > 16
      sink_mass_first4 : same for positions 0-3
      sink_top_layer   : max per-layer sink_mass_first (sinks concentrate)

    Paper reading: HLA's salience/distance channels claim to REDUCE the
    model's need for a garbage-collector sink (attention has a content-
    conditioned place to put mass). base-vs-HLA sink_mass on the twins is
    the direct test - and connects our story to the attention-sink line
    of work reviewers know. ~1/T at random init (no sink yet; measured),
    grows with training in vanilla transformers.
    """
    was_training = model.training
    prev_diag = bool(model.transformer.h[0].attn.capture_diagnostics)
    prev_attn = bool(model.transformer.h[0].attn.capture_attention)
    model.eval()
    model.set_diagnostics(enabled=True, capture_attention=True)
    try:
        T = min(256, model.config.block_size)
        if T < 32:
            return {"sink_mass_first": float("nan"),
                    "sink_mass_first4": float("nan"),
                    "sink_top_layer": float("nan")}
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T),
                               generator=g).to(device)
        model(tokens)
        per_layer_first = []
        per_layer_first4 = []
        for block in model.transformer.h:
            att = getattr(block.attn, "last_attn", None)
            if att is None:
                continue
            att = att.detach().float()          # (B, H, Tq, Tk)
            rows = att[:, :, 16:, :]            # queries past the sink zone
            per_layer_first.append(float(rows[..., 0].mean()))
            per_layer_first4.append(float(rows[..., :4].sum(-1).mean()))
        if not per_layer_first:
            return {"sink_mass_first": float("nan"),
                    "sink_mass_first4": float("nan"),
                    "sink_top_layer": float("nan")}
        return {"sink_mass_first": float(sum(per_layer_first) / len(per_layer_first)),
                "sink_mass_first4": float(sum(per_layer_first4) / len(per_layer_first4)),
                "sink_top_layer": float(max(per_layer_first))}
    finally:
        model.set_diagnostics(enabled=prev_diag, capture_attention=prev_attn)
        model.train(was_training)


@torch.no_grad()
def activation_outlier_stats(model, device="cpu", seed: int = 42,
                             batch_size: int = 2) -> Dict[str, float]:
    """Residual-stream activation outliers (Diff Transformer Sec. 3.7,
    Bondarenko et al.) - excess kurtosis and max/rms ratio of the residual
    stream at the last block.

    Paper reading: architectures that clean attention noise empirically
    reduce activation outliers (Diff Tr. Table: quantization robustness).
    If HLA's twins diverge here, it is a free secondary claim reviewers
    from the quantization community will recognize. Gaussian baseline:
    excess kurtosis ~0 (measured ~0.05 at random init), max/rms ~4-5 for
    n_embd-sized samples.
    """
    was_training = model.training
    model.eval()
    try:
        T = min(256, model.config.block_size)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, T),
                               generator=g).to(device)
        feats = {}

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            feats["h"] = h.detach().float()

        hd = model.transformer.h[-1].register_forward_hook(hook)
        try:
            model(tokens)
        finally:
            hd.remove()
        h = feats["h"].reshape(-1)
        mu, sd = h.mean(), h.std().clamp_min(1e-12)
        z = (h - mu) / sd
        kurt = float((z ** 4).mean()) - 3.0
        max_over_rms = float(h.abs().max() / h.pow(2).mean().sqrt().clamp_min(1e-12))
        return {"act_excess_kurtosis": kurt,
                "act_max_over_rms": max_over_rms}
    finally:
        model.train(was_training)


__all__ = [
    "evaluate_induction",
    "evaluate_distractor_induction",
    "measure_attention_entropy",
    "phase_statistics",
    "hla_statistics",
    "svd_statistics",
    "head_interference_statistics",
    "depth_profile_statistics",
    "perturbation_bounds",
    "per_layer_gate_analysis",
    "mechanism_gradient_statistics",
    "attention_head_similarity",
    "mechanism_knockout",
    "gate_redundancy_statistics",
    "positional_recall_curve",
    "attention_needle_snr",
    "per_position_loss_curve",
    "attention_sink_stats",
    "activation_outlier_stats",
    "prefix_matching_score",
]
