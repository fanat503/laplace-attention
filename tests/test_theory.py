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



"""Numerical verification of every theorem in docs/THEORY.md.

Each test cites its theorem. If a refactor breaks a mathematical guarantee,
this file is where CI catches it.
"""
from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402


def cfg(**kw):
    base = dict(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=32,
                gradient_checkpointing=False)
    base.update(kw)
    return GPTConfig(**base)


ALL_ON = dict(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
              use_salience_bias=True, salience_alpha=1.0,
              use_distance_laplace=True, distance_laplace_alpha=0.5,
              use_forget_gate=True, forget_alpha=1.0,
              use_rope=True, use_wpe=False)
ALL_OFF = dict(phase_mult=0.0, use_laplace=True, laplace_alpha=0.0,
               use_salience_bias=True, salience_alpha=0.0,
               use_distance_laplace=True, distance_laplace_alpha=0.0,
               use_forget_gate=True, forget_alpha=0.0,
               use_rope=True, use_wpe=False)


class TestTheorem1Identity:
    """HLA(x; backbone, 0) == GPT(x; backbone), exactly."""

    def test_bit_exact_all_mechanisms(self):
        torch.manual_seed(0)
        base = GPT(cfg(**ALL_OFF)).eval()
        hla = GPT(cfg(**ALL_ON)).eval()
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        for _ in range(3):
            x = torch.randint(0, 256, (2, 32))
            lb, _ = base(x)
            lh, _ = hla(x)
            assert torch.equal(lb, lh)


class TestTheorem2StrictInclusion:
    """F_base is a strict subset of F_HLA: the salience witness."""

    def test_base_with_zero_wq_is_uniform(self):
        model = GPT(cfg(use_salience_bias=True, salience_alpha=1.0)).eval()
        attn = model.transformer.h[0].attn
        with torch.no_grad():
            attn.c_attn.weight[: attn.n_embd, :].zero_()  # W_Q = 0
        model.set_diagnostics(enabled=True, capture_attention=True)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        row = model.transformer.h[0].attn.last_attn[0, 0][10, :11]
        assert float(row.max() - row.min()) < 1e-6, "W_Q=0 must give uniform attention in base"

    def test_salience_escapes_base_class(self):
        model = GPT(cfg(use_salience_bias=True, salience_alpha=1.0, salience_range=2.0)).eval()
        attn = model.transformer.h[0].attn
        with torch.no_grad():
            attn.c_attn.weight[: attn.n_embd, :].zero_()
            attn.W_gate_sal.weight.normal_(0, 1.0)
        model.set_diagnostics(enabled=True, capture_attention=True)
        x = torch.randint(0, 256, (1, 16))
        model(x)
        row = model.transformer.h[0].attn.last_attn[0, 0][10, :11]
        assert float(row.max() - row.min()) > 0.01, "salience must produce key-dependent attention at W_Q=0"


class TestTheorem3Isometry:
    def test_norm_preservation_random_angles(self):
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        for _ in range(3):
            z = torch.randn(2, 4, 8, attn.head_dim)
            th = torch.randn(2, 4, 8, attn.head_dim // 2) * 3.0
            r = attn._rotate_pairwise(z, torch.cos(th), torch.sin(th))
            assert torch.allclose(z.norm(dim=-1), r.norm(dim=-1), atol=1e-5)

    def test_group_action(self):
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        z = torch.randn(1, 2, 4, attn.head_dim)
        a = torch.randn(1, 2, 4, attn.head_dim // 2)
        b = torch.randn(1, 2, 4, attn.head_dim // 2)
        ab = attn._rotate_pairwise(attn._rotate_pairwise(z, torch.cos(a), torch.sin(a)),
                                   torch.cos(b), torch.sin(b))
        direct = attn._rotate_pairwise(z, torch.cos(a + b), torch.sin(a + b))
        assert torch.allclose(ab, direct, atol=1e-5)


class TestTheorem4Envelopes:
    """Score perturbation bounded regardless of weights/inputs."""

    def test_total_bias_bounded_at_saturation(self):
        c = cfg(**ALL_ON)
        model = GPT(c).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_sal.weight.fill_(1000.0)
                blk.attn.W_gate_f.weight.fill_(1000.0)
                blk.attn.W_gate_k.weight.fill_(1000.0)
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (1, 64))
        logits, _ = model(x)
        assert torch.isfinite(logits).all()
        lm = 1.0  # layer_dependent_gate off in this cfg
        bound = c.salience_clip + c.distance_laplace_clip * lm + c.forget_clip
        for blk in model.transformer.h:
            total = (float(blk.attn.last_salience_bias_abs_mean)
                     + float(blk.attn.last_distance_bias_abs_mean)
                     + float(blk.attn.last_forget_bias_abs_mean))
            assert total <= bound + 1e-4

    def test_mix_envelope_exact_endpoints(self):
        c = cfg(**ALL_ON)
        model = GPT(c).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)   # tanh -> sign(sum x): +-1
                blk.attn.W_range_k.fill_(1000.0)          # range at +25%
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (1, 32))
        model(x)
        eff = min(c.laplace_alpha * c.laplace_range_k * 1.25, c.k_log_clip)
        expected_hi = (1 - c.beta_k) + c.beta_k * math.exp(eff)
        expected_lo = (1 - c.beta_k) + c.beta_k * math.exp(-eff)
        # gate = tanh(1000 * sum(x)) = exactly +1 or -1 per position (sign of
        # the input projection), so the mix tensor must sit exactly at the two
        # envelope endpoints - and never beyond.
        mix = model.transformer.h[0].attn.last_mix_k.float()
        assert abs(float(mix.max()) - expected_hi) < 1e-3
        assert abs(float(mix.min()) - expected_lo) < 1e-3


class TestTheorem5FirstOrderSignal:
    def test_mechanism_gradients_nonzero_at_identity(self):
        torch.manual_seed(3)
        model = GPT(cfg(**ALL_ON))
        model.train()
        x = torch.randint(0, 256, (4, 32))
        _, loss = model(x, x)
        loss.backward()
        mech_sq = 0.0
        for n, p in model.named_parameters():
            if p.grad is not None and any(m in n for m in ("W_phase", "W_gate", "W_range")):
                mech_sq += float(p.grad.pow(2).sum())
        assert mech_sq > 0.0, "identity point must not be a saddle for the mechanisms"

    def test_tanh_derivative_at_zero_is_one(self):
        """The reason Theorem 5 works: gradient passes tanh(0) undamped."""
        z = torch.zeros(5, requires_grad=True)
        torch.tanh(z).sum().backward()
        assert torch.allclose(z.grad, torch.ones(5))


class TestTheorem6GaugeInvariance:
    """LeCun attack: what IS the geometry? Answer: score depends only on
    relative phase phi - theta; common shifts are unobservable (gauge)."""

    def test_common_phase_shift_cancels(self):
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        q = torch.randn(1, 2, 4, attn.head_dim)
        k = torch.randn(1, 2, 4, attn.head_dim)
        th = torch.rand(1, 2, 4, attn.head_dim // 2) * 2.0
        ph = torch.rand(1, 2, 4, attn.head_dim // 2) * 2.0
        c = torch.rand(1) * 3.14

        def score(t, p):
            qr = attn._rotate_pairwise(q, torch.cos(t), torch.sin(t))
            kr = attn._rotate_pairwise(k, torch.cos(p), torch.sin(p))
            return (qr * kr).sum(-1)

        s1 = score(th, ph)
        s2 = score(th + c, ph + c)
        assert torch.allclose(s1, s2, atol=1e-5), "gauge invariance violated"

    def test_relative_phase_form(self):
        """R(th)q . R(ph)k == q . R(ph-th)k - the canonical relative form."""
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        q = torch.randn(1, 2, 4, attn.head_dim)
        k = torch.randn(1, 2, 4, attn.head_dim)
        th = torch.rand(1, 2, 4, attn.head_dim // 2)
        ph = torch.rand(1, 2, 4, attn.head_dim // 2)
        qr = attn._rotate_pairwise(q, torch.cos(th), torch.sin(th))
        kr = attn._rotate_pairwise(k, torch.cos(ph), torch.sin(ph))
        lhs = (qr * kr).sum(-1)
        krel = attn._rotate_pairwise(k, torch.cos(ph - th), torch.sin(ph - th))
        rhs = (q * krel).sum(-1)
        assert torch.allclose(lhs, rhs, atol=1e-5)


class TestTheorem1bGradientIdentity:
    """Sutskever attack: identity must hold for GRADIENTS too - the first
    optimizer step of HLA moves the backbone exactly as base does."""

    def test_backbone_gradients_bit_identical(self):
        torch.manual_seed(0)
        base = GPT(cfg(**ALL_OFF))
        hla = GPT(cfg(**ALL_ON))
        hla.load_state_dict(base.state_dict(), strict=True)
        hla.reset_hla_identity()
        x = torch.randint(0, 256, (2, 32))
        _, lb = base(x, x)
        lb.backward()
        _, lh = hla(x, x)
        lh.backward()
        for (nb, pb), (nh, ph) in zip(base.named_parameters(), hla.named_parameters()):
            if pb.grad is None:
                continue
            assert torch.equal(pb.grad, ph.grad), f"gradient mismatch at {nb}"


class TestTheorem7PositionShift:
    """Phase == content-conditioned RoPE position displacement (per plane)."""

    def test_rope_phase_composition_is_single_rotation(self):
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        q = torch.randn(1, 2, 8, attn.head_dim)
        pos = torch.rand(1, 1, 8, attn.head_dim // 2) * 2.0     # RoPE angles
        theta = torch.rand(1, 2, 8, attn.head_dim // 2) * 2.0   # content phase
        seq = attn._rotate_pairwise(
            attn._rotate_pairwise(q, torch.cos(pos), torch.sin(pos)),
            torch.cos(theta), torch.sin(theta))
        one = attn._rotate_pairwise(q, torch.cos(pos + theta), torch.sin(pos + theta))
        assert torch.allclose(seq, one, atol=1e-5), "RoPE∘Phase must equal R(pos+theta)"

    def test_phase_equals_shifted_rope_position(self):
        """Explicit displacement form: R_phase(theta)R_rope(w*p) q == R_rope(w*(p+delta)) q
        with delta = theta/w - the positional reading of the mechanism."""
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        q = torch.randn(1, 2, 8, attn.head_dim)
        w = torch.rand(attn.head_dim // 2) * 0.5 + 0.1          # per-plane freqs
        p = 7.0
        theta = torch.rand(attn.head_dim // 2)
        delta = theta / w
        a = attn._rotate_pairwise(
            attn._rotate_pairwise(q, torch.cos(w * p).expand(1, 2, 8, -1), torch.sin(w * p).expand(1, 2, 8, -1)),
            torch.cos(theta).expand(1, 2, 8, -1), torch.sin(theta).expand(1, 2, 8, -1))
        b = attn._rotate_pairwise(
            q, torch.cos(w * (p + delta)).expand(1, 2, 8, -1), torch.sin(w * (p + delta)).expand(1, 2, 8, -1))
        assert torch.allclose(a, b, atol=1e-4)


class TestKVCacheCompatibility:
    """Sutskever attack: incremental decoding viability. Every key-side
    quantity must depend only on the prefix <= j => cacheable."""

    def test_key_side_quantities_prefix_stable(self):
        c = cfg(**ALL_ON)
        model = GPT(c).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 0.3)
                blk.attn.W_gate_sal.weight.normal_(0, 0.3)
                blk.attn.W_gate_f.weight.normal_(0, 0.3)
                blk.attn.W_phase_k.normal_(0, 0.3)
        model.set_diagnostics(enabled=True)
        x = torch.randint(0, 256, (1, 32))
        model(x[:, :20])
        mix_prefix = model.transformer.h[0].attn.last_mix_k[:, :, :20].clone()
        model(x)
        mix_full = model.transformer.h[0].attn.last_mix_k[:, :, :20]
        assert torch.allclose(mix_prefix, mix_full, atol=1e-6), \
            "K-gate must be prefix-causal (KV-cache compatible)"

    def test_logits_prefix_stable(self):
        """The decisive cache test: logits at position t computed from a
        prefix equal logits at t from the full sequence.

        HARDENED (round 6): randomize EVERY mechanism weight. The original
        version left W_gate_d at identity zero - which hid a real prefix
        -stability bug: the distance bias normalized by the CURRENT batch
        length (T-1) instead of the model constant (block_size-1), so the
        same (i,j) pair got a different bias depending on batch length
        (full-vs-prefix logit diff ~4e-3 once W_gate_d was nonzero).
        A probe is only as good as the weights it excites."""
        model = GPT(cfg(**ALL_ON, use_qtemp=True, qtemp_alpha=1.0)).eval()
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.2)
                blk.attn.W_phase_k.normal_(0, 0.2)
                blk.attn.W_gate_k.weight.normal_(0, 0.3)
                blk.attn.W_gate_v.weight.normal_(0, 0.3)
                blk.attn.W_gate_sal.weight.normal_(0, 0.3)
                blk.attn.W_gate_f.weight.normal_(0, 0.2)
                blk.attn.W_gate_d.weight.normal_(0, 0.5)   # <- the one that hid the bug
                blk.attn.W_qtemp.weight.normal_(0, 0.3)
        x = torch.randint(0, 256, (1, 32))
        l_prefix, _ = model(x[:, :20])
        l_full, _ = model(x)
        assert torch.allclose(l_prefix[0, 19], l_full[0, 19], atol=1e-5), (
            "prefix logits must equal full-sequence logits at the same "
            "position for EVERY active mechanism (KV-cache correctness)")


class TestTheorem8PairingEquivalence:
    """chunk (x_i, x_{i+d/2}) and interleaved (x_2i, x_2i+1) pairings are
    isomorphic via a fixed coordinate permutation - implementation detail,
    not a modeling choice."""

    def test_chunk_equals_permuted_interleaved(self):
        model = GPT(cfg(**ALL_ON)).eval()
        attn = model.transformer.h[0].attn
        hd = attn.head_dim
        x = torch.randn(2, 3, 5, hd)
        ang = torch.rand(2, 3, 5, hd // 2)

        def rot_inter(z, cos, sin):
            z1, z2 = z[..., 0::2], z[..., 1::2]
            out = torch.empty_like(z)
            out[..., 0::2] = z1 * cos - z2 * sin
            out[..., 1::2] = z1 * sin + z2 * cos
            return out

        P = torch.empty(hd, dtype=torch.long)
        P[0::2] = torch.arange(hd // 2)
        P[1::2] = torch.arange(hd // 2) + hd // 2
        r_chunk = attn._rotate_pairwise(x, torch.cos(ang), torch.sin(ang))
        r_inter = rot_inter(x[..., P], torch.cos(ang), torch.sin(ang))
        assert torch.equal(r_chunk, r_inter[..., torch.argsort(P)])


class TestRelatedMechanismSection:
    """THEORY 4b: the GLA / Selective-RoPE / Bondarenko differentiation must
    stay in the doc - reviewers of the linear-attention line ask this first,
    and the franken experiment's boundary claim depends on the decomposition
    argument written there."""

    def test_neighbors_are_differentiated(self):
        t = open(os.path.join(ROOT, "docs", "THEORY.md"), encoding="utf-8").read()
        assert "## 4b." in t
        for needle in ("Gated DeltaNet", "Selective/adaptive-RoPE",
                       "Bondarenko", "Corollary 7.1", "franken"):
            assert needle in t, f"THEORY 4b lost its {needle} differentiation"
        # the load-bearing claim: decomposition (pre- vs post-softmax gates)
        assert "pre-softmax" in t and "post-softmax" in t
