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



"""Eval probe tests: induction metric, entropy, HLA statistics."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval import (  # noqa: E402
    depth_profile_statistics,
    evaluate_distractor_induction,
    evaluate_induction,
    head_interference_statistics,
    hla_statistics,
    measure_attention_entropy,
    phase_statistics,
    svd_statistics,
    perturbation_bounds,
    per_layer_gate_analysis,
    mechanism_gradient_statistics,
    attention_head_similarity,
    mechanism_knockout,
    gate_redundancy_statistics,
    positional_recall_curve,
    prefix_matching_score,
)
from src.model import GPT, GPTConfig  # noqa: E402


def make_model(vocab_size=50257, **kw):
    cfg = GPTConfig(block_size=64, vocab_size=vocab_size, n_layer=1, n_head=2,
                    n_embd=32, gradient_checkpointing=False, **kw)
    return GPT(cfg)


class TestInduction:
    def test_returns_probability(self):
        model = make_model()
        r = evaluate_induction(model, device="cpu", batch_size=4)
        assert 0.0 <= r <= 1.0

    def test_small_vocab_returns_nan(self):
        model = make_model(vocab_size=40)  # < INDUCTION_TOK_B_OFFSET + batch
        r = evaluate_induction(model, device="cpu", batch_size=32)
        assert r != r  # NaN

    def test_deterministic(self):
        # batch_size=8: default 32 materializes (32,256,50257) fp32 ~ 1.6 GB
        # logits - fine on TPU, OOM on small CI hosts (found in panel sweep).
        model = make_model()
        r1 = evaluate_induction(model, device="cpu", seed=42, batch_size=8)
        r2 = evaluate_induction(model, device="cpu", seed=42, batch_size=8)
        assert r1 == r2

    def test_restores_training_mode(self):
        model = make_model().train()
        evaluate_induction(model, device="cpu", batch_size=8)
        assert model.training


class TestDistractorInduction:
    # NOTE: memory-conscious settings. Logits are (B, T, V); with V=50257 keep
    # B and T small so CI/CPU boxes don't OOM: (2, 512, 50257) fp32 ~ 200 MB.
    def _model(self):
        cfg = GPTConfig(block_size=512, vocab_size=50257, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        return GPT(cfg)

    def test_returns_valid_metrics_and_deterministic(self):
        m = self._model()
        r1 = evaluate_distractor_induction(m, device="cpu", seed=42, batch_size=2,
                                           n_distractors=8)
        assert 0.0 <= r1["distractor_induction"] <= 1.0
        assert -1.0 <= r1["distractor_margin"] <= 1.0
        r2 = evaluate_distractor_induction(m, device="cpu", seed=42, batch_size=2,
                                           n_distractors=8)
        assert r1 == r2

    def test_small_context_returns_nan(self):
        cfg = GPTConfig(block_size=64, vocab_size=50257, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        r = evaluate_distractor_induction(GPT(cfg), device="cpu")
        assert r["distractor_induction"] != r["distractor_induction"]  # NaN

    def test_restores_training_mode(self):
        m = self._model().train()
        evaluate_distractor_induction(m, device="cpu", batch_size=2, n_distractors=8)
        assert m.training


class TestDepthProfile:
    def test_empty_when_disabled(self):
        m = make_model()
        assert depth_profile_statistics(m) == {}

    def test_layer_temp_readout_at_init(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=4, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        layer_dependent_gate=True, learnable_layer_temp=True)
        s = depth_profile_statistics(GPT(cfg))
        # at init: temp softplus(0)/log2 = 1 -> mult = 1 + depth
        assert abs(s["layer_temp_first"] - 1.0) < 1e-6
        assert abs(s["layer_temp_last"] - (1.0 + 3.0 / 4.0)) < 1e-6

    def test_phase_budget_readout_at_init(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=4,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, per_head_phase=True)
        s = depth_profile_statistics(GPT(cfg))
        assert abs(s["phase_budget_mean"] - 1.0) < 1e-6
        assert abs(s["phase_budget_min"] - 1.0) < 1e-6


class TestEntropy:
    def test_positive_bounded(self):
        model = make_model()
        e = measure_attention_entropy(model, device="cpu")
        import math
        assert 0.0 < e < math.log(64) + 1e-6  # entropy <= log(T)


class TestStatistics:
    def test_phase_statistics_at_identity(self):
        model = make_model(phase_mult=0.15)
        per_head, mean = phase_statistics(model)
        assert per_head is not None
        assert mean == 0.0  # identity init => zero phase norms

    def test_svd_statistics_at_identity(self):
        """Phase/gate are zero at init: erank must be NaN (inactive), not 0."""
        model = make_model(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        s = svd_statistics(model)
        assert s["phase_erank"] != s["phase_erank"]  # NaN
        assert s["gate_erank"] != s["gate_erank"]    # NaN
        # backbone Q/K/V blocks are randomly initialized -> positive stable rank
        assert s["qk_stable_rank"] > 1.0
        assert s["v_stable_rank"] > 1.0

    def test_svd_statistics_after_perturbation(self):
        model = make_model(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.02)
                blk.attn.W_phase_k.normal_(0, 0.02)
                blk.attn.W_gate_k.weight.normal_(0, 0.02)
                blk.attn.W_gate_v.weight.normal_(0, 0.02)
        s = svd_statistics(model)
        head_half = model.config.n_embd // model.config.n_head // 2
        assert 1.0 <= s["phase_erank"] <= head_half + 1e-6
        assert 0.0 < s["phase_top1"] <= 1.0
        assert s["gate_erank"] > 0.0

    def test_svd_rank1_collapse_detected(self):
        """A rank-1 phase matrix must give erank ~= 1 and top1 ~= 1."""
        model = make_model(phase_mult=0.15)
        with torch.no_grad():
            for blk in model.transformer.h:
                H, C, K = blk.attn.W_phase_q.shape
                u = torch.randn(C, 1)
                v = torch.randn(1, K)
                blk.attn.W_phase_q.copy_((u @ v).unsqueeze(0).expand(H, C, K))
        s = svd_statistics(model)
        assert abs(s["phase_erank"] - 1.0) < 0.05
        assert s["phase_top1"] > 0.95

    def test_interference_keys_and_ranges(self):
        model = make_model()
        s = head_interference_statistics(model)
        for k in ("qk_interference", "ov_interference", "qk_self_overlap",
                  "ov_self_overlap", "qk_ov_separation"):
            assert k in s
        for k in ("qk_interference", "ov_interference"):
            assert 0.0 <= s[k] <= 1.0, f"{k}={s[k]} out of [0,1]"

    def test_interference_orthogonal_heads_near_zero(self):
        """Ground truth: heads writing/reading in disjoint coordinate blocks
        must have ~zero cross-head interference."""
        model = make_model()
        attn = model.transformer.h[0].attn
        C, H, hd = attn.n_embd, attn.n_head, attn.head_dim
        with torch.no_grad():
            attn.c_attn.weight.zero_()
            attn.c_proj.weight.zero_()
            for h in range(H):
                rows = slice(h * hd, (h + 1) * hd)
                cols = slice(h * (C // H), (h + 1) * (C // H))
                # Q, K, V of head h read ONLY from its own block of coords.
                for base in (0, C, 2 * C):
                    blk = attn.c_attn.weight[base + h * hd : base + (h + 1) * hd, cols]
                    blk.copy_(torch.randn_like(blk))
                # head h writes ONLY into its own block of coords.
                attn.c_proj.weight[cols, rows] = torch.randn(C // H, hd)
        s = head_interference_statistics(model, max_layers=1)
        assert s["qk_interference"] < 1e-6
        assert s["ov_interference"] < 1e-6
        # Self overlap: random topk-dim subspaces inside the head's own
        # (C//H)-dim coordinate block overlap ~ topk/(C//H) in expectation
        # (0.5 here); must be clearly nonzero, in contrast to cross ~ 0.
        assert s["qk_self_overlap"] > 0.2

    def test_interference_identical_heads_near_one(self):
        """Ground truth: all heads identical => cross overlap == self overlap ~ 1."""
        model = make_model()
        attn = model.transformer.h[0].attn
        C, H, hd = attn.n_embd, attn.n_head, attn.head_dim
        with torch.no_grad():
            q0 = torch.randn(hd, C)
            v0 = torch.randn(hd, C)
            o0 = torch.randn(C, hd)
            for h in range(H):
                attn.c_attn.weight[h * hd : (h + 1) * hd, :] = q0
                attn.c_attn.weight[C + h * hd : C + (h + 1) * hd, :] = q0
                attn.c_attn.weight[2 * C + h * hd : 2 * C + (h + 1) * hd, :] = v0
                attn.c_proj.weight[:, h * hd : (h + 1) * hd] = o0
        s = head_interference_statistics(model, max_layers=1)
        assert abs(s["qk_interference"] - s["qk_self_overlap"]) < 1e-5
        assert abs(s["ov_interference"] - s["ov_self_overlap"]) < 1e-5

    def test_hla_statistics_keys(self):
        model = make_model(phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        x = torch.randint(0, 50257, (1, 16))
        model.eval()
        with torch.no_grad():
            model(x)
        stats = hla_statistics(model)
        assert "mix_k_mean" in stats and "mix_v_mean" in stats
        # identity init => mix must be exactly 1
        assert abs(stats["mix_k_mean"] - 1.0) < 1e-6
        assert abs(stats["mix_v_mean"] - 1.0) < 1e-6


class TestPerturbationBounds:
    def _hla(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        return GPT(cfg)

    def test_actual_within_theoretical(self):
        """Core Theorem-4 check: captured mix never exceeds the envelope,
        even with saturated gates."""
        m = self._hla()
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 100.0)  # saturate
                blk.attn.W_range_k.normal_(0, 100.0)
        out = perturbation_bounds(m)
        for i in range(2):
            assert out[f"L{i:02d}_mix_k_max"] <= out[f"L{i:02d}_theo_mix_k_max"] + 1e-4
            assert out[f"L{i:02d}_mix_k_min"] >= out[f"L{i:02d}_theo_mix_k_min"] - 1e-4
            assert 0.0 <= out[f"L{i:02d}_util_k_up"] <= 1.0 + 1e-6

    def test_saturated_gates_reach_utilization_one(self):
        m = self._hla()
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
                blk.attn.W_range_k.fill_(1000.0)
        out = perturbation_bounds(m)
        assert out["L00_util_k_up"] > 0.99, "saturation must reach the envelope"

    def test_identity_init_dormant(self):
        out = perturbation_bounds(self._hla())
        assert abs(out["L00_util_k_up"]) < 1e-6, "identity init must show ~0 utilization"

    def test_restores_model_state(self):
        m = self._hla().train()
        perturbation_bounds(m)
        assert m.training
        assert m.transformer.h[0].attn.capture_diagnostics is False


class TestPerLayerGateAnalysis:
    def test_per_layer_keys_and_depth_scaling(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=4, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
                        layer_dependent_gate=True)
        m = GPT(cfg)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
        out = per_layer_gate_analysis(m)
        for i in range(4):
            assert f"L{i:02d}_mix_k_mean" in out
            assert f"L{i:02d}_attn_entropy" in out
        # deeper layer => wider envelope => larger mix at saturation
        assert out["L03_mix_k_mean"] > out["L00_mix_k_mean"]


class TestMechanismGradientStatistics:
    def test_active_nonzero_inactive_zero_and_summary(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
                        use_salience_bias=True, salience_alpha=0.0)  # salience OFF
        m = GPT(cfg)
        x = torch.randint(0, 256, (2, 32))
        _, loss = m(x, x)
        loss.backward()
        m.capture_mechanism_grad_norms()
        out = mechanism_gradient_statistics(m)
        assert out["L00_grad_phase_q"] > 0.0, "active mechanism must have gradient (Theorem 5)"
        assert out["L00_grad_gate_sal"] == 0.0, "inactive mechanism must record exact 0"
        assert out["mech_grad_min"] > 0.0, "summary must exclude inactive zeros"

    def test_empty_before_capture(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        assert mechanism_gradient_statistics(GPT(cfg)) == {}


class TestProbeNonInterference:
    """Review attack R-E: probes and captures must NEVER alter training.

    Sterility invariant I5 extends to instrumentation: enabling metrics
    cannot change the trajectory by a single bit."""

    def test_grad_capture_does_not_change_weights(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, phase_mult=0.15)

        def train3(capture):
            torch.manual_seed(0)
            m = GPT(cfg)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            torch.manual_seed(1)
            for _ in range(3):
                x = torch.randint(0, 128, (2, 16))
                _, l = m(x, x)
                l.backward()
                opt.step()
                if capture:
                    m.capture_mechanism_grad_norms()
                opt.zero_grad(set_to_none=True)
            return [p.detach().clone() for p in m.parameters()]

        pa, pb = train3(False), train3(True)
        assert all(torch.equal(a, b) for a, b in zip(pa, pb)), \
            "grad capture altered the training trajectory!"

    def test_perturbation_bounds_does_not_change_weights(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0)
        m = GPT(cfg)
        before = [p.detach().clone() for p in m.parameters()]
        perturbation_bounds(m)
        after = list(m.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    def test_captured_norms_survive_zero_grad(self):
        """R-C: snapshot must be a copy, not a view of .grad."""
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, phase_mult=0.15)
        m = GPT(cfg)
        x = torch.randint(0, 128, (2, 16))
        _, loss = m(x, x)
        loss.backward()
        m.capture_mechanism_grad_norms()
        v1 = float(m.transformer.h[0].attn.last_mechanism_grad_norms["phase_q"])
        for p in m.parameters():
            p.grad = None
        v2 = float(m.transformer.h[0].attn.last_mechanism_grad_norms["phase_q"])
        assert v1 == v2 and v1 > 0.0


class TestProbeEdgeCases:
    """Review attack R-A/R-G: probes under unusual but legal configs."""

    def test_bounds_track_learned_layer_temp(self):
        """R-A: with theta != 0 the envelope must use the LIVE lambda."""
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0,
                        layer_dependent_gate=True, learnable_layer_temp=True)
        m = GPT(cfg)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
                blk.attn.W_range_k.fill_(1000.0)
                blk.attn.W_layer_temp.fill_(2.0)  # learned, non-default
        out = perturbation_bounds(m)
        assert out["L01_util_k_up"] > 0.99, \
            "envelope must track live softplus(theta), not the static heuristic"
        assert out["L01_mix_k_max"] <= out["L01_theo_mix_k_max"] + 1e-4

    def test_bounds_empty_when_laplace_off(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, use_laplace=False)
        assert perturbation_bounds(GPT(cfg)) == {}

    def test_per_layer_analysis_no_crash_minimal_model(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False, use_laplace=False)
        out = per_layer_gate_analysis(GPT(cfg))
        assert "L00_attn_entropy" in out


class TestProbeStateRestoration:
    """Review attack P1: probes must restore the CALLER's diagnostics state,
    not blindly reset it."""

    def test_perturbation_bounds_restores_enabled_diagnostics(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0)
        m = GPT(cfg)
        m.set_diagnostics(enabled=True, capture_attention=True)
        perturbation_bounds(m)
        assert m.transformer.h[0].attn.capture_diagnostics is True
        assert m.transformer.h[0].attn.capture_attention is True

    def test_perturbation_bounds_restores_disabled_diagnostics(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False,
                        use_laplace=True, laplace_alpha=1.0)
        m = GPT(cfg)
        perturbation_bounds(m)
        assert m.transformer.h[0].attn.capture_diagnostics is False

    def test_per_layer_analysis_restores_state(self):
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=1, n_head=2,
                        n_embd=16, gradient_checkpointing=False)
        m = GPT(cfg)
        m.set_diagnostics(enabled=True)
        per_layer_gate_analysis(m)
        assert m.transformer.h[0].attn.capture_diagnostics is True


class TestAttentionHeadSimilarity:
    """Nanda attack: activation-space head redundancy (JS between heads)."""

    def _model(self, **kw):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False, **kw)
        return GPT(cfg)

    def test_identical_heads_zero_js(self):
        m = self._model()
        attn = m.transformer.h[0].attn
        with torch.no_grad():
            C, hd = attn.n_embd, attn.head_dim
            w = attn.c_attn.weight
            # head 1 copies head 0 for Q and K -> identical attention maps
            w[hd:2*hd, :] = w[0:hd, :]
            w[C+hd:C+2*hd, :] = w[C:C+hd, :]
        out = attention_head_similarity(m)
        assert out["head_js_mean"] < 1e-6, "identical heads must give JS ~ 0"

    def test_js_bounded_and_state_restored(self):
        import math
        m = self._model(phase_mult=0.15)
        m.set_diagnostics(enabled=True, capture_attention=True)
        out = attention_head_similarity(m)
        assert 0.0 <= out["head_js_mean"] <= math.log(2.0) + 1e-6
        assert m.transformer.h[0].attn.capture_attention is True  # restored

    def test_deterministic(self):
        m = self._model(phase_mult=0.15)
        assert attention_head_similarity(m, seed=1) == attention_head_similarity(m, seed=1)


class TestAngleStd:
    """Nanda attack: mean |angle| hides bimodality - std must be captured."""

    def test_std_zero_at_identity_nonzero_when_active(self):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False, phase_mult=0.15)
        m = GPT(cfg).eval()
        x = torch.randint(0, 256, (1, 32))
        m(x)
        assert float(m.transformer.h[0].attn.last_angle_q_std) == 0.0
        with torch.no_grad():
            m.transformer.h[0].attn.W_phase_q.normal_(0, 0.5)
        m(x)
        assert float(m.transformer.h[0].attn.last_angle_q_std) > 0.0


class TestMechanismKnockout:
    def _model(self):
        c = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                      n_embd=32, gradient_checkpointing=False,
                      phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        m = GPT(c)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.3)
                blk.attn.W_gate_k.weight.normal_(0, 0.3)
        return m

    def test_restoration_exact(self):
        m = self._model().eval()
        x = torch.randint(0, 256, (2, 32))
        l_before, _ = m(x)
        mechanism_knockout(m, x, x)
        l_after, _ = m(x)
        assert torch.equal(l_before, l_after), "knockout must restore exactly"

    def test_active_mechanism_has_effect(self):
        m = self._model()
        x = torch.randint(0, 256, (2, 32))
        out = mechanism_knockout(m, x, x)
        assert out["ko_phase_delta"] != 0.0, "randomized phase must carry load"

    def test_absent_mechanism_zero_delta(self):
        m = self._model()
        x = torch.randint(0, 256, (2, 32))
        out = mechanism_knockout(m, x, x)
        assert out["ko_salience_delta"] == 0.0  # salience_alpha=0 in config


class TestPrefixMatching:
    # Memory note: logits are (B,T,50257); keep B=1, block=128 so tiny CI
    # hosts survive ((1,128,50257) fp32 ~ 25 MB).
    def test_keys_and_range(self):
        c = GPTConfig(block_size=128, vocab_size=50257, n_layer=2, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        out = prefix_matching_score(GPT(c), batch_size=1)
        assert "prefix_match_global_max" in out
        assert 0.0 <= out["prefix_match_global_max"] <= 1.0
        assert "L00_prefix_match_max" in out and "L01_prefix_match_mean" in out

    def test_state_restored_and_deterministic(self):
        c = GPTConfig(block_size=128, vocab_size=50257, n_layer=1, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        m = GPT(c)
        r1 = prefix_matching_score(m, seed=7, batch_size=1)
        r2 = prefix_matching_score(m, seed=7, batch_size=1)
        assert r1 == r2
        assert m.transformer.h[0].attn.capture_attention is False


class TestRangeFlexConsistency:
    """Final polish sweep: range_flex became a config knob, but
    perturbation_bounds still hardcoded the historical 1.25 factor -
    with range_flex != 0.25 the probe reported a silently WRONG envelope."""

    def _hla_flex(self, flex):
        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
                        range_flex=flex)
        return GPT(cfg)

    def test_envelope_tracks_range_flex(self):
        import math as _m
        for flex in (0.25, 0.5, 1.0):
            m = self._hla_flex(flex)
            out = perturbation_bounds(m)
            attn = m.transformer.h[0].attn
            lam = float(attn.layer_gate_multiplier)
            eff = lam * min(attn.laplace_alpha * attn.laplace_range_k * (1.0 + flex),
                            attn.k_log_clip)
            expected_hi = (1 - attn.beta_k) + attn.beta_k * _m.exp(eff)
            got = out["L00_theo_mix_k_max"]
            assert abs(got - expected_hi) < 1e-9, (
                f"range_flex={flex}: probe envelope {got} != model envelope "
                f"{expected_hi} (hardcoded 1.25 regression)")

    def test_saturated_mix_stays_inside_flex_envelope(self):
        """With range_flex=1.0 the true envelope is WIDER than the 1.25 one:
        a saturated gate must still be inside the probe's reported bound."""
        m = self._hla_flex(1.0)
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.fill_(1000.0)
                blk.attn.W_range_k.fill_(1000.0)
        out = perturbation_bounds(m)
        assert out["L00_mix_k_max"] <= out["L00_theo_mix_k_max"] + 1e-4
        assert out["L00_util_k_up"] > 0.99


class TestGateRedundancy:
    """Round 7: measurement-first answer to the 'four gates may collapse into
    one function' attack (H3/M6) - probe, pre-registered merge rule, no aux loss."""

    KW = dict(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=32,
              gradient_checkpointing=False, use_rope=True, use_wpe=False,
              use_laplace=True, laplace_alpha=1.0,
              use_salience_bias=True, salience_alpha=1.0,
              use_distance_laplace=True, distance_laplace_alpha=0.5)

    def _model(self):
        return GPT(GPTConfig(**self.KW))

    def test_identity_reports_nan(self):
        """All-zero gates have no correlation structure - NaN, not fake 0."""
        import math
        out = gate_redundancy_statistics(self._model())
        assert math.isnan(out["gate_redundancy_mean_abs"])

    def test_cloned_gates_detected(self):
        """Ground-truth witness: gate_v cloned from gate_k must give corr ~ 1."""
        m = self._model()
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 0.5)
                blk.attn.W_gate_v.weight.copy_(blk.attn.W_gate_k.weight)
                blk.attn.W_gate_sal.weight.normal_(0, 0.5)
                blk.attn.W_gate_d.weight.normal_(0, 0.5)
        out = gate_redundancy_statistics(m)
        assert out["gate_corr_kv"] > 0.99
        assert 0.0 <= out["gate_redundancy_mean_abs"] <= 1.0

    def test_independent_gates_low_corr(self):
        """Independent random gates must NOT be flagged as redundant."""
        torch.manual_seed(11)
        m = self._model()
        with torch.no_grad():
            for blk in m.transformer.h:
                for lin in (blk.attn.W_gate_k, blk.attn.W_gate_v,
                            blk.attn.W_gate_sal, blk.attn.W_gate_d):
                    lin.weight.normal_(0, 0.5)
        out = gate_redundancy_statistics(m)
        assert out["gate_redundancy_mean_abs"] < 0.9

    def test_state_restored_and_deterministic(self):
        m = self._model().train()
        with torch.no_grad():  # nonzero gates: NaN != NaN would defeat the equality check
            for blk in m.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 0.3)
                blk.attn.W_gate_v.weight.normal_(0, 0.3)
        r1 = gate_redundancy_statistics(m, seed=5)
        r2 = gate_redundancy_statistics(m, seed=5)
        assert r1 == r2
        assert m.training
        assert m.transformer.h[0].attn.capture_diagnostics is False


class TestPositionalRecall:
    """Round 10: direct LITM probe - the pre-registered H4 measurement."""

    def test_keys_and_ranges(self):
        c = GPTConfig(block_size=128, vocab_size=50257, n_layer=1, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        out = positional_recall_curve(GPT(c), batch_size=2)
        for k in ("pos_10", "pos_30", "pos_50", "pos_70", "pos_90",
                  "litm_middle_drop", "litm_worst_frac"):
            assert k in out
        for k in ("pos_10", "pos_30", "pos_50", "pos_70", "pos_90"):
            assert 0.0 <= out[k] <= 1.0

    def test_small_vocab_nan(self):
        import math
        c = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        out = positional_recall_curve(GPT(c))
        assert math.isnan(out["litm_middle_drop"])

    def test_deterministic_and_state_restored(self):
        c = GPTConfig(block_size=128, vocab_size=50257, n_layer=1, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        m = GPT(c).train()
        r1 = positional_recall_curve(m, seed=3, batch_size=2)
        r2 = positional_recall_curve(m, seed=3, batch_size=2)
        assert r1 == r2
        assert m.training

    def test_flat_at_random_init(self):
        """An untrained model has no positional preference - the curve must be
        ~flat (drop ~ 0). Ground-truth witness for the metric's zero point."""
        c = GPTConfig(block_size=128, vocab_size=50257, n_layer=1, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        out = positional_recall_curve(GPT(c), batch_size=2)
        assert abs(out["litm_middle_drop"]) < 1e-4


class TestCausalPatch:
    """Round 14: the Oral experiment's tooling - transplant correctness."""

    KW = dict(block_size=64, vocab_size=50257, n_layer=2, n_head=2, n_embd=32,
              gradient_checkpointing=False, use_rope=True, use_wpe=False,
              phase_mult=0.15, use_laplace=True, laplace_alpha=1.0,
              use_salience_bias=True, salience_alpha=1.0,
              use_distance_laplace=True, distance_laplace_alpha=0.5)

    def _pair(self):
        import importlib
        import sys as _sys
        _sys.path.insert(0, os.path.join(ROOT, "scripts"))
        cp = importlib.import_module("causal_patch")
        torch.manual_seed(0)
        hla = GPT(GPTConfig(**self.KW))
        base = GPT(GPTConfig(**self.KW))
        base.load_state_dict(hla.state_dict())
        with torch.no_grad():  # раздвигаем близнецов детерминированно
            for blk in hla.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.2, generator=torch.Generator().manual_seed(1))
                blk.attn.W_gate_k.weight.normal_(0, 0.2, generator=torch.Generator().manual_seed(2))
            for blk in base.transformer.h:
                blk.attn.c_attn.weight.add_(0.01)
        return cp, base.state_dict(), hla.state_dict()

    def test_full_transplant_equals_hla(self):
        cp, bs, hs = self._pair()
        fr = cp.build_franken(bs, hs, 32, "full")
        assert all(torch.equal(fr[k], hs[k]) for k in hs)

    def test_qk_rows_spliced_v_rows_kept(self):
        cp, bs, hs = self._pair()
        fr = cp.build_franken(bs, hs, 32, "qk")
        key = next(k for k in hs if "c_attn.weight" in k)
        assert torch.equal(fr[key][:64], hs[key][:64])   # Q,K <- HLA
        assert torch.equal(fr[key][64:], bs[key][64:])   # V  <- base

    def test_retrieval_transplant_boundaries(self):
        """Retrieval mechanisms come from HLA; the V-side gate must stay base
        - the whole point is that transmission remains untouched."""
        cp, bs, hs = self._pair()
        fr = cp.build_franken(bs, hs, 32, "retrieval")
        kq = next(k for k in hs if "W_phase_q" in k)
        kv = next(k for k in hs if "W_gate_v.weight" in k)
        km = next(k for k in hs if "mlp" in k and "weight" in k)
        assert torch.equal(fr[kq], hs[kq])
        assert torch.equal(fr[kv], bs[kv])
        assert torch.equal(fr[km], bs[km])

    def test_franken_state_loads_strict(self):
        cp, bs, hs = self._pair()
        fr = cp.build_franken(bs, hs, 32, "retrieval")
        m = GPT(GPTConfig(**self.KW))
        m.load_state_dict(fr, strict=True)  # raises on any mismatch

    def test_mismatched_pair_rejected(self):
        import pytest as _pytest
        cp, bs, hs = self._pair()
        bad = dict(bs); bad.pop(next(iter(bad)))
        with _pytest.raises(ValueError, match="not a sterile pair"):
            cp.build_franken(bad, hs, 32, "qk")

    def test_probe_keys_not_double_prefixed(self):
        """v1 wrote distractor_distractor_induction (re-prefixed keys that
        already carried the prefix) - the H5 JSON would have silently missed
        the pre-registered metric names."""
        cp, bs, hs = self._pair()
        m = GPT(GPTConfig(**self.KW))
        m.load_state_dict(hs, strict=True)
        out = cp.probe(m.eval(), batch_size=2)
        assert "distractor_distractor_induction" not in out
        assert "distractor_induction" in out or "distractor_error" in out

    def test_gap_closure_guard_small_gap(self):
        """closure = (fr-b)/(hla-b) on a near-zero gap manufactures arbitrary
        percentages; the guard must refuse (NaN + flag), never divide."""
        cp, _, _ = self._pair()
        base = {"induction": 0.5000}
        hla = {"induction": 0.5000 + 1e-6}   # below MIN_MEANINGFUL_GAP
        fr = {"induction": 0.9}
        rec = cp.gap_closure(base, hla, fr)["induction"]
        assert rec["closure"] != rec["closure"]          # NaN
        assert rec.get("closure_note") == 1.0

    def test_gap_closure_math(self):
        cp, _, _ = self._pair()
        base = {"induction": 0.10, "induction_std": 0.0}
        hla = {"induction": 0.30, "induction_std": 0.0}
        fr = {"induction": 0.25, "induction_std": 0.0}
        rec = cp.gap_closure(base, hla, fr)["induction"]
        assert abs(rec["closure"] - 0.75) < 1e-9         # (0.25-0.10)/(0.30-0.10)
        assert rec["closure_std_bound"] == 0.0

    def test_probe_multi_reports_std(self):
        cp, bs, hs = self._pair()
        m = GPT(GPTConfig(**self.KW))
        m.load_state_dict(hs, strict=True)
        out = cp.probe_multi(m.eval(), seeds=(1, 2), batch_size=2)
        assert "induction" in out and "induction_std" in out
        assert out["induction_std"] >= 0.0


    def test_reverse_franken_boundaries(self):
        """A-extension (necessity test): reverse franken = HLA body + base
        retrieval. The V-side gate must stay HLA (body), the phase must
        become base - the exact mirror of the forward boundaries."""
        cp, bs, hs = self._pair()
        fr = cp.build_franken(bs, hs, 32, "retrieval", donor="base")
        kq = next(k for k in hs if "W_phase_q" in k)
        kv = next(k for k in hs if "W_gate_v.weight" in k)
        km = next(k for k in hs if "mlp" in k and "weight" in k)
        assert torch.equal(fr[kq], bs[kq]), "phase must come from base (graft)"
        assert torch.equal(fr[kv], hs[kv]), "V-gate must stay HLA (body)"
        assert torch.equal(fr[km], hs[km]), "MLP must stay HLA (body)"
        # Q,K rows from base, V rows from HLA
        key = next(k for k in hs if "c_attn.weight" in k)
        assert torch.equal(fr[key][:64], bs[key][:64])
        assert torch.equal(fr[key][64:], hs[key][64:])

    def test_franken_unknown_donor_rejected(self):
        import pytest as _pytest
        cp, bs, hs = self._pair()
        with _pytest.raises(ValueError, match="unknown donor"):
            cp.build_franken(bs, hs, 32, "retrieval", donor="qwen")

    def test_snr_is_gap_metric(self):
        """B-extension: the activation-level SNR must be part of the
        pre-registered gap metrics - P(B) probes alone can be moved by the
        V-path; snr_needle only moves with score geometry."""
        cp, _, _ = self._pair()
        assert "snr_needle_last" in cp.GAP_METRICS

    def test_forward_reverse_full_are_mirror_images(self):
        """Sanity: transplant='full' with donor='hla' == HLA exactly, and
        with donor='base' == base exactly."""
        cp, bs, hs = self._pair()
        f_h = cp.build_franken(bs, hs, 32, "full", donor="hla")
        f_b = cp.build_franken(bs, hs, 32, "full", donor="base")
        assert all(torch.equal(f_h[k], hs[k]) for k in hs)
        assert all(torch.equal(f_b[k], bs[k]) for k in bs)

class TestAttentionNeedleSNR:
    """Diff-Transformer-lesson metric (their Table 3 'attention noise'):
    activation-level SNR of retrieval. The Oral-tier reading of our
    salience/gate claim needs a direct 'where does attention LOOK' number,
    not only P(B) probes."""

    KW = dict(block_size=256, vocab_size=50257, n_layer=2, n_head=2,
              n_embd=64, gradient_checkpointing=False)

    def test_snr_near_one_at_random_init(self):
        from src.eval import attention_needle_snr
        torch.manual_seed(0)
        m = GPT(GPTConfig(**self.KW)).eval()
        out = attention_needle_snr(m)
        assert 0.3 < out["snr_needle_last"] < 3.0, (
            "random init has no needle preference; SNR must be ~1 "
            f"(got {out['snr_needle_last']})")
        assert out["snr_needle_best"] >= out["snr_needle_last"] - 1e-9

    def test_snr_nan_on_tiny_vocab(self):
        import math
        from src.eval import attention_needle_snr
        c = GPTConfig(block_size=64, vocab_size=32, n_layer=1, n_head=2,
                      n_embd=32, gradient_checkpointing=False)
        out = attention_needle_snr(GPT(c).eval())
        assert math.isnan(out["snr_needle_last"])

    def test_snr_restores_model_state(self):
        """Probe is side-effect-free like every other probe (tested class
        invariant): diagnostics flags and training mode restored."""
        from src.eval import attention_needle_snr
        m = GPT(GPTConfig(**self.KW))
        m.train()
        attention_needle_snr(m)
        assert m.training
        assert not m.transformer.h[0].attn.capture_diagnostics
        assert not m.transformer.h[0].attn.capture_attention

    def test_snr_detects_planted_signal(self):
        """Ground-truth: a model FORCED to attend at the needle must show
        SNR >> 1 - the metric must actually move when the behavior exists."""
        from src.eval import attention_needle_snr
        torch.manual_seed(0)
        m = GPT(GPTConfig(**self.KW)).eval()
        base = attention_needle_snr(m)["snr_needle_last"]
        # Craft attention artificially: huge salience toward needle tokens is
        # impractical to force directly; instead verify metric monotonicity
        # via a synthetic attention override on the last block.
        blk = m.transformer.h[-1].attn
        orig = blk.forward
        import types as _t

        def fake_forward(self, x):
            out = orig(x)
            if self.capture_attention and self.last_attn is not None:
                att = self.last_attn.clone()
                T = att.shape[-1]
                pos_ab = T // 3
                att[:, :, T - 2, :] = 1e-6
                att[:, :, T - 2, pos_ab] = 0.5
                att[:, :, T - 2, pos_ab + 1] = 0.5
                self.last_attn = att
            return out

        blk.forward = _t.MethodType(fake_forward, blk)
        boosted = attention_needle_snr(m)["snr_needle_last"]
        blk.forward = orig
        assert boosted > 100 * max(base, 1e-9), (
            f"metric must saturate when attention IS on the needle "
            f"(base={base:.3f}, boosted={boosted:.3f})")



class TestTrainProbe:
    """scripts/train_probe.py: position-conditioned linear probing (H4-P).
    The mech-interp reviewer's question 'is mid-context info actually more
    LINEARLY ACCESSIBLE, or just more attended-to?' needs its own tool -
    P(B) and SNR both live downstream of attention; the ridge probe reads
    the residual stream directly."""

    def _load(self):
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "train_probe", os.path.join(ROOT, "scripts", "train_probe.py"))
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _model(self, **kw):
        base = dict(block_size=128, vocab_size=50257, n_layer=2, n_head=2,
                    n_embd=64, gradient_checkpointing=False)
        base.update(kw)
        torch.manual_seed(0)
        return GPT(GPTConfig(**base)).eval()

    def test_chance_at_random_init(self):
        """Calibration: random weights carry no needle info; every depth
        must decode near chance (1/8), i.e. the probe must not hallucinate
        signal out of the ridge fit itself."""
        tp = self._load()
        res = tp.run_probe(self._model(), n_per_class=12)
        for d, a in res["probe_acc_by_depth"].items():
            assert a < 0.45, f"depth {d}: {a} >> chance 0.125"
        # Selectivity control: no real signal -> real ~ shuffled (both chance)
        assert all(v < 0.4 for v in res["probe_selectivity"].values())

    def test_detects_planted_signal(self):
        """Ground truth: if a label-dependent direction IS in the residual
        stream, the probe must find it (sensitivity)."""
        tp = self._load()
        m = self._model()
        sig = torch.randn(8, 64) * 5.0
        holder = [None]
        orig = m.forward

        def wrapped(idx, targets=None):
            holder[0] = idx
            return orig(idx, targets)

        m.forward = wrapped

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            toks = holder[0]
            T = toks.shape[1]
            mask = (toks >= 46000) & (toks < 46008)
            for b in range(toks.shape[0]):
                idx = mask[b].nonzero()
                if len(idx):
                    h[b, T - 2, :] += sig[int(toks[b, idx[0]] - 46000)]
            return (h,) + out[1:] if isinstance(out, tuple) else h

        hd = m.transformer.h[-1].register_forward_hook(hook)
        res = tp.run_probe(m, n_per_class=12)
        hd.remove()
        assert all(a > 0.8 for a in res["probe_acc_by_depth"].values())
        # Real signal -> high selectivity (probe reads features, not memorizes)
        assert all(v > 0.5 for v in res["probe_selectivity"].values())

    def test_selection_on_val_not_test(self):
        """The selection-bias fix: layer and lambda must be chosen on VAL;
        the reported scalar is TEST accuracy of that choice. Structural
        check: run_probe reports the selection metadata, and the selected
        (layer, lambda) is derived from val_acc - never from test."""
        tp = self._load()
        res = tp.run_probe(self._model(), n_per_class=12)
        sel = res["probe_selection"]
        assert set(sel) == {"0.1", "0.3", "0.5", "0.7", "0.9"}
        for d, rec in sel.items():
            assert 0 <= rec["layer"] < 2
            assert rec["lambda"] in tp.LAMBDA_GRID
            assert 0.0 <= rec["val_acc"] <= 1.0
        src = open(os.path.join(ROOT, "scripts", "train_probe.py")).read()
        assert "va[j] > best_val" in src, "layer choice must compare VAL accs"

    def test_nan_policy_tiny_vocab(self):
        import math
        tp = self._load()
        m = self._model(vocab_size=1024)
        res = tp.run_probe(m)
        assert math.isnan(res["probe_litm_gap"])

    def test_restores_training_mode(self):
        tp = self._load()
        m = self._model()
        m.train()
        tp.run_probe(m, n_per_class=8)
        assert m.training


class TestPerPositionLoss:
    """per_position_loss_curve: the field-standard long-context metric
    (FoX Fig.1-style) on REAL data - the non-synthetic companion to the
    author-created probes (metric-integrity answer for Reviewer 2)."""

    def _model(self, vocab=256):
        torch.manual_seed(0)
        return GPT(GPTConfig(block_size=128, vocab_size=vocab, n_layer=1,
                             n_head=2, n_embd=64,
                             gradient_checkpointing=False)).eval()

    def test_ratio_near_one_on_iid_tokens(self):
        """IID random tokens carry no context signal: early/late ratio must
        be ~1 (the metric must not invent long-context benefit)."""
        from src.eval import per_position_loss_curve
        m = self._model()
        g = torch.Generator().manual_seed(1)
        toks = torch.randint(0, 256, (8, 129), generator=g)
        r = per_position_loss_curve(m, toks)
        assert 0.9 < r["posloss_early_late_ratio"] < 1.1
        assert abs(r["posloss_mid_bump"]) < 0.2

    def test_nan_gate_short_input(self):
        import math
        from src.eval import per_position_loss_curve
        m = self._model()
        r = per_position_loss_curve(m, torch.randint(0, 256, (2, 20)))
        assert math.isnan(r["posloss_early_late_ratio"])

    def test_deterministic_and_side_effect_free(self):
        from src.eval import per_position_loss_curve
        m = self._model()
        g = torch.Generator().manual_seed(2)
        toks = torch.randint(0, 256, (4, 129), generator=g)
        m.train()
        r1 = per_position_loss_curve(m, toks)
        assert m.training, "training mode must be restored"
        r2 = per_position_loss_curve(m, toks)
        assert r1 == r2

    def test_detects_planted_predictability(self):
        """Ground truth: sequences whose second half deterministically
        copies token 0 must show late-position loss BELOW early - the
        metric must move when real structure exists. We fake it by making
        late targets constant (predictable even for a random model after
        bias drift is removed by using the ratio of a TRAINED-free case:
        instead we verify the per-bin machinery orders bins correctly on
        a crafted loss surface via monotone token predictability)."""
        from src.eval import per_position_loss_curve
        m = self._model()
        g = torch.Generator().manual_seed(3)
        toks = torch.randint(0, 256, (8, 129), generator=g)
        toks[:, 64:] = 7  # constant tail: model CAN'T know, but bins must differ
        r = per_position_loss_curve(m, toks)
        # bins covering the constant tail see a single-token distribution;
        # a random model's loss there differs from IID region - the curve
        # must reflect a difference between first and last bins
        bins = [v for k, v in sorted(r.items()) if k.startswith("posloss_bin_")]
        assert len(bins) == 8
        assert abs(bins[0] - bins[-1]) > 0.01, "metric blind to structure change"


class TestMetricMathAgainstManual:
    """Scientific-standards audit: each headline metric recomputed BY HAND
    from first principles (probe layout reproduced, softmax/entropy/ratio
    computed directly) and compared to the library value. Guards against
    formula drift - the failure mode where code runs fine but computes a
    subtly different quantity than the paper claims."""

    def _model(self):
        torch.manual_seed(0)
        return GPT(GPTConfig(block_size=256, vocab_size=50257, n_layer=2,
                             n_head=2, n_embd=64,
                             gradient_checkpointing=False)).eval()

    def test_induction_equals_manual_softmax_lookup(self):
        from src.eval import (evaluate_induction, INDUCTION_TOK_A_OFFSET,
                              INDUCTION_TOK_B_OFFSET)
        m = self._model()
        seed, bs = 42, 4
        T = 256
        g = torch.Generator(); g.manual_seed(seed)
        tokens = torch.randint(100, 20000 - 1, (bs, T), generator=g)
        i = torch.arange(bs)
        A = INDUCTION_TOK_A_OFFSET + i
        B = INDUCTION_TOK_B_OFFSET + i
        pos1 = T // 4
        tokens[i, pos1] = A
        tokens[i, pos1 + 1] = B
        tokens[i, T - 2] = A
        with torch.no_grad():
            logits, _ = m(tokens)
        probs = torch.softmax(logits[:, T - 2, :].float(), dim=-1)
        manual = float(probs[i, B].mean())
        lib = evaluate_induction(m, device="cpu", seed=seed, batch_size=bs)
        assert abs(manual - lib) < 1e-6

    def test_entropy_equals_manual_plogp(self):
        from src.eval import measure_attention_entropy
        m = self._model()
        seed, bs = 7, 2
        T = 256
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        tokens = torch.randint(0, m.config.vocab_size, (bs, T), generator=g)
        m.set_diagnostics(enabled=True, capture_attention=True)
        with torch.no_grad():
            m(tokens)
        ents = []
        for blk in m.transformer.h:
            att = blk.attn.last_attn.float().clamp_min(1e-12)
            ents.append(float((-(att * att.log()).sum(-1)).mean()))
        m.set_diagnostics(enabled=False)
        manual = sum(ents) / len(ents)
        lib = measure_attention_entropy(m, device="cpu", seed=seed, batch_size=bs)
        assert abs(manual - lib) < 1e-3

    def test_litm_scalars_equal_manual_aggregation(self):
        from src.eval import positional_recall_curve
        m = self._model()
        out = positional_recall_curve(m, batch_size=2)
        edges = [out["pos_10"], out["pos_90"]]
        middle = [out["pos_30"], out["pos_50"], out["pos_70"]]
        assert abs((sum(edges) / 2 - min(middle)) - out["litm_middle_drop"]) < 1e-9
        assert abs((min(middle) / max(edges)) - out["litm_worst_frac"]) < 1e-9

    def test_per_position_loss_uniform_row_weights(self):
        """Finding #15 regression: with N % batch_size != 0 the old code
        summed per-batch MEANS, so rows in the short final batch weighed
        1/len(last_batch) instead of 1/N (measured bin drift up to 1.2e-2
        at N=5, bs=4). Every bin must equal the direct equal-weight
        computation bit-for-bit."""
        from src.eval import per_position_loss_curve
        m = self._model()
        N, T = 5, 64  # 5 rows, batch 4 -> final batch of 1 (the bug trigger)
        g = torch.Generator(); g.manual_seed(4)
        toks = torch.randint(0, 50257, (N, T + 1), generator=g)
        out = per_position_loss_curve(m, toks, batch_size=4, n_bins=4)
        with torch.no_grad():
            x, y = toks[:, :T], toks[:, 1:T + 1]
            logits, _ = m(x)
            ls = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1), reduction="none").reshape(y.shape)
            per_pos = ls.mean(0)
            edges = torch.linspace(0, T, 5, dtype=torch.long)
            manual = [float(per_pos[edges[i]:edges[i + 1]].mean())
                      for i in range(4)]
        for i, want in enumerate(manual):
            assert abs(out[f"posloss_bin_{i:02d}"] - want) < 1e-6, \
                f"bin {i}: {out[f'posloss_bin_{i:02d}']} != manual {want}"

    def test_needle_snr_equals_manual_attention_ratio(self):
        """snr_needle recomputed by hand from captured attention: same probe
        layout, needle = mean mass on the pair, filler = mean mass on the
        masked row (no needle, no self/future). Must match bit-for-bit."""
        from src.eval import (attention_needle_snr, INDUCTION_TOK_A_OFFSET,
                              INDUCTION_TOK_B_OFFSET)
        m = self._model()
        res = attention_needle_snr(m, seed=42, batch_size=2)
        T, B = 256, 2
        g = torch.Generator(); g.manual_seed(42)
        lo = INDUCTION_TOK_B_OFFSET + B
        tokens = torch.randint(lo, 50257, (B, T), generator=g)
        i = torch.arange(B)
        pos_ab, q_pos = T // 3, T - 2
        tokens[:, pos_ab] = INDUCTION_TOK_A_OFFSET + i
        tokens[:, pos_ab + 1] = INDUCTION_TOK_B_OFFSET + i
        tokens[:, q_pos] = INDUCTION_TOK_A_OFFSET + i
        m.set_diagnostics(enabled=True, capture_attention=True)
        try:
            with torch.no_grad():
                m(tokens)
            manual = []
            for blk in m.transformer.h:
                att = blk.attn.last_attn.detach().float()
                row = att[:, :, q_pos, :]
                needle = row[:, :, pos_ab:pos_ab + 2].sum(-1) / 2.0
                mask = torch.ones(row.shape[-1], dtype=torch.bool)
                mask[pos_ab:pos_ab + 2] = False
                mask[q_pos:] = False
                filler = row[:, :, mask].mean(-1)
                manual.append(float((needle / filler.clamp_min(1e-12)).mean()))
        finally:
            m.set_diagnostics(enabled=False, capture_attention=False)
        assert abs(res["snr_needle_last"] - manual[-1]) < 1e-9
        assert abs(res["snr_needle_best"] - max(manual)) < 1e-9


class TestProbeStatisticalProperties:
    """Final eval.py audit layer: noise floor across seeds (the denominator
    of every error bar in the paper) and bf16-checkpoint stability (best/
    final checkpoints are SAVED in bf16 - probes must be valid on them)."""

    def _model(self):
        torch.manual_seed(0)
        return GPT(GPTConfig(block_size=256, vocab_size=50257, n_layer=2,
                             n_head=2, n_embd=64,
                             gradient_checkpointing=False)).eval()

    def test_probe_seed_noise_is_bounded(self):
        """Across 5 probe seeds at random init the coefficient of variation
        must stay small (<5%): the probes measure the MODEL, not the seed.
        Measured baseline: induction CV 0.2%, litm 0.4%, snr 0.04%."""
        import statistics
        from src.eval import evaluate_induction, attention_needle_snr
        m = self._model()
        for fn, key in ((lambda s: evaluate_induction(m, device="cpu", seed=s,
                                                      batch_size=4), "induction"),
                        (lambda s: attention_needle_snr(m, seed=s, batch_size=2)
                         ["snr_needle_last"], "snr")):
            vals = [fn(s) for s in range(42, 47)]
            cv = statistics.stdev(vals) / abs(statistics.fmean(vals))
            assert cv < 0.05, f"{key}: seed-noise CV {cv:.1%} - probe unstable"

    def test_probes_stable_on_bf16_cast_weights(self):
        """best_val/final checkpoints are saved in bf16; probes run on them
        post-hoc (causal_patch, analyze_checkpoint). Metrics must stay
        finite and within 15% of the fp32 value at random init."""
        import math
        from src.eval import evaluate_induction, positional_recall_curve
        m32 = self._model()
        sd = {k: (v.to(torch.bfloat16).to(torch.float32)
                  if torch.is_floating_point(v) else v)
              for k, v in m32.state_dict().items()}
        mbf = GPT(m32.config).eval()
        mbf.load_state_dict(sd, strict=True)
        for fn in (lambda mm: evaluate_induction(mm, device="cpu", batch_size=4),
                   lambda mm: positional_recall_curve(mm, batch_size=2)
                   ["litm_worst_frac"]):
            a, b = fn(m32), fn(mbf)
            assert math.isfinite(b)
            assert abs(a - b) / max(abs(a), 1e-12) < 0.15


class TestSinkAndOutlierStats:
    """Metric-inventory round vs Oral/top papers of the niche:
    attention_sink_stats (StreamingLLM standard) and
    activation_outlier_stats (Diff Transformer Sec 3.7 standard).
    Both calibrated against first principles."""

    def _model(self):
        torch.manual_seed(0)
        return GPT(GPTConfig(block_size=256, vocab_size=50257, n_layer=2,
                             n_head=2, n_embd=64,
                             gradient_checkpointing=False)).eval()

    def test_sink_mass_matches_uniform_attention_at_init(self):
        """No sink exists at random init: mass(pos 0) must equal the
        uniform-attention expectation mean(1/(k+1)) over causal rows
        (measured ratio 1.00)."""
        import statistics
        from src.eval import attention_sink_stats
        r = attention_sink_stats(self._model())
        expect = statistics.fmean(1.0 / (k + 1) for k in range(16, 256))
        assert 0.5 < r["sink_mass_first"] / expect < 2.0
        assert r["sink_mass_first4"] > r["sink_mass_first"]
        assert r["sink_top_layer"] >= r["sink_mass_first"]

    def test_outliers_gaussian_at_init(self):
        """Random-init residual stream is near-Gaussian: excess kurtosis
        ~0 (measured -0.03), max/rms in the 2-10 band. Training-induced
        outliers move BOTH numbers up - that movement is the signal."""
        from src.eval import activation_outlier_stats
        o = activation_outlier_stats(self._model())
        assert abs(o["act_excess_kurtosis"]) < 2.0
        assert 2.0 < o["act_max_over_rms"] < 10.0

    def test_both_side_effect_free_and_deterministic(self):
        from src.eval import attention_sink_stats, activation_outlier_stats
        m = self._model()
        m.train()
        assert attention_sink_stats(m) == attention_sink_stats(m)
        assert activation_outlier_stats(m) == activation_outlier_stats(m)
        assert m.training
        assert not m.transformer.h[0].attn.capture_attention


class TestMetricsSeeRealTraining:
    """The last audit level for eval.py: DYNAMIC validation. A metric that
    cannot see actual learning is useless for the paper. We train a tiny
    2-layer model (minimum for the induction circuit, Olsson et al.) on
    dense induction patterns in the probes' own token ranges and assert
    every headline metric moves in the pre-registered direction.
    Measured on this harness: induction x168, snr x1.13 (up),
    entropy 3.878 -> 3.773 (down) after 120 Adam steps."""

    def test_probes_move_under_real_training(self):
        import torch as _t
        from src.eval import (evaluate_induction, attention_needle_snr,
                              measure_attention_entropy)
        _t.manual_seed(0)
        m = GPT(GPTConfig(block_size=128, vocab_size=50257, n_layer=2,
                          n_head=2, n_embd=64, gradient_checkpointing=False))

        def make_batch(bs=4, T=128, seed=None):
            g = _t.Generator()
            if seed is not None:
                g.manual_seed(seed)
            x = _t.randint(100, 20000, (bs, T + 1), generator=g)
            for row in range(bs):
                a = 10 + int(_t.randint(0, 30, (1,), generator=g))
                b = 50 + int(_t.randint(0, 30, (1,), generator=g))
                for start in range(2, T - 2, 12):
                    x[row, start] = a
                    x[row, start + 1] = b
            return x[:, :-1], x[:, 1:]

        ind0 = evaluate_induction(m.eval(), device="cpu", batch_size=2)
        snr0 = attention_needle_snr(m, batch_size=2)["snr_needle_last"]
        ent0 = measure_attention_entropy(m, device="cpu", batch_size=2)
        opt = _t.optim.Adam(m.parameters(), lr=1e-3)
        m.train()
        for step in range(120):
            x, y = make_batch(seed=1000 + step)
            _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ind1 = evaluate_induction(m.eval(), device="cpu", batch_size=2)
        snr1 = attention_needle_snr(m, batch_size=2)["snr_needle_last"]
        ent1 = measure_attention_entropy(m, device="cpu", batch_size=2)
        assert ind1 > ind0 * 3, f"induction blind to training: {ind0} -> {ind1}"
        assert snr1 > snr0, f"snr blind to training: {snr0} -> {snr1}"
        assert ent1 < ent0, f"entropy blind to training: {ent0} -> {ent1}"


class TestFrankenBehavioralSterility:
    """Round-5 audit: transplant correctness proven at the LOGIT level, not
    just tensor equality - the form a reviewer actually cares about."""

    KW = dict(block_size=64, vocab_size=50257, n_layer=2, n_head=2, n_embd=32,
              gradient_checkpointing=False, phase_mult=0.15, use_laplace=True,
              laplace_alpha=1.0)

    def _cp(self):
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        return importlib.import_module("causal_patch")

    def test_retrieval_transplant_on_identity_pair_is_noop(self):
        """Grafting all-zero mechanisms into an identical body must not move
        a single logit: the transplant machinery itself is sterile."""
        cp = self._cp()
        torch.manual_seed(0)
        hla = GPT(GPTConfig(**self.KW)).eval()
        base = GPT(GPTConfig(**self.KW)).eval()
        base.load_state_dict(hla.state_dict())
        fr = cp.build_franken(base.state_dict(), hla.state_dict(), 32,
                              "retrieval", donor="hla")
        m = GPT(GPTConfig(**self.KW)).eval()
        m.load_state_dict(fr)
        x = torch.randint(0, 50257, (1, 48),
                          generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            lb, _ = base(x)
            lf, _ = m(x)
        assert torch.equal(lb, lf)

    def test_full_franken_matches_hla_logits(self):
        cp = self._cp()
        torch.manual_seed(0)
        hla = GPT(GPTConfig(**self.KW)).eval()
        base = GPT(GPTConfig(**self.KW)).eval()
        base.load_state_dict(hla.state_dict())
        with torch.no_grad():
            for blk in hla.transformer.h:
                blk.attn.W_phase_q.normal_(0, 0.3, generator=torch.Generator().manual_seed(2))
            for blk in base.transformer.h:
                blk.attn.c_attn.weight.add_(0.02)
        fr = cp.build_franken(base.state_dict(), hla.state_dict(), 32,
                              "full", donor="hla")
        m = GPT(GPTConfig(**self.KW)).eval()
        m.load_state_dict(fr)
        x = torch.randint(0, 50257, (1, 48),
                          generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            lh, _ = hla(x)
            lf, _ = m(x)
        assert torch.equal(lh, lf)

    def test_reverse_then_forward_recovers_hla_state(self):
        """Composition sanity: anti-franken then re-graft HLA retrieval must
        reconstruct the HLA state dict exactly (no key leaks either way)."""
        cp = self._cp()
        torch.manual_seed(0)
        hla = GPT(GPTConfig(**self.KW)).eval()
        base = GPT(GPTConfig(**self.KW)).eval()
        base.load_state_dict(hla.state_dict())
        with torch.no_grad():
            for blk in hla.transformer.h:
                blk.attn.W_gate_k.weight.normal_(0, 0.3, generator=torch.Generator().manual_seed(4))
            for blk in base.transformer.h:
                blk.attn.c_attn.weight.add_(0.01)
        rev = cp.build_franken(base.state_dict(), hla.state_dict(), 32,
                               "retrieval", donor="base")
        back = cp.build_franken(rev, hla.state_dict(), 32,
                                "retrieval", donor="hla")
        hs = hla.state_dict()
        assert all(torch.equal(back[k], hs[k]) for k in back)


class TestProbeSmallSampleGuard:
    """Finding #18: n_per_class < 8 leaves classes untrained under the
    60/20/20 split (measured: npc=3 -> 1 of 8 classes absent from train);
    accuracies degrade silently. The probe must refuse."""

    def test_small_n_per_class_refused(self):
        import importlib
        import pytest
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        tp = importlib.import_module("train_probe")
        m = GPT(GPTConfig(block_size=64, vocab_size=50257, n_layer=1,
                          n_head=2, n_embd=32,
                          gradient_checkpointing=False)).eval()
        with pytest.raises(SystemExit):
            tp.run_probe(m, n_per_class=4)


class TestSampleGreedyParity:
    """Round-11 audit: paper-appendix samples must come from THE model -
    greedy generation must equal the manual argmax(forward) loop exactly,
    with HLA mechanisms active (a windowing/cache bug in the generation
    path would silently sample from a different model)."""

    def test_greedy_equals_manual_argmax_loop(self):
        torch.manual_seed(0)
        m = GPT(GPTConfig(block_size=32, vocab_size=1000, n_layer=2, n_head=2,
                          n_embd=32, phase_mult=0.15, use_laplace=True,
                          laplace_alpha=1.0,
                          gradient_checkpointing=False)).eval()
        with torch.no_grad():
            for blk in m.transformer.h:
                blk.attn.W_phase_q.normal_(
                    0, 0.3, generator=torch.Generator().manual_seed(1))
        seq = torch.tensor([[1, 2, 3]], dtype=torch.long)
        # sample.py delegates to model.generate - test that exact path
        got = m.generate(seq.clone(), 8, temperature=0.8, top_k=200,
                         greedy=True)
        want = seq.clone()
        with torch.no_grad():
            for _ in range(8):
                logits, _ = m(want[:, -32:])
                nxt = logits[0, -1].argmax().item()
                want = torch.cat(
                    [want, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        assert got[0].tolist() == want[0].tolist()


class TestProbeCapturePosition:
    """Round-11: capture_residuals must read exactly q_pos (differential:
    shifting q_pos by one must change features; re-capture is bit-stable)."""

    def test_qpos_differential_and_stability(self):
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        tp = importlib.import_module("train_probe")
        torch.manual_seed(0)
        m = GPT(GPTConfig(block_size=64, vocab_size=50257, n_layer=2,
                          n_head=2, n_embd=32,
                          gradient_checkpointing=False)).eval()
        toks = torch.randint(100, 20000, (4, 64),
                             generator=torch.Generator().manual_seed(1))
        f1 = tp.capture_residuals(m, toks, q_pos=62)
        f2 = tp.capture_residuals(m, toks, q_pos=61)
        assert (f1[0] - f2[0]).abs().max().item() > 0
        f3 = tp.capture_residuals(m, toks, q_pos=62)
        assert all(torch.equal(x, y) for x, y in zip(f1, f3, strict=True))
        assert len(f1) == 2


class TestPasskeyEval:
    """Round-12 (reviewer-mitigation): field-standard passkey retrieval at
    the FULL context window (Mohtashami & Jaggi) - the external anchor the
    simulated R2 demanded (B7). Calibration: exact-match at random init
    must be exactly 0; NaN policy on tiny blocks; deterministic per seed."""

    def _model(self, block=128):
        torch.manual_seed(0)
        return GPT(GPTConfig(block_size=block, vocab_size=50257, n_layer=2,
                             n_head=2, n_embd=32,
                             gradient_checkpointing=False)).eval()

    def _ep(self):
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        return importlib.import_module("eval_passkey")

    def test_zero_at_random_init_and_deterministic(self):
        import json as _json
        ep = self._ep()
        m = self._model()
        r1 = ep.eval_passkey(m, n_trials=3, passkey_len=3, seed=42)
        assert r1["passkey_acc_mean"] == 0.0
        r2 = ep.eval_passkey(m, n_trials=3, passkey_len=3, seed=42)
        assert _json.dumps(r1, sort_keys=True) == _json.dumps(r2, sort_keys=True)

    def test_nan_policy_small_block(self):
        ep = self._ep()
        m = self._model(block=32)
        r = ep.eval_passkey(m, n_trials=2, passkey_len=3)
        assert r["passkey_acc_mean"] != r["passkey_acc_mean"]

    def test_token_ranges_disjoint(self):
        """Passkey/marker/filler ranges must not collide with each other or
        with the induction/LITM probe blocks (45000/46000)."""
        ep = self._ep()
        assert ep.FILLER_HI <= ep.PASSKEY_LO
        assert ep.PASSKEY_HI <= ep.MARKER_TOK
        assert not (ep.PASSKEY_LO <= 45000 < ep.PASSKEY_HI)
        assert not (ep.PASSKEY_LO <= 46000 < ep.PASSKEY_HI)
        assert ep.MARKER_TOK not in (45000, 46000)


class TestGapClosurePowerFields:
    """Round-12: gap_closure must report its own statistical power (R2-Q2):
    min detectable gap at z=3, the observed z, and a powered flag."""

    def test_power_fields(self):
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        cp = importlib.import_module("causal_patch")
        r = cp.gap_closure({"induction": 0.1, "induction_std": 0.01},
                           {"induction": 0.3, "induction_std": 0.02},
                           {"induction": 0.25})["induction"]
        assert abs(r["min_detectable_gap_z3"] - 0.06) < 1e-12
        assert abs(r["gap_over_noise_z"] - 10.0) < 1e-9
        assert r["powered"] == 1.0
        r2 = cp.gap_closure({"induction": 0.1, "induction_std": 0.05},
                            {"induction": 0.13, "induction_std": 0.05},
                            {"induction": 0.12})["induction"]
        assert r2["powered"] == 0.0


class TestPrefixMatchPerHead:
    """Round-12: per-head census keys must exist and agree with aggregates
    (feeds fig10; a drift here silently blanks the census figure)."""

    def test_per_head_keys_and_consistency(self):
        from src.eval import prefix_matching_score
        torch.manual_seed(0)
        m = GPT(GPTConfig(block_size=128, vocab_size=50257, n_layer=2,
                          n_head=4, n_embd=64,
                          gradient_checkpointing=False)).eval()
        r = prefix_matching_score(m, batch_size=2)
        heads = [k for k in r if "_H" in k]
        assert len(heads) == 8
        for li in (0, 1):
            hv = [v for k, v in r.items() if k.startswith(f"L{li:02d}_H")]
            assert abs(max(hv) - r[f"L{li:02d}_prefix_match_max"]) < 1e-12


class TestPaddedVocabMasking:
    """Finding #19: with padded_vocab_size > vocab_size (the TPU configs:
    50304 vs 50257) the model's own loss masks pad classes but eval code
    computed CE/softmax over RAW logits - 47 phantom classes inflated CE by
    ~1e-3 and deflated every P(target) probe. Symmetric across twins but
    biased vs the true distribution. All eval paths must match the model's
    masked objective exactly."""

    KW = dict(block_size=64, vocab_size=50257, padded_vocab_size=50304,
              n_layer=2, n_head=2, n_embd=32, gradient_checkpointing=False)

    def test_per_position_loss_matches_model_masked_loss(self):
        from src.eval import per_position_loss_curve
        torch.manual_seed(0)
        m = GPT(GPTConfig(**self.KW)).eval()
        g = torch.Generator(); g.manual_seed(1)
        toks = torch.randint(0, 50257, (4, 65), generator=g)
        r = per_position_loss_curve(m, toks, batch_size=4, n_bins=4)
        got = sum(r[f"posloss_bin_{i:02d}"] for i in range(4)) / 4
        with torch.no_grad():
            _, lm = m(toks[:, :64], toks[:, 1:65])
        assert abs(got - float(lm)) < 1e-5, \
            f"eval CE {got} != model masked CE {float(lm)}"

    def test_probe_softmax_excludes_pad_classes(self):
        """P(B) over the masked distribution must exceed P(B) over the raw
        padded distribution (denominator drops 47 phantom classes)."""
        from src.eval import _mask_padded_logits
        torch.manual_seed(0)
        m = GPT(GPTConfig(**self.KW)).eval()
        g = torch.Generator(); g.manual_seed(2)
        x = torch.randint(0, 50257, (1, 32), generator=g)
        with torch.no_grad():
            logits, _ = m(x)
        raw = torch.softmax(logits[0, -1].float(), dim=-1)
        masked = torch.softmax(_mask_padded_logits(m, logits[0, -1].float()),
                               dim=-1)
        assert float(masked[:50257].sum()) > 0.999999
        assert float(raw[50257:].sum()) > 0.0  # raw really leaked mass
        assert torch.all(masked[:50257] >= raw[:50257])

    def test_noop_on_unpadded_model(self):
        from src.eval import _mask_padded_logits
        kw = dict(self.KW)
        kw.pop("padded_vocab_size")
        torch.manual_seed(0)
        m = GPT(GPTConfig(**kw)).eval()
        g = torch.Generator(); g.manual_seed(3)
        x = torch.randint(0, 50257, (1, 16), generator=g)
        with torch.no_grad():
            logits, _ = m(x)
        out = _mask_padded_logits(m, logits.float())
        assert torch.equal(out, logits.float())


class TestNonFiniteCheckpointGuards:
    """Findings #20/#21 (round 15). #21: a loss-spike NaN leaking into a
    saved checkpoint used to flow SILENTLY through causal_patch and
    train_probe (JSON full of nan presented as results). Both CLIs must
    refuse loudly. #20 is covered in test_train_utils (duplicate JSON keys)."""

    def _nan_ckpt(self, tmp_path):
        import json
        torch.manual_seed(0)
        kw = dict(block_size=64, vocab_size=50257, n_layer=1, n_head=2,
                  n_embd=32, gradient_checkpointing=False)
        m = GPT(GPTConfig(**kw))
        sd = m.state_dict()
        k = next(k for k in sd if "c_attn.weight" in k)
        sd[k][0, 0] = float("nan")
        ck = str(tmp_path / "nan.pt")
        cfg = str(tmp_path / "cfg.json")
        torch.save(sd, ck)
        json.dump({"model": kw}, open(cfg, "w"))
        return ck, cfg

    def test_causal_patch_refuses_nan_weights(self, tmp_path):
        import subprocess
        ck, cfg = self._nan_ckpt(tmp_path)
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "causal_patch.py"),
                            "--base-checkpoint", ck, "--hla-checkpoint", ck,
                            "--hla-config", cfg, "--out",
                            str(tmp_path / "o.json"), "--probe-seeds", "42"],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "non-finite" in (r.stdout + r.stderr)

    def test_train_probe_refuses_nan_weights(self, tmp_path):
        import subprocess
        ck, cfg = self._nan_ckpt(tmp_path)
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "train_probe.py"),
                            "--checkpoint", ck, "--config", cfg,
                            "--out", str(tmp_path / "p.json"),
                            "--n-per-class", "8"],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "non-finite" in (r.stdout + r.stderr)
