# Copyright 2026 Slyatski Ilya
"""Round 79: mini 124M-class (gpt2-small) speed-smoke matrix contract (BD/BF).

Mini configs let anyone (incl. top labs) validate a backend in minutes
before committing to full runs. Contract: tiny arch is IDENTICAL across
the matrix except attention_backend; mechanisms stay ON for hla variants;
GPU minis must not point into /kaggle.
"""
import glob
import json

MINIS = sorted(glob.glob("configs/mini_124m_*.json") +
               glob.glob("configs/gpu_mini_124m_*.json"))
ARCH_KEYS = ("n_embd", "n_layer", "n_head", "block_size", "vocab_size")

class TestMiniSpeedMatrix:
    def test_matrix_present(self):
        names = {p.split("/")[-1] for p in MINIS}
        assert names == {
            "mini_124m_base_sdpa_s42.json",
            "mini_124m_hla_fold_s42.json",
            "mini_124m_hla_pallas_s42.json",
            "gpu_mini_124m_base_flash_s42.json",
            "gpu_mini_124m_hla_flash_fold_s42.json",
        }

    def test_arch_is_gpt2_small_class(self):
        for p in MINIS:
            c = json.load(open(p))["model"]
            assert (c["n_embd"], c["n_layer"], c["n_head"]) == (768, 12, 12), p

    def test_arch_identical_across_matrix(self):
        cfgs = [json.load(open(p))["model"] for p in MINIS]
        ref = {k: cfgs[0][k] for k in ARCH_KEYS}
        for c in cfgs[1:]:
            assert {k: c[k] for k in ARCH_KEYS} == ref

    def test_backends_correct(self):
        for p in MINIS:
            c = json.load(open(p))["model"]
            name = p.split("/")[-1]
            if "pallas" in name:
                assert c["attention_backend"] == "pallas"
            elif "fold" in name:
                assert c["attention_backend"] == "sdpa_fold"
            else:
                assert c["attention_backend"] == "sdpa"

    def test_hla_mechanisms_on_base_off(self):
        for p in MINIS:
            c = json.load(open(p))["model"]
            if "hla" in p.split("/")[-1]:
                assert c["laplace_alpha"] == 1.0 and c["phase_mult"] > 0
            else:
                assert c["laplace_alpha"] == 0.0 and c["phase_mult"] == 0.0

    def test_gpu_minis_use_local_paths(self):
        for p in MINIS:
            if "gpu_" not in p:
                continue
            c = json.load(open(p))
            for key in ("save_dir", "train_path", "val_path"):
                assert not str(c[key]).startswith("/kaggle"), (p, key)

    def test_smoke_budget_small(self):
        for p in MINIS:
            c = json.load(open(p))
            assert c["max_steps"] <= 500 and c["resume_every"] == 0
