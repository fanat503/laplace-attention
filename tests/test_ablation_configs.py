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



"""Tests for the ablation-matrix generator: single-factor discipline."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_ablation_configs import ARMS, ZERO_KEYS, make_arm  # noqa: E402


@pytest.fixture()
def templates():
    with open(os.path.join(ROOT, "configs/200m_base_v2_s42.json")) as f:
        base = json.load(f)
    with open(os.path.join(ROOT, "configs/200m_hla_v2_s42.json")) as f:
        hla = json.load(f)
    return base, hla


class TestSingleFactor:
    def test_base_arm_all_zero(self, templates):
        base, hla = templates
        _, cfg = make_arm(base, hla, "base", 42, "/tmp/x")
        for k in ZERO_KEYS:
            assert cfg["model"][k] == 0.0, f"{k} must be off in base arm"

    @pytest.mark.parametrize("arm,active_key", [
        ("phase", "phase_mult"),
        ("gates", "laplace_alpha"),
        ("salience", "salience_alpha"),
        ("distance", "distance_laplace_alpha"),
        ("forget", "forget_alpha"),
    ])
    def test_single_mechanism_arms(self, templates, arm, active_key):
        """Each single arm must activate EXACTLY one alpha."""
        base, hla = templates
        _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
        m = cfg["model"]
        for k in ZERO_KEYS:
            if k == active_key:
                assert m[k] != 0.0, f"{arm}: {k} should be active"
            else:
                assert m[k] == 0.0, f"{arm}: {k} leaked into single-factor arm"

    def test_full_matches_template(self, templates):
        base, hla = templates
        _, cfg = make_arm(base, hla, "full", 42, "/tmp/x")
        for k in ZERO_KEYS:
            if k in hla["model"]:
                assert cfg["model"][k] == hla["model"][k]

    def test_shared_init_per_seed(self, templates):
        """All arms of one seed must point to the SAME init checkpoint."""
        base, hla = templates
        inits = set()
        for arm in ARMS:
            _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
            inits.add(cfg["init_ckpt"])
        assert len(inits) == 1, "arms must share one sterile init per seed"

    def test_seeds_get_distinct_names(self, templates):
        base, hla = templates
        n42, _ = make_arm(base, hla, "phase", 42, "/tmp/x")
        n43, _ = make_arm(base, hla, "phase", 43, "/tmp/x")
        assert n42 != n43

    def test_structural_flags_uniform(self, templates):
        """use_* switches identical in every arm => parameter matching holds
        across the whole matrix."""
        base, hla = templates
        flags = ("use_laplace", "use_distance_laplace", "use_salience_bias", "use_forget_gate", "use_qtemp")
        reference = None
        for arm in ARMS:
            _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
            vals = tuple(cfg["model"][f] for f in flags)
            if reference is None:
                reference = vals
            assert vals == reference, f"{arm}: structural flags differ"


class TestCLI:
    def test_end_to_end_generation(self, tmp_path, templates):
        out = str(tmp_path / "abl")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/make_ablation_configs.py"),
             "--base", os.path.join(ROOT, "configs/200m_base_v2_s42.json"),
             "--hla", os.path.join(ROOT, "configs/200m_hla_v2_s42.json"),
             "--outdir", out, "--seeds", "42", "43"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        manifest = json.load(open(os.path.join(out, "MANIFEST.json")))
        assert len(manifest["configs"]) == 2 * len(ARMS)
        # every emitted config must construct a valid GPTConfig
        from src.model import GPTConfig
        for p in manifest["configs"]:
            cfg = json.load(open(p))
            GPTConfig(**cfg["model"])


class TestLeaveOneOut:
    @pytest.mark.parametrize("arm,dropped", [
        ("no_phase", "phase_mult"),
        ("no_gates", "laplace_alpha"),
        ("no_salience", "salience_alpha"),
        ("no_distance", "distance_laplace_alpha"),
    ])
    def test_drops_exactly_one(self, templates, arm, dropped):
        base, hla = templates
        _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
        m = cfg["model"]
        assert m[dropped] == 0.0, f"{arm}: {dropped} must be OFF"
        for k in ZERO_KEYS:
            if k == dropped or k == "forget_alpha":
                continue  # forget not in HLA template -> stays 0
            if k in hla["model"] and hla["model"][k] != 0.0:
                assert m[k] == hla["model"][k], f"{arm}: {k} must keep template value"


class TestArmConfigValidation:
    """Review attack P2: every generated arm pair must pass validate_configs
    (forget_alpha was missing from the allow-list before this test existed)."""

    def test_all_arms_pass_pair_validator(self, templates, tmp_path):
        import subprocess
        base, hla = templates
        base_name, base_cfg = make_arm(base, hla, "base", 42, str(tmp_path))
        base_path = tmp_path / f"{base_name}.json"
        with open(base_path, "w") as f:
            json.dump(base_cfg, f)
        for arm in ARMS:
            if arm == "base":
                continue
            name, cfg = make_arm(base, hla, arm, 42, str(tmp_path))
            p = tmp_path / f"{name}.json"
            with open(p, "w") as f:
                json.dump(cfg, f)
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts/validate_configs.py"),
                 "--base", str(base_path), "--hla", str(p)],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, f"arm '{arm}' rejected by validator:\n{r.stdout}{r.stderr}"


class TestSharedInitCompatibility:
    """Review attack N6/N7: one init per seed must strict-load into EVERY arm."""

    def test_base_arm_init_loads_into_all_arms(self, templates):
        from src.model import GPT, GPTConfig
        base, hla = templates
        small = dict(n_layer=2, n_head=2, n_embd=32, block_size=64,
                     vocab_size=256, gradient_checkpointing=False)

        def small_model(arm):
            _, cfg = make_arm(base, hla, arm, 42, "/tmp/x")
            mc = dict(cfg["model"]); mc.update(small)
            return GPT(GPTConfig(**mc))

        donor = small_model("base").state_dict()
        for arm in ARMS:
            m = small_model(arm)
            m.load_state_dict(donor, strict=True)  # raises on mismatch

    def test_manifest_contains_init_commands(self, templates, tmp_path):
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/make_ablation_configs.py"),
             "--base", os.path.join(ROOT, "configs/200m_base_v2_s42.json"),
             "--hla", os.path.join(ROOT, "configs/200m_hla_v2_s42.json"),
             "--outdir", str(tmp_path), "--seeds", "42", "43"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        man = json.load(open(tmp_path / "MANIFEST.json"))
        assert "init_commands" in man
        assert "42" in man["init_commands"] and "43" in man["init_commands"]
        assert "make_init" in man["init_commands"]["42"]


class TestDocsConsistency:
    """Review attack K1 (Karpathy-style 'the README must not lie'):
    documentation numbers are cross-checked against reality by CI."""

    def test_readme_test_count_matches_collected(self):
        import re
        import subprocess
        readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        # Only TOTAL-count claims (badge, layout total, quick-start echo,
        # section header) - NOT per-file mentions like "test_theory.py (9 tests)".
        claimed = set(int(m) for m in re.findall(r"tests-(\d+)%20passing", readme))
        claimed |= set(int(m) for m in re.findall(r"·\s*(\d+) tests", readme))
        claimed |= set(int(m) for m in re.findall(r"(\d+) CPU tests", readme))
        claimed |= set(int(m) for m in re.findall(r"→ (\d+) passed", readme))
        # \D{0,4} (non-digits!) so backtracking can't split "188" into 18+8
        claimed |= set(int(m) for m in re.findall(r"## \D{0,4}(\d+) tests", readme))
        r = subprocess.run(
            [sys.executable, "-m", "pytest", os.path.join(ROOT, "tests"),
             "--collect-only", "-q", "-p", "no:cacheprovider"],
            capture_output=True, text=True)
        # pytest -q --collect-only prints per-file lines "path: N"
        per_file = re.findall(r": (\d+)\s*$", r.stdout, re.M)
        actual = sum(int(n) for n in per_file) if per_file else -1
        assert actual > 0, f"collection failed: {r.stdout[-300:]}"
        wrong = {c for c in claimed if c != actual}
        assert not wrong, (
            f"README claims test counts {sorted(wrong)} but actual is {actual}. "
            f"Update README badges/sections."
        )


class TestToolingIntegrity:
    """Final review sweep: helper scripts must work on the SHIPPED files."""

    def test_check_environment_parses_range_specs(self):
        """F1: parse_pins must understand range specs, not only pkg==x.y -
        previously requirements.txt parsed to {} and the check passed vacuously."""
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        ce = importlib.import_module("check_environment")
        specs = ce.parse_pins(os.path.join(ROOT, "requirements.txt"))
        assert "torch" in specs and specs["torch"], "torch range spec must be parsed"
        assert ce.spec_satisfied("2.5.0", [(">=", "2.4"), ("<", "2.15")])
        assert not ce.spec_satisfied("2.15.1", [(">=", "2.4"), ("<", "2.15")])

    def test_check_environment_rejects_empty_specfile(self, tmp_path):
        import subprocess
        p = tmp_path / "empty.txt"
        p.write_text("# only comments\n")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/check_environment.py"),
             "--requirements", str(p)], capture_output=True, text=True)
        assert r.returncode != 0, "empty spec file must fail, not pass vacuously"

    def test_preflight_references_existing_files(self):
        """F2: every path preflight.sh invokes must exist in the repo."""
        sh = open(os.path.join(ROOT, "scripts/preflight.sh")).read()
        assert "requirements_tpu.txt" not in sh, "preflight references a non-existent file"
        assert "python src/make_init.py" in sh, "make_init lives in src/"

    def test_configs_readme_matches_validator(self):
        """F3: configs/README allow-list must be a superset-consistent view of
        the actual validator (script is the source of truth)."""
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        vc = importlib.import_module("validate_configs")
        readme = open(os.path.join(ROOT, "configs/README.md")).read()
        for key in sorted(vc.ALLOWED_MAIN_DIFFS):
            assert f"`{key}`" in readme, f"configs/README.md missing allowed key {key}"

    def test_profile_flops_analytic_no_model_instantiation(self):
        """OOM fix: profiling must not allocate model weights."""
        src = open(os.path.join(ROOT, "scripts/profile_flops.py")).read()
        assert "no weights allocated" in src
        assert "hidden = int(8 * C / 3)" in src, "MLP hidden must not be scaled by L (old bug)"


class TestFinalPolishSweep:
    """Regression tests for the final polish round: every doc/tool must track
    the range_flex knob and the qtemp/W_gate_d mechanisms (no silent drift)."""

    def test_audit_uses_range_flex_not_hardcoded(self):
        """audit_config_values must derive the envelope from the config's
        range_flex, not the historical hardcoded 1.25."""
        src = open(os.path.join(ROOT, "scripts/audit_config_values.py")).read()
        assert 'm.get("range_flex", 0.25)' in src, "audit must read range_flex from config"
        assert "alpha * rng * 1.25" not in src, "hardcoded 1.25 must be gone from envelope math"
        assert "alpha * rng_k * 1.25" not in src, "hardcoded 1.25 must be gone from E8 too"

    def test_perturbation_bounds_uses_model_range_flex(self):
        src = open(os.path.join(ROOT, "src/eval.py")).read()
        assert 'getattr(attn, "range_flex", 0.25)' in src, (
            "perturbation_bounds must read range_flex from the model")

    def test_profile_flops_counts_qtemp_and_distance_gate(self):
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        pf = importlib.import_module("profile_flops")
        from src.model import GPTConfig
        common = dict(block_size=128, vocab_size=512, n_layer=2, n_head=2,
                      n_embd=64, gradient_checkpointing=False)
        base = type("M", (), {"config": GPTConfig(**common)})()
        qt = type("M", (), {"config": GPTConfig(use_qtemp=True, qtemp_alpha=1.0, **common)})()
        fb = pf.count_forward_flops(base, batch_size=1, seq_len=64)
        fq = pf.count_forward_flops(qt, batch_size=1, seq_len=64)
        assert fb["qtemp"] == 0 and fq["qtemp"] > 0, "qtemp FLOPs must be counted when active"
        assert fq["total"] > fb["total"]
        # distance must include its own W_gate_d projection (C->H), not just T*T
        dm = type("M", (), {"config": GPTConfig(use_distance_laplace=True,
                                                distance_laplace_alpha=0.5, **common)})()
        fd = pf.count_forward_flops(dm, batch_size=1, seq_len=64)
        B, T, C, H, L = 1, 64, 64, 2, 2
        assert fd["distance_bias"] >= L * (2 * B * T * C * H + 2 * B * H * T * T), (
            "distance FLOPs must include the dedicated W_gate_d projection")

    def test_trainer_csv_logs_qtemp_columns(self):
        """qtemp diagnostics existed in hla_statistics but never reached the
        CSV - the qtemp ablation arm would have produced no mechanism curve."""
        src = open(os.path.join(ROOT, "src/train_xla.py")).read()
        assert '"qtemp_mean"' in src, "CSV header must include qtemp_mean"
        assert '"qtemp_sat_frac"' in src, "CSV header must include qtemp_sat_frac"
        assert 'metrics.get("qtemp_mean"' in src, "CSV row must write qtemp_mean"
        assert 'metrics.get("qtemp_sat_frac"' in src, "CSV row must write qtemp_sat_frac"

    def test_trainer_csv_header_row_same_length(self):
        """Header list and row list in train_xla.py must stay in lockstep."""
        import re as _re
        src = open(os.path.join(ROOT, "src/train_xla.py")).read()
        hdr_m = _re.search(r'writer\.writerow\(\[\s*"step",(.*?)\]\)', src, _re.S)
        assert hdr_m, "CSV header writerow not found"
        n_header = len(_re.findall(r'"[a-z_0-9]+"', hdr_m.group(0)))
        row_m = _re.search(r'writer\.writerow\(\[\s*completed_step,(.*?)\]\)', src, _re.S)
        assert row_m, "CSV data writerow not found"
        body = row_m.group(0)
        n_row = body.count("\n") - 1  # one entry per line in the literal
        assert n_header == n_row, (
            f"CSV header has {n_header} columns but data row writes {n_row}")

    def test_audit_has_qtemp_check(self, templates, tmp_path):
        """E11: the qtemp arm's values must be auditable, not invisible."""
        import subprocess
        base, hla = templates
        _, cfg = make_arm(base, hla, "qtemp", 42, str(tmp_path))
        p = tmp_path / "qtemp_arm.json"
        with open(p, "w") as f:
            json.dump(cfg, f)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/audit_config_values.py"),
             "--config", str(p)], capture_output=True, text=True)
        assert "E11" in r.stdout, "audit must report E11 for an active qtemp config"
        assert "FAIL" not in r.stdout.replace("0 fail", ""), r.stdout

    def test_docs_arm_count_matches_code(self):
        """Anti-drift: every 'N arms' claim in README/EXPERIMENT_CARD must
        equal len(ARMS). (Caught a real 15-vs-14 drift during review.)"""
        import re as _re
        n = len(ARMS)
        for rel in ("README.md", "docs/EXPERIMENT_CARD.md"):
            text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
            claims = [int(m) for m in _re.findall(r"(\d+) arms", text)]
            for c in claims:
                assert c == n, f"{rel} claims {c} arms but code has {n}"

    def test_structural_flags_cover_all_use_switches(self):
        """The uniformity test above is only as good as its flag list: it must
        include every use_* mechanism switch GPTConfig knows about."""
        import dataclasses
        from src.model import GPTConfig
        cfg_flags = {f.name for f in dataclasses.fields(GPTConfig)
                     if f.name.startswith("use_") and f.name not in ("use_wpe", "use_rope")}
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "test_ablation_configs.py"), encoding="utf-8").read()
        for flag in cfg_flags:
            assert f'"{flag}"' in src, (
                f"structural-flags test does not check {flag}; add it to the tuple")


class TestExternalReviewRound2:
    """Regression tests for the second external review (B1/B4/B6/B7)."""

    def test_dual_positional_encoding_rejected(self):
        """B4: use_wpe=True + use_rope=True must raise, not silently dual-encode."""
        from src.model import GPTConfig
        with pytest.raises(ValueError, match="exactly one positional scheme"):
            GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                      n_embd=32, use_wpe=True, use_rope=True)

    def test_trainer_importable_and_helpful_without_xla(self):
        """B1: on a CPU-only host the trainer must import cleanly and give a
        clear SystemExit from main(), not ModuleNotFoundError at line 1."""
        import subprocess
        code = (
            "import sys; sys.modules['torch_xla'] = None\n"  # simulate absence even if installed
            "import importlib.util, os\n"
            "spec = importlib.util.spec_from_file_location('train_xla', os.path.join(%r, 'src', 'train_xla.py'))\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "assert m.xm is None or m.xm is not None  # import survived\n"
            "print('IMPORT_OK')\n"
        ) % ROOT
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert "IMPORT_OK" in r.stdout, f"trainer import must survive missing torch_xla:\n{r.stderr[-500:]}"

    def test_forget_arm_rejects_global_bf16_env(self, monkeypatch):
        """B7: forget_alpha != 0 + XLA_USE_BF16=1 must fail fast (bf16 cumsum
        silently swallows forgetting steps near |S|~200)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_xla_b7", os.path.join(ROOT, "src", "train_xla.py"))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pytest.skip("trainer refused to import in this env")
        cfg = {"seed": 1, "save_dir": "/tmp/x", "train_path": "a", "val_path": "b",
               "batch_size_per_device": 1, "eval_batch_size_per_device": 1,
               "grad_accum": 1, "max_steps": 1, "lr": 1e-4, "min_lr": 1e-5,
               "warmup": 0, "model": {"forget_alpha": 1.0}}
        monkeypatch.setenv("XLA_USE_BF16", "1")
        with pytest.raises(ValueError, match="fp32 cumsum"):
            mod.validate_config(cfg)
        monkeypatch.delenv("XLA_USE_BF16")
        monkeypatch.setenv("XLA_DOWNCAST_BF16", "1")
        with pytest.raises(ValueError, match="fp32 cumsum"):
            mod.validate_config(cfg)

    def test_experiment_card_declares_mechanism_sets(self):
        """B6: EXPERIMENT_CARD must state which mechanisms each shipped config
        actually activates (capacity != default arm)."""
        card = open(os.path.join(ROOT, "docs", "EXPERIMENT_CARD.md"), encoding="utf-8").read()
        assert "Active mechanism sets per config" in card
        assert "capacity, not the" in card.replace("\n", " ")

    def test_theory_has_rope_phase_commutation(self):
        """A4 (Nanda): the before/after-RoPE ordering question is settled by
        Corollary 7.1, and the numeric claim in it must be true."""
        theory = open(os.path.join(ROOT, "docs", "THEORY.md"), encoding="utf-8").read()
        assert "Corollary 7.1" in theory
        import math
        import torch
        from src.model import GPT, GPTConfig
        torch.manual_seed(0)
        cfg = GPTConfig(block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=32,
                        gradient_checkpointing=False, use_rope=True, use_wpe=False,
                        phase_mult=0.3)
        a = GPT(cfg).transformer.h[0].attn
        B, H, T, hs = 2, 2, 16, 16
        q = torch.randn(B, H, T, hs)
        pos = torch.arange(T, dtype=torch.float32)
        rope_ang = torch.einsum("t,k->tk", pos, a.rope_inv_freq).view(1, 1, T, hs // 2)
        ph_ang = 0.3 * math.pi * torch.randn(B, H, T, hs // 2)

        def rot(x, ang):
            return a._rotate_pairwise(x, torch.cos(ang), torch.sin(ang))

        q_rope_then_phase = rot(rot(q, rope_ang), ph_ang)
        q_phase_then_rope = rot(rot(q, ph_ang), rope_ang)
        q_sum = rot(q, rope_ang + ph_ang)
        assert (q_rope_then_phase - q_phase_then_rope).abs().max() < 1e-5
        assert (q_rope_then_phase - q_sum).abs().max() < 1e-5


class TestFinalHardening:
    """Final 'fix it so it stays fixed' round: every analysis entrypoint must
    work on a CPU-only, memory-constrained host (the post-hoc analysis
    environment), and artifact dirs must be untrackable."""

    def test_prepare_c4_help_works_without_data_deps(self):
        """--help must not require tiktoken/datasets/tqdm (same class as B1)."""
        import subprocess
        code = ("import sys\n"
                "for m in ('tiktoken','datasets','tqdm'):\n"
                "    sys.modules[m] = None\n"  # simulate absence
                "import runpy, sys as s\n"
                "s.argv = ['prepare_c4_data.py', '--help']\n"
                "try:\n"
                "    runpy.run_path(%r, run_name='__main__')\n"
                "except SystemExit as e:\n"
                "    raise SystemExit(0 if e.code in (0, None) else 1)\n"
                % os.path.join(ROOT, "scripts", "prepare_c4_data.py"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, f"--help must survive missing data deps:\n{r.stderr[-400:]}"

    def test_analysis_scripts_default_device_is_auto(self):
        """post-hoc analysis runs on CPU hosts: default --device xla crashed
        with ModuleNotFoundError before this round."""
        for rel in ("scripts/analyze_checkpoint.py", "scripts/compare_attention_kl.py"):
            src = open(os.path.join(ROOT, rel)).read()
            assert 'default="auto"' in src, f"{rel}: --device default must be auto"
            assert 'if name == "auto"' in src, f"{rel}: resolve_device must handle auto"

    def test_compare_kl_clamps_seq_len(self):
        src = open(os.path.join(ROOT, "scripts/compare_attention_kl.py")).read()
        assert 'min(int(seq_len), int(cfg["model"]["block_size"]))' in src, (
            "unclamped --seq-len crashed on block_size+1 sequences")

    def test_divergence_micro_batches(self):
        """plot_divergence held two (B,T,vocab) fp32 logit tensors -> OOM 137
        on small hosts; must accumulate per-row."""
        src = open(os.path.join(ROOT, "scripts/plot_divergence.py")).read()
        assert "for i in range(n_rows)" in src
        assert "kl_sum" in src and "nb2" in src, "cosine must be accumulated, not materialized"

    def test_induction_by_distance_micro_chunks(self):
        src = open(os.path.join(ROOT, "scripts/analyze_checkpoint.py")).read()
        assert "for s in range(0, batch_size, chunk)" in src, (
            "batch=16 forward materializes (16,T,vocab) logits -> OOM")

    def test_analyze_checkpoint_has_knockout_curve(self):
        """The long-context evidence figure must be produced by the standard
        checkpoint analysis, not require custom code at paper time."""
        src = open(os.path.join(ROOT, "scripts/analyze_checkpoint.py")).read()
        assert "knockout_by_context_length" in src
        assert "prefix_matching" in src

    def test_make_plots_has_mechanism_dashboard(self):
        """Every diagnostic CSV column family must be plottable out of the box."""
        src = open(os.path.join(ROOT, "scripts/make_plots.py")).read()
        assert "mechanism_dashboard" in src
        for col in ("qk_interference", "distractor_margin", "mech_grad_mean",
                    "qtemp_sat_frac", "layer_temp_last", "svd_phase_erank"):
            assert col in src, f"dashboard must cover {col}"

    def test_gitignore_blocks_experiment_artifacts(self):
        """Token files are ~21 GB; a stray `git add -A` after data prep must
        not be able to commit them."""
        gi = open(os.path.join(ROOT, ".gitignore")).read()
        for pat in ("data/", "runs/", "inits/", "*.pt", "*.bin"):
            assert f"\n{pat}\n" in gi or gi.endswith(f"\n{pat}") or f"\n{pat}\n" in gi + "\n", (
                f".gitignore must contain {pat}")

    def test_citation_version_matches_pyproject(self):
        import re as _re
        cff = open(os.path.join(ROOT, "CITATION.cff")).read()
        pyp = open(os.path.join(ROOT, "pyproject.toml")).read()
        cff_v = _re.search(r'^version: "([\d.]+)"', cff, _re.M).group(1)
        pyp_v = _re.search(r'^version = "([\d.]+)"', pyp, _re.M).group(1)
        assert pyp_v.startswith(cff_v), (
            f"CITATION.cff version {cff_v} vs pyproject {pyp_v} - keep in sync")


class TestRound5Hardening:
    """Fifth sweep: config typo guard, CRLF hygiene, CI e2e helpers."""

    def _load_trainer(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_xla_r5", os.path.join(ROOT, "src", "train_xla.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_unknown_top_level_key_rejected(self):
        """A typo like warmup_steps (vs warmup) silently fell back to defaults
        and trained with wrong hyperparameters. Must raise, checked BEFORE
        file-existence so it fires on any host."""
        mod = self._load_trainer()
        cfg = {"seed": 1, "save_dir": "s", "train_path": "t", "val_path": "v",
               "batch_size_per_device": 1, "eval_batch_size_per_device": 1,
               "grad_accum": 1, "max_steps": 1, "lr": 1e-4, "min_lr": 1e-5,
               "warmup": 0, "model": {}, "warmup_steps": 500}
        with pytest.raises(KeyError, match="Unknown top-level config keys"):
            mod.validate_config(cfg)

    def test_unknown_key_escape_hatch(self):
        mod = self._load_trainer()
        assert "allow_unknown_config_keys" in mod.KNOWN_TOP_LEVEL_KEYS

    def test_every_known_key_is_actually_used(self):
        """The allow-list must not drift: every KNOWN key (minus meta) must
        appear in the trainer source, else it's a stale entry."""
        mod = self._load_trainer()
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        meta = {"allow_unknown_config_keys", "_doc", "variant"}
        for key in mod.KNOWN_TOP_LEVEL_KEYS - meta:
            assert f'"{key}"' in src, f"stale KNOWN_TOP_LEVEL_KEYS entry: {key}"

    def test_shipped_configs_have_no_unknown_keys(self):
        """Every shipped config must pass the new guard (keys subset check
        only - paths may not exist on CI)."""
        import glob as _glob
        mod = self._load_trainer()
        for path in sorted(_glob.glob(os.path.join(ROOT, "configs", "*.json"))):
            cfg = json.load(open(path))
            unknown = set(cfg) - mod.KNOWN_TOP_LEVEL_KEYS
            assert not unknown, f"{os.path.basename(path)}: unknown keys {unknown}"

    def test_no_crlf_in_tracked_text_files(self):
        """CRLF snuck in via web uploads (prepare_data.py, utils.py) - breaks
        patches and produces noisy diffs. .gitattributes now enforces LF; this
        test keeps the tree itself clean."""
        import glob as _glob
        bad = []
        for pattern in ("src/*.py", "scripts/*.py", "tests/*.py", "docs/*.md",
                        "*.md", "*.toml", "*.txt", ".github/workflows/*.yml"):
            for f in _glob.glob(os.path.join(ROOT, pattern)):
                if b"\r\n" in open(f, "rb").read():
                    bad.append(os.path.relpath(f, ROOT))
        assert not bad, f"CRLF line endings in: {bad}"

    def test_gitattributes_enforces_lf(self):
        ga = os.path.join(ROOT, ".gitattributes")
        assert os.path.exists(ga), ".gitattributes must exist (LF enforcement)"
        content = open(ga).read()
        assert "eol=lf" in content

    def test_ci_check_analysis_script(self, tmp_path):
        """CI e2e gate helper: accepts a good analysis JSON, rejects broken."""
        import subprocess
        good = {"val_loss": 10.9, "induction_by_distance": {}, "prefix_matching": {},
                "attention_entropy": [], "knockout_by_context_length": {"64": {"loss_full": 10.9}}}
        p = tmp_path / "good.json"
        p.write_text(json.dumps(good))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ci_check_analysis.py"), str(p)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        bad = dict(good); del bad["knockout_by_context_length"]
        p2 = tmp_path / "bad.json"
        p2.write_text(json.dumps(bad))
        r2 = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ci_check_analysis.py"), str(p2)],
                            capture_output=True, text=True)
        assert r2.returncode != 0

    def test_ci_workflow_has_cli_and_e2e_smoke(self):
        wf = open(os.path.join(ROOT, ".github", "workflows", "tests.yml")).read()
        assert "CLI smoke" in wf, "CI must gate every script's --help"
        assert "E2E pipeline smoke" in wf, "CI must run the dummy-data pipeline"
        assert "ci_check_analysis.py" in wf


class TestRound6Science:
    """Round 6 (top-reviewer pass): numerical semantics and doc-code parity."""

    def test_distance_bias_normalized_by_model_constant(self):
        """The distance bias must use block_size-1 (model constant), not the
        current batch length: T-normalization broke prefix-stability the
        moment W_gate_d trained (full-vs-prefix diff ~4e-3)."""
        src = open(os.path.join(ROOT, "src", "model.py")).read()
        assert "self.block_size - 1" in src, "distance must normalize by block_size-1"
        import torch
        from src.model import GPT, GPTConfig
        torch.manual_seed(0)
        m = GPT(GPTConfig(block_size=64, vocab_size=128, n_layer=1, n_head=2,
                          n_embd=32, gradient_checkpointing=False,
                          use_rope=True, use_wpe=False,
                          use_distance_laplace=True, distance_laplace_alpha=0.5)).eval()
        with torch.no_grad():
            m.transformer.h[0].attn.W_gate_d.weight.normal_(0, 0.5)
        x = torch.randint(0, 128, (1, 32))
        full, _ = m(x)
        prefix, _ = m(x[:, :16])
        assert torch.allclose(full[0, :16], prefix[0], atol=1e-5), (
            "distance bias must be prefix-stable for a trained gate")

    def test_theory_defines_distance_normalization(self):
        theory = open(os.path.join(ROOT, "docs", "THEORY.md"), encoding="utf-8").read()
        assert "block_size−1" in theory or "block_size-1" in theory, (
            "THEORY must define d(i,j) including its normalization constant")

    def test_metrics_documents_every_csv_column(self):
        """Every diagnostic column the trainer writes must be documented in
        METRICS.md - the paper appendix is generated from it."""
        import re as _re
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        m = _re.search(r'writer\.writerow\(\[\s*"step",(.*?)\]\)', src, _re.S)
        cols = _re.findall(r'"([a-z_0-9]+)"', '"step",' + m.group(1))
        doc = open(os.path.join(ROOT, "docs", "METRICS.md"), encoding="utf-8").read()
        core = {"step", "tokens_seen", "lr", "val_tokens", "steps_per_sec",
                "tokens_per_sec", "wall_time_sec", "eta_hours", "best_val",
                "phase_norm", "induction", "entropy", "train_loss", "val_loss",
                "val_ppl", "grad_norm"}
        undocumented = [c for c in cols if c not in core and f"`{c}`" not in doc
                        and not any(f"`{c.rsplit('_', 1)[0]}_" in line and c.rsplit('_', 1)[1] in line
                                    for line in doc.splitlines() if "`" in line)]
        assert not undocumented, f"CSV columns missing from METRICS.md: {undocumented}"

    def test_audit_envelope_matches_model_envelope(self):
        """The E1/E3 audit formula and the actual model envelope must agree
        numerically at saturation (audit is only useful if it audits the real
        model)."""
        import json as _json
        import math as _m
        import torch
        from src.model import GPT, GPTConfig
        m_cfg = _json.load(open(os.path.join(ROOT, "configs", "200m_hla_v2_s42.json")))["model"]
        small = dict(m_cfg)
        small.update(block_size=64, vocab_size=256, n_layer=1, n_head=2,
                     n_embd=32, gradient_checkpointing=False)
        model = GPT(GPTConfig(**small)).eval()
        with torch.no_grad():
            model.transformer.h[0].attn.W_gate_k.weight.fill_(1000.0)
            model.transformer.h[0].attn.W_range_k.fill_(1000.0)
        model.set_diagnostics(enabled=True)
        model(torch.randint(0, 256, (1, 48)))
        attn = model.transformer.h[0].attn
        flex = 1.0 + m_cfg.get("range_flex", 0.25)
        lm = attn.layer_gate_multiplier
        eff = min(m_cfg["laplace_alpha"] * m_cfg["laplace_range_k"] * flex * lm,
                  m_cfg["k_log_clip"] * lm)
        hi_pred = (1 - m_cfg["beta_k"]) + m_cfg["beta_k"] * _m.exp(eff)
        assert abs(hi_pred - float(attn.last_mix_k.max())) < 1e-3

    def test_dataloader_deterministic_across_workers(self):
        """Sterility: batch order must not depend on num_workers."""
        import subprocess
        import torch
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_dummy_data.py")],
                       capture_output=True, cwd=ROOT)
        from src.data import FixedDataset, fixed_token_collate, worker_init_fn
        from torch.utils.data import DataLoader

        def first(nw):
            ds = FixedDataset(os.path.join(ROOT, "data", "train_fixed_tokens.pt"), 128)
            dl = DataLoader(ds, batch_size=4, num_workers=nw, shuffle=False,
                            collate_fn=fixed_token_collate, worker_init_fn=worker_init_fn)
            out = []
            for i, b in enumerate(dl):
                out.append(b["input_ids"].clone())
                if i >= 1:
                    break
            return torch.cat(out)
        assert torch.equal(first(0), first(2))


class TestRound8Tooling:
    """Round 8: nanoGPT-parity entrypoints (sample.py, bench.py) and their
    contracts - qualitative sampling and honest wall-clock overhead."""

    def test_sample_script_e2e_greedy(self, tmp_path):
        """sample.py must produce tokens from an init checkpoint on CPU,
        without a tokenizer (--start-ids path)."""
        import subprocess
        ckpt = tmp_path / "init.pt"
        r0 = subprocess.run(
            [sys.executable, os.path.join(ROOT, "src", "make_init.py"),
             "--config", os.path.join(ROOT, "configs", "smoke_hla_s42.json"),
             "--out", str(ckpt)], capture_output=True, text=True, cwd=ROOT)
        assert r0.returncode == 0, r0.stderr[-300:]
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "sample.py"),
             "--checkpoint", str(ckpt),
             "--config", os.path.join(ROOT, "configs", "smoke_hla_s42.json"),
             "--start-ids", "1,2,3", "--max-new-tokens", "5", "--greedy"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-300:]
        assert "sample 1" in r.stdout

    def test_sample_deterministic_greedy(self, tmp_path):
        import subprocess
        ckpt = tmp_path / "init.pt"
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "make_init.py"),
                        "--config", os.path.join(ROOT, "configs", "smoke_hla_s42.json"),
                        "--out", str(ckpt)], capture_output=True, cwd=ROOT)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "sample.py"),
               "--checkpoint", str(ckpt),
               "--config", os.path.join(ROOT, "configs", "smoke_hla_s42.json"),
               "--start-ids", "5,6", "--max-new-tokens", "8", "--greedy"]
        a = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
        b = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
        assert a == b, "greedy sampling must be deterministic"

    def test_bench_script_reports_ratio(self, tmp_path):
        """bench.py must report the wall-clock HLA/base ratio - the honest
        companion number to the analytic FLOPs overhead."""
        import subprocess
        import json as _json
        out = tmp_path / "bench.json"
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "bench.py"),
             "--base", os.path.join(ROOT, "configs", "smoke_hla_s42.json"),
             "--hla", os.path.join(ROOT, "configs", "smoke_hla_s42.json"),
             "--steps", "2", "--batch", "1", "--seq-len", "32", "--n-layer", "1",
             "--out", str(out)],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-300:]
        d = _json.loads(out.read_text())
        assert "hla_over_base_ratio" in d and d["hla_over_base_ratio"] > 0
        assert d["base"]["params"] == d["hla"]["params"]


class TestIsoFlopsPairs:
    """Round 13: both comparison axes must stay internally consistent."""

    RATIO = 1.0693  # profile_flops, v2 recipe @ 2048

    def _steps(self, name):
        cfg = json.load(open(os.path.join(ROOT, "configs", name)))
        return int(cfg["max_steps"])

    def test_kaggle_iso_flops_pair_matched(self):
        base = self._steps("200m_base_v2_s42.json")
        hla = self._steps("200m_hla_v2_flops_s42.json")
        assert hla < base, "iso-FLOPs on Kaggle must be hla-shorter"
        mismatch = abs(hla * self.RATIO - base) / base
        assert mismatch < 0.001, f"FLOPs mismatch {mismatch:.4%} exceeds 0.1%"

    def test_tpu_base_longer_pair_matched(self):
        base = self._steps("tpu3_200m_base_v2_flopslong_s42.json")
        hla = self._steps("tpu3_200m_hla_v2_s42.json")
        assert base > hla, "TPU iso-FLOPs must be base-longer (strongest control)"
        mismatch = abs(hla * self.RATIO - base) / base
        assert mismatch < 0.001, f"FLOPs mismatch {mismatch:.4%} exceeds 0.1%"

    def test_kaggle_200m_fits_dataset(self):
        """17900 steps x 262144 tok/step must fit the 4.7B-token dataset."""
        for name in ("200m_base_s42.json", "200m_hla_s42.json",
                     "200m_base_v2_s42.json", "200m_hla_v2_s42.json"):
            cfg = json.load(open(os.path.join(ROOT, "configs", name)))
            tokens = cfg["max_steps"] * cfg["batch_size_per_device"] * cfg["grad_accum"] * 8 * 2048
            assert tokens <= 4_700_000_000, f"{name}: needs {tokens/1e9:.2f}B > 4.7B dataset"

    def test_experiment_card_documents_axes(self):
        card = open(os.path.join(ROOT, "docs", "EXPERIMENT_CARD.md"), encoding="utf-8").read()
        assert "Two comparison axes" in card
        assert "base-longer" in card


class TestPaperFactsConsistency:
    """Reviewer-facing numbers must agree across README / DATA_CARD - a
    single mismatched token count is the kind of detail that costs trust
    (and did exist: README said 5.4B while the produced set is 4.7B)."""

    def test_readme_token_count_matches_data_card(self):
        import re
        readme = open(os.path.join(ROOT, "README.md")).read()
        card = open(os.path.join(ROOT, "docs", "DATA_CARD.md")).read()
        m = re.search(r"--train-tokens (\d+)", readme)
        assert m, "README quick-start must show the real prepare command"
        readme_tokens = int(m.group(1))
        m2 = re.search(r"num_tokens \(train\) \| ([\d,]+)", card)
        assert m2, "DATA_CARD must pin the produced num_tokens"
        card_tokens = int(m2.group(1).replace(",", ""))
        assert readme_tokens == card_tokens, (
            f"README --train-tokens {readme_tokens} != DATA_CARD {card_tokens}")

    def test_data_card_pins_stream_fingerprint(self):
        card = open(os.path.join(ROOT, "docs", "DATA_CARD.md")).read()
        assert "content_sha256_stream" in card and "4edd8fa4" in card, (
            "DATA_CARD must pin the produced dataset fingerprint")
        assert "1588ec45" in card, "DATA_CARD must pin the C4 revision"


class TestPaperFigures:
    """Paper figures are generated by scripts/make_paper_figures.py from run
    artifacts - the figure code must exist, parse, and expose one function
    per load-bearing figure (fig1 twins, fig2 LITM, fig3 H5)."""

    def test_script_parses_and_has_three_figures(self):
        import ast
        path = os.path.join(ROOT, "scripts", "make_paper_figures.py")
        tree = ast.parse(open(path).read())
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for fn in ("fig1_twin_divergence", "fig2_litm_curves", "fig3_gap_closure"):
            assert fn in fns, f"missing figure function {fn}"

    def test_fig3_draws_preregistered_thresholds(self):
        """The 50%/20% H5 decision lines are the pre-registered reading of
        the experiment - the figure must draw them, not leave them to text."""
        src = open(os.path.join(ROOT, "scripts", "make_paper_figures.py")).read()
        assert "0.2" in src and "0.5" in src and "gap_closure" in src

    def test_repo_has_no_unused_imports(self):
        """pyflakes-clean is a shipping standard for this repo (was violated
        by 23 files once). AST-level check: no obviously-unused top-level
        'import X' in src/ and scripts/ (cheap approximation, no pyflakes
        dependency on CI)."""
        import ast
        import glob as _glob
        offenders = []
        for path in _glob.glob(os.path.join(ROOT, "src", "*.py")) + \
                _glob.glob(os.path.join(ROOT, "scripts", "*.py")):
            src = open(path).read()
            tree = ast.parse(src)
            body_src = src
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for a in node.names:
                        name = (a.asname or a.name).split(".")[0]
                        # crude usage count: name appears beyond the import line
                        uses = body_src.count(name)
                        if uses <= 1 and name not in ("annotations",):
                            offenders.append(f"{os.path.basename(path)}:{name}")
        assert not offenders, f"unused imports: {offenders}"

    def test_paper_figures_has_five_figures(self):
        """fig4 (knockout vs context: H2 long-range evidence) and fig5
        (mechanism trajectories: GDN/Diff 'inner dynamics' reviewer ask)
        joined fig1-3; the figure pipeline must expose all five."""
        import ast
        path = os.path.join(ROOT, "scripts", "make_paper_figures.py")
        tree = ast.parse(open(path).read())
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for fn in ("fig1_twin_divergence", "fig2_litm_curves",
                   "fig3_gap_closure", "fig4_knockout_context",
                   "fig5_mechanism_trajectories"):
            assert fn in fns, f"missing {fn}"


class TestLogToolsCrashResumeContract:
    """Finding #14: the 200m runs WILL produce crash-truncated rows and
    autoresume duplicate-step seams. Every CSV consumer must handle both:
    misaligned columns are a silently-wrong figure; an IndexError in a CI
    gate is a missing verdict. Each fix here was demonstrated as a live
    failure before patching (read_log shifted val_loss by one row on a
    truncated line; validate_log/check_litm_csv crashed on the seam)."""

    HEADER = ["step", "tokens_seen", "lr", "train_loss", "val_loss", "val_ppl"]

    def _write(self, path, rows, header=None):
        import csv as _csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(header or self.HEADER)
            for r in rows:
                w.writerow(r)

    def test_paper_figures_read_log_no_column_shift_on_short_row(self, tmp_path):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mpf = importlib.import_module("make_paper_figures")
        p = str(tmp_path / "log.csv")
        self._write(p, [[1, 100, 5.0], [2, 200], [3, 300, 3.0]],
                    header=["step", "tokens_seen", "val_loss"])
        log = mpf.read_log(p)
        assert len(log["val_loss"]) == len(log["tokens_seen"]) == 3
        pairs = [(t, v) for t, v in zip(log["tokens_seen"], log["val_loss"], strict=True)
                 if v == v]
        # 3.0 belongs to tokens=300; the old zip attributed it to 200
        assert pairs == [(100.0, 5.0), (300.0, 3.0)]

    def test_paper_figures_read_log_resume_seam_keep_last(self, tmp_path):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mpf = importlib.import_module("make_paper_figures")
        p = str(tmp_path / "log.csv")
        import csv as _csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["step", "tokens_seen", "val_loss"])
            for s in (1, 2, 3, 4):
                w.writerow([s, s * 100, 10 - s])
            w.writerow(["# resumed", "x"])
            for s in (3, 4, 5):
                w.writerow([s, s * 100, 20 - s])
        log = mpf.read_log(p)
        assert log["step"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert log["val_loss"][2] == 17.0  # post-resume row won

    def test_make_plots_read_log_resume_seam_keep_last(self, tmp_path):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mp = importlib.import_module("make_plots")
        p = str(tmp_path / "log.csv")
        import csv as _csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["step", "tokens_seen", "val_loss"])
            for s in (1, 2, 3):
                w.writerow([s, s * 100, 10 - s])
            w.writerow(["# resumed", "x"])
            for s in (2, 3, 4):
                w.writerow([s, s * 100, 20 - s])
        log = mp.read_log(p)
        assert log["step"] == [1.0, 2.0, 3.0, 4.0]
        assert log["val_loss"][1] == 18.0

    def test_validate_log_tolerates_crash_row_and_seam(self, tmp_path):
        p = str(tmp_path / "log.csv")
        import csv as _csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(self.HEADER)
            w.writerow([1, 100, 3e-4, 5.0, 5.1, 160.0])
            w.writerow([2, 200, 3e-4, 4.9, "nan", "nan"])
            w.writerow([3, 300])  # truncated crash row
            w.writerow(["# resumed", "t"])
            w.writerow([2, 200, 3e-4, 4.9, "nan", "nan"])  # seam re-log
            w.writerow([3, 300, 3e-4, 4.0, 4.1, 60.0])
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "validate_log.py"), p],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "resume_dups=1" in r.stdout and "truncated_rows=1" in r.stdout

    def test_validate_log_still_rejects_backwards_tokens(self, tmp_path):
        p = str(tmp_path / "log.csv")
        self._write(p, [[1, 100, 3e-4, 5.0, 5.1, 160.0],
                        [2, 50, 3e-4, 4.9, "nan", "nan"]])
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "validate_log.py"), p],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "non-monotonic" in r.stdout + r.stderr

    def test_check_litm_csv_tolerates_truncated_row(self, tmp_path):
        cols = ["step", "tokens_seen", "val_loss", "pos_10", "pos_30", "pos_50",
                "pos_70", "pos_90", "litm_middle_drop", "litm_worst_frac"]
        p = str(tmp_path / "log.csv")
        self._write(p, [[1, 100, 5.0, .1, .1, .1, .1, .1, 0.0, 1.0], [2, 200]],
                    header=cols)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_litm_csv.py"),
                            "--csv", p], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

    def test_check_litm_csv_still_fails_all_nan(self, tmp_path):
        cols = ["step", "tokens_seen", "val_loss", "pos_10", "pos_30", "pos_50",
                "pos_70", "pos_90", "litm_middle_drop", "litm_worst_frac"]
        p = str(tmp_path / "log.csv")
        self._write(p, [[1, 100, 5.0] + ["nan"] * 7], header=cols)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_litm_csv.py"),
                            "--csv", p], capture_output=True, text=True)
        assert r.returncode != 0

    def test_paper_figures_has_seven_figures(self):
        """fig6 (per-position loss, FoX Fig.1 convention: WHERE the gain
        lives) and fig7 (sink mass + outliers: softmax-pathology side
        effects) joined the pipeline - all seven must exist."""
        import ast
        path = os.path.join(ROOT, "scripts", "make_paper_figures.py")
        tree = ast.parse(open(path).read())
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for fn in ("fig6_per_position_loss", "fig7_sink_outliers"):
            assert fn in fns, f"missing {fn}"

    def test_fig6_fig7_run_on_minimal_analysis_json(self, tmp_path):
        """Executable check on synthetic analyze_checkpoint JSONs - the
        figure code must produce files, not just parse."""
        pytest.importorskip("matplotlib")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mpf = importlib.import_module("make_paper_figures")
        base = {"per_position_loss": {f"posloss_bin_{i:02d}": 10.0 - 0.01 * i for i in range(8)},
                "attention_sink": {"sink_mass_first": 0.02, "sink_mass_first4": 0.08,
                                   "sink_top_layer": 0.02},
                "activation_outliers": {"act_excess_kurtosis": 0.5, "act_max_over_rms": 6.0}}
        hla = {"per_position_loss": {f"posloss_bin_{i:02d}": 10.0 - 0.02 * i for i in range(8)},
               "attention_sink": {"sink_mass_first": 0.01, "sink_mass_first4": 0.05,
                                  "sink_top_layer": 0.01},
               "activation_outliers": {"act_excess_kurtosis": 0.2, "act_max_over_rms": 4.0}}
        pb, ph = str(tmp_path / "b.json"), str(tmp_path / "h.json")
        json.dump(base, open(pb, "w"))
        json.dump(hla, open(ph, "w"))
        o6 = str(tmp_path / "fig6.png")
        o7 = str(tmp_path / "fig7.png")
        mpf.fig6_per_position_loss(pb, ph, o6)
        mpf.fig7_sink_outliers(pb, ph, o7)
        assert os.path.getsize(o6) > 1000 and os.path.getsize(o7) > 1000

    def test_paper_figures_has_fig8_probe_depth(self):
        """fig8: linear-probe accuracy vs depth with chance line + shuffled
        control (Hewitt-Liang selectivity) - the representation-level
        counterpart of fig2's behavioral LITM curve."""
        import ast
        path = os.path.join(ROOT, "scripts", "make_paper_figures.py")
        tree = ast.parse(open(path).read())
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "fig8_probe_depth" in fns

    def test_fig8_runs_on_minimal_probe_json(self, tmp_path):
        pytest.importorskip("matplotlib")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mpf = importlib.import_module("make_paper_figures")
        rec = {"probe_acc_by_depth": {"0.1": 0.4, "0.3": 0.35, "0.5": 0.3,
                                      "0.7": 0.36, "0.9": 0.41},
               "probe_shuffled_by_depth": {"0.1": 0.13, "0.3": 0.12, "0.5": 0.13,
                                           "0.7": 0.12, "0.9": 0.13},
               "chance": 0.125}
        pb, ph = str(tmp_path / "b.json"), str(tmp_path / "h.json")
        json.dump(rec, open(pb, "w"))
        rec2 = dict(rec)
        rec2["probe_acc_by_depth"] = {k: v + 0.05 for k, v in rec["probe_acc_by_depth"].items()}
        json.dump(rec2, open(ph, "w"))
        out = str(tmp_path / "fig8.png")
        mpf.fig8_probe_depth(pb, ph, out)
        assert os.path.getsize(out) > 1000

    def test_fig8_fails_loud_on_missing_probe_block(self, tmp_path):
        pytest.importorskip("matplotlib")  # finding #27: CI hosts may lack it
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mpf = importlib.import_module("make_paper_figures")
        p = str(tmp_path / "empty.json")
        json.dump({}, open(p, "w"))
        with pytest.raises(SystemExit):
            mpf.fig8_probe_depth(p, p, str(tmp_path / "x.png"))

    def test_fig1_inset_refuses_on_misaligned_token_grids(self, tmp_path):
        """Finding #16: with asymmetric resumes the twins' eval rows can land
        on different tokens_seen grids; the old code silently DROPPED the
        gap/seed-sigma inset - fig1's decision quantity - shipping a figure
        without its argument. Now it must refuse loudly when --seed-std is
        requested but no common grid exists (and still work without it)."""
        pytest.importorskip("matplotlib")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib, csv as _csv
        mpf = importlib.import_module("make_paper_figures")
        def wlog(p, shift):
            with open(p, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["step", "tokens_seen", "val_loss"])
                for s in (1, 2, 3):
                    w.writerow([s, s * 100 + shift, 5.0 - s * 0.1])
        pb, ph = str(tmp_path / "b.csv"), str(tmp_path / "h.csv")
        wlog(pb, 0)
        wlog(ph, 50)  # misaligned grid
        with pytest.raises(SystemExit):
            mpf.fig1_twin_divergence(pb, ph, str(tmp_path / "f.png"), seed_std=0.01)
        # without seed-std the figure itself must still render
        mpf.fig1_twin_divergence(pb, ph, str(tmp_path / "f2.png"))
        assert os.path.getsize(str(tmp_path / "f2.png")) > 1000

    def test_fig3_draws_reverse_arm_when_present(self, tmp_path):
        """fig3 must carry BOTH halves of H5: forward franken (sufficiency)
        and anti-franken (necessity). A causal JSON with a reverse block
        must yield 4 bars; the title must state both closures."""
        pytest.importorskip("matplotlib")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mpf = importlib.import_module("make_paper_figures")
        rec = {"base": 0.1, "hla": 0.3, "franken": 0.28, "gap": 0.2,
               "closure": 0.9, "closure_std_bound": 0.02}
        rev = {"base": 0.1, "hla": 0.3, "franken": 0.12, "gap": 0.2,
               "closure": 0.1, "closure_std_bound": 0.02}
        blob = {"gap_closure": {"induction": rec},
                "gap_closure_reverse": {"induction": rev}}
        p = str(tmp_path / "c.json")
        json.dump(blob, open(p, "w"))
        out = str(tmp_path / "f3.png")
        mpf.fig3_gap_closure(p, out)
        assert os.path.getsize(out) > 1000
        # forward-only JSON must still work (V7 runs forward first)
        json.dump({"gap_closure": {"induction": rec}}, open(p, "w"))
        mpf.fig3_gap_closure(p, str(tmp_path / "f3b.png"))
        assert os.path.getsize(str(tmp_path / "f3b.png")) > 1000

    def test_fig9_fig10_run_and_refuse(self, tmp_path):
        """Round-12 reviewer-mitigation figures: fig9 passkey-at-full-window
        (external anchor, B7) and fig10 induction-head census (Olsson
        convention, mech-interp ask). Executable on minimal JSONs, loud
        refusal on wrong inputs."""
        pytest.importorskip("matplotlib")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        mpf = importlib.import_module("make_paper_figures")
        pb = {f"passkey_acc_{d:02d}": 0.5 for d in range(10, 100, 10)}
        pb["context_length"] = 2048
        b, h = str(tmp_path / "b.json"), str(tmp_path / "h.json")
        json.dump(pb, open(b, "w"))
        json.dump(pb, open(h, "w"))
        mpf.fig9_passkey_depth(b, h, str(tmp_path / "f9.png"))
        assert os.path.getsize(str(tmp_path / "f9.png")) > 1000
        pm = {"prefix_matching": {
            f"L{li:02d}_H{hi:02d}_prefix_match": 0.1
            for li in range(2) for hi in range(4)}}
        ab, ah = str(tmp_path / "ab.json"), str(tmp_path / "ah.json")
        json.dump(pm, open(ab, "w"))
        json.dump(pm, open(ah, "w"))
        mpf.fig10_head_census(ab, ah, str(tmp_path / "f10.png"))
        assert os.path.getsize(str(tmp_path / "f10.png")) > 1000
        with pytest.raises(SystemExit):
            mpf.fig9_passkey_depth(ab, ab, str(tmp_path / "x.png"))
        with pytest.raises(SystemExit):
            mpf.fig10_head_census(b, b, str(tmp_path / "x.png"))

    def test_docs_cover_shipped_metrics(self):
        """Round-17 (AEC audit): docs/METRICS.md must document every metric
        family the code actually ships, and EXPERIMENT_CARD must register
        the post-R12 additions. Docs drifting from code is the top artifact-
        evaluation complaint."""
        md = open(os.path.join(ROOT, "docs", "METRICS.md")).read()
        for needle in ("posloss_bin", "passkey_acc", "min_detectable_gap_z3",
                       "gap_over_noise_z", "prefix_match", "snr_needle_last",
                       "sink_mass_first"):
            assert needle in md, f"METRICS.md missing {needle}"
        ec = open(os.path.join(ROOT, "docs", "EXPERIMENT_CARD.md")).read().lower()
        for needle in ("passkey", "powered", "census", "wake", "s43"):
            assert needle in ec, f"EXPERIMENT_CARD missing {needle}"
        th = open(os.path.join(ROOT, "docs", "THEORY.md")).read()
        assert "Theorem 5.1" in th and "wake order" in th.lower(), \
            "THEORY.md must scope Theorem 5 with the wake-order refinement"
        st = open(os.path.join(ROOT, "docs", "STERILITY.md")).read().lower()
        for needle in ("adamw", "franken", "knockout", "resume"):
            assert needle in st, f"STERILITY.md missing level: {needle}"
        # Budget numbers in the card must match the ACTUAL 9h-pair configs
        # (finding #24: the card described the retired 17900-step plan only).
        import json as _json
        c = _json.load(open(os.path.join(
            ROOT, "configs", "kaggle_200m_hla_9h_s42.json")))
        assert str(c["max_steps"]) in ec, \
            "EXPERIMENT_CARD must state the 9h-pair max_steps"
        assert "262,144" in ec or "262144" in ec.replace(",", ""), \
            "EXPERIMENT_CARD must state tokens/update"

    def test_fig1_legend_never_under_inset(self):
        """Finding #26 (fresh-eyes audit on REALISTIC curves): the default
        legend lands upper-right - exactly under the gap/sigma inset - when
        both loss curves decrease monotonically (which real training curves
        do). The legend must be pinned away from the inset region."""
        src = open(os.path.join(ROOT, "scripts", "make_paper_figures.py")).read()
        i = src.find("def fig1_twin_divergence")
        j = src.find("def fig2")
        body = src[i:j]
        assert 'legend(loc="lower left")' in body, \
            "fig1 legend must be pinned (default lands under the inset)"


class TestSpeedBackendConfigMatrix:
    """#30/#30b config matrix: every speed config must load through GPTConfig,
    declare exactly the intended backend, satisfy the fold-legality invariant
    (|alpha*range| <= clip for each active folded bias), and GPU configs must
    not request the TPU-only pallas backend."""

    TPU = {
        "kaggle_200m_base_speed_sdpa_s42": "sdpa",
        "kaggle_200m_hla_speed_fold_s42": "sdpa_fold",
        "kaggle_200m_hla_speed_pallas_s42": "pallas",
        "700m_base_14b_sdpa_s42": "sdpa",
        "700m_hla_14b_fold_s42": "sdpa_fold",
        "700m_hla_14b_pallas_s42": "pallas",
    }
    GPU = {
        "gpu_200m_base_flash_s42": "sdpa",
        "gpu_200m_hla_flash_fold_s42": "sdpa_fold",
        "gpu_700m_base_flash_s42": "sdpa",
        "gpu_700m_hla_flash_fold_s42": "sdpa_fold",
    }

    def _check(self, name, backend):
        import json, os
        from src.model import GPTConfig
        path = os.path.join(ROOT, "configs", name + ".json")
        m = json.load(open(path))["model"]
        cfg = GPTConfig(**m)
        assert cfg.attention_backend == backend, name
        if m.get("use_distance_laplace") and m.get("distance_laplace_alpha", 0):
            assert abs(m["distance_laplace_alpha"] * m.get("distance_laplace_range", 1.0)) \
                <= m.get("distance_laplace_clip", 1.0), name
        if m.get("use_salience_bias") and m.get("salience_alpha", 0):
            assert abs(m["salience_alpha"] * m.get("salience_range", 1.0)) \
                <= m.get("salience_clip", 2.0), name

    def test_tpu_matrix(self):
        for name, backend in self.TPU.items():
            self._check(name, backend)

    def test_gpu_matrix_no_pallas(self):
        for name, backend in self.GPU.items():
            self._check(name, backend)
            assert backend != "pallas"

    def test_speed_docs_exist(self):
        import os
        assert os.path.exists(os.path.join(ROOT, "docs", "SPEED_BACKENDS.md"))


class TestBenchSpeedProvenance:
    """Attack round H (speed-claim methodology). #33: the bench JSON must
    record WHICH backend was actually measured (plus shapes and torch
    version) - without provenance a reviewer can dismiss the speed table.
    H3a: self-vs-self ratio is the noise floor (sanity). H3b: pallas on a
    CPU host fails loud in bench too - a silent manual fallback would fake
    the speed table."""

    def _cfg(self, tmp_path, name, **extra):
        import json
        kw = dict(block_size=128, vocab_size=256, padded_vocab_size=256,
                  n_layer=1, n_head=2, n_embd=64,
                  gradient_checkpointing=False, **extra)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"model": kw}))
        return str(p)

    def test_bench_records_backend_provenance(self, tmp_path):
        import json, subprocess, sys, os
        m = self._cfg(tmp_path, "m", attention_backend="manual")
        f = self._cfg(tmp_path, "f", phase_mult=0.15,
                      use_distance_laplace=True, distance_laplace_alpha=0.5,
                      attention_backend="sdpa_fold")
        out = tmp_path / "b.json"
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "bench.py"),
                            "--base", m, "--hla", f, "--steps", "2",
                            "--batch", "1", "--seq-len", "64",
                            "--out", str(out)],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-300:]
        b = json.loads(out.read_text())
        assert b["base"]["attention_backend"] == "manual"
        assert b["hla"]["attention_backend"] == "sdpa_fold"
        assert "torch_version" in b["base"]

    def test_bench_pallas_on_cpu_fails_loud(self, tmp_path):
        import subprocess, sys, os
        try:
            import torch_xla  # noqa: F401
            import pytest
            pytest.skip("torch_xla present")
        except ImportError:
            pass
        m = self._cfg(tmp_path, "m")
        p = self._cfg(tmp_path, "p", phase_mult=0.15,
                      use_distance_laplace=True, distance_laplace_alpha=0.5,
                      attention_backend="pallas")
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "bench.py"),
                            "--base", m, "--hla", p, "--steps", "1",
                            "--batch", "1", "--seq-len", "64"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "torch_xla" in r.stderr


class TestFigureCoverageGuard:
    """Regression #34 (attack I2): with multi-session autoresume runs a
    tail-only CSV (single resume-session log) passed against a full log
    used to produce a silently-wrong figure (curves covering different
    token ranges). fig1/fig2/fig5 must refuse loudly; the legit fix -
    concatenated session CSVs (read_log keep-last seam handling, #14) -
    must still draw."""

    def _mk(self, tmp_path, name, lo, hi):
        hdr = "step,tokens_seen,lr,train_loss,val_loss\n"
        rows = [f"{s},{s*262144},0.0003,{4.0-s*1e-5:.6f},"
                f"{(f'{5.0-s*1e-4:.6f}' if s % 500 == 0 else 'nan')}\n"
                for s in range(lo, hi + 1, 50)]
        p = tmp_path / f"{name}.csv"
        p.write_text(hdr + "".join(rows))
        return str(p)

    def _mpf(self):
        import importlib.util, os
        pytest.importorskip("matplotlib")
        spec = importlib.util.spec_from_file_location(
            "mpf", os.path.join(ROOT, "scripts", "make_paper_figures.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_tail_only_log_refused_loud(self, tmp_path):
        mpf = self._mpf()
        full = self._mk(tmp_path, "full", 50, 15000)
        tail = self._mk(tmp_path, "tail", 10550, 15000)
        with pytest.raises(SystemExit, match="coverage mismatch"):
            mpf.fig1_twin_divergence(full, tail, str(tmp_path / "f.png"))

    def test_concatenated_sessions_accepted(self, tmp_path):
        import os
        mpf = self._mpf()
        full = self._mk(tmp_path, "full2", 50, 15000)
        s1 = open(self._mk(tmp_path, "s1", 50, 10700)).read()
        s2 = open(self._mk(tmp_path, "s2", 10550, 15000)).read()
        cat = tmp_path / "cat.csv"
        cat.write_text(s1 + s2)  # includes mid-file header: read_log handles
        out = tmp_path / "ok.png"
        mpf.fig1_twin_divergence(full, str(cat), str(out))
        assert os.path.exists(out)
