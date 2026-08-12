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



"""make_init.py tests: sterile shared-backbone init generation."""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.make_init import (  # noqa: E402
    assert_hla_identity,
    copy_shared_backbone,
    is_hla_key,
    parameter_report,
    stable_hash,
)
from src.model import GPT, GPTConfig  # noqa: E402


def make_pair():
    base_cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                         n_embd=32, phase_mult=0.0, use_laplace=True,
                         laplace_alpha=0.0, gradient_checkpointing=False)
    hla_cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, phase_mult=0.15, use_laplace=True,
                        laplace_alpha=1.0, gradient_checkpointing=False)
    torch.manual_seed(1)
    base = GPT(base_cfg)
    torch.manual_seed(2)  # deliberately different seed
    hla = GPT(hla_cfg)
    return base, hla


class TestIsHlaKey:
    @pytest.mark.parametrize("key,expected", [
        ("transformer.h.0.attn.W_phase_q", True),
        ("transformer.h.11.attn.W_gate_k.weight", True),
        ("transformer.h.0.attn.c_attn.weight", False),
        ("transformer.wte.weight", False),
        ("lm_head.weight", False),
    ])
    def test_patterns(self, key, expected):
        assert is_hla_key(key) is expected


class TestSharedBackbone:
    def test_backbone_identical_after_copy(self):
        base, hla = make_pair()
        manifest = copy_shared_backbone(base, hla)
        assert manifest["num_copied_non_hla_keys"] > 0
        base_sd, hla_sd = base.state_dict(), hla.state_dict()
        for k in manifest["copied_non_hla_keys"]:
            assert torch.equal(base_sd[k], hla_sd[k]), k

    def test_hla_identity_after_copy(self):
        base, hla = make_pair()
        copy_shared_backbone(base, hla)
        assert hla.hla_identity_error() == 0.0
        assert base.hla_identity_error() == 0.0

    def test_logits_identical_after_shared_init(self):
        base, hla = make_pair()
        copy_shared_backbone(base, hla)
        base.eval(); hla.eval()
        x = torch.randint(0, 256, (2, 32))
        lb, _ = base(x)
        lh, _ = hla(x)
        assert torch.equal(lb, lh), "shared-backbone init must give identical outputs"


class TestAssertIdentity:
    def test_passes_on_fresh_model(self):
        base, _ = make_pair()
        assert_hla_identity(base)

    def test_fails_on_perturbed_model(self):
        _, hla = make_pair()
        with torch.no_grad():
            hla.transformer.h[0].attn.W_phase_q.add_(0.01)
        with pytest.raises(RuntimeError):
            assert_hla_identity(hla)


class TestParameterReport:
    def test_groups_sum_to_total(self):
        base, _ = make_pair()
        rep = parameter_report(base)
        parts = rep["embedding"] + rep["attention_base"] + rep["mlp"] + rep["norm"] + rep["hla"] + rep["other"]
        assert parts == rep["total"]
        assert rep["hla"] > 0  # W_phase/W_range/W_gate present even in base (ablated)


class TestStableHash:
    def test_deterministic_and_order_independent(self):
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})


class TestInitAudit:
    """Line-by-line audit round of make_init.py: determinism, seed
    plumbing, and the seed-mismatch sterility gate."""

    def _small(self, tmp_path, seed_base=42, seed_hla=42):
        import json
        cfgs = []
        for var, seed in (("base", seed_base), ("hla", seed_hla)):
            src = json.load(open(os.path.join(
                ROOT, "configs", f"kaggle_smoke_{var}_s42.json")))
            src["model"].update(n_layer=1, n_head=2, n_embd=64, block_size=128)
            src["seed"] = seed
            p = tmp_path / f"{var}_{seed}.json"
            p.write_text(json.dumps(src))
            cfgs.append(str(p))
        return cfgs

    def test_shared_backbone_bit_deterministic(self, tmp_path):
        """Two runs of the same pair must produce bit-identical states -
        the whole experiment chain assumes init checkpoints are pure
        functions of (config, seed)."""
        import subprocess
        import sys as _sys
        import hashlib
        import torch as _t
        bc, hc = self._small(tmp_path)

        def run(tag):
            ob, oh = str(tmp_path / f"{tag}b.pt"), str(tmp_path / f"{tag}h.pt")
            r = subprocess.run([_sys.executable,
                                os.path.join(ROOT, "src", "make_init.py"),
                                "--shared-backbone", "--base-config", bc,
                                "--hla-config", hc, "--out-base", ob,
                                "--out-hla", oh], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-400:]
            return ob

        def h(path):
            sd = _t.load(path, map_location="cpu", weights_only=False)["model"]
            m = hashlib.sha256()
            for k in sorted(sd):
                m.update(k.encode())
                m.update(sd[k].numpy().tobytes())
            return m.hexdigest()

        assert h(run("a")) == h(run("b"))

    def test_seed_mismatch_rejected(self, tmp_path):
        """Sterility gate: differing base/hla seeds diverge the twins' data
        order (trainer reads seed from each config) - must fail loudly."""
        import subprocess
        import sys as _sys
        bc, hc = self._small(tmp_path, seed_base=42, seed_hla=77)
        r = subprocess.run([_sys.executable,
                            os.path.join(ROOT, "src", "make_init.py"),
                            "--shared-backbone", "--base-config", bc,
                            "--hla-config", hc,
                            "--out-base", str(tmp_path / "b.pt"),
                            "--out-hla", str(tmp_path / "h.pt")],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "STERILITY VIOLATION" in (r.stdout + r.stderr)

    def test_hla_seed_does_not_leak_into_backbone(self, tmp_path):
        """The shared backbone must be a function of the BASE seed only:
        creating models in any order / with polluted global RNG must not
        change the copied tensors (verified via model_from_config's full
        reseed)."""
        import json
        import torch as _t
        from src.make_init import model_from_config
        bc, hc = self._small(tmp_path)
        cfg = json.load(open(hc))
        m1 = model_from_config(cfg)
        _t.manual_seed(12345)
        _t.randn(4096)  # pollute global RNG
        m2 = model_from_config(cfg)
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        assert all(_t.equal(sd1[k], sd2[k]) for k in sd1)

    def test_identity_survives_bf16_cast(self, tmp_path):
        """--dtype bf16 must not perturb the exact-zero identity params
        (0.0 is exactly representable, but a regression that added noise
        before the cast would slip through unnoticed otherwise)."""
        import subprocess
        import sys as _sys
        import torch as _t
        bc, hc = self._small(tmp_path)
        out = str(tmp_path / "s.pt")
        r = subprocess.run([_sys.executable,
                            os.path.join(ROOT, "src", "make_init.py"),
                            "--config", hc, "--out", out, "--dtype", "bf16"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-400:]
        sd = _t.load(out, map_location="cpu", weights_only=False)["model"]
        for k, v in sd.items():
            if "W_phase_q" in k or "W_gate_k.weight" in k or "W_qtemp" in k:
                assert float(v.float().abs().max()) == 0.0, k

    def test_allow_shape_mismatch_leaves_audit_trail(self, tmp_path):
        """The escape hatch must not be silent: manifest.shape_mismatches
        must record every non-copied tensor, or a non-sterile pair could
        masquerade as sterile."""
        import json
        import subprocess
        import sys as _sys
        bc, hc = self._small(tmp_path)
        ch = json.loads(open(hc).read())
        ch["model"]["vocab_size"] = 50304
        ch["expected_vocab_size"] = 50304
        hc2 = tmp_path / "hla_v2.json"
        hc2.write_text(json.dumps(ch))
        args = [_sys.executable, os.path.join(ROOT, "src", "make_init.py"),
                "--shared-backbone", "--base-config", bc,
                "--hla-config", str(hc2),
                "--out-base", str(tmp_path / "b.pt"),
                "--out-hla", str(tmp_path / "h.pt")]
        r = subprocess.run(args, capture_output=True, text=True)
        assert r.returncode != 0, "shape mismatch without flag must fail"
        r = subprocess.run(args + ["--allow-shape-mismatch"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
        man = json.loads(open(str(tmp_path / "h.pt.manifest.json")).read())
        assert man["shape_mismatches"], "audit trail lost"

    def test_pair_manifests_identical(self, tmp_path):
        """base and hla manifest files must be the same document - one
        audit record for one pair."""
        import json
        import subprocess
        import sys as _sys
        bc, hc = self._small(tmp_path)
        r = subprocess.run([_sys.executable,
                            os.path.join(ROOT, "src", "make_init.py"),
                            "--shared-backbone", "--base-config", bc,
                            "--hla-config", hc,
                            "--out-base", str(tmp_path / "b.pt"),
                            "--out-hla", str(tmp_path / "h.pt")],
                           capture_output=True, text=True)
        assert r.returncode == 0
        m1 = json.loads(open(str(tmp_path / "b.pt.manifest.json")).read())
        m2 = json.loads(open(str(tmp_path / "h.pt.manifest.json")).read())
        assert m1 == m2

    def test_payload_passes_trainer_gate(self, tmp_path):
        """End-to-end contract: the payload make_init writes must pass the
        trainer's validate_init_config_compatibility for its own config and
        fail for a shape-modified one (correct argument order: current
        run config first, saved init config second)."""
        import json
        import subprocess
        import sys as _sys
        import copy
        import torch as _t
        bc, hc = self._small(tmp_path)
        out = str(tmp_path / "h.pt")
        r = subprocess.run([_sys.executable,
                            os.path.join(ROOT, "src", "make_init.py"),
                            "--config", hc, "--out", out],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
        from src.train_xla import validate_init_config_compatibility
        payload = _t.load(out, map_location="cpu", weights_only=False)
        cfg = json.loads(open(hc).read())
        validate_init_config_compatibility(cfg, payload["config"])  # no raise
        bad = copy.deepcopy(cfg)
        bad["model"]["n_embd"] = 256
        import pytest as _pytest
        with _pytest.raises(ValueError):
            validate_init_config_compatibility(bad, payload["config"])


class TestCompareInitsRoleGate:
    """Finding #17 (mutation testing): compare_inits' tensor checks are
    blind to a role swap by design - the shared backbone is bit-identical,
    so base-as-hla and swapped-args both passed silently. A swapped pair
    trains the wrong variant from the right weights. The role stamped by
    make_init must be enforced."""

    def _pair(self, tmp_path):
        import json
        import subprocess
        import sys as _sys
        cfgs = []
        for var in ("base", "hla"):
            src = json.load(open(os.path.join(
                ROOT, "configs", f"kaggle_smoke_{var}_s42.json")))
            src["model"].update(n_layer=1, n_head=2, n_embd=64, block_size=128)
            p = tmp_path / f"{var}.json"
            p.write_text(json.dumps(src))
            cfgs.append(str(p))
        ob, oh = str(tmp_path / "ib.pt"), str(tmp_path / "ih.pt")
        r = subprocess.run([_sys.executable,
                            os.path.join(ROOT, "src", "make_init.py"),
                            "--shared-backbone", "--base-config", cfgs[0],
                            "--hla-config", cfgs[1], "--out-base", ob,
                            "--out-hla", oh], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return ob, oh

    def _compare(self, base, hla):
        import subprocess
        import sys as _sys
        return subprocess.run([_sys.executable,
                               os.path.join(ROOT, "scripts", "compare_inits.py"),
                               "--base", base, "--hla", hla],
                              capture_output=True, text=True)

    def test_honest_pair_passes(self, tmp_path):
        ob, oh = self._pair(tmp_path)
        r = self._compare(ob, oh)
        assert r.returncode == 0, r.stdout + r.stderr

    def test_swapped_args_caught(self, tmp_path):
        ob, oh = self._pair(tmp_path)
        r = self._compare(oh, ob)
        assert r.returncode != 0
        assert "role" in (r.stdout + r.stderr)

    def test_base_as_both_caught(self, tmp_path):
        ob, _ = self._pair(tmp_path)
        r = self._compare(ob, ob)
        assert r.returncode != 0
        assert "role" in (r.stdout + r.stderr)


class TestPipelineSterilityEndToEnd:
    """Round-14: the WHOLE V8 path in one test - make_init pair -> bf16
    save/load (the exact final_*_bf16.pt format) -> retrieval-franken on
    the identity pair == base bit-for-bit, and eval probes agree across
    the twins exactly. Every link was proven separately before; this welds
    the chain, so a regression in ANY link fails one obvious test."""

    def test_pair_to_franken_to_probes(self, tmp_path):
        import json
        import subprocess
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        cp = importlib.import_module("causal_patch")
        from src.eval import evaluate_induction, positional_recall_curve

        cfgs = {}
        for var in ("base", "hla"):
            c = json.load(open(os.path.join(
                ROOT, "configs", f"kaggle_smoke_{var}_s42.json")))
            c["model"].update(n_layer=1, n_head=2, n_embd=64, block_size=128)
            p = str(tmp_path / f"{var}.json")
            json.dump(c, open(p, "w"))
            cfgs[var] = c["model"]
        ib, ih = str(tmp_path / "ib.pt"), str(tmp_path / "ih.pt")
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "src", "make_init.py"),
                            "--shared-backbone",
                            "--base-config", str(tmp_path / "base.json"),
                            "--hla-config", str(tmp_path / "hla.json"),
                            "--out-base", ib, "--out-hla", ih],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

        def bf16_roundtrip(path, cfg):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            sd = payload["model"] if "model" in payload else payload
            sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                  for k, v in sd.items()}
            sd = {k: (v.float() if v.is_floating_point() else v)
                  for k, v in sd.items()}
            m = GPT(GPTConfig(**cfg)).eval()
            m.load_state_dict(sd, strict=True)
            return m, sd

        mb, sb = bf16_roundtrip(ib, cfgs["base"])
        mh, sh = bf16_roundtrip(ih, cfgs["hla"])
        # franken on the identity pair must equal base at the logit level
        fr = cp.build_franken(sb, sh, 64, "retrieval", donor="hla")
        mfr = GPT(GPTConfig(**cfgs["hla"])).eval()
        mfr.load_state_dict(fr, strict=True)
        g = torch.Generator(); g.manual_seed(9)
        x = torch.randint(0, 50257, (1, 128), generator=g)
        with torch.no_grad():
            lb, _ = mb(x)
            lf, _ = mfr(x)
        assert torch.equal(lb, lf)
        # probes agree exactly across twins at identity
        assert float(evaluate_induction(mb, device="cpu", batch_size=2)) == \
            float(evaluate_induction(mh, device="cpu", batch_size=2))
        pb = positional_recall_curve(mb, batch_size=2)
        ph = positional_recall_curve(mh, batch_size=2)
        assert all(pb[k] == ph[k] for k in pb)
