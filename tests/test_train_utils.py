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



"""Unit tests for pure trainer helpers (no TPU needed).

train_xla.py imports torch_xla at module level; on CPU-only CI we inject a
minimal stub so the pure-Python helpers (lr schedule, sharded samplers,
config validation, overrides) can be tested. The stub is NOT used by the
functions under test.
"""
from __future__ import annotations

import math
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _install_torch_xla_stub() -> None:
    if "torch_xla" in sys.modules:
        return
    xla = types.ModuleType("torch_xla")
    core = types.ModuleType("torch_xla.core")
    xm = types.ModuleType("torch_xla.core.xla_model")
    xm.get_ordinal = lambda: 0
    xm.get_local_ordinal = lambda: 0
    xm.xrt_world_size = lambda: 1
    xm.mark_step = lambda: None
    xm.rendezvous = lambda tag: None
    dist = types.ModuleType("torch_xla.distributed")
    pl = types.ModuleType("torch_xla.distributed.parallel_loader")
    pl.MpDeviceLoader = object
    xmp = types.ModuleType("torch_xla.distributed.xla_multiprocessing")
    xmp.spawn = lambda *a, **k: None
    debug = types.ModuleType("torch_xla.debug")
    metrics = types.ModuleType("torch_xla.debug.metrics")
    metrics.metrics_report = lambda: ""
    xla.core = core
    core.xla_model = xm
    xla.distributed = dist
    dist.parallel_loader = pl
    dist.xla_multiprocessing = xmp
    xla.debug = debug
    debug.metrics = metrics
    for name, mod in {
        "torch_xla": xla,
        "torch_xla.core": core,
        "torch_xla.core.xla_model": xm,
        "torch_xla.distributed": dist,
        "torch_xla.distributed.parallel_loader": pl,
        "torch_xla.distributed.xla_multiprocessing": xmp,
        "torch_xla.debug": debug,
        "torch_xla.debug.metrics": metrics,
    }.items():
        sys.modules[name] = mod


_install_torch_xla_stub()

from src.train_xla import (  # noqa: E402
    EvenShardedSequentialSampler,
    ShardedSequentialSampler,
    apply_override,
    get_lr,
    validate_config,
    validate_init_config_compatibility,
    validate_resume_config_compatibility,
)


class TestLrSchedule:
    def test_warmup_starts_above_zero_and_reaches_base(self):
        # step 0 must have non-zero lr (dead first update otherwise)
        assert get_lr(0, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4) > 0
        assert math.isclose(
            get_lr(99, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4), 1e-3
        )

    def test_cosine_endpoints(self):
        # right at warmup end: base_lr; at max_steps: min_lr
        assert math.isclose(get_lr(100, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4), 1e-3)
        assert math.isclose(get_lr(1000, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4), 1e-4)

    def test_monotone_decay_after_warmup(self):
        prev = float("inf")
        for s in range(100, 1001, 50):
            lr = get_lr(s, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4)
            assert lr <= prev + 1e-12
            prev = lr

    def test_never_below_min_lr(self):
        for s in range(0, 1100, 37):
            lr = get_lr(s, warmup=100, max_steps=1000, base_lr=1e-3, min_lr=1e-4)
            if s >= 100:
                assert lr >= 1e-4 - 1e-12


class TestEvenSampler:
    def test_no_duplicates_no_out_of_range(self):
        for n, ws, bs in [(103, 4, 2), (64, 8, 1), (1000, 8, 4), (17, 2, 3)]:
            seen = []
            for r in range(ws):
                s = EvenShardedSequentialSampler(n, rank=r, world_size=ws, batch_size=bs)
                seen.extend(list(s))
            assert len(seen) == len(set(seen)), f"dup at n={n},ws={ws},bs={bs}"
            assert all(0 <= i < n for i in seen)

    def test_equal_length_across_ranks(self):
        for r in range(8):
            s = EvenShardedSequentialSampler(1001, rank=r, world_size=8, batch_size=2)
            assert len(s) == len(EvenShardedSequentialSampler(1001, rank=0, world_size=8, batch_size=2))

    def test_resume_is_exact_suffix(self):
        full = list(EvenShardedSequentialSampler(103, rank=1, world_size=4, batch_size=2))
        resumed = list(EvenShardedSequentialSampler(103, rank=1, world_size=4, batch_size=2,
                                                    start_local_sample=6))
        assert full[6:] == resumed

    def test_resume_beyond_end_is_empty(self):
        s = EvenShardedSequentialSampler(100, rank=0, world_size=4, batch_size=2,
                                         start_local_sample=10_000)
        assert len(list(s)) == 0

    def test_length_multiple_of_batch(self):
        for start in (0, 1, 3, 7):
            s = EvenShardedSequentialSampler(103, rank=0, world_size=4, batch_size=2,
                                             start_local_sample=start)
            assert len(s) % 2 == 0


class TestValSampler:
    def test_covers_everything_once(self):
        n, ws = 101, 8
        seen = []
        for r in range(ws):
            seen.extend(list(ShardedSequentialSampler(n, r, ws)))
        assert sorted(seen) == list(range(n))

    def test_len_matches_iter(self):
        for r in range(8):
            s = ShardedSequentialSampler(101, r, 8)
            assert len(list(s)) == len(s)


class TestApplyOverride:
    def test_nested_json_types(self):
        cfg = {"model": {"n_layer": 2}, "lr": 1e-3}
        apply_override(cfg, "model.n_layer=24")
        apply_override(cfg, "lr=0.0005")
        apply_override(cfg, "model.use_rope=true")
        apply_override(cfg, "run_name=test_run")
        assert cfg["model"]["n_layer"] == 24
        assert cfg["lr"] == 0.0005
        assert cfg["model"]["use_rope"] is True
        assert cfg["run_name"] == "test_run"

    def test_invalid_format_rejected(self):
        with pytest.raises(ValueError):
            apply_override({}, "no_equals_sign")


class TestOptimizerGroups:
    """HLA params must be excluded from weight decay: zero IS their identity
    state, so decay would be a constant force against the mechanisms."""

    def test_hla_params_in_nodecay_group(self):
        from src.model import GPT, GPTConfig
        from src.train_xla import make_optimizer

        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False,
                        phase_mult=0.15, use_laplace=True, laplace_alpha=1.0)
        model = GPT(cfg)
        opt = make_optimizer(model, {"lr": 1e-3, "weight_decay": 0.1})
        decay_params = {id(p) for p in opt.param_groups[0]["params"]}
        hla_names = ("W_phase", "W_gate", "W_range", "W_layer_temp")
        leaked = [n for n, p in model.named_parameters()
                  if any(m in n for m in hla_names) and id(p) in decay_params]
        assert leaked == [], f"HLA params leaked into decay group: {leaked}"

    def test_backbone_matrices_still_decayed(self):
        from src.model import GPT, GPTConfig
        from src.train_xla import make_optimizer

        cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2,
                        n_embd=32, gradient_checkpointing=False)
        model = GPT(cfg)
        opt = make_optimizer(model, {"lr": 1e-3, "weight_decay": 0.1})
        decay_params = {id(p) for p in opt.param_groups[0]["params"]}
        attn_w = dict(model.named_parameters())["transformer.h.0.attn.c_attn.weight"]
        assert id(attn_w) in decay_params


class TestValidateConfig:
    def _minimal(self, tmp_path):
        p = str(tmp_path / "t.pt")
        torch.save(torch.ones(10, dtype=torch.int32), p)
        return {
            "seed": 42, "save_dir": "/tmp/x", "train_path": p, "val_path": p,
            "batch_size_per_device": 1, "eval_batch_size_per_device": 1,
            "grad_accum": 1, "max_steps": 10, "lr": 1e-3, "min_lr": 1e-4,
            "warmup": 2, "model": {"block_size": 64, "vocab_size": 256},
        }

    def test_valid_passes(self, tmp_path):
        validate_config(self._minimal(tmp_path))

    def test_missing_key_rejected(self, tmp_path):
        cfg = self._minimal(tmp_path)
        del cfg["lr"]
        with pytest.raises(KeyError):
            validate_config(cfg)

    def test_nonexistent_data_rejected(self, tmp_path):
        cfg = self._minimal(tmp_path)
        cfg["train_path"] = "/nonexistent/file.bin"
        with pytest.raises(FileNotFoundError):
            validate_config(cfg)


class TestCompatChecks:
    def test_resume_mismatch_rejected(self):
        cur = {"seed": 42, "model": {"n_layer": 24}}
        saved = {"seed": 43, "model": {"n_layer": 24}}
        with pytest.raises(ValueError):
            validate_resume_config_compatibility(cur, saved)

    def test_resume_match_ok(self):
        cur = {"seed": 42, "model": {"n_layer": 24}}
        validate_resume_config_compatibility(cur, dict(cur))

    def test_init_shape_mismatch_rejected(self):
        cur = {"model": {"n_embd": 1024}}
        saved = {"model": {"n_embd": 1408}}
        with pytest.raises(ValueError):
            validate_init_config_compatibility(cur, saved)

    def test_init_positional_scheme_mismatch_rejected(self):
        """R19 (adversarial review round 2): an init checkpoint trained with
        wpe must not silently load into a RoPE-only model."""
        cur = {"model": {"n_embd": 32, "use_rope": True, "use_wpe": False}}
        saved = {"model": {"n_embd": 32, "use_rope": False, "use_wpe": True}}
        with pytest.raises(ValueError):
            validate_init_config_compatibility(cur, saved)

    def test_init_padded_vocab_mismatch_rejected(self):
        # NOTE: None in the saved config means "unknown" (legacy checkpoint)
        # and is deliberately skipped; only a *different concrete value* is
        # an error. That is what we test here.
        cur = {"model": {"n_embd": 32, "padded_vocab_size": 50304}}
        saved = {"model": {"n_embd": 32, "padded_vocab_size": 50432}}
        with pytest.raises(ValueError):
            validate_init_config_compatibility(cur, saved)

    def test_none_saved_is_ok(self):
        validate_resume_config_compatibility({"seed": 1}, None)
        validate_init_config_compatibility({"model": {}}, None)


class TestTpuDay1Fixes:
    """Regression tests for the six day-1-on-real-TPU findings."""

    def test_kaggle_env_scrubbed_before_xla_import(self):
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        scrub = src.find("TPU_PROCESS_ADDRESSES")
        xla_import = src.find("import torch_xla.core.xla_model")
        assert 0 < scrub < xla_import, "Kaggle env scrub must run BEFORE torch_xla import"

    def test_rank_world_use_modern_runtime_api(self):
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        assert "xr.global_ordinal()" in src, "rank must prefer torch_xla.runtime"
        assert "xr.world_size()" in src
        assert "import torch_xla.runtime as xr" in src

    def test_spawn_maps_nprocs_for_pjrt(self):
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        assert "nprocs=(1 if nprocs == 1 else None)" in src, (
            "PJRT accepts only nprocs=1 or None")

    def test_softmax_no_fp32_materialization(self):
        """The fp32 att copy per layer cost 12 x 4 GB at B=16,T=2048 (measured
        OOM). softmax(dtype=fp32) is bit-identical (verified) without it."""
        src = open(os.path.join(ROOT, "src", "model.py")).read()
        assert "F.softmax(att, dim=-1, dtype=torch.float32)" in src
        assert "att = att.float()\n        att = att - att.amax" not in src

    def test_softmax_numerics_unchanged(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(0)
        att = torch.randn(2, 4, 32, 32)
        mask = torch.tril(torch.ones(32, 32, dtype=torch.bool))
        a = att.masked_fill(~mask, float("-inf"))
        old = F.softmax(a.float() - a.float().amax(-1, keepdim=True), dim=-1)
        new = F.softmax(a, dim=-1, dtype=torch.float32)
        assert torch.equal(old, new)

    def test_checkpoint_fallback_wrapper_present(self):
        src = open(os.path.join(ROOT, "src", "model.py")).read()
        assert "has no attribute 'xla'" in src, (
            "grad-checkpointing must fall back (not crash) on torch/xla mismatch")

    def test_kaggle_200m_configs_disable_grad_checkpointing(self):
        import glob as _glob
        import json as _json
        for f in _glob.glob(os.path.join(ROOT, "configs", "kaggle_*.json")) + \
                 _glob.glob(os.path.join(ROOT, "configs", "200m_*.json")):
            cfg = _json.load(open(f))
            assert cfg["model"].get("gradient_checkpointing") is False, (
                f"{os.path.basename(f)}: 200M fits without checkpointing; "
                "the torch/xla checkpoint bug makes it a liability")


class TestPilotReadiness:
    """Pre-pilot hardening: findings from the day-1 postmortem that would
    only bite AFTER a run finished (wrong numbers, missing curves)."""

    def test_world_size_mismatch_fails_fast(self):
        """The day-1 ordinal bug (world_size=1 x8 processes) was SILENT: the
        run trained, saved checkpoints and logged plausible CSV rows - with
        every token count 8x wrong and racing saves. Must die before step 0."""
        from src.train_xla import check_world_size
        cfg = {"num_cores": 8}
        with pytest.raises(RuntimeError, match="World-size mismatch"):
            check_world_size(cfg, world_size=1, rank=0)
        # Healthy worlds pass.
        check_world_size(cfg, world_size=8, rank=3)
        # Single-core debug mode is exempt (no sharding to corrupt).
        check_world_size({"num_cores": 1}, world_size=1, rank=0)

    def test_trainer_calls_world_size_guard(self):
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        assert "check_world_size(config, world_size=world_size, rank=rank)" in src, (
            "guard must run inside the worker, after ordinals are read")

    def test_litm_curve_logged_in_csv(self):
        """H4 needs a training-time trajectory ('when does the U-curve
        flatten?'), not just a final-checkpoint probe. The columns must exist
        in header AND row (lockstep is covered by the generic CSV test)."""
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        for col in ("pos_10", "pos_50", "pos_90",
                    "litm_middle_drop", "litm_worst_frac"):
            assert f'"{col}"' in src, f"CSV header must include {col}"
            assert f'metrics.get("{col}"' in src, f"CSV row must write {col}"
        assert "positional_recall_curve" in src, (
            "trainer must import and call the LITM probe")

    def test_run_state_records_world_and_tokens_per_update(self):
        """run_state_*.json is the budget-planning source of truth; after the
        day-1 bug it must prove sharding was real."""
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        assert '"world_size": int(world_size)' in src
        assert '"tokens_per_update": int(tokens_per_update)' in src

    def test_plot_dashboard_grid_fits_all_panels(self):
        """zip() over a hard-coded 3x2 grid silently dropped panel #7 (LITM).
        The grid must be derived from MECHANISM_PANELS."""
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "make_plots", os.path.join(ROOT, "scripts", "make_plots.py"))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert any("pos_10" in cols for _, _, cols in mod.MECHANISM_PANELS), (
            "LITM panel missing from dashboard")
        src = open(os.path.join(ROOT, "scripts", "make_plots.py")).read()
        assert "len(MECHANISM_PANELS)" in src, (
            "subplot grid must be computed from the panel list, not hard-coded")


class TestDryRunHarness:
    """scripts/dry_run_cpu.py runs the REAL trainer loop on CPU. Before it
    existed, the training loop itself was only ever executed on TPU - every
    integration bug (CSV wiring, probe calls, save paths) cost a Kaggle
    session to discover."""

    def test_dry_run_script_exists_and_parses(self):
        path = os.path.join(ROOT, "scripts", "dry_run_cpu.py")
        assert os.path.exists(path)
        import ast
        ast.parse(open(path).read())

    def test_dry_run_forces_single_core(self):
        """The world-size guard (correctly) kills num_cores=8 with world=1;
        the dry run must therefore always pin num_cores=1, or it would die
        on every multi-core config."""
        src = open(os.path.join(ROOT, "scripts", "dry_run_cpu.py")).read()
        assert 'config["num_cores"] = 1' in src
        assert 'config["num_workers"] = 0' in src

    def test_dry_run_stubs_modern_runtime_module(self):
        """train_xla.py prefers torch_xla.runtime ordinals (day-1 fix); a stub
        without that module would silently exercise only the legacy path."""
        src = open(os.path.join(ROOT, "scripts", "dry_run_cpu.py")).read()
        assert "torch_xla.runtime" in src
        assert "global_ordinal" in src and "world_size" in src


class TestHbmBudget:
    """Day-1 finding #7 (the REAL cause of the first OOM, discovered after
    the world-size fix): fp32 softmax materializes [B*H, T, T] per layer and
    the backward pass keeps ALL layers' buffers alive. At b=16, T=2048, H=16:
    4 GB x 12 layers = 48 GB >> 15.75 GB HBM - the config could never fit,
    on any runtime. Measured: 'Used 68.25G of 15.75G hbm', twelve
    f32[256,2048,2048] allocations."""

    HBM_BYTES = 15.75 * 2**30
    # Leave headroom for params+optimizer (~2.5 GB at 200m) and XLA temps.
    ATTN_BUDGET = 10.0 * 2**30

    @staticmethod
    def _attn_bytes(cfg):
        m = cfg["model"]
        b = int(cfg["batch_size_per_device"])
        # Finding #10 (v5e smoke): the fp32 softmax buffer is NOT alone.
        # Every active score-bias mechanism (distance / salience / forget)
        # materializes its own [B, H, T, T] tensor that backward keeps
        # alive per layer. The base twin passed at b=2 while HLA hit
        # 'Attempting to reserve 9.37G ... 9.28G free' - because this
        # counter ignored the bias tensors. Count them.
        n_buffers = 1
        n_buffers += 1 if m.get("use_distance_laplace") else 0
        n_buffers += 1 if m.get("use_salience_bias") else 0
        n_buffers += 1 if m.get("use_forget_gate") or m.get("forget_alpha") else 0
        return (int(m["n_layer"]) * b * int(m["n_head"])
                * int(m["block_size"]) ** 2 * 4 * n_buffers)

    def test_tpu_v3_configs_fit_hbm(self):
        import glob as _glob
        import json as _json
        # Finding #13: pattern lists rot (tpu3_abl_* escaped findings 7/10).
        # Budget-check EVERY 8-core config with a 2048 block instead.
        checked = 0
        for p in sorted(_glob.glob(os.path.join(ROOT, "configs", "*.json"))):
            cfg = _json.load(open(p))
            if cfg.get("num_cores", 8) != 8:
                continue
            if int(cfg.get("model", {}).get("block_size", 0)) < 2048:
                continue
            if True:
                got = self._attn_bytes(cfg)
                assert got <= self.ATTN_BUDGET, (
                    f"{os.path.basename(p)}: fp32-softmax residency "
                    f"{got/2**30:.1f} GB > {self.ATTN_BUDGET/2**30:.1f} GB budget "
                    f"(HBM 15.75). Lower batch_size_per_device, raise grad_accum.")
                checked += 1
        assert checked >= 20

    def test_tokens_per_update_invariant_preserved(self):
        """The OOM fixes (b16->2->1, accum 1->8->16) must NOT change the
        optimization trajectory: tokens/update stays 262,144 everywhere."""
        import glob as _glob
        import json as _json
        pats = ("kaggle_*.json", "200m_*.json", "tpu3_200m_*.json")
        for pat in pats:
            for p in sorted(_glob.glob(os.path.join(ROOT, "configs", pat))):
                cfg = _json.load(open(p))
                tpu = (int(cfg["batch_size_per_device"]) * 8
                       * int(cfg["model"]["block_size"]) * int(cfg["grad_accum"]))
                assert tpu == 262144, f"{os.path.basename(p)}: tokens/update {tpu}"

    def test_grad_accum_preserves_update_math(self):
        """The HBM fix changes b16/accum1 -> b2/accum8. This must be the SAME
        optimization step: sum of (loss/accum).backward() over 8 microbatches
        equals one full-batch backward (fp32, linearity of gradients).
        Measured on this repo's GPT: max weight diff 3.7e-09 after 1 update."""
        import torch as _t
        from src.model import GPT, GPTConfig
        cfg = GPTConfig(block_size=32, vocab_size=256, n_layer=1, n_head=2,
                        n_embd=32, gradient_checkpointing=False)

        def one_update(accum):
            _t.manual_seed(0)
            m = GPT(cfg)
            opt = _t.optim.SGD(m.parameters(), lr=0.1)
            _t.manual_seed(42)
            X = _t.randint(0, 256, (16, 32))
            Y = _t.randint(0, 256, (16, 32))
            opt.zero_grad()
            mb = 16 // accum
            for i in range(accum):
                _, loss = m(X[i * mb:(i + 1) * mb], Y[i * mb:(i + 1) * mb])
                (loss / accum).backward()
            opt.step()
            return _t.cat([p.detach().flatten() for p in m.parameters()])

        diff = (one_update(1) - one_update(8)).abs().max().item()
        assert diff < 1e-5, f"accum changed the update: max weight diff {diff}"

    def test_atomic_save_tmp_name_deterministic(self):
        """Finding #8: on PJRT the xm.save writer process and the
        xr-rank-0 replacer can be DIFFERENT processes (observed rank=0 on
        local_rank=3, pid 827). A time+pid tmp name diverges between them:
        writer writes its own tmp, replacer renames a nonexistent one ->
        FileNotFoundError best_val_*.pt.tmp.1785696336.827 (real crash).
        The tmp name must contain no per-process entropy, and a barrier must
        sit between write and replace."""
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        start = src.find("def atomic_xm_save")
        body = src[start:src.find("\ndef ", start + 1)]
        assert 'tmp = f"{path}.tmp"' in body, "tmp name must be deterministic"
        assert "os.getpid()" not in body, "pid in tmp name re-opens finding #8"
        assert "time.time()" not in body, "timestamp in tmp name re-opens finding #8"
        assert "rendezvous(" in body, "need write->replace barrier across processes"

    def test_validate_log_accepts_real_log_shape(self, tmp_path):
        """Finding #9 (pre-pilot audit): the trainer writes a row every
        log_every steps but evaluates every val_every steps, so real logs
        have nan val_loss on most rows. validate_log.py's default mode
        rejected EVERY real pilot log. Contract now: nan val rows fine,
        present val values must be finite, and >=1 eval row must exist."""
        import subprocess
        import sys as _sys
        p = tmp_path / "log.csv"
        p.write_text(
            "step,tokens_seen,train_loss,val_loss\n"
            "1,100,10.5,10.6\n"
            "2,200,10.4,nan\n"
            "3,300,10.3,10.5\n")
        r = subprocess.run([_sys.executable,
                            os.path.join(ROOT, "scripts", "validate_log.py"),
                            str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "eval_rows=2" in r.stdout
        # But a log where eval NEVER ran must fail.
        p2 = tmp_path / "log2.csv"
        p2.write_text("step,tokens_seen,train_loss,val_loss\n1,100,10.5,nan\n")
        r2 = subprocess.run([_sys.executable,
                             os.path.join(ROOT, "scripts", "validate_log.py"),
                             str(p2)], capture_output=True, text=True)
        assert r2.returncode != 0

    def test_9h_configs_fit_one_session(self):
        """Session policy (user + measured V4 numbers): a run must fit one
        9h batch session AT THE MEASURED SPEED (0.53 steps/sec, V4), and if
        hardware turns out slower the auto-resume tail must be bounded
        (resume_every <= 500). 15000 steps = 8.3h at 0.53 -> single session;
        worst case -20% -> ~1.8h follow-up, <=500 steps lost. This trades
        the old hard-pessimism bound for +0.78B tokens (14.5 -> 18.1
        tok/param), which strengthens the primary comparison."""
        import glob as _glob
        import json as _json
        found = 0
        for p in sorted(_glob.glob(os.path.join(ROOT, "configs", "kaggle_200m_*_9h_*.json"))):
            c = _json.load(open(p))
            steps = int(c["max_steps"])
            sps = 0.53  # V4-measured steady-state
            total_h = (steps / sps
                       + steps / int(c["val_every"]) * 25
                       + steps / int(c["svd_every"]) * 110) / 3600
            assert total_h <= 8.5, (
                f"{os.path.basename(p)}: {total_h:.1f}h > 8.5h at measured "
                "speed - does not fit a 9h session even optimistically")
            # twin-pair invariants
            assert (c["batch_size_per_device"] * 8 * c["model"]["block_size"]
                    * c["grad_accum"]) == 262144
            assert steps * 262144 <= 4_700_000_000
            assert int(c["resume_every"]) <= 500, "need resume for session-splits"
            found += 1
        # s42 pair + s43 pair (R1-W2 seed-replication, round 12)
        assert found in (2, 4), "expected the base+hla 9h pair(s)"
        assert found % 2 == 0, "9h configs must come in base/hla pairs"

    def test_crash_save_master_only(self):
        """Finding #11 (V5): all 8 ranks wrote full fp32 model+Adam crash
        payloads (~2.4 GB each at 200m = 19 GB per crash) - the session
        disk died before anything could be salvaged (output 0 B). Ranks
        are data-parallel twins after the last completed step, so one
        file carries all information: only rank 0 may write."""
        src = open(os.path.join(ROOT, "src", "train_xla.py")).read()
        i = src.find("Training crashed")
        assert i > 0
        window = src[i:i + 600]
        assert "if rank != 0:" in window and "raise" in window, (
            "non-master ranks must re-raise BEFORE the crash-save block")


@pytest.mark.slow
class TestBitExactResume:
    """Round-8 audit, THE V7 contract: an interrupted-then-resumed run must
    equal the continuous run BIT-FOR-BIT - final weights identical and the
    per-step train-loss trajectory identical after seam dedup. The 9h 200m
    sessions WILL resume; if resume shifted the data order or optimizer
    state even slightly, the sterile-twin comparison would silently rot.
    Proven live in the sandbox first (worst weight diff 0.000e+00)."""

    def _dry(self, cfg_path, *overrides):
        import subprocess
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "dry_run_cpu.py"),
               "--config", cfg_path]
        for ov in overrides:
            cmd += ["--override", ov]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    def test_resumed_equals_continuous(self, tmp_path):
        import json
        import csv
        import subprocess
        # 3 sequential trainer subprocesses at ~0.5GB each on top of the
        # pytest parent: skip on very low-memory sandboxes (verified live
        # there instead; CI/Codespaces run it in full).
        try:
            avail_kb = int(next(ln for ln in open("/proc/meminfo")
                                if "MemAvailable" in ln).split()[1])
            if avail_kb < 1_200_000:
                pytest.skip("needs ~1.2GB free for trainer subprocesses")
        except (OSError, StopIteration):
            pass
        base = json.load(open(os.path.join(
            ROOT, "configs", "smoke_hla_s42.json")))
        base["model"].update(n_layer=1, n_head=2, n_embd=32, block_size=32)
        common = dict(num_workers=0, max_steps=4, val_every=99, svd_every=0,
                      resume_every=2, save_every=0, val_batches=1,
                      min_free_gb=0, log_every=1)
        # dummy data + tiny init
        r = subprocess.run([sys.executable, os.path.join(
            ROOT, "scripts", "make_dummy_data.py")],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        init = str(tmp_path / "init.pt")
        cfgs = {}
        for name in ("cont", "resu"):
            c = dict(base)
            c.update(common)
            c.update(run_name=f"{name}_x", save_dir=str(tmp_path / name),
                     init_ckpt=init)
            p = str(tmp_path / f"{name}.json")
            json.dump(c, open(p, "w"))
            cfgs[name] = p
        r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "make_init.py"),
                            "--config", cfgs["cont"], "--out", init],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        # continuous 4 steps
        r = self._dry(cfgs["cont"])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        # interrupted at 2, resumed to 4
        r = self._dry(cfgs["resu"], "max_steps=2")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        latest = str(tmp_path / "resu" / "latest_resu_x_resume.pt")
        assert os.path.exists(latest)
        r = self._dry(cfgs["resu"], "max_steps=4",
                      f'resume_ckpt="{latest}"')
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        # weights bit-equal
        a = torch.load(str(tmp_path / "cont" / "final_cont_x_bf16.pt"),
                       map_location="cpu", weights_only=True)
        b = torch.load(str(tmp_path / "resu" / "final_resu_x_bf16.pt"),
                       map_location="cpu", weights_only=True)
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), f"resume diverged at {k}"
        # per-step trajectory equal after seam dedup

        def rows(p):
            dd = {}
            for row in csv.reader(open(p)):
                if row and row[0] and not row[0].startswith("#") and row[0] != "step":
                    dd[row[0]] = row[3]
            return dd
        ca = rows(str(tmp_path / "cont" / "train_log_cont_x.csv"))
        cb = rows(str(tmp_path / "resu" / "train_log_resu_x.csv"))
        assert ca == cb, "train-loss trajectories differ after resume"


class TestCheckpointCleanup:
    """Round-10 audit: cleanup_old_checkpoints on a real filesystem. The V7
    sessions rotate step_* checkpoints on a 20GB disk; a glob that matched
    best/latest/final would kill resume, a wrong sort would keep the OLD
    steps. Proven live before becoming a test."""

    def _mk(self, d, run, steps):
        for s in steps:
            for suffix in (f"step_{s}_{run}.pt", f"step_{s}_{run}_resume.pt"):
                open(os.path.join(d, suffix), "w").write("x")
        protected = [f"best_val_{run}.pt", f"final_{run}_bf16.pt",
                     f"latest_{run}_resume.pt", f"train_log_{run}.csv"]
        for f in protected:
            open(os.path.join(d, f), "w").write("x")
        return protected

    def test_keeps_last_n_steps_and_protects_specials(self, tmp_path):
        from src.train_xla import cleanup_old_checkpoints
        run = "kaggle_200m_base_9h_s42"
        d = str(tmp_path)
        protected = self._mk(d, run, (500, 1000, 1500, 2000, 2500))
        cleanup_old_checkpoints(d, run, keep_last=2)
        left = sorted(os.listdir(d))
        steps = sorted({int(f.split("_")[1]) for f in left
                        if f.startswith("step_")})
        assert steps == [2000, 2500], f"wrong survivors: {steps}"
        for f in protected:
            assert f in left, f"protected file deleted: {f}"

    def test_keep_last_zero_is_noop_and_foreign_runs_untouched(self, tmp_path):
        from src.train_xla import cleanup_old_checkpoints
        run = "runA"
        d = str(tmp_path)
        self._mk(d, run, (100, 200))
        open(os.path.join(d, "step_50_runB.pt"), "w").write("x")
        before = sorted(os.listdir(d))
        cleanup_old_checkpoints(d, run, keep_last=0)
        assert sorted(os.listdir(d)) == before
        cleanup_old_checkpoints(d, run, keep_last=1)
        assert os.path.exists(os.path.join(d, "step_50_runB.pt"))


class TestDuplicateJsonKeys:
    """Finding #20 (round 15): json.load keeps the LAST duplicate key, so a
    hand-edited config with a second "lr" silently trains with the wrong
    value while every downstream gate sees only the collapsed dict. Both
    config entry points must hard-fail on duplicates at any level."""

    def test_trainer_load_config_rejects_duplicates(self, tmp_path):
        from src.train_xla import load_config
        src = open(os.path.join(
            ROOT, "configs", "kaggle_200m_base_9h_s42.json")).read()
        evil = src.rstrip().rstrip("}") + ',  "lr": 0.006\n}\n'
        p = str(tmp_path / "dup.json")
        open(p, "w").write(evil)
        with pytest.raises(ValueError, match="duplicate key"):
            load_config(p)
        nested = src.replace('"n_layer":', '"n_layer": 24, "n_layer":', 1)
        p2 = str(tmp_path / "dup2.json")
        open(p2, "w").write(nested)
        with pytest.raises(ValueError, match="duplicate key"):
            load_config(p2)
        # clean config still loads
        cfg = load_config(os.path.join(
            ROOT, "configs", "kaggle_200m_base_9h_s42.json"))
        assert cfg["lr"] == 0.0006

    def test_validate_configs_rejects_duplicates(self, tmp_path):
        import subprocess
        src = open(os.path.join(
            ROOT, "configs", "kaggle_200m_base_9h_s42.json")).read()
        evil = src.rstrip().rstrip("}") + ',  "lr": 0.006\n}\n'
        p = str(tmp_path / "dup.json")
        open(p, "w").write(evil)
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "validate_configs.py"),
                            "--base", p, "--hla",
                            os.path.join(ROOT, "configs",
                                         "kaggle_200m_hla_9h_s42.json")],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "duplicate key" in (r.stdout + r.stderr)
