# Copyright 2026 Slyatski Ilya
"""Regression tests for finding #79 and its whole class (r122-r123).

#79 (battle-caught 2026-09-30): the v15 mini-124M matrix shipped
resume_every=0, which validate_config rejects (">= 1 when provided") —
all three MINI_SMOKE arms failed on TPU before training started. The
contract tests checked architecture fields but never ran the on-device
gates themselves. These tests close the class, not just the instance:

  1. every shipped config passes validate_config (the exact #79 gate);
  2. every shipped config instantiates GPTConfig (frozen dataclass:
     catches unknown model.* keys and bad attention_backend values);
  3. the LR schedule is finite and positive across the FULL budget of
     every config (warmup/cosine edge cases);
  4. replication pair contracts: s43 base-vs-hla may differ ONLY in the
     five mechanism amplitudes (+ naming/init bookkeeping); the same arm
     across s42/s43 may differ ONLY in seed (+ naming/init bookkeeping);
     init_ckpt filenames must embed the config's own seed.

train_path/val_path are REQUIRED keys pointing at Kaggle mounts; the
validator checks existence only when present, so tests rewrite both to a
dummy file — everything else runs exactly as on the device.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

from src.model import GPTConfig
from src.train_xla import get_lr, validate_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(REPO, "configs", "*.json")))

MECHANISM_FIELDS = {
    "baseline_type",
    "distance_laplace_alpha",
    "laplace_alpha",
    "phase_mult",
    "salience_alpha",
}
NAMING_FIELDS = {"run_name", "save_dir", "variant", "init_ckpt", "experiment"}


def _load(path, tmp_path=None):
    with open(path) as f:
        config = json.load(f)
    if tmp_path is not None:
        dummy = tmp_path / "dummy_tokens.bin"
        if not dummy.exists():
            dummy.write_bytes(b"\x00\x00")
        for key in ("train_path", "val_path"):
            if key in config:
                config[key] = str(dummy)
    return config


def test_configs_discovered():
    assert len(CONFIGS) >= 40, f"configs/ glob suspiciously small: {len(CONFIGS)}"


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p) for p in CONFIGS])
def test_config_passes_all_device_gates(path, tmp_path):
    config = _load(path, tmp_path)
    validate_config(config)                      # gate 1: the exact #79 gate
    GPTConfig(**config["model"])                 # gate 2: unknown keys/backends
    warmup, max_steps = int(config["warmup"]), int(config["max_steps"])
    lr, min_lr = float(config["lr"]), float(config["min_lr"])
    for step in range(1, max_steps + 1):         # gate 3: full-budget LR sweep
        v = get_lr(step, warmup=warmup, max_steps=max_steps,
                   base_lr=lr, min_lr=min_lr)
        assert v > 0.0 and v == v and v != float("inf"), (path, step, v)


def test_resume_every_zero_is_rejected(tmp_path):
    """The exact #79 failure mode stays impossible to reintroduce silently."""
    config = _load(CONFIGS[0], tmp_path)
    config["resume_every"] = 0
    with pytest.raises(ValueError, match="resume_every"):
        validate_config(config)


def _diff(a, b, section=None):
    da = a[section] if section else {k: v for k, v in a.items() if k != "model"}
    db = b[section] if section else {k: v for k, v in b.items() if k != "model"}
    return {k for k in set(da) | set(db) if da.get(k) != db.get(k)}


@pytest.mark.parametrize("seed", ["s42", "s43"])
def test_pair_contract_base_vs_hla(seed):
    """Twin arms may differ ONLY in mechanism amplitudes + naming/init."""
    base = _load(os.path.join(REPO, "configs", f"kaggle_200m_base_9h_{seed}.json"))
    hla = _load(os.path.join(REPO, "configs", f"kaggle_200m_hla_9h_{seed}.json"))
    model_diff = _diff(base, hla, "model")
    top_diff = _diff(base, hla)
    assert model_diff == MECHANISM_FIELDS, f"model diff drifted: {sorted(model_diff)}"
    assert top_diff <= NAMING_FIELDS, f"non-naming top-level diff: {sorted(top_diff - NAMING_FIELDS)}"


@pytest.mark.parametrize("arm", ["base", "hla"])
def test_pair_contract_s42_vs_s43(arm):
    """Replication config = same arm, only seed + naming/init may move."""
    c42 = _load(os.path.join(REPO, "configs", f"kaggle_200m_{arm}_9h_s42.json"))
    c43 = _load(os.path.join(REPO, "configs", f"kaggle_200m_{arm}_9h_s43.json"))
    assert _diff(c42, c43, "model") == set(), "model must be identical across seeds"
    top_diff = _diff(c42, c43)
    assert top_diff <= NAMING_FIELDS | {"seed"}, f"unexpected diff: {sorted(top_diff)}"
    assert c42["seed"] == 42 and c43["seed"] == 43


@pytest.mark.parametrize("path", [p for p in CONFIGS
                                  if "init_ckpt" in json.load(open(p))
                                  and not p.endswith("_template.json")],
                         ids=lambda p: os.path.basename(p))
def test_init_ckpt_embeds_own_seed(path):
    """The init filename must embed the config's own seed — a copy-paste of
    an s42 init path into an s43 config would silently destroy the
    replication (same init => same trajectory => fake 'replication').
    Two honest naming eras exist: infix init_200m_s42_base.pt (kaggle) and
    suffix init_200m_base_s42.pt (legacy tpu3) — both encode the seed as a
    delimited token, so the contract is r'_s{seed}[._]'. *_template.json
    blueprints carry placeholder init names by design (users rename them
    per ablation) and are excluded HERE ONLY — validator/GPTConfig/LR
    gates above still cover them."""
    import re
    config = json.load(open(path))
    name = os.path.basename(config["init_ckpt"])
    assert re.search(rf"_s{config['seed']}[._]", name), (name, config["seed"])
