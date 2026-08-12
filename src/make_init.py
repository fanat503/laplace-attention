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
Sterile initialization checkpoint generator for base/HLA runs.

Why this exists:
  - `src/__init__.py` is only a Python package/export file.
  - `make_init.py` creates reproducible model initialization checkpoints.

Modes:
  1) Single init:
     python make_init.py --config configs/800m_hla.json --out data/init_hla.pt

  2) Shared-backbone init for matched base vs HLA runs:
     python make_init.py \
       --base-config configs/800m_base.json --hla-config configs/800m_hla.json \
       --out-base data/init_base.pt --out-hla data/init_hla.pt \
       --shared-backbone

Shared-backbone mode copies every matching non-HLA tensor from the base model into
HLA, then resets all HLA tensors to identity. This makes the common backbone as
identical as possible while preserving the provided model logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from src.model import GPT, GPTConfig, RMSNorm  # noqa: E402


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_name(name: str) -> Optional[torch.dtype]:
    if name == "fp32":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


# Match keys that END with these markers (last component of path).
HLA_KEY_PATTERNS = (
    re.compile(r"\.W_phase_q$"),
    re.compile(r"\.W_phase_k$"),
    re.compile(r"\.W_range_k$"),
    re.compile(r"\.W_range_v$"),
    re.compile(r"\.W_gate_k\.weight$"),  # nn.Linear weights
    re.compile(r"\.W_gate_v\.weight$"),
    re.compile(r"\.W_gate_sal\.weight$"),
    re.compile(r"\.W_gate_f\.weight$"),
    re.compile(r"\.W_gate_d\.weight$"),
    re.compile(r"\.W_qtemp\.weight$"),
    re.compile(r"\.W_range_f$"),
    re.compile(r"\.W_layer_temp$"),
    re.compile(r"\.W_phase_scale$"),
    re.compile(r"^memory_slots\.slot_values$"),
    re.compile(r"^memory_slots\.slot_queries$"),
    re.compile(r"^memory_slots\.w_scale$"),
    re.compile(r"^memory_slots\.lambda_mem$"),
)

def is_hla_key(key):
    return any(p.search(key) for p in HLA_KEY_PATTERNS)


def model_from_config(config: Dict[str, Any], *, seed: Optional[int] = None) -> GPT:
    if seed is None:
        seed = int(config["seed"])
    seed_everything(seed)
    model = GPT(GPTConfig(**config["model"]))
    model.eval()
    model.reset_hla_identity()
    return model


def cast_state(state: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        t = v.detach().cpu().contiguous()
        if dtype is not None and torch.is_floating_point(t):
            t = t.to(dtype)
        out[k] = t
    return out


# Module-level: which params are checked for exact-zero vs fixed-init value.
HLA_PARAMS_MUST_BE_ZERO = frozenset({
    "W_phase_q", "W_phase_k",
    "W_range_k", "W_range_v",
    "W_gate_k", "W_gate_v", "W_gate_sal", "W_gate_f", "W_range_f",
    "W_gate_d", "W_qtemp",
    "W_layer_temp", "W_phase_scale",
})

HLA_PARAMS_FIXED_INIT = frozenset({
    # memory_slots.* params IF use_memory_slots=True.
    "memory_slots.slot_values",  # zero at init
    "memory_slots.w_scale",       # zero at init
})

HLA_PARAMS_FIXED_INIT_NONZERO = frozenset({
    # memory_slots params with non-zero init values.
    "memory_slots.lambda_mem",  # 0.5 at init
})


def assert_hla_identity(model: GPT, *, atol: float = 0.0) -> None:
    """Verify HLA-specific params are at identity state.

    Must-be-zero: max_abs value < atol.
    Memory slots must-be-zero: same check.
    Memory slots fixed-init non-zero: mean ≈ expected value.

    Skips params not present in state_dict (e.g., memory_slots.* when
    use_memory_slots=False).
    """
    bad: List[str] = []
    sd = model.state_dict()

    for k, v in sd.items():
        # Check parameter-level identity.
        # Match by exact key or suffix for nn.Linear weights.
        is_hla_attn = any(
            k == p or k.startswith(f"transformer.h.{i}.attn.{p}")
            for p in HLA_PARAMS_MUST_BE_ZERO
            for i in range(model.config.n_layer)
        )

        if is_hla_attn or k in HLA_PARAMS_FIXED_INIT:
            max_abs = float(v.detach().float().abs().max().item()) if v.numel() > 0 else 0.0
            if max_abs > atol:
                bad.append(f"{k}: max_abs={max_abs} (must be zero)")
        elif k in HLA_PARAMS_FIXED_INIT_NONZERO:
            mean = float(v.detach().float().mean().item()) if v.numel() > 0 else 0.0
            if k == "memory_slots.lambda_mem":
                expected = 0.5
                if abs(mean - expected) > 0.01:
                    bad.append(f"{k}: mean={mean} (expected {expected})")

    if bad:
        raise RuntimeError(
            "HLA identity check failed:\n  " + "\n  ".join(bad[:32])
        )


def atomic_torch_save(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_json_atomic(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def parameter_report(model: GPT) -> Dict[str, int]:
    """Count parameters by walking module hierarchy."""
    groups = {
        "embedding": 0, "attention_base": 0, "mlp": 0, "norm": 0, "hla": 0, "other": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        parts = name.split(".")
        module_path = ".".join(parts[:-1]) if len(parts) > 1 else ""
        try:
            mod = model.get_submodule(module_path) if module_path else model
        except Exception:
            groups["other"] += n
            continue
        # Categorise by owning module.
        if is_hla_key(name):
            groups["hla"] += n
        elif isinstance(mod, nn.Embedding):
            groups["embedding"] += n
        elif isinstance(mod, nn.Linear):
            if any(k in name for k in ("c_attn", "c_proj")):
                groups["attention_base"] += n
            else:
                groups["mlp"] += n
        elif isinstance(mod, RMSNorm):
            groups["norm"] += n
        else:
            groups["other"] += n
    groups["total"] = sum(groups.values())
    groups["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return groups


def make_payload(
    *,
    model: GPT,
    config: Dict[str, Any],
    dtype: Optional[torch.dtype],
    role: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    assert_hla_identity(model)
    state = cast_state(model.state_dict(), dtype)
    return {
        "model": state,
        "config": config,
        "config_hash": stable_hash(config),
        "seed": int(config["seed"]),
        "role": role,
        "param_report": parameter_report(model),
        "manifest": manifest,
    }


def save_single_init(config: Dict[str, Any], out: str, *, dtype: Optional[torch.dtype]) -> None:
    model = model_from_config(config)
    manifest = {
        "mode": "single",
        "description": "single sterile initialization",
        "hla_identity_reset": True,
    }
    payload = make_payload(model=model, config=config, dtype=dtype, role="single", manifest=manifest)
    atomic_torch_save(payload, out)
    write_json_atomic(payload["manifest"], f"{out}.manifest.json")
    print(f"[single] saved {out}")
    print(f"  config_hash={payload['config_hash']} params={payload['param_report']['total']:,}")


def copy_shared_backbone(base: GPT, hla: GPT, *, allow_shape_mismatch: bool = False) -> Dict[str, Any]:
    """Copy matching non-HLA tensors from base into HLA.

    Does not copy HLA tensors; those are reset to identity in both models.
    """
    base_sd = base.state_dict()
    hla_sd = hla.state_dict()
    copied: List[str] = []
    skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    skipped_hla: List[str] = []
    missing_in_hla: List[str] = []

    new_hla_sd = dict(hla_sd)
    for k, v in base_sd.items():
        if is_hla_key(k):
            skipped_hla.append(k)
            continue
        if k not in hla_sd:
            missing_in_hla.append(k)
            continue
        if tuple(v.shape) != tuple(hla_sd[k].shape):
            skipped_shape.append((k, tuple(v.shape), tuple(hla_sd[k].shape)))
            continue
        new_hla_sd[k] = v.detach().clone()
        copied.append(k)

    hla.load_state_dict(new_hla_sd, strict=True)
    base.reset_hla_identity()
    hla.reset_hla_identity()
    assert_hla_identity(base)
    assert_hla_identity(hla)

    # Verify all copied tensors are exactly equal after load.
    hla_sd_after = hla.state_dict()
    unequal = []
    for k in copied:
        if not torch.equal(base_sd[k].cpu(), hla_sd_after[k].cpu()):
            unequal.append(k)
    if unequal:
        raise RuntimeError(f"Shared-backbone copy verification failed for keys: {unequal[:16]}")
    if skipped_shape and not allow_shape_mismatch:
        preview = [f"{k}: base={bs}, hla={hs}" for k, bs, hs in skipped_shape[:16]]
        raise RuntimeError(
            "Shape mismatches in non-HLA shared-backbone tensors. "
            "This is not sterile unless explicitly allowed.\n" + "\n".join(preview)
        )
    if missing_in_hla and not allow_shape_mismatch:
        raise RuntimeError(
            "Base non-HLA keys missing in HLA model. This is not sterile unless explicitly allowed.\n"
            + "\n".join(missing_in_hla[:16])
        )

    return {
        "copied_non_hla_keys": copied,
        "num_copied_non_hla_keys": len(copied),
        "skipped_hla_keys": skipped_hla,
        "num_skipped_hla_keys": len(skipped_hla),
        "missing_in_hla": missing_in_hla,
        "shape_mismatches": [
            {"key": k, "base_shape": bs, "hla_shape": hs} for k, bs, hs in skipped_shape
        ],
    }


def save_shared_backbone_init(
    *,
    base_config: Dict[str, Any],
    hla_config: Dict[str, Any],
    out_base: str,
    out_hla: str,
    dtype: Optional[torch.dtype],
    allow_shape_mismatch: bool = False,
) -> None:
    # Sterility gate (init-audit finding): the trainer reads data-order seed
    # from EACH run's own config. If base and HLA seeds differ, the twins see
    # different token order - silently NOT token-matched, even though the
    # shared backbone itself would still verify. validate_configs.py catches
    # this too, but make_init is callable directly (Kaggle cells do) -
    # defence in depth: fail here as well.
    if int(base_config["seed"]) != int(hla_config["seed"]):
        raise SystemExit(
            f"STERILITY VIOLATION: base seed={base_config['seed']} != "
            f"hla seed={hla_config['seed']}. Twins must share the seed, or "
            "their data order diverges and the pair is not token-matched."
        )
    # Use base seed for base. HLA seed is irrelevant for copied backbone but used
    # for any non-copied non-HLA tensors if shapes differ.
    base = model_from_config(base_config, seed=int(base_config["seed"]))
    hla = model_from_config(hla_config, seed=int(hla_config["seed"]))

    manifest = copy_shared_backbone(base, hla, allow_shape_mismatch=allow_shape_mismatch)
    manifest.update(
        {
            "mode": "shared_backbone",
            "base_config_hash": stable_hash(base_config),
            "hla_config_hash": stable_hash(hla_config),
            "hla_identity_reset": True,
        }
    )

    base_payload = make_payload(
        model=base,
        config=base_config,
        dtype=dtype,
        role="base_shared_backbone",
        manifest=manifest,
    )
    hla_payload = make_payload(
        model=hla,
        config=hla_config,
        dtype=dtype,
        role="hla_shared_backbone",
        manifest=manifest,
    )

    atomic_torch_save(base_payload, out_base)
    atomic_torch_save(hla_payload, out_hla)
    write_json_atomic(manifest, f"{out_base}.manifest.json")
    write_json_atomic(manifest, f"{out_hla}.manifest.json")

    print(f"[shared] saved base: {out_base}")
    print(f"         hash={base_payload['config_hash']} params={base_payload['param_report']['total']:,}")
    print(f"[shared] saved hla : {out_hla}")
    print(f"         hash={hla_payload['config_hash']} params={hla_payload['param_report']['total']:,}")
    print(f"[shared] copied_non_hla_keys={manifest['num_copied_non_hla_keys']} skipped_hla_keys={manifest['num_skipped_hla_keys']}")
    if manifest["shape_mismatches"]:
        print(f"[shared] WARNING shape mismatches: {len(manifest['shape_mismatches'])}")
        print(json.dumps(manifest["shape_mismatches"][:8], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sterile init checkpoints for HLA-v4/base runs")

    # Single mode
    parser.add_argument("--config", type=str, default=None, help="Single config JSON")
    parser.add_argument("--out", type=str, default=None, help="Output checkpoint for single mode")

    # Shared-backbone mode
    parser.add_argument("--shared-backbone", action="store_true", help="Create matched base/HLA init checkpoints")
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--hla-config", type=str, default=None)
    parser.add_argument("--out-base", type=str, default=None)
    parser.add_argument("--out-hla", type=str, default=None)

    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--allow-shape-mismatch", action="store_true", help="Allow non-HLA shape mismatches in shared-backbone mode")
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)

    if args.shared_backbone:
        required = [args.base_config, args.hla_config, args.out_base, args.out_hla]
        if any(x is None for x in required):
            raise SystemExit("--shared-backbone requires --base-config --hla-config --out-base --out-hla")
        save_shared_backbone_init(
            base_config=load_config(args.base_config),
            hla_config=load_config(args.hla_config),
            out_base=args.out_base,
            out_hla=args.out_hla,
            dtype=dtype,
            allow_shape_mismatch=args.allow_shape_mismatch,
        )
    else:
        if args.config is None or args.out is None:
            raise SystemExit("single mode requires --config and --out")
        save_single_init(load_config(args.config), args.out, dtype=dtype)


if __name__ == "__main__":
    main()
