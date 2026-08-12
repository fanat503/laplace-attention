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
Production-grade TPU/XLA trainer for HLA-v4-safe-scale and base GPT runs.

This file is deliberately a *trainer*, not an architecture file. It expects:
  - src/model.py : GPTConfig, GPT
  - src/data.py  : FixedDataset or compatible dataset returning {"input_ids": ...}
  - src/eval.py  : evaluate_induction, measure_attention_entropy, phase_statistics

Main public entrypoint:
    train_worker_xla(config: dict)

CLI:
    python train_xla.py --config configs/800m_base_s42.json
    python train_xla.py --config configs/800m_hla_s42.json --override max_steps=1000

Design goals for long TRC/TPU runs:
  - deterministic sharded data order;
  - exact-ish resume with data skipping;
  - full resume checkpoints + lightweight bf16 weights;
  - crash-save best effort;
  - token-weighted distributed validation;
  - global gradient clipping across TPU replicas;
  - minimal architecture branching inside the trainer;
  - clear CSV/run metadata for paper plots.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

# B1 fix (external review): torch_xla is imported lazily so that
# `python src/train_xla.py --help`, imports and doc tooling work on CPU-only
# hosts. Actual training still requires torch_xla and fails fast with a clear
# message in main() instead of an ImportError at line 1 of the traceback.
# Kaggle TPU day-1 fix: Kaggle exports TPU_PROCESS_ADDRESSES/CLOUD_TPU_TASK_ID
# which confuse the modern PJRT runtime into "Expected 8 worker addresses,
# got 1" during multiprocess init. Scrub them BEFORE torch_xla import.
# Harmless elsewhere: the variables simply do not exist on TPU-VM/GCE.
for _kaggle_var in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID",
                    "TPU_MESH_CONTROLLER_ADDRESS", "TPU_MESH_CONTROLLER_PORT"):
    os.environ.pop(_kaggle_var, None)
os.environ.setdefault("PJRT_DEVICE", "TPU")

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:  # pragma: no cover - CPU-only host
    xm = None
    pl = None
    xmp = None
try:
    import torch_xla.runtime as xr  # modern PJRT world/ordinal API
except Exception:  # pragma: no cover
    xr = None
try:
    import torch_xla.debug.metrics as xla_metrics
except Exception:  # pragma: no cover - version dependent
    xla_metrics = None

# Allow `python src/train_xla.py` from the project root or from src/.
# B2-adjacent fix (external review): imports below are `from src.model ...`,
# so the REPO ROOT (parent of src/) must be importable - not src/ itself.
# The old code inserted src/ only; the mistake was masked by the (now lazy)
# torch_xla import failing first on CPU hosts.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model import GPT, GPTConfig, RMSNorm  # noqa: E402
from src.data import FixedDataset, fixed_token_collate, worker_init_fn  # noqa: E402
from src.eval import evaluate_induction, measure_attention_entropy, phase_statistics  # noqa: E402

try:  # Optional but recommended: gate/mix/angle statistics for HLA runs.
    from src.eval import hla_statistics  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    hla_statistics = None

try:  # Optional: spectral (SVD) diagnostics of attention weights.
    from src.eval import svd_statistics  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    svd_statistics = None

try:  # Optional: cross-head QK/OV subspace interference (Transformer Circuits).
    from src.eval import head_interference_statistics  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    head_interference_statistics = None

try:  # Optional: induction-under-noise probe + learned depth/head profiles.
    from src.eval import evaluate_distractor_induction, depth_profile_statistics  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    evaluate_distractor_induction = None
    depth_profile_statistics = None

try:  # Optional: Theorem-5 gradient-norm readout (captured at svd cadence).
    from src.eval import mechanism_gradient_statistics  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    mechanism_gradient_statistics = None

try:  # Optional: Lost-in-the-Middle recall curve (H4), captured at svd cadence.
    from src.eval import positional_recall_curve  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    positional_recall_curve = None


# =============================================================================
# Process / XLA helpers
# =============================================================================

_SHOULD_STOP = False


def handle_signal(signum: int, frame: Any) -> None:  # pragma: no cover - runtime only
    global _SHOULD_STOP
    _SHOULD_STOP = True
    try:
        print(f"[Signal] received signal={signum}; will save and stop at next safe point", flush=True)
    except Exception:
        pass


def install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handle_signal)
        except Exception:
            pass


def xla_rank() -> int:
    # PJRT day-1 fix: on the modern runtime the legacy xm.get_ordinal() may
    # report 0 in EVERY spawned process ("world_size=1, rank=0" x8), which
    # breaks data sharding (all cores chew the same batch -> 8x memory, OOM)
    # and master-only saves (8 self-declared masters -> tmp-file rename race).
    # torch_xla.runtime.global_ordinal() is the authoritative source.
    if xr is not None:
        try:
            return int(xr.global_ordinal())
        except Exception:
            pass
    try:
        return int(xm.get_ordinal())
    except Exception:
        return int(os.environ.get("RANK", "0"))


def xla_local_rank() -> int:
    if xr is not None:
        try:
            return int(xr.local_ordinal())
        except Exception:
            pass
    try:
        return int(xm.get_local_ordinal())
    except Exception:
        return int(os.environ.get("LOCAL_RANK", "0"))


def xla_world_size() -> int:
    if xr is not None:
        try:
            return int(xr.world_size())
        except Exception:
            pass
    try:
        return int(xm.xrt_world_size())
    except Exception:
        try:
            return int(xm.world_size())  # type: ignore[attr-defined]
        except Exception:
            return int(os.environ.get("WORLD_SIZE", "1"))


def is_master() -> bool:
    if xr is not None:
        try:
            return int(xr.global_ordinal()) == 0
        except Exception:
            pass
    try:
        return bool(xm.is_master_ordinal(local=False))
    except Exception:
        return xla_rank() == 0


def check_world_size(config: Dict[str, Any], *, world_size: int, rank: int) -> None:
    # Day-1 postmortem guard: when the runtime misreports ordinals (legacy
    # xm.get_ordinal() on PJRT returned rank=0/world=1 in EVERY process), the
    # failure is SILENT - 8 processes chew identical batches, "Tokens/update"
    # is 8x smaller than reality, checkpoint saves race, and every downstream
    # number is garbage. A config asking for N cores that observes a smaller
    # world must die loudly before any training step.
    expected_world = int(config.get("num_cores", 8))
    if expected_world > 1 and int(world_size) < expected_world:
        raise RuntimeError(
            f"World-size mismatch: config num_cores={expected_world} but the "
            f"XLA runtime reports world_size={world_size} (rank={rank}). "
            "This is the day-1 PJRT ordinal bug signature - data sharding and "
            "master-only saves would be silently wrong. Fix the runtime "
            "(torch_xla.runtime ordinals) or set num_cores to match."
        )


def master_print(*args: Any, **kwargs: Any) -> None:
    if is_master():
        print(*args, **kwargs, flush=True)


def rendezvous(tag: str) -> None:
    xm.rendezvous(tag)


def mark_step() -> None:
    xm.mark_step()


def make_mp_device_loader(loader: DataLoader, device: torch.device, config: Dict[str, Any]):
    """Create MpDeviceLoader with explicit prefetch knobs when supported.

    XLA versions differ slightly in constructor kwargs, so this function falls
    back to the minimal signature if needed. This keeps training portable while
    still enabling CPU/device prefetch on modern torch_xla.
    """
    kwargs = {
        "loader_prefetch_size": int(config.get("xla_loader_prefetch_size", 8)),
        "device_prefetch_size": int(config.get("xla_device_prefetch_size", 4)),
    }
    try:
        return pl.MpDeviceLoader(loader, device, **kwargs)
    except TypeError:
        return pl.MpDeviceLoader(loader, device)


# =============================================================================
# General helpers
# =============================================================================


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # XLA/TPU RNG seed. No-op if torch_xla not loaded.
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    except Exception:
        pass


def free_disk_gb(path: str) -> float:
    p = path if os.path.isdir(path) else os.path.dirname(path)
    if p == "":
        p = "."
    return shutil.disk_usage(p).free / (1024 ** 3)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def fmt(x: float, spec: str = ".6f") -> str:
    try:
        x = float(x)
        return format(x, spec) if math.isfinite(x) else "nan"
    except Exception:
        return "nan"


def safe_torch_load(path: str, *, map_location: str = "cpu", weights_only: Optional[bool] = None) -> Any:
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def dataclass_or_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"Cannot convert object of type {type(obj)} to dict")


def environment_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
    try:
        import torch_xla  # noqa
        snap["torch_xla"] = getattr(torch_xla, "__version__", "unknown")
    except Exception:
        snap["torch_xla"] = "unavailable"
    for k in [
        "PJRT_DEVICE",
        "TPU_NAME",
        "XRT_TPU_CONFIG",
        "XLA_USE_BF16",
        "XLA_DOWNCAST_BF16",
        "LIBTPU_INIT_ARGS",
    ]:
        if k in os.environ:
            snap[f"env_{k}"] = os.environ[k]
    return snap


KNOWN_TOP_LEVEL_KEYS = frozenset({
    # identity / paths
    "seed", "run_name", "save_dir", "variant", "_doc",
    "train_path", "val_path", "init_ckpt", "resume_ckpt",
    # batching / schedule
    "batch_size_per_device", "eval_batch_size_per_device", "grad_accum",
    "max_steps", "lr", "min_lr", "warmup", "weight_decay", "betas",
    "adam_eps", "grad_clip",
    # cadence
    "log_every", "val_every", "val_batches", "save_every", "resume_every",
    "svd_every", "xla_metrics_every", "keep_last_checkpoints",
    # data / loader
    "expected_vocab_size", "num_workers", "dataloader_prefetch_factor",
    "allow_multiple_epochs", "validate_dataset_full",
    "xla_loader_prefetch_size", "xla_device_prefetch_size",
    # runtime
    "num_cores", "xmp_start_method", "xla_autocast", "min_free_gb",
    # guards / escape hatches
    "check_init_hla_identity", "init_strict", "fail_on_nonfinite_val",
    "allow_resume_config_mismatch", "allow_init_config_mismatch",
    "allow_unknown_config_keys",
    # model subtree (validated by GPTConfig itself, which rejects typos)
    "model",
})


def validate_config(config: Dict[str, Any]) -> None:
    # B7 fix (external review): the forget-gate cumulative sum is only correct
    # in fp32 (bf16 ulp ~0.78 near |S|~200 swallows per-token steps ~0.1; the
    # .float() cast in model.py is load-bearing). XLA_USE_BF16=1 /
    # XLA_DOWNCAST_BF16=1 physically downcast fp32 tensors on TPU, silently
    # breaking the mechanism - fail fast instead of training garbage.
    # Final hardening: a TYPO in a top-level key (warmup_steps vs warmup)
    # silently fell back to defaults - the run trained with wrong hparams and
    # nobody was told. GPTConfig already rejects unknown model.* keys (frozen
    # dataclass); this closes the same hole one level up.
    unknown = set(config) - KNOWN_TOP_LEVEL_KEYS
    if unknown and not bool(config.get("allow_unknown_config_keys", False)):
        raise KeyError(
            f"Unknown top-level config keys (typo?): {sorted(unknown)}. "
            "Set allow_unknown_config_keys=true only if intentional."
        )
    model_cfg = config.get("model", {}) or {}
    if float(model_cfg.get("forget_alpha", 0.0) or 0.0) != 0.0:
        for env_key in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
            if os.environ.get(env_key, "") not in ("", "0"):
                raise ValueError(
                    f"forget_alpha != 0 requires fp32 cumsum, but {env_key}="
                    f"{os.environ[env_key]!r} downcasts fp32 on TPU. Unset it "
                    "(use bf16 via model dtype, not global env downcast)."
                )
    required = [
        "seed",
        "save_dir",
        "train_path",
        "val_path",
        "batch_size_per_device",
        "eval_batch_size_per_device",
        "grad_accum",
        "max_steps",
        "lr",
        "min_lr",
        "warmup",
        "model",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
    for k in ["batch_size_per_device", "eval_batch_size_per_device", "grad_accum", "max_steps"]:
        if int(config[k]) < 1:
            raise ValueError(f"{k} must be >= 1")
    for k in ["log_every", "val_every", "resume_every"]:
        if k in config and int(config[k]) < 1:
            raise ValueError(f"{k} must be >= 1 when provided")
    if "save_every" in config and int(config["save_every"]) < 0:
        raise ValueError("save_every must be >= 0")
    if "block_size" not in config["model"]:
        raise KeyError("config['model']['block_size'] is required")
    if int(config["model"]["block_size"]) < 1:
        raise ValueError("model.block_size must be >= 1")
    if "vocab_size" in config["model"] and int(config["model"]["vocab_size"]) < 1:
        raise ValueError("model.vocab_size must be >= 1")
    # After existing checks:
    for path_key in ("train_path", "val_path"):
        p = config.get(path_key)
        if p is not None and not os.path.exists(p):
            raise FileNotFoundError(f"config['{path_key}'] points to non-existent file: {p}")


def _get_nested(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_resume_config_compatibility(current: Dict[str, Any], saved: Optional[Dict[str, Any]]) -> None:
    """Fail fast if a resume checkpoint is used with incompatible critical config."""
    if not saved:
        return
    critical = [
        "seed",
        "train_path",
        "val_path",
        "batch_size_per_device",
        "eval_batch_size_per_device",
        "grad_accum",
        "model.block_size",
        "model.vocab_size",
        "model.n_layer",
        "model.n_head",
        "model.n_embd",
    ]
    mismatches = []
    for k in critical:
        a = _get_nested(current, k)
        b = _get_nested(saved, k)
        if b is not None and a != b:
            mismatches.append((k, a, b))
    if mismatches and not bool(current.get("allow_resume_config_mismatch", False)):
        text = "\n".join(f"  {k}: current={a!r}, checkpoint={b!r}" for k, a, b in mismatches)
        raise ValueError(
            "Resume config mismatch in critical fields. Set "
            "allow_resume_config_mismatch=true only if you know exactly what you are doing.\n" + text
        )


def validate_init_config_compatibility(current: Dict[str, Any], saved: Optional[Dict[str, Any]]) -> None:
    """Validate that an init checkpoint is shape-compatible with the run config."""
    if not saved:
        return
    critical = [
        "model.block_size",
        "model.vocab_size",
        "model.n_layer",
        "model.n_head",
        "model.n_embd",
        # Shape/architecture toggles: an init checkpoint with a different
        # positional scheme or padded vocab is structurally incompatible;
        # catch it here with a clear message rather than in strict-load (or,
        # worse, silently when init_strict=false).
        "model.use_wpe",
        "model.use_rope",
        "model.padded_vocab_size",
        "model.fused_swiglu",
        "model.ffn_hidden_multiple_of",
    ]
    mismatches = []
    for k in critical:
        a = _get_nested(current, k)
        b = _get_nested(saved, k)
        if b is not None and a != b:
            mismatches.append((k, a, b))
    if mismatches and not bool(current.get("allow_init_config_mismatch", False)):
        text = "\n".join(f"  {k}: current={a!r}, init_ckpt={b!r}" for k, a, b in mismatches)
        raise ValueError(
            "Init checkpoint config mismatch in model shape fields. Set "
            "allow_init_config_mismatch=true only if intentional.\n" + text
        )


def write_json_atomic(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# =============================================================================
# LR / params / optimizer
# =============================================================================


def get_lr(step: int, *, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count parameters by module path, not by name substring match.

    More robust: walks module hierarchy and categorises by parent module.
    """
    groups = {
        "embedding": 0,
        "attention_base": 0,
        "mlp": 0,
        "norm": 0,
        "phase": 0,
        "gate_laplace": 0,
        "other": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        # Walk up the module hierarchy.
        parts = name.split(".")
        # Find parameter's owning module.
        module_path = ".".join(parts[:-1])
        try:
            mod = model.get_submodule(module_path)
        except Exception:
            groups["other"] += n
            continue
        # Categorise by module class or attribute name.
        if isinstance(mod, nn.Embedding):
            groups["embedding"] += n
        elif hasattr(mod, "W_phase_q") or hasattr(mod, "W_phase_k"):
            groups["phase"] += n
        elif hasattr(mod, "W_gate_k") or hasattr(mod, "W_gate_v") or hasattr(mod, "W_range_k") or hasattr(mod, "W_range_v"):
            groups["gate_laplace"] += n
        elif isinstance(mod, nn.Linear):
            # Distinguish attention projections from MLP projections.
            if any(k in name for k in ("c_attn", "c_proj")):
                groups["attention_base"] += n
            else:
                groups["mlp"] += n
        elif isinstance(mod, RMSNorm):
            groups["norm"] += n
        else:
            groups["other"] += n
    total = sum(groups.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups["total"] = total
    groups["trainable"] = trainable
    groups["hla_extra_estimate"] = groups["phase"] + groups["gate_laplace"]
    return groups


HLA_NODECAY_MARKERS = (
    "W_phase_q", "W_phase_k", "W_phase_scale",
    "W_range_k", "W_range_v", "W_range_f",
    "W_gate_k", "W_gate_v", "W_gate_sal", "W_gate_f", "W_gate_d", "W_qtemp",
    "W_layer_temp",
)


def make_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only, EXCLUDING all HLA parameters.

    Rationale: weight decay pulls parameters toward zero. For HLA parameters
    zero *is* the identity state, so decaying them applies a constant force
    against the mechanisms themselves (the 3-D W_phase tensors and the 2-D
    gate Linears would otherwise fall into the decay group). Gates/scales are
    conventionally excluded from decay for exactly this reason. This also
    keeps base-vs-HLA sterile: in base these params receive no gradient and
    AdamW skips grad-less params, so excluding them changes nothing for base.
    """
    decay: List[torch.nn.Parameter] = []
    nodecay: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(m in name for m in HLA_NODECAY_MARKERS):
            nodecay.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            nodecay.append(p)
    groups = [
        {"params": decay, "weight_decay": float(config.get("weight_decay", 0.1))},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=float(config["lr"]),
        betas=tuple(config.get("betas", (0.9, 0.95))),
        eps=float(config.get("adam_eps", 1e-8)),
    )


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def xla_reduce_gradients(optimizer: torch.optim.Optimizer) -> None:
    """All-reduce/average gradients across TPU replicas without stepping.

    torch_xla exposes this helper in normal versions. If unavailable, we fall
    back to xm.optimizer_step elsewhere; the main path should use this function
    so clipping is applied to the already-reduced gradients, matching standard
    distributed training semantics.
    """
    if not hasattr(xm, "reduce_gradients"):
        raise RuntimeError("torch_xla.core.xla_model.reduce_gradients is unavailable in this torch_xla version")
    xm.reduce_gradients(optimizer)


def local_grad_norm_and_clip(parameters: Iterator[torch.nn.Parameter], max_norm: float, device: torch.device) -> torch.Tensor:
    """Clip already-reduced local gradients and return their norm.

    Call xla_reduce_gradients(optimizer) before this. After reduction, every
    replica has the averaged gradient, so local clip == global synchronized clip.
    """
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return torch.tensor(0.0, device=device)
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if not torch.is_tensor(norm):
        norm = torch.tensor(float(norm), device=device)
    return norm.detach().to(device).float()


def optimizer_step_with_reduced_clip(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> torch.Tensor:
    """XLA optimizer step with correct ordering: reduce -> clip -> step.

    xm.optimizer_step() reduces gradients internally, which means clipping before
    it clips unreduced per-replica gradients. For paper-grade reproducibility we
    explicitly reduce first, then clip the averaged gradients, then call the raw
    optimizer step.
    """
    xla_reduce_gradients(optimizer)
    grad_norm = local_grad_norm_and_clip(model.parameters(), grad_clip, device)
    optimizer.step()
    return grad_norm


# =============================================================================
# Data loading
# =============================================================================


class EvenShardedSequentialSampler(Sampler[int]):
    """Equal-length no-padding sequential shard sampler for synchronized training.

    It uses only a prefix whose size is divisible by world_size * batch_size, so
    every rank has exactly the same number of samples and batches. This avoids
    silent duplicate-padding and avoids rank desynchronization at epoch end.
    Resume is O(1): pass start_local_sample = completed_step * grad_accum * batch_size.
    """

    def __init__(
        self,
        dataset_len: int,
        *,
        rank: int,
        world_size: int,
        batch_size: int,
        start_local_sample: int = 0,
    ):
        self.dataset_len = int(dataset_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size = int(batch_size)
        self.start_local_sample = int(start_local_sample)
        if self.world_size <= 0 or self.batch_size <= 0:
            raise ValueError("world_size and batch_size must be positive")
        self.global_batch = self.world_size * self.batch_size
        self.usable_global = self.dataset_len - (self.dataset_len % self.global_batch)
        self.local_total = self.usable_global // self.world_size
        self.local_start = min(max(0, self.start_local_sample), self.local_total)
        # Keep local length divisible by batch_size after resume offset.
        remaining = self.local_total - self.local_start
        self.local_len = remaining - (remaining % self.batch_size)

    def __iter__(self) -> Iterator[int]:
        start = self.local_start
        end = self.local_start + self.local_len
        for local_i in range(start, end):
            yield local_i * self.world_size + self.rank

    def __len__(self) -> int:
        return self.local_len

    @property
    def num_batches(self) -> int:
        return self.local_len // self.batch_size


class ShardedSequentialSampler(Sampler[int]):
    """No-padding sequential shard sampler for exact validation accounting."""

    def __init__(self, dataset_len: int, rank: int, world_size: int):
        self.dataset_len = int(dataset_len)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_len, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.dataset_len:
            return 0
        return (self.dataset_len - 1 - self.rank) // self.world_size + 1


def build_train_loader(
    *,
    path: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    num_workers: int,
    prefetch_factor: int,
    start_local_sample: int = 0,
    expected_vocab_size: Optional[int] = None,
    validate_dataset_full: bool = False,
) -> Tuple[DataLoader, EvenShardedSequentialSampler, FixedDataset]:
    del seed  # deterministic sequential order; no RNG used
    ds = FixedDataset(
        path,
        seq_len,
        expected_vocab_size=expected_vocab_size,
        validate_full=validate_dataset_full,
        cast_to_long=False,
    )
    sampler = EvenShardedSequentialSampler(
        len(ds),
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        start_local_sample=start_local_sample,
    )
    if len(sampler) == 0 and start_local_sample == 0:
        # R22 (adversarial review round 2): with dataset_len < world_size *
        # batch_size the even sampler drops EVERYTHING and training would
        # silently loop over zero batches. Fail loudly with the fix hint.
        raise ValueError(
            f"Train dataset too small for even sharding: {len(ds)} sequences "
            f"< world_size*batch_size = {world_size * batch_size}. "
            f"Reduce batch_size_per_device/num_cores or provide more data."
        )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        collate_fn=fixed_token_collate,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return loader, sampler, ds


def build_val_loader(
    *,
    path: str,
    seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int,
    prefetch_factor: int,
    expected_vocab_size: Optional[int] = None,
    validate_dataset_full: bool = False,
) -> Tuple[DataLoader, ShardedSequentialSampler, FixedDataset]:
    ds = FixedDataset(
        path,
        seq_len,
        expected_vocab_size=expected_vocab_size,
        validate_full=validate_dataset_full,
        cast_to_long=False,
    )
    sampler = ShardedSequentialSampler(len(ds), rank=rank, world_size=world_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        collate_fn=fixed_token_collate,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return loader, sampler, ds


# =============================================================================
# XLA reductions / model output
# =============================================================================


def xla_sum_scalar(value: torch.Tensor | float, device: torch.device) -> float:
    if not torch.is_tensor(value):
        value = torch.tensor(float(value), device=device)
    value = value.detach().float()
    try:
        reduced = xm.all_reduce(xm.REDUCE_SUM, value)
    except Exception:
        reduced = value
    return float(reduced.cpu().item())


def xla_mean_scalar(value: torch.Tensor | float, device: torch.device) -> float:
    return xla_sum_scalar(value, device) / max(1, xla_world_size())


def unpack_model_output(out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(out, tuple):
        if len(out) == 2:
            return out[0], out[1]
        if len(out) == 1:
            return out[0], None
    if isinstance(out, dict):
        return out["logits"], out.get("loss")
    raise TypeError(f"Unsupported model output type: {type(out)}")


def get_autocast_context(config: Dict[str, Any]):
    # By default, TRC/XLA bf16 is controlled by XLA_USE_BF16 or XLA_DOWNCAST_BF16.
    # torch.autocast('xla') is not equally stable across torch_xla versions, so it
    # is opt-in only.
    if bool(config.get("xla_autocast", False)):
        return torch.autocast("xla", dtype=torch.bfloat16)
    return nullcontext()


# =============================================================================
# Checkpointing
# =============================================================================


def _cast_state_dict_for_weights(state: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]) -> Dict[str, torch.Tensor]:
    if dtype is None:
        return state
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            out[k] = v.detach().to(dtype)
        else:
            out[k] = v
    return out


def atomic_xm_save(payload: Any, path: str, *, master_only: bool = True) -> None:
    """Atomic save on master. All ranks may call safely.

    Day-1 finding #8: the tmp name must be IDENTICAL in every process.
    xm.save(master_only=True) writes through the RUNTIME's master process,
    which on PJRT need not be the xr.global_ordinal()==0 process that later
    calls os.replace (observed: rank=0 lived on local_rank=3). With time+pid
    in the name each process computed its OWN tmp - the writer wrote to its
    tmp while rank 0 replaced a name nobody created -> FileNotFoundError
    ('...tmp.1785696336.827'). A deterministic shared name makes writer and
    replacer agree no matter which process the runtime picks.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    xm.save(payload, tmp, master_only=master_only)
    # Writer must be done before the replace (they can be different
    # processes); all ranks reach this call, so the barrier is safe.
    rendezvous("atomic_save_" + os.path.basename(path))
    if is_master():
        os.replace(tmp, path)


def save_model_weights_xla(model: torch.nn.Module, path: str, dtype: Optional[torch.dtype] = torch.bfloat16) -> None:
    state = _cast_state_dict_for_weights(model.state_dict(), dtype)
    atomic_xm_save(state, path, master_only=True)


def save_resume_checkpoint_xla(
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    best_val: float,
    config: Dict[str, Any],
    path: str,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "best_val": float(best_val),
        "config": config,
        "config_hash": config_hash(config),
        "xla_rank": xla_rank(),
        "xla_world_size": xla_world_size(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_xm_save(payload, path, master_only=True)


def cleanup_old_checkpoints(save_dir: str, run_name: str, keep_last: int) -> None:
    """Keep the last N periodic checkpoint *steps* and delete older step_* files."""
    if keep_last <= 0 or not is_master():
        return
    try:
        import re

        by_step: Dict[int, List[str]] = {}
        pattern = re.compile(r"^step_(\d+)_.+")
        for fn in os.listdir(save_dir):
            if not (fn.startswith("step_") and run_name in fn):
                continue
            m = pattern.match(fn)
            if m is None:
                continue
            step = int(m.group(1))
            by_step.setdefault(step, []).append(os.path.join(save_dir, fn))

        steps = sorted(by_step)
        for step in steps[:-keep_last]:
            for f in by_step[step]:
                try:
                    os.remove(f)
                    master_print(f"[Cleanup] removed old checkpoint {f}")
                except OSError:
                    pass
    except Exception as e:
        master_print(f"[Cleanup] failed: {e}")


# =============================================================================
# Evaluation / logging
# =============================================================================


@torch.no_grad()
def validation_loss_xla(
    *,
    model: torch.nn.Module,
    val_loader_raw: DataLoader,
    device: torch.device,
    max_batches: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Distributed token-weighted validation.

    Uses a no-padding validation sampler so the final metric is not biased by
    duplicate-padding validation samplers.
    """
    was_training = model.training
    model.eval()

    loader = make_mp_device_loader(val_loader_raw, device, config)
    local_loss_tokens = torch.tensor(0.0, device=device)
    local_tokens = torch.tensor(0.0, device=device)
    local_bad = torch.tensor(0.0, device=device)
    local_batches = torch.tensor(0.0, device=device)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"] if isinstance(batch, dict) else batch
        x = ids[:, :-1]
        y = ids[:, 1:]
        with get_autocast_context(config):
            logits, loss = unpack_model_output(model(x, y))
            if loss is None:
                raise RuntimeError("Model did not return validation loss")
            loss = loss.mean()
        finite = torch.isfinite(loss)
        n_tok = torch.tensor(float(y.numel()), device=device)
        local_loss_tokens = local_loss_tokens + torch.where(finite, loss.detach().float() * n_tok, torch.zeros_like(n_tok))
        local_tokens = local_tokens + torch.where(finite, n_tok, torch.zeros_like(n_tok))
        local_bad = local_bad + torch.where(finite, torch.zeros_like(local_bad), torch.ones_like(local_bad))
        local_batches = local_batches + 1.0
        mark_step()

    loss_tokens = xla_sum_scalar(local_loss_tokens, device)
    tokens = xla_sum_scalar(local_tokens, device)
    bad = xla_sum_scalar(local_bad, device)
    batches = xla_sum_scalar(local_batches, device)

    if was_training:
        model.train()

    if bad > 0 and bool(config.get("fail_on_nonfinite_val", True)):
        raise RuntimeError(f"Non-finite validation loss in {bad} distributed val batches")
    val_loss = loss_tokens / tokens if tokens > 0 else float("nan")
    return {"val_loss": val_loss, "val_tokens": tokens, "val_bad_batches": bad, "val_batches": batches}


@torch.no_grad()
def master_extra_metrics(
    model: torch.nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    *,
    with_svd: bool = False,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    was_training = model.training
    model.eval()
    try:
        try:
            metrics["induction"] = float(evaluate_induction(model, device=device, seed=int(config.get("seed", 42))))
        except Exception as e:
            master_print(f"[WARN] evaluate_induction failed: {e}")
            metrics["induction"] = float("nan")

        try:
            metrics["entropy"] = float(measure_attention_entropy(model, device=device, seed=int(config.get("seed", 42))))
        except Exception as e:
            master_print(f"[WARN] measure_attention_entropy failed: {e}")
            metrics["entropy"] = float("nan")

        try:
            _, phase_norm = phase_statistics(model)
            metrics["phase_norm"] = float(phase_norm)
        except Exception as e:
            master_print(f"[WARN] phase_statistics failed: {e}")
            metrics["phase_norm"] = float("nan")

        if evaluate_distractor_induction is not None:
            try:
                for k, v in evaluate_distractor_induction(
                    model, device=device, seed=int(config.get("seed", 42))
                ).items():
                    metrics[str(k)] = float(v)
            except Exception as e:
                master_print(f"[WARN] evaluate_distractor_induction failed: {e}")

        if depth_profile_statistics is not None:
            try:
                for k, v in depth_profile_statistics(model).items():
                    metrics[str(k)] = float(v)
            except Exception as e:
                master_print(f"[WARN] depth_profile_statistics failed: {e}")

        if hla_statistics is not None:
            try:
                for k, v in hla_statistics(model).items():
                    try:
                        metrics[str(k)] = float(v)
                    except Exception:
                        pass
            except Exception as e:
                master_print(f"[WARN] hla_statistics failed: {e}")

        # Spectral (SVD) diagnostics are relatively expensive (CPU SVD of
        # c_attn blocks and per-head phase matrices), so they run only when
        # explicitly requested via with_svd (see svd_every config knob).
        if with_svd and svd_statistics is not None:
            try:
                t0 = time.time()
                for k, v in svd_statistics(model).items():
                    metrics[f"svd_{k}"] = float(v)
                master_print(f"[SVD] computed in {time.time() - t0:.1f}s")
            except Exception as e:
                master_print(f"[WARN] svd_statistics failed: {e}")

        if with_svd and head_interference_statistics is not None:
            try:
                t0 = time.time()
                for k, v in head_interference_statistics(model).items():
                    metrics[str(k)] = float(v)
                master_print(f"[Interference] computed in {time.time() - t0:.1f}s")
            except Exception as e:
                master_print(f"[WARN] head_interference_statistics failed: {e}")

        if with_svd and mechanism_gradient_statistics is not None:
            try:
                mg = mechanism_gradient_statistics(model)
                for k in ("mech_grad_mean", "mech_grad_min"):
                    if k in mg:
                        metrics[k] = float(mg[k])
            except Exception as e:
                master_print(f"[WARN] mechanism_gradient_statistics failed: {e}")

        # H4 (Lost-in-the-Middle) needs a TRAINING-TIME trajectory, not just a
        # final-checkpoint number: the claim is that HLA flattens the U-curve
        # relative to the base twin, and reviewers will ask WHEN the flattening
        # emerges. Same cadence as SVD (expensive: 5 depth fractions x forward).
        if with_svd and positional_recall_curve is not None:
            try:
                t0 = time.time()
                for k, v in positional_recall_curve(
                    model, device=device, seed=int(config.get("seed", 42))
                ).items():
                    metrics[str(k)] = float(v)
                master_print(f"[LITM] computed in {time.time() - t0:.1f}s")
            except Exception as e:
                master_print(f"[WARN] positional_recall_curve failed: {e}")
    finally:
        model.train(was_training)
    return metrics


def open_csv(save_dir: str, run_name: str, config: Dict[str, Any], param_report: Dict[str, int], append: bool) -> Tuple[Any, csv.writer]:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"train_log_{run_name}.csv")
    f = open(path, "a" if append else "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not append:
        writer.writerow(["# experiment", run_name])
        writer.writerow(["# timestamp", time.strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["# config_hash", config_hash(config)])
        writer.writerow(["# variant", config.get("variant", "unknown")])
        writer.writerow(["# params_total", param_report.get("total", -1)])
        writer.writerow(["# params_hla_extra_estimate", param_report.get("hla_extra_estimate", -1)])
        writer.writerow(["# config_json", canonical_json(config)])
        writer.writerow([])
        writer.writerow([
            "step",
            "tokens_seen",
            "lr",
            "train_loss",
            "val_loss",
            "val_ppl",
            "val_tokens",
            "induction",
            "entropy",
            "phase_norm",
            "grad_norm",
            "steps_per_sec",
            "tokens_per_sec",
            "wall_time_sec",
            "eta_hours",
            "best_val",
            "angle_q_abs_mean",
            "angle_k_abs_mean",
            "gate_k_mean",
            "gate_v_mean",
            "mix_k_mean",
            "mix_v_mean",
            "svd_phase_erank",
            "svd_phase_top1",
            "svd_qk_stable_rank",
            "svd_v_stable_rank",
            "svd_gate_erank",
            "qk_interference",
            "ov_interference",
            "qk_ov_separation",
            "angle_q_sat_frac",
            "angle_k_sat_frac",
            "gate_k_sat_frac",
            "gate_v_sat_frac",
            "distractor_induction",
            "distractor_margin",
            "layer_temp_last",
            "phase_budget_mean",
            "mech_grad_mean",
            "mech_grad_min",
            "qtemp_mean",
            "qtemp_sat_frac",
            "pos_10",
            "pos_30",
            "pos_50",
            "pos_70",
            "pos_90",
            "litm_middle_drop",
            "litm_worst_frac",
        ])
    else:
        writer.writerow([])
        writer.writerow(["# resumed", time.strftime("%Y-%m-%d %H:%M:%S")])
    f.flush()
    return f, writer


# =============================================================================
# Worker
# =============================================================================


def _train_worker_fn(index: int, config: Dict[str, Any]) -> None:
    del index  # use xm ordinals instead
    install_signal_handlers()
    gc.collect()

    validate_config(config)
    base_seed = int(config["seed"])

    # Identical model initialization on all ranks.
    seed_everything(base_seed)

    device = xm.xla_device()
    rank = xla_rank()
    local_rank = xla_local_rank()
    world_size = xla_world_size()
    master = is_master()

    check_world_size(config, world_size=world_size, rank=rank)

    save_dir = str(config["save_dir"])
    run_name = str(config.get("run_name", "xla_run"))
    os.makedirs(save_dir, exist_ok=True)

    if master:
        with open(os.path.join(save_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        with open(os.path.join(save_dir, "environment.json"), "w", encoding="utf-8") as f:
            json.dump(environment_snapshot(), f, indent=2, sort_keys=True)
    rendezvous("setup_saved")

    # ------------------------- model / optimizer / load ----------------------
    model_config = GPTConfig(**config["model"])
    model = GPT(model_config)
    param_report = count_parameters(model)

    completed_step = 0
    best_val = float("inf")
    resume_ckpt = config.get("resume_ckpt")
    init_ckpt = config.get("init_ckpt")
    init_strict = bool(config.get("init_strict", True))
    loaded_resume_payload: Optional[Dict[str, Any]] = None

    if resume_ckpt:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt}")
        master_print(f"[Resume] loading {resume_ckpt}")
        loaded_resume_payload = safe_torch_load(resume_ckpt, map_location="cpu", weights_only=False)
        if "model" not in loaded_resume_payload:
            raise KeyError("resume checkpoint must contain key 'model'")
        validate_resume_config_compatibility(config, loaded_resume_payload.get("config"))
        incompat = model.load_state_dict(loaded_resume_payload["model"], strict=True)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError(f"Strict resume failed: missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}")
        completed_step = int(loaded_resume_payload.get("step", 0))
        best_val = float(loaded_resume_payload.get("best_val", float("inf")))
        master_print(f"[Resume] model loaded, step={completed_step}, best_val={best_val}")
    elif init_ckpt:
        if not os.path.exists(init_ckpt):
            raise FileNotFoundError(f"init_ckpt not found: {init_ckpt}")
        master_print(f"[Init] loading {init_ckpt}, strict={init_strict}")
        init_payload = safe_torch_load(init_ckpt, map_location="cpu", weights_only=False)
        if isinstance(init_payload, dict) and "model" in init_payload:
            validate_init_config_compatibility(config, init_payload.get("config"))
            state = init_payload["model"]
            if master and "manifest" in init_payload:
                write_json_atomic(os.path.join(save_dir, f"init_manifest_{run_name}.json"), init_payload["manifest"])
        else:
            state = init_payload
        incompat = model.load_state_dict(state, strict=init_strict)
        master_print(f"[Init] missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")
        if init_strict and (incompat.missing_keys or incompat.unexpected_keys):
            raise RuntimeError(f"Strict init failed: missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}")
        if bool(config.get("check_init_hla_identity", True)) and hasattr(model, "hla_identity_error"):
            hla_err = float(model.hla_identity_error())
            if hla_err != 0.0:
                raise RuntimeError(f"Init checkpoint violates HLA identity: max_abs={hla_err}")
            master_print("[Init] HLA identity check passed")
    else:
        master_print("[Init] fresh random initialization")

    model.to(device)

    optimizer = make_optimizer(model, config)
    if loaded_resume_payload is not None:
        if loaded_resume_payload.get("optimizer") is None:
            raise KeyError("resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(loaded_resume_payload["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        loaded_resume_payload = None  # release CPU memory

    # Rank-offset seed after identical model init/load.
    seed_everything(base_seed + rank)

    # ------------------------------- data ------------------------------------
    block_size = int(config["model"]["block_size"])
    batch_size = int(config["batch_size_per_device"])
    eval_batch_size = int(config["eval_batch_size_per_device"])
    grad_accum = int(config["grad_accum"])
    num_workers = int(config.get("num_workers", 0))
    dataloader_prefetch_factor = int(config.get("dataloader_prefetch_factor", 4))
    if dataloader_prefetch_factor < 1:
        raise ValueError("dataloader_prefetch_factor must be >= 1")

    # Resume is handled by starting the deterministic shard sampler at the exact
    # local sample offset. This avoids O(step) dataloader skipping.
    start_local_sample = completed_step * grad_accum * batch_size
    expected_vocab_size = config.get("expected_vocab_size", config["model"].get("vocab_size"))
    validate_dataset_full = bool(config.get("validate_dataset_full", False)) and master

    train_loader_raw, train_sampler, train_ds = build_train_loader(
        path=str(config["train_path"]),
        seq_len=block_size,
        batch_size=batch_size,
        seed=base_seed,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        prefetch_factor=dataloader_prefetch_factor,
        start_local_sample=start_local_sample,
        expected_vocab_size=expected_vocab_size,
        validate_dataset_full=validate_dataset_full,
    )
    val_loader_raw, _, val_ds = build_val_loader(
        path=str(config["val_path"]),
        seq_len=block_size,
        batch_size=eval_batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        prefetch_factor=dataloader_prefetch_factor,
        expected_vocab_size=expected_vocab_size,
        validate_dataset_full=validate_dataset_full,
    )
    train_loader = make_mp_device_loader(train_loader_raw, device, config)

    if master:
        try:
            write_json_atomic(os.path.join(save_dir, "train_dataset_info.json"), train_ds.info.to_dict())
            write_json_atomic(os.path.join(save_dir, "val_dataset_info.json"), val_ds.info.to_dict())
        except Exception as e:
            master_print(f"[WARN] failed to save dataset info: {e}")

    tokens_per_update = batch_size * world_size * block_size * grad_accum
    planned_tokens = int(config["max_steps"]) * tokens_per_update
    local_batches_needed = max(0, int(config["max_steps"]) - completed_step) * grad_accum

    if master:
        with open(os.path.join(save_dir, "param_report.json"), "w", encoding="utf-8") as f:
            json.dump(param_report, f, indent=2, sort_keys=True)
        master_print("=" * 88)
        master_print(f"Run: {run_name}")
        master_print(f"Rank/world: rank={rank} local_rank={local_rank} world_size={world_size} device={device}")
        master_print(f"Config hash: {config_hash(config)}")
        master_print(f"Params: {json.dumps(param_report, indent=2)}")
        master_print(f"Train sequences total: {len(train_ds):,}")
        master_print(f"Val sequences total: {len(val_ds):,}")
        master_print(f"Local train batches/epoch: {len(train_loader_raw):,}")
        master_print(f"Local batches needed remaining: {local_batches_needed:,}")
        master_print(f"Tokens/update: {tokens_per_update:,}")
        master_print(f"Planned tokens: {planned_tokens:,}")
        if len(train_loader_raw) < local_batches_needed and not bool(config.get("allow_multiple_epochs", False)):
            master_print("[WARNING] train dataset will be exhausted before max_steps; set allow_multiple_epochs=true if intended")
        master_print("=" * 88)
    rendezvous("data_ready")

    # ------------------------------- logging ---------------------------------
    csv_file = None
    writer = None
    if master:
        csv_path = os.path.join(save_dir, f"train_log_{run_name}.csv")
        append = completed_step > 0 and os.path.exists(csv_path)
        csv_file, writer = open_csv(save_dir, run_name, config, param_report, append=append)

    # ----------------------------- resume position ----------------------------
    # Data resume is O(1): build_train_loader() started the sampler at
    # completed_step * grad_accum * batch_size local samples.
    start_epoch = 0
    train_iter = iter(train_loader)
    rendezvous("resume_position_ready")

    # ------------------------------- train -----------------------------------
    max_steps = int(config["max_steps"])
    warmup = int(config.get("warmup", 0))
    base_lr = float(config["lr"])
    min_lr = float(config.get("min_lr", 0.0))
    grad_clip = float(config.get("grad_clip", 1.0))
    log_every = int(config.get("log_every", 100))
    val_every = int(config.get("val_every", log_every))
    # SVD diagnostics cadence: every N optimizer steps (0 = disabled).
    # Runs on master only, inside the eval window. ~seconds per call on CPU
    # for <=1B models; keep it a multiple of val_every.
    svd_every = int(config.get("svd_every", 0))
    resume_every = int(config.get("resume_every", max(1, log_every)))
    save_every = int(config.get("save_every", 0))
    val_batches = int(config.get("val_batches", 50))
    min_free_gb = float(config.get("min_free_gb", 5.0))
    keep_last_checkpoints = int(config.get("keep_last_checkpoints", 4))
    xla_metrics_every = int(config.get("xla_metrics_every", 0))

    model.train()
    optimizer.zero_grad(set_to_none=True)

    micro_in_update = 0
    running_micro_loss_t = torch.tensor(0.0, device=device)
    nonfinite_micro_t = torch.tensor(0.0, device=device)
    log_loss_accum_t = torch.tensor(0.0, device=device)
    log_grad_accum_t = torch.tensor(0.0, device=device)
    log_steps_accum = 0

    epoch = start_epoch
    train_start_time = time.time()
    last_log_time = train_start_time
    last_log_step = completed_step

    def save_run_state(tag: str) -> None:
        if master:
            write_json_atomic(
                os.path.join(save_dir, f"run_state_{tag}.json"),
                {
                    "run_name": run_name,
                    "step": int(completed_step),
                    "best_val": float(best_val),
                    "tokens_seen": int(completed_step * tokens_per_update),
                    "epoch": int(epoch),
                    "wall_time_sec": float(time.time() - train_start_time),
                    "config_hash": config_hash(config),
                    # Throughput accounting for budget planning: world_size
                    # proves sharding was real (day-1 bug wrote plausible
                    # run_states with 8x-wrong token counts), tokens_per_update
                    # makes tokens/sec recomputable from this file alone.
                    "world_size": int(world_size),
                    "tokens_per_update": int(tokens_per_update),
                },
            )

    def save_latest(tag: str = "latest") -> None:
        save_resume_checkpoint_xla(
            model=model,
            optimizer=optimizer,
            step=completed_step,
            best_val=best_val,
            config=config,
            path=os.path.join(save_dir, f"{tag}_{run_name}_resume.pt"),
        )
        save_run_state(tag)

    try:
        while completed_step < max_steps:
            if _SHOULD_STOP:
                master_print("[Signal] stopping at safe point before next microbatch")
                break

            try:
                batch = next(train_iter)
            except StopIteration:
                if not bool(config.get("allow_multiple_epochs", False)):
                    master_print("[Train] dataset exhausted; stopping")
                    break
                epoch += 1
                # Rebuild a fresh full-epoch sampler. Multiple epochs are not the
                # default for fixed-token paper runs, but this path is deterministic.
                train_loader_raw, train_sampler, _ = build_train_loader(
                    path=str(config["train_path"]),
                    seq_len=block_size,
                    batch_size=batch_size,
                    seed=base_seed + epoch,
                    rank=rank,
                    world_size=world_size,
                    num_workers=num_workers,
                    prefetch_factor=dataloader_prefetch_factor,
                    start_local_sample=0,
                    expected_vocab_size=expected_vocab_size,
                    validate_dataset_full=False,
                )
                train_loader = make_mp_device_loader(train_loader_raw, device, config)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            lr = get_lr(completed_step, warmup=warmup, max_steps=max_steps, base_lr=base_lr, min_lr=min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            ids = batch["input_ids"] if isinstance(batch, dict) else batch
            x = ids[:, :-1]
            y = ids[:, 1:]

            with get_autocast_context(config):
                logits, loss = unpack_model_output(model(x, y))
                if loss is None:
                    raise RuntimeError("Model did not return train loss")
                loss = loss.mean()

            finite_loss = torch.isfinite(loss)
            nonfinite_micro_t = nonfinite_micro_t + torch.where(
                finite_loss, torch.zeros_like(nonfinite_micro_t), torch.ones_like(nonfinite_micro_t)
            )
            safe_loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            running_micro_loss_t = running_micro_loss_t + safe_loss.detach().float()
            (safe_loss / grad_accum).backward()
            micro_in_update += 1
            mark_step()

            if micro_in_update < grad_accum:
                continue

            bad_loss_count = xla_sum_scalar(nonfinite_micro_t, device)
            if bad_loss_count > 0:
                raise RuntimeError(
                    f"Non-finite train loss detected before optimizer step={completed_step}: "
                    f"bad_microbatches_across_replicas={bad_loss_count}"
                )
            nonfinite_micro_t = torch.tensor(0.0, device=device)

            # Theorem-5 evidence: snapshot mechanism grad norms at svd cadence
            # (post-reduce+clip, pre-zero). Device tensors only - no host sync.
            _next = completed_step + 1
            want_grad_capture = svd_every > 0 and (
                _next % svd_every == 0 or _next >= max_steps
            )
            grad_norm_t = optimizer_step_with_reduced_clip(
                model=model, optimizer=optimizer, grad_clip=grad_clip, device=device
            )
            if want_grad_capture and hasattr(model, "capture_mechanism_grad_norms"):
                model.capture_mechanism_grad_norms()
            optimizer.zero_grad(set_to_none=True)
            mark_step()

            completed_step += 1
            update_loss_local_t = running_micro_loss_t / float(grad_accum)
            running_micro_loss_t = torch.tensor(0.0, device=device)
            micro_in_update = 0

            # Keep training metrics on device and reduce only at logging cadence.
            # This avoids a host sync every optimizer step.
            log_loss_accum_t = log_loss_accum_t + update_loss_local_t.detach().float()
            log_grad_accum_t = log_grad_accum_t + grad_norm_t.detach().float()
            log_steps_accum += 1

            should_log = completed_step == 1 or completed_step % log_every == 0 or completed_step >= max_steps
            should_eval = completed_step == 1 or completed_step % val_every == 0 or completed_step >= max_steps
            should_save_latest = completed_step % resume_every == 0
            should_save_periodic = save_every > 0 and completed_step % save_every == 0

            if should_log or should_eval or should_save_latest or should_save_periodic:
                rendezvous("pre_log_eval_save")

            if should_log or should_eval:
                steps_in_window = max(1, log_steps_accum)
                smooth_loss = xla_mean_scalar(log_loss_accum_t / float(steps_in_window), device)
                smooth_grad = xla_mean_scalar(log_grad_accum_t / float(steps_in_window), device)
                log_loss_accum_t = torch.tensor(0.0, device=device)
                log_grad_accum_t = torch.tensor(0.0, device=device)
                log_steps_accum = 0

                now = time.time()
                elapsed = now - last_log_time
                steps_since = max(1, completed_step - last_log_step)
                steps_per_sec = steps_since / max(elapsed, 1e-9)
                tokens_per_sec = steps_per_sec * tokens_per_update
                wall = now - train_start_time
                remaining_steps_for_eta = max(0, max_steps - completed_step)
                eta_hours = (remaining_steps_for_eta / max(steps_per_sec, 1e-12)) / 3600.0
                last_log_time = now
                last_log_step = completed_step

                val_loss = float("nan")
                val_tokens = 0.0
                val_ppl = float("nan")
                metrics: Dict[str, float] = {"induction": float("nan"), "entropy": float("nan"), "phase_norm": float("nan")}

                if should_eval:
                    val = validation_loss_xla(
                        model=model,
                        val_loader_raw=val_loader_raw,
                        device=device,
                        max_batches=val_batches,
                        config=config,
                    )
                    val_loss = float(val["val_loss"])
                    val_tokens = float(val["val_tokens"])
                    val_ppl = math.exp(val_loss) if math.isfinite(val_loss) and val_loss < 20 else float("inf")
                    rendezvous("post_val_loss")
                    if master:
                        want_svd = svd_every > 0 and (
                            completed_step % svd_every == 0 or completed_step >= max_steps
                        )
                        metrics.update(
                            master_extra_metrics(model, device, config, with_svd=want_svd)
                        )
                    rendezvous("post_master_metrics")

                    # Save best immediately after eval, before latest, so latest has updated best_val.
                    if math.isfinite(val_loss) and val_loss < best_val:
                        best_val = val_loss
                        if free_disk_gb(save_dir) < min_free_gb:
                            raise RuntimeError(f"Not enough free disk for best checkpoint: {free_disk_gb(save_dir):.2f} GB")
                        rendezvous("pre_best_save")
                        save_model_weights_xla(model, os.path.join(save_dir, f"best_val_{run_name}.pt"), dtype=torch.bfloat16)
                        save_resume_checkpoint_xla(
                            model=model,
                            optimizer=optimizer,
                            step=completed_step,
                            best_val=best_val,
                            config=config,
                            path=os.path.join(save_dir, f"best_val_{run_name}_resume.pt"),
                        )
                        rendezvous("post_best_save")
                        save_run_state("best")
                        master_print(f"[Saved best] step={completed_step} val_loss={best_val:.6f}")

                if master and writer is not None:
                    tokens_seen = completed_step * tokens_per_update
                    writer.writerow([
                        completed_step,
                        tokens_seen,
                        f"{lr:.8e}",
                        fmt(smooth_loss),
                        fmt(val_loss),
                        f"{val_ppl:.3f}" if math.isfinite(val_ppl) else ("inf" if val_ppl == float("inf") else "nan"),
                        f"{val_tokens:.0f}",
                        fmt(metrics.get("induction", float("nan"))),
                        fmt(metrics.get("entropy", float("nan"))),
                        fmt(metrics.get("phase_norm", float("nan"))),
                        fmt(smooth_grad),
                        f"{steps_per_sec:.4f}",
                        f"{tokens_per_sec:.2f}",
                        f"{wall:.1f}",
                        f"{eta_hours:.3f}",
                        fmt(best_val),
                        fmt(metrics.get("angle_q_abs_mean", float("nan"))),
                        fmt(metrics.get("angle_k_abs_mean", float("nan"))),
                        fmt(metrics.get("gate_k_mean", float("nan"))),
                        fmt(metrics.get("gate_v_mean", float("nan"))),
                        fmt(metrics.get("mix_k_mean", float("nan"))),
                        fmt(metrics.get("mix_v_mean", float("nan"))),
                        fmt(metrics.get("svd_phase_erank", float("nan"))),
                        fmt(metrics.get("svd_phase_top1", float("nan"))),
                        fmt(metrics.get("svd_qk_stable_rank", float("nan"))),
                        fmt(metrics.get("svd_v_stable_rank", float("nan"))),
                        fmt(metrics.get("svd_gate_erank", float("nan"))),
                        fmt(metrics.get("qk_interference", float("nan"))),
                        fmt(metrics.get("ov_interference", float("nan"))),
                        fmt(metrics.get("qk_ov_separation", float("nan"))),
                        fmt(metrics.get("angle_q_sat_frac", float("nan"))),
                        fmt(metrics.get("angle_k_sat_frac", float("nan"))),
                        fmt(metrics.get("gate_k_sat_frac", float("nan"))),
                        fmt(metrics.get("gate_v_sat_frac", float("nan"))),
                        fmt(metrics.get("distractor_induction", float("nan"))),
                        fmt(metrics.get("distractor_margin", float("nan"))),
                        fmt(metrics.get("layer_temp_last", float("nan"))),
                        fmt(metrics.get("phase_budget_mean", float("nan"))),
                        fmt(metrics.get("mech_grad_mean", float("nan"))),
                        fmt(metrics.get("mech_grad_min", float("nan"))),
                        fmt(metrics.get("qtemp_mean", float("nan"))),
                        fmt(metrics.get("qtemp_sat_frac", float("nan"))),
                        fmt(metrics.get("pos_10", float("nan"))),
                        fmt(metrics.get("pos_30", float("nan"))),
                        fmt(metrics.get("pos_50", float("nan"))),
                        fmt(metrics.get("pos_70", float("nan"))),
                        fmt(metrics.get("pos_90", float("nan"))),
                        fmt(metrics.get("litm_middle_drop", float("nan"))),
                        fmt(metrics.get("litm_worst_frac", float("nan"))),
                    ])
                    csv_file.flush()
                    master_print(
                        f"\nStep {completed_step} | tokens_seen={tokens_seen:,} | lr={lr:.2e}\n"
                        f"  Train Loss : {smooth_loss:.6f}\n"
                        f"  Val Loss   : {val_loss:.6f}\n"
                        f"  Val PPL    : {val_ppl:.3f}\n"
                        f"  Induction  : {metrics.get('induction', float('nan')):.6f}\n"
                        f"  Entropy    : {metrics.get('entropy', float('nan')):.6f}\n"
                        f"  Phase Norm : {metrics.get('phase_norm', float('nan')):.6f}\n"
                        f"  Angle Q/K  : {metrics.get('angle_q_abs_mean', float('nan')):.6f} / {metrics.get('angle_k_abs_mean', float('nan')):.6f}\n"
                        f"  Gate K/V   : {metrics.get('gate_k_mean', float('nan')):.6f} / {metrics.get('gate_v_mean', float('nan')):.6f}\n"
                        f"  Mix K/V    : {metrics.get('mix_k_mean', float('nan')):.6f} / {metrics.get('mix_v_mean', float('nan')):.6f}\n"
                        f"  Grad Norm  : {smooth_grad:.6f}\n"
                        f"  Steps/sec  : {steps_per_sec:.4f}\n"
                        f"  Tokens/sec : {tokens_per_sec:.2f}\n"
                        f"  Best Val   : {best_val:.6f}\n"
                        f"  ETA        : {eta_hours:.2f}h\n"
                        f"  Wall time  : {wall:.1f}s"
                    )
                    if xla_metrics_every > 0 and completed_step % xla_metrics_every == 0 and xla_metrics is not None:
                        try:
                            master_print("\n[XLA metrics report]\n" + xla_metrics.metrics_report())
                        except Exception as e:
                            master_print(f"[WARN] failed to print XLA metrics: {e}")

                model.train()

            if should_save_latest:
                rendezvous("pre_latest_save")
                save_latest("latest")
                rendezvous("post_latest_save")
                master_print(f"[Saved latest resume] step={completed_step}")

            if should_save_periodic:
                if free_disk_gb(save_dir) >= min_free_gb:
                    rendezvous("pre_periodic_save")
                    save_model_weights_xla(model, os.path.join(save_dir, f"step_{completed_step}_{run_name}.pt"), dtype=torch.bfloat16)
                    save_resume_checkpoint_xla(
                        model=model,
                        optimizer=optimizer,
                        step=completed_step,
                        best_val=best_val,
                        config=config,
                        path=os.path.join(save_dir, f"step_{completed_step}_{run_name}_resume.pt"),
                    )
                    rendezvous("post_periodic_save")
                    cleanup_old_checkpoints(save_dir, run_name, keep_last=keep_last_checkpoints)
                    master_print(f"[Saved periodic] step={completed_step}")
                else:
                    master_print(f"[Skip periodic save] low disk: {free_disk_gb(save_dir):.2f} GB")

            if should_log or should_eval or should_save_latest or should_save_periodic:
                rendezvous("post_log_eval_save")

        # Final save at natural end or signal-safe stop.
        rendezvous("pre_final_save")
        if free_disk_gb(save_dir) >= min_free_gb:
            save_model_weights_xla(model, os.path.join(save_dir, f"final_{run_name}_bf16.pt"), dtype=torch.bfloat16)
            save_resume_checkpoint_xla(
                model=model,
                optimizer=optimizer,
                step=completed_step,
                best_val=best_val,
                config=config,
                path=os.path.join(save_dir, f"final_{run_name}_resume.pt"),
            )
            save_run_state("final")
            master_print(f"[Final] saved step={completed_step}, best_val={best_val:.6f}")
        else:
            master_print(f"[Final] skip save due low disk: {free_disk_gb(save_dir):.2f} GB")
        rendezvous("post_final_save")

    except Exception:
        master_print("[ERROR] Training crashed. Attempting best-effort crash save...")
        if master:
            traceback.print_exc()
        if rank != 0:
            raise  # finding #11: non-masters skip crash save (8x2.4GB killed V5 disk)
        try:
            # Finding #11 (V5 disk blow-up): ALL 8 ranks used to write a full
            # fp32 model+Adam payload (~2.4 GB each at 200m -> 19 GB per
            # crash), which killed the session's disk before anything could
            # be salvaged. The twins are data-parallel: after the last
            # completed optimizer step every rank holds IDENTICAL state, so
            # one crash file carries all information. Rank 0 writes the full
            # resume payload; other ranks write nothing (if rank 0 died
            # before the save, the resume_every checkpoint is the fallback).
            crash_path = os.path.join(
                save_dir, f"crash_rank{rank}_{run_name}_step{completed_step}_resume.pt"
            )
            # Build payload locally (no xm.save dependency).
            def _to_cpu(obj: Any) -> Any:
                # Recursively move tensors to CPU. optimizer.state_dict() contains
                # both dicts ('state') and lists ('param_groups'), so a flat
                # dict-comprehension would crash here - exactly when we least
                # want the crash-saver itself to crash.
                if torch.is_tensor(obj):
                    return obj.detach().cpu()
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_to_cpu(v) for v in obj)
                return obj

            payload = {
                "model": _to_cpu(model.state_dict()),
                "optimizer": _to_cpu(optimizer.state_dict()) if optimizer is not None else None,
                "step": int(completed_step),
                "best_val": float(best_val),
                "config": config,
                "rank": rank,
                "world_size": world_size,
            }
            os.makedirs(os.path.dirname(crash_path), exist_ok=True)
            tmp = f"{crash_path}.tmp.{os.getpid()}"
            torch.save(payload, tmp)
            os.replace(tmp, crash_path)
            master_print(f"[CrashSave] rank={rank} saved {crash_path}")
        except Exception as e:
            master_print(f"[CrashSave] rank={rank} failed: {e}")
        raise

    finally:
        if master and csv_file is not None:
            csv_file.close()
        try:
            rendezvous("trainer_done")
        except Exception:
            pass


# =============================================================================
# Public entrypoint / CLI
# =============================================================================


def train_worker_xla(config: Dict[str, Any]) -> None:
    validate_config(config)
    nprocs = int(config.get("num_cores", 8))
    start_method = str(config.get("xmp_start_method", "fork"))
    # PJRT accepts only nprocs=1 or None (None = all local devices).
    # num_cores=1 -> single-process debug; anything else -> None (8 cores).
    xmp.spawn(_train_worker_fn, args=(config,),
              nprocs=(1 if nprocs == 1 else None), start_method=start_method)


def _reject_duplicate_keys(pairs):
    """Finding #20: json.load silently keeps the LAST duplicate key, so a
    hand-edited config with an accidental second "lr" trains with the wrong
    value while every gate sees only the collapsed dict. Duplicates at ANY
    nesting level are a hard error."""
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r} in config JSON")
        d[k] = v
    return d


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=_reject_duplicate_keys)


def apply_override(config: Dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"Invalid override {item!r}; expected dotted.key=value")
    key, raw = item.split("=", 1)
    try:
        value: Any = json.loads(raw)
    except Exception:
        value = raw
    cur: Dict[str, Any] = config
    parts = key.split(".")
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="TPU/XLA trainer for base/HLA GPT runs")
    parser.add_argument("--config", required=True, type=str, help="Path to JSON config")
    parser.add_argument("--override", action="append", default=[], help="Override as dotted.key=json_value")
    args = parser.parse_args()

    if xm is None:
        raise SystemExit(
            "torch_xla is not installed - training requires a TPU host.\n"
            "Install per requirements.txt, e.g.:\n"
            "  pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 "
            "-f https://storage.googleapis.com/libtpu-releases/index.html\n"
            "(CPU-only hosts can still run tests, audits and all scripts/.)"
        )

    config = load_config(args.config)
    for item in args.override:
        apply_override(config, item)
    train_worker_xla(config)


if __name__ == "__main__":
    main()
