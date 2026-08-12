# === Common helpers extracted from train_xla.py and make_init.py ===

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all random sources for paper-grade reproducibility.

    Covers Python, NumPy, PyTorch CPU, optional CUDA, and XLA/TPU.
    Idempotent. Safe to call multiple times.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    except Exception:
        pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def atomic_torch_save(payload: Any, path: str) -> None:
    """Atomically save payload (typically checkpoint dict)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> None:
    """Atomically write JSON. Requires writable target directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of config dict for reproducibility."""
    return stable_hash(config)
