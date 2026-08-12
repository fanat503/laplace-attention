"""Strictly compare base/HLA init checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HLA_MARKERS = ("W_phase_q", "W_phase_k", "W_range_k", "W_range_v", "W_gate_k", "W_gate_v")


def is_hla(k: str) -> bool:
    return any(m in k for m in HLA_MARKERS)


def load_state(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload


def load_role(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("role") if isinstance(payload, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--hla", required=True)
    ap.add_argument("--allow-shape-mismatch", action="store_true")
    args = ap.parse_args()

    # Finding #17 (mutation testing): the tensor checks are provably blind to
    # a role swap - base-as-hla and swapped args both passed, because the
    # shared-backbone tensors are bit-identical BY DESIGN. A swapped pair
    # trains the wrong variant from the right weights and the whole sterile
    # comparison silently dies. make_init stamps `role`; enforce it here.
    role_b, role_h = load_role(args.base), load_role(args.hla)
    if role_b is not None and "base" not in str(role_b):
        raise RuntimeError(
            f"--base file has role={role_b!r} (expected a base init). "
            f"Arguments swapped?")
    if role_h is not None and "hla" not in str(role_h):
        raise RuntimeError(
            f"--hla file has role={role_h!r} (expected an hla init). "
            f"Arguments swapped?")

    base = load_state(args.base)
    hla = load_state(args.hla)
    mismatched = []
    missing = []
    compared = 0
    for k, v in base.items():
        if is_hla(k):
            continue
        if k not in hla:
            missing.append(k)
            continue
        if tuple(v.shape) != tuple(hla[k].shape):
            mismatched.append((k, tuple(v.shape), tuple(hla[k].shape)))
            continue
        if not torch.equal(v.cpu(), hla[k].cpu()):
            raise RuntimeError(f"non-HLA tensor differs: {k}")
        compared += 1
    if missing and not args.allow_shape_mismatch:
        raise RuntimeError(f"missing non-HLA keys in HLA init: {missing[:16]}")
    if mismatched and not args.allow_shape_mismatch:
        raise RuntimeError(f"shape mismatches: {mismatched[:16]}")
    hla_err = 0.0
    for k, v in hla.items():
        if is_hla(k) and torch.is_tensor(v) and v.numel() > 0:
            hla_err = max(hla_err, float(v.float().abs().max().item()))
    if hla_err != 0.0:
        raise RuntimeError(f"HLA init is not identity: max_abs={hla_err}")
    print(f"INIT COMPARISON VALID: compared_non_hla_tensors={compared} hla_identity_max_abs={hla_err}")


if __name__ == "__main__":
    main()
