"""Validate base/HLA experiment configs for sterile matched runs."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

ALLOWED_MAIN_DIFFS = {
    "variant",
    "run_name",
    "save_dir",
    "init_ckpt",
    "model.phase_mult",
    "model.laplace_alpha",
    "model.distance_laplace_alpha",
    "model.salience_alpha",
    "model.forget_alpha",
    "model.qtemp_alpha",
    # Identity-at-init structural toggles: with their learned params at zero
    # the forward pass is bit-identical to base (verified by tests), so they
    # are treatment switches of the same class as the alphas above.
    "model.layer_dependent_gate",
    "model.learnable_layer_temp",
    "model.per_head_phase",
    "model.layer_dependent_phase",
    "model.baseline_type",
    # Free-text documentation; never affects the run.
    "_doc",
}

# Only allowed when the pair is explicitly declared FLOPs-matched:
# HLA has slightly more compute per token, so it gets fewer steps.
FLOPS_MATCHED_EXTRA_DIFFS = {
    "max_steps",
}


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        def _no_dups(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise SystemExit(
                        f"duplicate key {k!r} in {path} - json.load would "
                        f"silently keep the LAST value (finding #20)")
                d[k] = v
            return d
        return json.load(f, object_pairs_hook=_no_dups)


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    fa, fb = flatten(a), flatten(b)
    keys = sorted(set(fa) | set(fb))
    return [(k, fa.get(k), fb.get(k)) for k in keys if fa.get(k) != fb.get(k)]


def tokens_per_update(cfg: Dict[str, Any]) -> int:
    return int(cfg["batch_size_per_device"]) * int(cfg["num_cores"]) * int(cfg["model"]["block_size"]) * int(cfg["grad_accum"])


def validate_pair(base_path: str, hla_path: str, *, flops_matched: bool = False) -> None:
    base, hla = load(base_path), load(hla_path)
    diffs = diff(base, hla)
    allowed = set(ALLOWED_MAIN_DIFFS)
    if flops_matched:
        allowed |= FLOPS_MATCHED_EXTRA_DIFFS
    illegal = [x for x in diffs if x[0] not in allowed]
    if illegal:
        text = "\n".join(f"  {k}: base={a!r}, hla={b!r}" for k, a, b in illegal)
        raise RuntimeError("Illegal base/HLA config differences:\n" + text)
    if flops_matched and any(k == "max_steps" for k, _, _ in diffs):
        bs, hs = int(base["max_steps"]), int(hla["max_steps"])
        if hs > bs:
            raise RuntimeError(
                f"FLOPs-matched pair expects HLA max_steps <= base (HLA costs more per token), "
                f"got base={bs}, hla={hs}"
            )
        print(f"[flops-matched] max_steps: base={bs:,} hla={hs:,} (hla {100*hs/bs:.1f}% of base)")

    for cfg, name in [(base, "base"), (hla, "hla")]:
        for k in ["train_path", "val_path", "init_ckpt", "save_dir", "run_name"]:
            if not cfg.get(k):
                raise RuntimeError(f"{name}: missing/empty {k}")
        if int(cfg["model"]["block_size"]) <= 0:
            raise RuntimeError(f"{name}: invalid block_size")
        if int(cfg["grad_accum"]) <= 0:
            raise RuntimeError(f"{name}: invalid grad_accum")

    print("CONFIG PAIR VALID")
    for label, cfg in [("base", base), ("hla", hla)]:
        tpu = tokens_per_update(cfg)
        total = tpu * int(cfg["max_steps"])
        print(f"{label}: tokens/update={tpu:,} total_tokens={total:,} max_steps={cfg['max_steps']:,}")
    print("allowed diffs:")
    for k, a, b in diffs:
        print(f"  {k}: {a!r} -> {b!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--hla", required=True)
    ap.add_argument(
        "--flops-matched",
        action="store_true",
        help="Pair is FLOPs-matched: allow max_steps to differ (HLA <= base).",
    )
    args = ap.parse_args()
    validate_pair(args.base, args.hla, flops_matched=args.flops_matched)


if __name__ == "__main__":
    main()
