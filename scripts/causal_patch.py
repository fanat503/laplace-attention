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


"""Causal patching: transplant HLA's RETRIEVAL geometry into the base twin.

THE Oral experiment (Type 5: architecture -> mechanism -> behavior). The
sterile pair guarantees base and HLA share init and data; after training they
differ only through the mechanisms' influence. This script builds a FRANKEN
model: HLA's retrieval side (Q/K projections, phases, K-gate, score biases,
q-temp) spliced onto base's transmission side (V projection, output proj,
MLPs, embeddings), then measures retrieval behavior (induction, distractor
margin, positional recall).

Causal logic, pre-registered in EXPERIMENT_CARD (H5):
  - franken ~ HLA on retrieval probes  => retrieval geometry CARRIES the gain
    (the mechanistic claim becomes causal, not correlational);
  - franken ~ base                     => the gain lives in the V/MLP path,
    and the "cleaner retrieval" story is NOT the cause - report honestly.

Transplant sets (--transplant):
  qk        : Q,K rows of every c_attn only (pure geometry, no mechanisms)
  phase     : qk + W_phase_q/k + W_phase_scale
  retrieval : phase + K-side gate/range + salience + distance + forget + qtemp
              (everything inside the softmax; V-side stays base)
  full      : every weight from HLA (sanity: franken == HLA exactly)

Note W_layer_temp is deliberately NOT transplanted in 'retrieval' (it scales
both K- and V-side envelopes - mixed allegiance; documented limitation).

Works on CPU. Ground-truth tests in tests/test_eval.py::TestCausalPatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402
from src.eval import (  # noqa: E402
    attention_needle_snr,
    evaluate_induction,
    evaluate_distractor_induction,
    positional_recall_curve,
)

RETRIEVAL_MECH_KEYS = (
    ".W_phase_q", ".W_phase_k", ".W_phase_scale",
    ".W_gate_k.weight", ".W_range_k",
    ".W_gate_sal.weight", ".W_gate_d.weight",
    ".W_gate_f.weight", ".W_range_f",
    ".W_qtemp.weight",
)
PHASE_KEYS = (".W_phase_q", ".W_phase_k", ".W_phase_scale")


def load_state(path: str) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    out = {k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
           for k, v in state.items()}
    # Finding #21: a loss-spike NaN that leaks into a saved checkpoint used
    # to flow SILENTLY through every probe (JSON full of nan presented as a
    # "result"). H5 inputs must be finite or the experiment is void.
    bad = [k for k, v in out.items()
           if torch.is_tensor(v) and torch.is_floating_point(v)
           and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(
            f"checkpoint {path} contains non-finite weights in {len(bad)} "
            f"tensors (first: {bad[0]}) - refusing to run causal patching "
            f"on a corrupted model")
    return out


def build_franken(base_state: Dict[str, torch.Tensor],
                  hla_state: Dict[str, torch.Tensor],
                  n_embd: int,
                  transplant: str,
                  donor: str = "hla") -> Dict[str, torch.Tensor]:
    """Splice the retrieval side of DONOR into the other twin's body.

    donor="hla" (forward): base body + HLA retrieval - the SUFFICIENCY test
      (does HLA's retrieval geometry carry the gain into a base body?).
    donor="base" (reverse / anti-franken): HLA body + base retrieval - the
      NECESSITY test (does removing HLA's retrieval geometry collapse the
      gain?). Pre-registered reading: forward closure >50% AND reverse
      closure <50% (gain collapses toward base) => retrieval geometry is
      both sufficient and necessary. Forward high but reverse ALSO high
      => the V/MLP path compensates - the boundary story needs revision;
      report honestly.
    """
    if set(base_state) != set(hla_state):
        raise ValueError("state dict key sets differ - not a sterile pair")
    if donor == "hla":
        body, graft = base_state, hla_state
    elif donor == "base":
        body, graft = hla_state, base_state
    else:
        raise ValueError(f"unknown donor: {donor}")
    # FIX #41 (attack P5): an unknown transplant name silently degraded to
    # qk-only splicing (the elif chain matched nothing beyond c_attn rows) -
    # an H5 caller passing a typo would get a VALID-LOOKING but wrong-arm
    # closure. The CLI is argparse-guarded; the function must be too.
    if transplant not in ("qk", "phase", "retrieval", "full"):
        raise ValueError(f"unknown transplant set: {transplant!r} "
                         f"(want qk|phase|retrieval|full)")
    if transplant == "full":
        return dict(graft)

    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in body.items()}
    for key in graft:
        if ".c_attn.weight" in key:
            # rows [0:2C] = Q,K (retrieval); rows [2C:3C] = V (transmission)
            spliced = out[key].clone()
            spliced[: 2 * n_embd] = graft[key][: 2 * n_embd]
            out[key] = spliced
        elif transplant in ("phase", "retrieval") and any(p in key for p in PHASE_KEYS) or transplant == "retrieval" and any(p in key for p in RETRIEVAL_MECH_KEYS):
            out[key] = graft[key].clone()
    return out


def probe(model: GPT, device: str = "cpu", seed: int = 42,
          batch_size: int = 8) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["induction"] = float(evaluate_induction(
        model, device=device, seed=seed, batch_size=batch_size))
    try:
        d = evaluate_distractor_induction(
            model, device=device, seed=seed, batch_size=batch_size)
        # Keys already carry the distractor_ prefix - do NOT re-prefix
        # (v1 wrote distractor_distractor_induction; regression-tested now).
        out.update({str(k): float(v) for k, v in d.items()})
    except Exception as e:
        out["distractor_error"] = str(e)  # type: ignore[assignment]
    try:
        out.update({f"posrec_{k}": float(v)
                    for k, v in positional_recall_curve(
                        model, device=device, seed=seed,
                        batch_size=max(2, batch_size // 2)).items()})
    except Exception:
        pass
    # B-extension: activation-level SNR. P(B) can move through the V-path;
    # snr_needle moves ONLY if the score geometry concentrates. A franken
    # that inherits BOTH closes the causal story on two independent levels.
    try:
        out.update({str(k): float(v) for k, v in attention_needle_snr(
            model, device=device, seed=seed,
            batch_size=max(2, batch_size // 2)).items()})
    except Exception:
        pass
    return out


def probe_multi(model: GPT, device: str = "cpu", seeds=(42, 43, 44, 45, 46),
                batch_size: int = 8) -> Dict[str, float]:
    """Probe under several seeds -> mean and std per metric.

    The H5 decision number needs an uncertainty: a gap-closure fraction
    without error bars is exactly the kind of single-number claim Reviewer 2
    strikes down. Seeds vary the synthetic probe content, not the model.
    """
    import statistics
    per_seed = [probe(model, device=device, seed=s, batch_size=batch_size)
                for s in seeds]
    keys = [k for k in per_seed[0]
            if all(isinstance(r.get(k), float) for r in per_seed)]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in per_seed if r[k] == r[k]]  # drop NaN
        if not vals:
            out[k] = float("nan")
            out[f"{k}_std"] = float("nan")
            continue
        out[k] = float(statistics.fmean(vals))
        out[f"{k}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return out


GAP_METRICS = ("induction", "distractor_induction", "distractor_margin",
               "posrec_litm_worst_frac", "snr_needle_last")
# Pre-registered (EXPERIMENT_CARD H5): >50% closure on retrieval probes =>
# retrieval geometry CAUSES the gain; <20% => story is NOT causal - report so.
MIN_MEANINGFUL_GAP = 1e-4


def gap_closure(base: Dict[str, float], hla: Dict[str, float],
                franken: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """closure = (franken - base) / (hla - base), guarded.

    Guards (each has a test):
      - |hla-base| < MIN_MEANINGFUL_GAP -> closure undefined (NaN), flagged:
        dividing by a near-zero gap manufactures arbitrary percentages;
      - propagated std via first-order bounds from per-metric stds.
    """
    out: Dict[str, Dict[str, float]] = {}
    for m in GAP_METRICS:
        if m not in base or m not in hla or m not in franken:
            continue
        gap = hla[m] - base[m]
        rec = {"base": base[m], "hla": hla[m], "franken": franken[m],
               "gap": gap}
        if not (gap == gap) or abs(gap) < MIN_MEANINGFUL_GAP:
            rec["closure"] = float("nan")
            rec["gap_too_small"] = 1.0  # #39: self-documenting numeric flag
        else:
            rec["closure"] = (franken[m] - base[m]) / gap
            noise = max(base.get(f"{m}_std", 0.0), hla.get(f"{m}_std", 0.0),
                        franken.get(f"{m}_std", 0.0))
            rec["closure_std_bound"] = 3.0 * noise / abs(gap)
            # R2-Q2 mitigation (power): the smallest gap this probe set can
            # attribute at z=3. If |gap| < mdg the closure number is inside
            # probe noise - the JSON says so explicitly instead of letting a
            # reader over-trust a percentage.
            rec["min_detectable_gap_z3"] = 3.0 * noise
            # FIX #40 (attack N): float("inf")/NaN serialize as Infinity/NaN
            # literals - the H5 artifact (a published paper artifact!) becomes
            # INVALID JSON per RFC 8259 (jq/browsers/strict parsers reject).
            # noise==0 means "measured exactly across seeds": encode as a
            # large finite sentinel + explicit flag instead of Infinity.
            if noise > 0:
                rec["gap_over_noise_z"] = abs(gap) / noise
            else:
                rec["gap_over_noise_z"] = 1e9
                rec["noise_exact_zero"] = 1.0
            rec["powered"] = 1.0 if (noise == 0.0 or abs(gap) >= 3.0 * noise) else 0.0
        out[m] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Transplant HLA retrieval geometry into the base twin")
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--hla-checkpoint", required=True)
    ap.add_argument("--hla-config", required=True,
                    help="HLA run config (franken runs with mechanisms ACTIVE)")
    ap.add_argument("--transplant", default="retrieval",
                    choices=["qk", "phase", "retrieval", "full"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe-seeds", default="42,43,44,45,46",
                    help="comma-separated probe seeds (error bars for H5)")
    ap.add_argument("--probe-batch", type=int, default=8)
    ap.add_argument("--direction", default="both",
                    choices=["forward", "reverse", "both"],
                    help="forward: base body + HLA retrieval (sufficiency); "
                         "reverse: HLA body + base retrieval (necessity)")
    args = ap.parse_args()

    # Parse seeds BEFORE loading checkpoints: on a 200m pair the loads take
    # minutes - a typo in --probe-seeds must fail in milliseconds, not after.
    try:
        seeds = tuple(int(s) for s in args.probe_seeds.split(","))
        if not seeds:
            raise ValueError("empty")
    except ValueError:
        raise SystemExit(
            f"--probe-seeds must be comma-separated ints, "
            f"got {args.probe_seeds!r}") from None

    cfg = json.load(open(args.hla_config, encoding="utf-8"))["model"]
    # FIX #32 (attack G1): H5 analysis must be bit-exact and CPU-runnable.
    # Speed configs may declare attention_backend="sdpa_fold" (different
    # reduction order, ~1e-7 drift in probe stats) or "pallas" (crashes
    # without torch_xla). Analysis semantics are backend-independent, so we
    # pin the exact manual path regardless of what the training config used.
    if cfg.get("attention_backend", "manual") != "manual":
        print(f"[causal_patch] overriding attention_backend="
              f"{cfg['attention_backend']!r} -> 'manual' (bit-exact analysis)")
        cfg["attention_backend"] = "manual"
    base_state = load_state(args.base_checkpoint)
    hla_state = load_state(args.hla_checkpoint)
    results: Dict[str, Dict[str, float]] = {}

    arms = [("base", base_state), ("hla", hla_state)]
    if args.direction in ("forward", "both"):
        arms.append(("franken", build_franken(
            base_state, hla_state, int(cfg["n_embd"]), args.transplant, donor="hla")))
    if args.direction in ("reverse", "both"):
        arms.append(("franken_rev", build_franken(
            base_state, hla_state, int(cfg["n_embd"]), args.transplant, donor="base")))

    for name, state in arms:
        model = GPT(GPTConfig(**cfg)).eval()
        model.load_state_dict(state, strict=True)
        results[name] = probe_multi(model, device=args.device, seeds=seeds,
                                    batch_size=args.probe_batch)
        print(f"{name:12s}: " + "  ".join(f"{k}={v:.5f}" for k, v in results[name].items()
                                          if isinstance(v, float) and "pos_" not in k
                                          and not k.endswith("_std")))

    if "franken" in results:
        closure = gap_closure(results["base"], results["hla"], results["franken"])
        results["gap_closure"] = closure  # type: ignore[assignment]
        print("\n=== H5 FORWARD closure (sufficiency; pre-reg: >0.50 causal, <0.20 not) ===")
        for m, rec in closure.items():
            if rec.get("gap_too_small"):
                print(f"  {m:24s}: gap={rec['gap']:+.6f} TOO SMALL to attribute (no claim)")
            else:
                print(f"  {m:24s}: closure={rec['closure']:+.3f} "
                      f"(±{rec.get('closure_std_bound', float('nan')):.3f}) "
                      f"gap={rec['gap']:+.6f}")
    if "franken_rev" in results:
        closure_r = gap_closure(results["base"], results["hla"], results["franken_rev"])
        results["gap_closure_reverse"] = closure_r  # type: ignore[assignment]
        print("\n=== H5 REVERSE closure (necessity; pre-reg: <0.50 = gain collapses) ===")
        for m, rec in closure_r.items():
            if rec.get("gap_too_small"):
                print(f"  {m:24s}: gap={rec['gap']:+.6f} TOO SMALL to attribute (no claim)")
            else:
                print(f"  {m:24s}: closure={rec['closure']:+.3f} "
                      f"(±{rec.get('closure_std_bound', float('nan')):.3f}) "
                      f"gap={rec['gap']:+.6f}")

    results["meta"] = {"transplant": args.transplant,  # type: ignore[assignment]
                       "direction": args.direction,
                       "base_checkpoint": args.base_checkpoint,
                       "hla_checkpoint": args.hla_checkpoint,
                       "probe_seeds": list(seeds),
                       "probe_batch": args.probe_batch}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        # FIX #40: allow_nan=False would crash on legit NaN probe values;
        # sanitize instead: NaN -> null (valid JSON, honest "no value").
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_clean(v) for v in o]
            if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
                return None
            return o
        json.dump(_clean(results), f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
