"""Numeric audit of HLA config values: are ranges/betas/clips/alphas sane?

Static analysis (no training): for every HLA knob computes the analytic
envelope, gradient liveness at init, clip slack, and softmax-level impact,
then flags values outside evidence-based comfort zones.

Usage:
  python scripts/audit_config_values.py --config configs/200m_hla_v2_s42.json
  python scripts/audit_config_values.py --all           # audit every config
  python scripts/audit_config_values.py --all --strict  # non-zero exit on WARN

Checks (each prints PASS / WARN / FAIL with the number that triggered it):
  E1  K-mix envelope width: max/min score multiplier within [1.5x, 12x] band
  E2  V-mix envelope narrower than K (content should move slower than salience)
  E3  Clip slack: clip must NOT bind (>=1.2x headroom above alpha*range_max)
  E4  Gradient liveness at init: d(mix)/d(gate) = beta*alpha*range in [0.02, 0.7]
  E5  Salience reach: suppression floor <= 0.25x weight, boost <= 20x
  E6  Phase budget: max rotation in [9deg, 90deg]; warn > 54deg (0.3)
  E7  Distance bias magnitude <= 1.0 nats/unit-distance at worst layer
  E8  Worst-layer combined K amplification <= 3x score scale
  E9  LR vs width: lr * n_embd within [0.25, 0.75] (muP-style heuristic band)
  E10 Warmup fraction within [1%, 10%] of max_steps
  E11 Q-temp reach: per-query temperature within [x0.2, x5] (<= ~1.6 nats)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

GREEN, YELLOW, RED, END = "\033[92m", "\033[93m", "\033[91m", "\033[0m"


def report(level: str, code: str, msg: str, counters: dict) -> None:
    color = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}[level]
    counters[level] += 1
    print(f"  {color}{level:4s}{END} {code}: {msg}")


def audit(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    m = cfg["model"]
    counters = {"PASS": 0, "WARN": 0, "FAIL": 0}
    print(f"\n=== {os.path.basename(path)} ===")

    alpha = float(m.get("laplace_alpha", 0.0))
    ldg = bool(m.get("layer_dependent_gate", False))
    lm = 2.0 if ldg else 1.0  # worst-layer multiplier (static heuristic bound)

    is_hla = alpha != 0.0 or float(m.get("phase_mult", 0.0)) != 0.0
    if not is_hla:
        print("  (base/ablated config: HLA knobs inactive, checking trainer knobs only)")

    # range_flex is a config knob since the final sweep (default 0.25 = the
    # historical +25% W_range headroom). Hardcoding 1.25 here was a silent-drift
    # bug: a config with range_flex != 0.25 would be audited against the WRONG
    # envelope while the model uses the right one.
    flex = 1.0 + float(m.get("range_flex", 0.25))

    # --- E1/E2/E3/E4: gate envelopes ---
    if is_hla and bool(m.get("use_laplace", False)):
        env = {}
        for side in ("k", "v"):
            rng = float(m[f"laplace_range_{side}"])
            beta = float(m[f"beta_{side}"])
            clip = float(m[f"{side}_log_clip"])
            pre_clip = alpha * rng * flex * lm          # |tanh|<=1, W_range at +range_flex
            eff = min(pre_clip, clip * lm)
            hi = (1 - beta) + beta * math.exp(eff)
            lo = (1 - beta) + beta * math.exp(-eff)
            env[side] = (lo, hi, pre_clip, clip * lm, beta, rng)

        lo_k, hi_k, _pre_k, _clip_k, beta_k, rng_k = env["k"]
        width_k = hi_k / lo_k
        if 1.5 <= width_k <= 12.0:
            report("PASS", "E1", f"K-mix envelope [{lo_k:.3f}, {hi_k:.3f}] width {width_k:.2f}x", counters)
        elif width_k < 1.5:
            report("WARN", "E1", f"K envelope too NARROW ({width_k:.2f}x < 1.5x): gating may be cosmetic. Raise beta_k/range_k", counters)
        else:
            report("WARN", "E1", f"K envelope very WIDE ({width_k:.2f}x > 12x): watch mix_k_mean drift + grad noise", counters)

        lo_v, hi_v, _pre_v, _clip_v, _beta_v, _rng_v = env["v"]
        if (hi_v / lo_v) < width_k:
            report("PASS", "E2", f"V envelope ({hi_v/lo_v:.2f}x) narrower than K ({width_k:.2f}x)", counters)
        else:
            report("WARN", "E2", f"V envelope ({hi_v/lo_v:.2f}x) >= K ({width_k:.2f}x): content modulated harder than salience - unusual, justify or fix", counters)

        for side, (_lo, _hi, pre, clip_eff, beta, rng) in env.items():
            slack = clip_eff / max(pre, 1e-9)
            if slack >= 1.2:
                report("PASS", "E3", f"{side.upper()}-clip slack {slack:.2f}x (clip never binds)", counters)
            elif slack >= 1.0:
                report("WARN", "E3", f"{side.upper()}-clip slack only {slack:.2f}x: clip may bind at saturation -> masks true envelope, widen clip", counters)
            else:
                report("FAIL", "E3", f"{side.upper()}-clip BINDS (slack {slack:.2f}x < 1): envelope silently truncated; clip must exceed alpha*range*(1+range_flex)*lm", counters)

            g0 = beta * alpha * rng   # d(mix)/d(raw_gate) at init
            if 0.02 <= g0 <= 0.7:
                report("PASS", "E4", f"{side.upper()}-gate grad@init {g0:.3f} (alive, stable)", counters)
            elif g0 < 0.02:
                report("WARN", "E4", f"{side.upper()}-gate grad@init {g0:.3f} < 0.02: mechanism may never wake up; raise beta/range", counters)
            else:
                report("WARN", "E4", f"{side.upper()}-gate grad@init {g0:.3f} > 0.7: early training may be jumpy; consider lower range", counters)

    # --- E5: salience ---
    if is_hla and bool(m.get("use_salience_bias", False)) and float(m.get("salience_alpha", 0.0)) != 0.0:
        s_reach = min(float(m["salience_alpha"]) * float(m["salience_range"]), float(m["salience_clip"]))
        supp, boost = math.exp(-s_reach), math.exp(s_reach)
        if supp <= 0.25 and boost <= 20.0:
            report("PASS", "E5", f"salience reach +-{s_reach:.1f} nats (x{supp:.3f} .. x{boost:.1f})", counters)
        elif supp > 0.25:
            report("WARN", "E5", f"salience too weak (min weight x{supp:.2f} > 0.25): can't truly silence distractors", counters)
        else:
            report("WARN", "E5", f"salience boost x{boost:.0f} > 20: risk of attention collapse onto boosted keys", counters)

    # --- E6: phase ---
    pm = float(m.get("phase_mult", 0.0))
    if pm != 0.0:
        max_deg = 180.0 * pm
        if bool(m.get("per_head_phase", False)):
            max_deg *= 2
        if bool(m.get("layer_dependent_phase", False)):
            max_deg *= lm
        if max_deg <= 54.0:
            report("PASS", "E6", f"max rotation {max_deg:.0f} deg (conservative)", counters)
        elif max_deg <= 90.0:
            report("WARN", "E6", f"max rotation {max_deg:.0f} deg: aggressive; monitor angle_sat_frac + induction", counters)
        else:
            report("FAIL", "E6", f"max rotation {max_deg:.0f} deg > 90: risks inverting similarities (cos flips sign)", counters)

    # --- E7: distance bias ---
    if is_hla and bool(m.get("use_distance_laplace", False)) and float(m.get("distance_laplace_alpha", 0.0)) != 0.0:
        d_reach = min(float(m["distance_laplace_alpha"]) * float(m["distance_laplace_range"]) * lm,
                      float(m["distance_laplace_clip"]) * lm)
        if d_reach <= 1.0:
            report("PASS", "E7", f"distance bias reach {d_reach:.2f} nats at max distance", counters)
        else:
            report("WARN", "E7", f"distance bias reach {d_reach:.2f} > 1.0 nats: strong positional prior, may fight RoPE", counters)

    # --- E11: Q-side temperature reach ---
    if bool(m.get("use_qtemp", False)) and float(m.get("qtemp_alpha", 0.0)) != 0.0:
        q_reach = min(float(m.get("qtemp_alpha", 0.0)) * float(m.get("qtemp_range", 0.5)),
                      float(m.get("qtemp_clip", 1.5)))
        if q_reach <= 1.61:  # ln(5): temperature within [x0.2, x5]
            report("PASS", "E11", f"q-temp reach +-{q_reach:.2f} nats (x{math.exp(-q_reach):.2f} .. x{math.exp(q_reach):.1f})", counters)
        else:
            report("WARN", "E11", f"q-temp reach +-{q_reach:.2f} > 1.61 nats: scores scaled >5x, softmax may saturate; lower qtemp_range/clip", counters)

    # --- E8: worst-layer combined K amplification ---
    if is_hla and bool(m.get("use_laplace", False)):
        beta_k, rng_k = float(m["beta_k"]), float(m["laplace_range_k"])
        k_max = (1 - beta_k) + beta_k * math.exp(min(alpha * rng_k * flex * lm, float(m["k_log_clip"]) * lm))
        if k_max <= 3.0:
            report("PASS", "E8", f"worst-layer K amplification x{k_max:.2f}", counters)
        else:
            report("WARN", "E8", f"worst-layer K amplification x{k_max:.2f} > 3: deep-layer scores may saturate softmax", counters)

    # --- E9/E10: trainer knobs. Smoke/pilot runs (<= 1000 steps) exist to
    # exercise infrastructure, not to train: their lr/warmup are deliberately
    # unrepresentative, and flagging them on every audit run teaches people
    # to ignore WARNs (alarm fatigue - reviewer attack N9). Skip with a note.
    if int(cfg.get("max_steps", 0)) <= 1000:
        print("  NOTE E9/E10 skipped: smoke/pilot profile (max_steps <= 1000)")
        return counters
    lr, C = float(cfg.get("lr", 0)), int(m.get("n_embd", 0))
    if lr and C:
        prod = lr * C
        if 0.25 <= prod <= 0.75:
            report("PASS", "E9", f"lr*n_embd = {prod:.2f} (muP-band [0.25, 0.75])", counters)
        else:
            report("WARN", "E9", f"lr*n_embd = {prod:.2f} outside [0.25, 0.75]: check against scaling heuristics", counters)

    warm, steps = int(cfg.get("warmup", 0)), int(cfg.get("max_steps", 1))
    frac = warm / max(steps, 1)
    if 0.01 <= frac <= 0.10:
        report("PASS", "E10", f"warmup {100*frac:.1f}% of steps", counters)
    else:
        report("WARN", "E10", f"warmup {100*frac:.1f}% outside [1%, 10%]", counters)

    return counters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any WARN/FAIL")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    paths = sorted(glob.glob(os.path.join(root, "*.json"))) if args.all else [args.config]
    if not paths or paths == [None]:
        raise SystemExit("provide --config PATH or --all")

    total = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for p in paths:
        c = audit(p)
        for k in total:
            total[k] += c[k]

    print(f"\nTOTAL: {GREEN}{total['PASS']} pass{END}, {YELLOW}{total['WARN']} warn{END}, {RED}{total['FAIL']} fail{END}")
    if total["FAIL"] or (args.strict and total["WARN"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
