# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Paper figures: the three plots the paper's argument stands on.

Each figure obeys one rule (Anthropic style): ONE takeaway per figure,
statable in the caption's first sentence.

  fig1_twin_divergence : val loss, base vs HLA twins (same init/data/steps)
                         + inset of the gap in std units of seed noise.
  fig2_litm_curves     : positional recall (pos_10..pos_90) at N checkpoints
                         - does HLA flatten the U while base keeps sagging?
  fig3_gap_closure     : H5 causal patching - base / franken / hla bars per
                         retrieval probe with the pre-registered 50%/20%
                         thresholds drawn as decision lines.

Inputs are the artifacts training already produces (train_log_*.csv,
causal JSON from scripts/causal_patch.py). Works on CPU.

    python scripts/make_paper_figures.py --base-log runs/b/train_log_b.csv \
        --hla-log runs/h/train_log_h.csv --causal-json runs/causal.json \
        --out-dir figures/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List


def require_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })
    return plt


BASE_COLOR, HLA_COLOR, FRANKEN_COLOR = "#606060", "#B0413E", "#3E6FB0"


def read_log(path: str) -> Dict[str, List[float]]:
    """Parse a trainer CSV into column lists, crash/resume-safe.

    Two artifact classes the 200m autoresume runs WILL produce (finding #14):
      - truncated crash rows: naive per-column zip silently SHIFTS later
        columns up one row (val_loss attributed to the wrong tokens_seen -
        a silently wrong figure, the worst failure mode);
      - resume seams: after autoresume from step S the trainer re-logs
        S+1..crash, duplicating steps. Keep-LAST semantics (post-resume rows
        supersede pre-crash rows; they are the ones a completed run stands on).
    Row-aligned dicts keep every column the same length (missing -> NaN).
    """
    header: List[str] = []
    by_step: Dict[float, Dict[str, str]] = {}
    order: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0] or row[0].startswith("#"):
                continue
            if not header:
                header = row
                continue
            rec = dict(zip(header, row, strict=False))  # short rows: missing keys
            try:
                step = float(rec.get("step", "nan"))
            except ValueError:
                step = float("nan")
            if step == step and step in by_step:
                by_step[step] = rec        # resume seam: last write wins
            else:
                by_step[step if step == step else float(len(order))] = rec
                order.append(step if step == step else float(len(order)))
    out: Dict[str, List[float]] = {k: [] for k in header}
    for s in order:
        rec = by_step[s]
        for k in header:
            try:
                out[k].append(float(rec.get(k, "nan")))
            except ValueError:
                out[k].append(float("nan"))
    return out


def fig1_twin_divergence(base_log, hla_log, out, seed_std: float = 0.0):
    plt = require_matplotlib()
    b, h = read_log(base_log), read_log(hla_log)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for log, name, color in ((b, "base", BASE_COLOR), (h, "HLA", HLA_COLOR)):
        xs = [t for t, v in zip(log["tokens_seen"], log["val_loss"], strict=True)
              if not math.isnan(v)]
        ys = [v for v in log["val_loss"] if not math.isnan(v)]
        ax.plot(xs, ys, label=name, color=color, lw=1.4)
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("val loss")
    # Finding #26: default legend goes upper-right - exactly under the inset
    # on realistic (monotone-decreasing) curves. Pin it away from the inset.
    ax.legend(loc="lower left")
    # Inset: the actual decision quantity - gap in units of seed noise.
    if seed_std > 0:
        bx = {t: v for t, v in zip(b["tokens_seen"], b["val_loss"], strict=True) if not math.isnan(v)}
        hx = {t: v for t, v in zip(h["tokens_seen"], h["val_loss"], strict=True) if not math.isnan(v)}
        common = sorted(set(bx) & set(hx))
        if common:
            ia = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
            ia.plot(common, [(bx[t] - hx[t]) / seed_std for t in common],
                    color=HLA_COLOR, lw=1.2)
            ia.axhline(0, color="k", lw=0.6)
            ia.set_title("gap / seed σ", fontsize=7)
            ia.tick_params(labelsize=6)
        else:
            # The inset is fig1's DECISION quantity - vanishing silently
            # (e.g. token grids misaligned after asymmetric resumes) would
            # ship a figure missing its argument. Refuse loudly instead.
            raise SystemExit(
                "fig1 inset: no common tokens_seen between base/HLA eval "
                "rows - token grids misaligned (asymmetric resume?). "
                "Align eval cadence or drop --seed-std.")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig2_litm_curves(base_log, hla_log, out, n_checkpoints: int = 3):
    plt = require_matplotlib()
    cols = ["pos_10", "pos_30", "pos_50", "pos_70", "pos_90"]
    depths = [10, 30, 50, 70, 90]
    b, h = read_log(base_log), read_log(hla_log)
    if "pos_10" not in b or all(math.isnan(v) for v in b["pos_10"]):
        raise SystemExit("no LITM columns in logs (need svd_every cadence rows)")
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(3.0 * n_checkpoints, 2.6),
                             sharey=True)
    for log, name, color in ((b, "base", BASE_COLOR), (h, "HLA", HLA_COLOR)):
        idx = [i for i, v in enumerate(log["pos_10"]) if not math.isnan(v)]
        picks = [idx[round(j * (len(idx) - 1) / (n_checkpoints - 1))]
                 for j in range(n_checkpoints)] if len(idx) >= n_checkpoints else idx
        for ax, i in zip(axes, picks, strict=False):  # picks may be < axes
            ax.plot(depths, [log[c][i] for c in cols], "o-", ms=3, lw=1.2,
                    label=name, color=color)
            ax.set_title(f"{log['tokens_seen'][i]:,.0f} tokens", fontsize=8)
            ax.set_xlabel("needle depth (%)")
    axes[0].set_ylabel("P(needle)")
    axes[0].legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig3_gap_closure(causal_json, out):
    """H5 decision figure. If the JSON carries the reverse (necessity) arm,
    a fourth bar joins each panel: forward franken shows SUFFICIENCY
    (does grafting HLA retrieval into a base body carry the gain?), the
    anti-franken shows NECESSITY (does removing it collapse the gain?).
    One figure, both halves of the causal claim - the pre-registered
    reading needs forward >0.5 AND reverse <0.5 together."""
    plt = require_matplotlib()
    res = json.load(open(causal_json, encoding="utf-8"))
    gc = res.get("gap_closure")
    if not gc:
        raise SystemExit("causal JSON has no gap_closure block (rerun causal_patch.py)")
    gcr = res.get("gap_closure_reverse") or {}
    metrics = [m for m, r in gc.items() if not r.get("closure_note")]
    if not metrics:
        raise SystemExit("all gaps flagged too-small - nothing to plot honestly")
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.2),
                             sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics, strict=True):
        r = gc[m]
        names = ["base", "franken", "HLA"]
        vals = [r["base"], r["franken"], r["hla"]]
        colors = [BASE_COLOR, FRANKEN_COLOR, HLA_COLOR]
        rr = gcr.get(m)
        title = f"{m}\nfwd={r['closure']:.2f}"
        if "closure_std_bound" in r:
            title += f"±{r['closure_std_bound']:.2f}"
        if rr and not rr.get("closure_note"):
            names.insert(2, "anti-frk")
            vals.insert(2, rr["franken"])
            colors.insert(2, "#8FB03E")
            title += f"  rev={rr['closure']:.2f}"
            if "closure_std_bound" in rr:
                title += f"±{rr['closure_std_bound']:.2f}"
        ax.bar(names, vals, color=colors, width=0.62)
        # Pre-registered decision lines on the base->hla span
        for frac, style in ((0.2, ":"), (0.5, "--")):
            ax.axhline(r["base"] + frac * r["gap"], color="k", ls=style, lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=6.5)
    fig.suptitle("H5 causal patching: sufficiency (franken) + necessity (anti-franken)",
                 fontsize=9, y=1.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig4_knockout_context(analysis_json, out):
    """Mechanism knockout Dloss vs context length (H2: the long-range
    mechanisms must MATTER MORE as context grows - the quantitative
    long-context evidence, one line per mechanism)."""
    plt = require_matplotlib()
    res = json.load(open(analysis_json, encoding="utf-8"))
    ko = res.get("knockout_by_context_length")
    if not ko:
        raise SystemExit("analysis JSON lacks knockout_by_context_length")
    lengths = sorted(int(k) for k in ko)
    mechs = sorted({m for L in ko.values() for m in L if m.startswith("ko_")})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for m in mechs:
        ys = [ko[str(L)].get(m, float("nan")) for L in lengths]
        ax.plot(lengths, ys, "o-", ms=3.5, lw=1.3,
                label=m.replace("ko_", "").replace("_delta", ""))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("knockout \u0394loss")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig5_mechanism_trajectories(base_log, hla_log, out):
    """Internal dynamics over training (GDN/Diff reviewer lesson: show WHEN
    mechanisms wake up, not just that they exist): gate/salience activity,
    saturation, and retrieval probes on one page, base vs HLA."""
    plt = require_matplotlib()
    panels = [
        ("Gate activity", ["gate_k_mean", "gate_v_mean"]),
        ("Mix envelope", ["mix_k_mean", "mix_v_mean"]),
        ("Saturation", ["gate_k_sat_frac", "angle_q_sat_frac"]),
        ("Retrieval probes", ["induction", "distractor_margin"]),
        ("LITM scalars", ["litm_middle_drop", "litm_worst_frac"]),
        ("Head interference", ["qk_interference", "qk_ov_separation"]),
    ]
    logs = {"base": read_log(base_log), "HLA": read_log(hla_log)}
    colors = {"base": BASE_COLOR, "HLA": HLA_COLOR}
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (title, cols) in zip(axes.flat, panels, strict=True):
        drew = False
        for name, log in logs.items():
            for i, col in enumerate(cols):
                if col not in log:
                    continue
                pairs = [(x, y) for x, y in zip(log["tokens_seen"], log[col], strict=True)
                         if not math.isnan(y)]
                if not pairs:
                    continue
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                        ["-", "--"][i % 2], color=colors[name], lw=1.2,
                        label=f"{name}:{col}", alpha=0.9)
                drew = True
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("tokens", fontsize=8)
        if drew:
            ax.legend(fontsize=5.5)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig6_per_position_loss(base_json, hla_json, out):
    """Per-position val loss, base vs HLA (FoX Fig.1 convention): the one
    plot that shows WHERE in the context the gain lives. A long-context
    mechanism must win at the BACK of the window; a uniform gain would
    instead say the mechanism is just extra capacity. Bottom panel: the
    delta (base - HLA) per position bin, positive = HLA wins there."""
    plt = require_matplotlib()
    curves = {}
    for name, path in (("base", base_json), ("HLA", hla_json)):
        res = json.load(open(path, encoding="utf-8"))
        pp = res.get("per_position_loss")
        if not pp or "error" in pp:
            raise SystemExit(f"{path} lacks per_position_loss (rerun analyze_checkpoint)")
        bins = sorted(k for k in pp if k.startswith("posloss_bin_"))
        curves[name] = [float(pp[k]) for k in bins]
    n = len(curves["base"])
    if n != len(curves["HLA"]):
        raise SystemExit("base/HLA bin counts differ - not a comparable pair")
    xs = list(range(n))
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(4.6, 4.2), sharex=True,
                                  height_ratios=[2.2, 1.0])
    for name, color in (("base", BASE_COLOR), ("HLA", HLA_COLOR)):
        ax.plot(xs, curves[name], "o-", ms=2.5, lw=1.3, label=name, color=color)
    ax.set_ylabel("val loss")
    ax.legend()
    delta = [b - h for b, h in zip(curves["base"], curves["HLA"], strict=True)]
    axd.bar(xs, delta, color=[HLA_COLOR if d > 0 else BASE_COLOR for d in delta],
            width=0.8, alpha=0.85)
    axd.axhline(0, color="k", lw=0.6)
    axd.set_xlabel("position bin (front -> back of context)")
    axd.set_ylabel("base - HLA")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig7_sink_outliers(base_json, hla_json, out):
    """Side-effect panel (Diff Transformer S3.7 / StreamingLLM / Bondarenko):
    attention-sink mass and activation outliers, base vs HLA. The claim it
    feeds: cleaner retrieval geometry should REDUCE the pathological mass
    the softmax parks on early tokens and the outlier kurtosis downstream -
    two independent literatures' symptoms, one mechanism. Honest by
    construction: bars show whatever the checkpoints say."""
    plt = require_matplotlib()
    specs = [("sink_mass_first", "attention_sink", "sink mass\n(first token)"),
             ("sink_mass_first4", "attention_sink", "sink mass\n(first 4)"),
             ("act_excess_kurtosis", "activation_outliers", "activation\nkurtosis"),
             ("act_max_over_rms", "activation_outliers", "max / RMS\nactivation")]
    data = {}
    for name, path in (("base", base_json), ("HLA", hla_json)):
        res = json.load(open(path, encoding="utf-8"))
        vals = []
        for key, block, _ in specs:
            blk = res.get(block) or {}
            v = blk.get(key)
            vals.append(float(v) if v is not None else float("nan"))
        data[name] = vals
    if all(v != v for v in data["base"]) and all(v != v for v in data["HLA"]):
        raise SystemExit("no sink/outlier stats in either JSON (rerun analyze_checkpoint)")
    fig, axes = plt.subplots(1, len(specs), figsize=(2.2 * len(specs), 2.8))
    for j, (ax, (_, _, label)) in enumerate(zip(axes, specs, strict=True)):
        for i, (name, color) in enumerate((("base", BASE_COLOR), ("HLA", HLA_COLOR))):
            v = data[name][j]
            if v == v:
                ax.bar([i], [v], color=color, width=0.6, label=name if j == 0 else None)
        ax.set_xticks([0, 1], ["base", "HLA"], fontsize=7)
        ax.set_title(label, fontsize=8)
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.suptitle("Softmax pathologies: sink mass & outliers (lower = cleaner)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig9_passkey_depth(base_json, hla_json, out):
    """Passkey retrieval at the FULL context window, base vs HLA (the
    external-anchor eval, Mohtashami & Jaggi convention). Exact-match per
    depth: chance is ~0, so any bar is signal; the middle-depth sag is the
    LITM signature on a FIELD-STANDARD task rather than an author probe."""
    plt = require_matplotlib()
    curves = {}
    for name, path in (("base", base_json), ("HLA", hla_json)):
        res = json.load(open(path, encoding="utf-8"))
        depths = sorted(int(k.rsplit("_", 1)[1]) for k in res
                        if k.startswith("passkey_acc_") and k[-1].isdigit()
                        and k != "passkey_acc_mean")
        if not depths:
            raise SystemExit(f"{path} lacks passkey_acc_XX (run eval_passkey)")
        curves[name] = (depths, [float(res[f"passkey_acc_{d:02d}"]) for d in depths],
                        res.get("context_length"))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    width = 3.4
    for i, (name, color) in enumerate((("base", BASE_COLOR), ("HLA", HLA_COLOR))):
        depths, accs, _t = curves[name]
        ax.bar([d + (i - 0.5) * width for d in depths], accs, width=width,
               color=color, label=name, alpha=0.9)
    t = curves["base"][2]
    ax.set_xlabel(f"passkey depth (%) at T={int(t) if t else '?'}")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig10_head_census(base_json, hla_json, out):
    """Induction-head census (Olsson et al. convention): per-head prefix-
    matching scores, base vs HLA on the shared-init twins. Answers the
    mech-interp reviewer directly: WHICH heads become induction heads, do
    the twins grow them in the same places, and does HLA concentrate or
    distribute the circuitry? Input: analyze_checkpoint JSONs containing
    prefix_matching per-head keys (LXX_HYY_prefix_match)."""
    plt = require_matplotlib()
    grids = {}
    for name, path in (("base", base_json), ("HLA", hla_json)):
        res = json.load(open(path, encoding="utf-8"))
        pm = res.get("prefix_matching") or {}
        cells = {k: float(v) for k, v in pm.items()
                 if "_H" in k and k.endswith("_prefix_match")}
        if not cells:
            raise SystemExit(f"{path} lacks per-head prefix_matching keys")
        L = 1 + max(int(k[1:3]) for k in cells)
        H = 1 + max(int(k.split("_H")[1][:2]) for k in cells)
        grid = [[cells.get(f"L{li:02d}_H{hi:02d}_prefix_match", float("nan"))
                 for hi in range(H)] for li in range(L)]
        grids[name] = grid
    vmax = max(v for g in grids.values() for row in g for v in row if v == v)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, (name, grid) in zip(axes, grids.items(), strict=True):
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0,
                       vmax=max(vmax, 1e-6))
        ax.set_title(f"{name}: prefix-matching per head", fontsize=9)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig8_probe_depth(base_json, hla_json, out):
    """Linear-probe accuracy vs needle depth with the two controls that make
    it publishable (Hewitt & Liang 2019): the CHANCE line and the SHUFFLED-
    label control band. The claim it feeds: if HLA flattens the LITM sag,
    mid-context information must be MORE linearly accessible in HLA's
    residual stream than base's - and the shuffled control proves the probe
    reads representation, not memorization. Inputs: train_probe.py JSONs."""
    plt = require_matplotlib()
    curves = {}
    for name, path in (("base", base_json), ("HLA", hla_json)):
        res = json.load(open(path, encoding="utf-8"))
        acc = res.get("probe_acc_by_depth")
        if not acc:
            raise SystemExit(f"{path} lacks probe_acc_by_depth (rerun train_probe)")
        depths = sorted(float(k) for k in acc)
        curves[name] = {
            "depths": depths,
            "acc": [float(acc[f"{d:g}"]) for d in depths],
            "shuf": [float(res.get("probe_shuffled_by_depth", {}).get(f"{d:g}", float("nan")))
                     for d in depths],
            "chance": float(res.get("chance", float("nan"))),
        }
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for name, color in (("base", BASE_COLOR), ("HLA", HLA_COLOR)):
        c = curves[name]
        xs = [d * 100 for d in c["depths"]]
        ax.plot(xs, c["acc"], "o-", ms=3.5, lw=1.4, color=color, label=name)
        if any(v == v for v in c["shuf"]):
            ax.plot(xs, c["shuf"], "--", lw=0.9, color=color, alpha=0.45,
                    label=f"{name} (shuffled ctrl)")
    ch = curves["base"]["chance"]
    if ch == ch:
        ax.axhline(ch, color="k", lw=0.7, ls=":", label=f"chance={ch:.3f}")
    ax.set_xlabel("needle depth (%)")
    ax.set_ylabel("probe accuracy")
    ax.legend(fontsize=6.5)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-grade figures from run artifacts")
    ap.add_argument("--base-log")
    ap.add_argument("--hla-log")
    ap.add_argument("--causal-json")
    ap.add_argument("--analysis-json")
    ap.add_argument("--base-analysis-json",
                    help="analyze_checkpoint JSON for the BASE twin (fig6/fig7)")
    ap.add_argument("--hla-analysis-json",
                    help="analyze_checkpoint JSON for the HLA twin (fig6/fig7)")
    ap.add_argument("--base-probe-json",
                    help="train_probe JSON for the BASE twin (fig8)")
    ap.add_argument("--hla-probe-json",
                    help="train_probe JSON for the HLA twin (fig8)")
    ap.add_argument("--base-passkey-json",
                    help="eval_passkey JSON for the BASE twin (fig9)")
    ap.add_argument("--hla-passkey-json",
                    help="eval_passkey JSON for the HLA twin (fig9)")
    ap.add_argument("--seed-std", type=float, default=0.0,
                    help="val-loss std across seeds (for the gap inset)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.base_log and args.hla_log:
        fig1_twin_divergence(args.base_log, args.hla_log,
                             os.path.join(args.out_dir, "fig1_twin_divergence.png"),
                             seed_std=args.seed_std)
        try:
            fig2_litm_curves(args.base_log, args.hla_log,
                             os.path.join(args.out_dir, "fig2_litm_curves.png"))
        except SystemExit as e:
            print(f"[skip fig2] {e}")
    if args.causal_json:
        try:
            fig3_gap_closure(args.causal_json,
                             os.path.join(args.out_dir, "fig3_gap_closure.png"))
        except SystemExit as e:
            print(f"[skip fig3] {e}")
    if args.base_log and args.hla_log:
        try:
            fig5_mechanism_trajectories(
                args.base_log, args.hla_log,
                os.path.join(args.out_dir, "fig5_mechanism_trajectories.png"))
        except SystemExit as e:
            print(f"[skip fig5] {e}")
    if args.analysis_json:
        try:
            fig4_knockout_context(args.analysis_json,
                                  os.path.join(args.out_dir, "fig4_knockout_context.png"))
        except (SystemExit, OSError) as e:
            print(f"[skip fig4] {e}")
    if args.base_analysis_json and args.hla_analysis_json:
        try:
            fig6_per_position_loss(args.base_analysis_json, args.hla_analysis_json,
                                   os.path.join(args.out_dir, "fig6_per_position_loss.png"))
        except (SystemExit, OSError) as e:
            print(f"[skip fig6] {e}")
        try:
            fig7_sink_outliers(args.base_analysis_json, args.hla_analysis_json,
                               os.path.join(args.out_dir, "fig7_sink_outliers.png"))
        except (SystemExit, OSError) as e:
            print(f"[skip fig7] {e}")
    if args.base_probe_json and args.hla_probe_json:
        try:
            fig8_probe_depth(args.base_probe_json, args.hla_probe_json,
                             os.path.join(args.out_dir, "fig8_probe_depth.png"))
        except (SystemExit, OSError) as e:
            print(f"[skip fig8] {e}")
    if args.base_passkey_json and args.hla_passkey_json:
        try:
            fig9_passkey_depth(args.base_passkey_json, args.hla_passkey_json,
                               os.path.join(args.out_dir, "fig9_passkey_depth.png"))
        except (SystemExit, OSError) as e:
            print(f"[skip fig9] {e}")
    if args.base_analysis_json and args.hla_analysis_json:
        try:
            fig10_head_census(args.base_analysis_json, args.hla_analysis_json,
                              os.path.join(args.out_dir, "fig10_head_census.png"))
        except (SystemExit, OSError) as e:
            print(f"[skip fig10] {e}")


if __name__ == "__main__":
    main()
