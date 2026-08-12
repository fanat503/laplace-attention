"""Generate paper-ready plots from HLA-v4 logs and analysis JSON files.

This script is optional and depends on matplotlib/pandas. It is intentionally not
part of TPU training requirements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise RuntimeError("Install matplotlib to use make_plots.py") from e


def read_log(path: str) -> Dict[str, List[float]]:
    header = None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
            else:
                rows.append(dict(zip(header, row, strict=False)))  # ragged rows allowed pre-crash
    if header is None:
        raise RuntimeError(f"No header in {path}")
    # Autoresume seams (finding #14): a resumed run re-logs steps from the
    # resume checkpoint. Keep-LAST per step, else every seam draws a sawtooth.
    dedup: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for r in rows:
        s = r.get("step", "")
        if s not in dedup:
            order.append(s)
        dedup[s] = r
    rows = [dedup[s] for s in order]
    out: Dict[str, List[float]] = {k: [] for k in header}
    for r in rows:
        for k in header:
            v = r.get(k, "nan")
            try:
                out[k].append(float(v))
            except Exception:
                out[k].append(float("nan"))
    return out


def plot_loss(base_log: str, hla_log: str, out: str) -> None:
    plt = require_matplotlib()
    base, hla = read_log(base_log), read_log(hla_log)
    plt.figure(figsize=(7, 4))
    plt.plot(base["tokens_seen"], base["val_loss"], label="base val")
    plt.plot(hla["tokens_seen"], hla["val_loss"], label="HLA val")
    plt.xlabel("tokens seen")
    plt.ylabel("validation loss")
    plt.legend()
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def heatmap(matrix, out: str, title: str, xlabel: str = "head", ylabel: str = "layer") -> None:
    plt = require_matplotlib()
    plt.figure(figsize=(8, 5))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_analysis(analysis_json: str, out_dir: str, prefix: str) -> None:
    with open(analysis_json, "r", encoding="utf-8") as f:
        a = json.load(f)
    if "attention_entropy_normalized" in a:
        heatmap(a["attention_entropy_normalized"], os.path.join(out_dir, f"{prefix}_entropy_norm.png"), f"{prefix} normalized attention entropy")
    if "attention_entropy" in a:
        heatmap(a["attention_entropy"], os.path.join(out_dir, f"{prefix}_entropy.png"), f"{prefix} attention entropy")

    # Gate/scale summaries by layer.
    if "layer_metrics" in a:
        plt = require_matplotlib()
        layers = [x["layer"] for x in a["layer_metrics"]]
        for key in ["gate_k", "gate_v", "mix_k", "mix_v"]:
            vals = [x[key]["mean"] for x in a["layer_metrics"]]
            plt.plot(layers, vals, label=key)
        plt.xlabel("layer")
        plt.ylabel("mean activation / scale")
        plt.legend()
        plt.tight_layout()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_gate_scale_means.png"))
        plt.close()


MECHANISM_PANELS = [
    # (title, ylabel, [columns]) - one subplot per entry, columns drawn per log.
    ("Retrieval probes", "probability / margin",
     ["induction", "distractor_induction", "distractor_margin"]),
    ("Head interference (Transformer-Circuits)", "subspace overlap",
     ["qk_interference", "ov_interference", "qk_ov_separation"]),
    ("Saturation fractions (|tanh|>0.99)", "fraction",
     ["angle_q_sat_frac", "angle_k_sat_frac", "gate_k_sat_frac",
      "gate_v_sat_frac", "qtemp_sat_frac"]),
    ("Mechanism gradient norms (Theorem 5, live)", "L2 norm",
     ["mech_grad_mean", "mech_grad_min"]),
    ("Learned profiles", "multiplier",
     ["layer_temp_last", "phase_budget_mean", "qtemp_mean"]),
    ("Spectral shape", "rank / share",
     ["svd_phase_erank", "svd_qk_stable_rank", "svd_v_stable_rank"]),
    ("Lost-in-the-Middle (H4)", "P(B) / drop",
     ["pos_10", "pos_30", "pos_50", "pos_70", "pos_90", "litm_middle_drop"]),
]


def plot_mechanism_dashboard(logs: Dict[str, Dict[str, List[float]]], out: str) -> None:
    """One figure, one panel per MECHANISM_PANELS entry: every diagnostic
    CSV column family vs tokens.

    Columns that are all-NaN in a log (mechanism inactive / cadence gated)
    are skipped silently, so the same dashboard works for base and HLA runs.
    """
    plt = require_matplotlib()
    # Grid derived from the panel list: zip() would SILENTLY drop panels
    # if the list outgrew a hard-coded grid (it did once: LITM panel #7).
    nrows = (len(MECHANISM_PANELS) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4 * nrows))
    for extra_ax in list(axes.flat)[len(MECHANISM_PANELS):]:
        extra_ax.set_visible(False)
    for ax, (title, ylabel, cols) in zip(axes.flat, MECHANISM_PANELS, strict=False):  # grid may exceed panels
        drew = False
        for name, log in logs.items():
            x = log.get("tokens_seen", [])
            for col in cols:
                ys = log.get(col, [])
                pairs = [(xi, yi) for xi, yi in zip(x, ys, strict=True) if not math.isnan(yi)]
                if not pairs:
                    continue
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                        label=f"{name}:{col}", alpha=0.85)
                drew = True
        ax.set_title(title)
        ax.set_xlabel("tokens seen")
        ax.set_ylabel(ylabel)
        if drew:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-log")
    ap.add_argument("--hla-log")
    ap.add_argument("--base-analysis")
    ap.add_argument("--hla-analysis")
    ap.add_argument("--kl-json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.base_log and args.hla_log:
        plot_loss(args.base_log, args.hla_log, os.path.join(args.out_dir, "loss_curves.png"))
        plot_mechanism_dashboard(
            {"base": read_log(args.base_log), "hla": read_log(args.hla_log)},
            os.path.join(args.out_dir, "mechanism_dashboard.png"))
    elif args.hla_log:
        plot_mechanism_dashboard({"hla": read_log(args.hla_log)},
                                 os.path.join(args.out_dir, "mechanism_dashboard.png"))
    if args.base_analysis:
        plot_analysis(args.base_analysis, args.out_dir, "base")
    if args.hla_analysis:
        plot_analysis(args.hla_analysis, args.out_dir, "hla")
    if args.kl_json:
        with open(args.kl_json, "r", encoding="utf-8") as f:
            kl = json.load(f)
        heatmap(kl["kl_base_to_hla_per_layer_head"], os.path.join(args.out_dir, "attention_kl.png"), "KL(base || HLA) attention")
    print(f"wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
