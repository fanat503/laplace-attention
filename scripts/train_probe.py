# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Position-conditioned linear probing of the residual stream (H4/feature axis).

THE QUESTION this answers (mech-interp reviewer, protocol Phase 2.3):
"P(B) probes and attention SNR show WHERE the model looks and WHAT it
outputs - but is mid-context information actually MORE LINEARLY ACCESSIBLE
in the residual stream of HLA than of its base twin?"

Method (closed-form, deterministic, CPU-friendly):
  1. Build N sequences with a needle pair [A][B] planted at a controlled
     DEPTH FRACTION; the query [A] sits at T-2 (same layout as the
     positional_recall_curve probe, so results are directly comparable).
  2. Run the model, capture the residual stream AFTER EVERY BLOCK at the
     query position (forward hooks; no model-code changes).
  3. For each (layer, depth): fit a RIDGE classifier (one-vs-all,
     closed-form via torch.linalg.solve - no SGD, no seeds beyond data)
     on a train split to decode WHICH B-token the needle carried
     (K classes), report held-out accuracy.
  4. Scalars per model: probe_acc_middle (depth 0.5), probe_acc_edge
     (mean of 0.1/0.9), probe_litm_gap = edge - middle (0 = flat access),
     each at the best layer, plus the full layer x depth grid.

Pre-registered reading (H4-P, registered before any trained pair existed):
  HLA flattens probe_litm_gap relative to base => mid-context information
  is genuinely more accessible, not merely more attended-to. Calibration:
  at random init accuracy ~= 1/K at every depth (verified by test).

Usage:
    python scripts/train_probe.py --checkpoint runs/.../best.pt \
        --config configs/kaggle_200m_hla_9h_s42.json --out probe.json
Works on CPU; ~1 min for a 200m checkpoint at default sizes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402

DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
N_CLASSES = 8          # which-B classification: chance = 1/8
TOK_A = 45000          # same block as positional_recall_curve needles
TOK_B0 = 46000


def build_dataset(vocab_size: int, block_size: int, n_per_class: int,
                  depth: float, seed: int):
    """Sequences with [A][B_k] at depth; label k. Query [A] at T-2."""
    g = torch.Generator().manual_seed(seed)
    T = min(512, block_size)
    n = N_CLASSES * n_per_class
    toks = torch.randint(100, min(20000, vocab_size - 1), (n, T), generator=g)
    labels = torch.arange(n) % N_CLASSES
    pos = max(1, min(int(depth * (T - 4)), T - 4))
    toks[:, pos] = TOK_A
    toks[torch.arange(n), pos + 1] = TOK_B0 + labels
    toks[:, T - 2] = TOK_A
    return toks, labels, T


@torch.no_grad()
def capture_residuals(model: GPT, tokens: torch.Tensor, q_pos: int,
                      batch: int = 16) -> List[torch.Tensor]:
    """Residual stream after every block at q_pos: list of (N, C) tensors."""
    feats: List[List[torch.Tensor]] = [[] for _ in model.transformer.h]
    hooks = []

    def mk_hook(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            feats[i].append(h[:, q_pos, :].detach().float().cpu())
        return hook

    for i, blk in enumerate(model.transformer.h):
        hooks.append(blk.register_forward_hook(mk_hook(i)))
    try:
        for s in range(0, tokens.shape[0], batch):
            model(tokens[s:s + batch])
    finally:
        for h in hooks:
            h.remove()
    return [torch.cat(f) for f in feats]


LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0)


def _ridge_fit_predict(Xtr, ytr, Xev, lam):
    """One-vs-all ridge, closed form; returns predictions on Xev."""
    mu = Xtr.mean(0, keepdim=True)
    Xtr = Xtr - mu
    Xev = Xev - mu
    Y = torch.nn.functional.one_hot(ytr, N_CLASSES).float()
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.T @ Y)
    return (Xev @ W).argmax(-1)


def ridge_probe_split(X: torch.Tensor, y: torch.Tensor,
                      split=(0.6, 0.2, 0.2), seed: int = 0):
    """Three-way split. Returns (val_acc_per_lambda, test_acc_per_lambda).

    Selection discipline: BOTH the layer and lambda are chosen on VAL;
    the reported number is TEST accuracy of that (layer, lambda) - never
    the max over anything test-derived. Fixes the selection-after-peeking
    bias of the original max-over-layers-on-heldout scheme (max of L noisy
    estimates is biased up by ~sigma*sqrt(2 ln L)).
    """
    n = X.shape[0]
    n_tr, n_va = int(n * split[0]), int(n * split[1])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    val_accs, test_accs = [], []
    for lam in LAMBDA_GRID:
        pv = _ridge_fit_predict(X[tr], y[tr], X[va], lam)
        pt = _ridge_fit_predict(X[tr], y[tr], X[te], lam)
        val_accs.append(float((pv == y[va]).float().mean()))
        test_accs.append(float((pt == y[te]).float().mean()))
    return val_accs, test_accs


def ridge_probe_accuracy(X: torch.Tensor, y: torch.Tensor,
                         train_frac: float = 0.75, lam: float = 1e-2) -> float:
    """Legacy two-way scheme (kept for the layer x depth GRID heatmap,
    where per-cell selection is not performed and the bias concern does
    not apply; scalars use ridge_probe_split)."""
    n = X.shape[0]
    n_tr = int(n * train_frac)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[:n_tr], perm[n_tr:]
    pred = _ridge_fit_predict(X[tr], y[tr], X[te], lam)
    return float((pred == y[te]).float().mean())


def run_probe(model: GPT, n_per_class: int = 24, seed: int = 42) -> Dict:
    # Finding #18: with small --n-per-class the 0.6/0.2/0.2 split can leave
    # classes with ZERO train examples (measured: npc=3 -> 1 of 8 classes
    # missing); the ridge then silently degrades and the JSON looks normal.
    # 8 per class puts >=~4 train examples per class in expectation; below
    # that the probe is statistically meaningless - refuse loudly.
    if n_per_class < 8:
        raise SystemExit(
            f"--n-per-class {n_per_class} < 8: the 60/20/20 split would "
            f"leave classes untrained (silently wrong accuracies). Use >= 8.")
    was_training = model.training
    model.eval()
    try:
        cfg = model.config
        if cfg.vocab_size < TOK_B0 + N_CLASSES or min(512, cfg.block_size) < 32:
            return {"probe_litm_gap": float("nan"), "note": "vocab/block too small"}
        grid: Dict[str, Dict[str, float]] = {}
        test_by_depth: Dict[float, float] = {}
        sel_by_depth: Dict[float, Dict[str, float]] = {}
        shuf_by_depth: Dict[float, float] = {}
        g_shuf = torch.Generator().manual_seed(seed + 1)
        for depth in DEPTHS:
            toks, labels, T = build_dataset(cfg.vocab_size, cfg.block_size,
                                            n_per_class, depth, seed)
            feats = capture_residuals(model, toks, q_pos=T - 2)
            # Grid heatmap (legacy two-way; no per-cell selection happens).
            grid[f"depth_{int(depth*100):02d}"] = {
                f"L{i:02d}": ridge_probe_accuracy(X, labels)
                for i, X in enumerate(feats)}
            # Scalars: three-way split; layer AND lambda chosen on VAL,
            # number reported from TEST (selection-bias-free).
            best_val, best_test, best_layer, best_lam = -1.0, float("nan"), -1, -1
            for i, X in enumerate(feats):
                va, te = ridge_probe_split(X, labels)
                j = max(range(len(va)), key=va.__getitem__)
                if va[j] > best_val:
                    best_val, best_test = va[j], te[j]
                    best_layer, best_lam = i, j
            test_by_depth[depth] = best_test
            sel_by_depth[depth] = {"layer": best_layer,
                                   "lambda": LAMBDA_GRID[best_lam],
                                   "val_acc": best_val}
            # Selectivity control (Hewitt & Liang): shuffled labels must
            # collapse to chance, else the probe memorizes examples.
            y_shuf = labels[torch.randperm(len(labels), generator=g_shuf)]
            Xb = feats[best_layer]
            _, te_shuf = ridge_probe_split(Xb, y_shuf)
            shuf_by_depth[depth] = te_shuf[best_lam]
        middle = test_by_depth[0.5]
        edge = 0.5 * (test_by_depth[0.1] + test_by_depth[0.9])
        selectivity = {f"{d:.1f}": test_by_depth[d] - shuf_by_depth[d]
                       for d in DEPTHS}
        return {
            "probe_acc_by_depth": {f"{d:.1f}": v for d, v in test_by_depth.items()},
            "probe_selection": {f"{d:.1f}": v for d, v in sel_by_depth.items()},
            "probe_shuffled_by_depth": {f"{d:.1f}": v for d, v in shuf_by_depth.items()},
            "probe_selectivity": selectivity,
            "probe_acc_middle": middle,
            "probe_acc_edge": edge,
            "probe_litm_gap": edge - middle,   # 0 = flat linear access
            "chance": 1.0 / N_CLASSES,
            "grid": grid,
            "n_per_class": n_per_class,
            "seed": seed,
        }
    finally:
        model.train(was_training)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Linear probe: is mid-context info linearly accessible?")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["model"]
        # FIX #32 (attack G1): analysis must be bit-exact and CPU-runnable;
    # speed configs may declare sdpa_fold/pallas. Pin manual for analysis.
    cfg = dict(cfg, attention_backend="manual")
    model = GPT(GPTConfig(**cfg)).eval()
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
             for k, v in state.items()}
    bad = [k for k, v in state.items()
           if torch.is_tensor(v) and torch.is_floating_point(v)
           and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(
            f"checkpoint {args.checkpoint} has non-finite weights "
            f"({len(bad)} tensors, first: {bad[0]}) - probe would silently "
            f"report garbage (finding #21)")
    model.load_state_dict(state, strict=True)

    res = run_probe(model, n_per_class=args.n_per_class, seed=args.seed)
    res["meta"] = {"checkpoint": args.checkpoint, "config": args.config}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    # FIX #42 (attack R2): run_probe's early-return path (e.g. the
    # "vocab/block too small" guard) yields a partial dict; formatting None
    # with :.3f crashed main AFTER the JSON was written -> rc=1 with a
    # valid artifact on disk, which an orchestration script would read as
    # total failure. Print defensively; the JSON remains the ground truth.
    def _f(x, spec="{:.3f}"):
        return spec.format(x) if isinstance(x, (int, float)) and x == x else "n/a"
    if res.get("note"):
        print(f"probe: partial result ({res['note']}) - see JSON")
    else:
        print(f"probe_acc middle={_f(res.get('probe_acc_middle'))} "
              f"edge={_f(res.get('probe_acc_edge'))} "
              f"litm_gap={_f(res.get('probe_litm_gap'), '{:+.3f}')} "
              f"(chance={_f(res.get('chance'))})")
    sel = res.get("probe_selectivity") or {}
    if sel:
        worst = min(sel.values())
        print(f"selectivity (real - shuffled), worst depth: {worst:+.3f} "
              f"{'OK' if worst > 0.1 else 'WARNING: probe may memorize'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
