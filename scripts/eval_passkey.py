# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Passkey retrieval at the FULL context window (external-anchor eval, B7).

Mitigates two review attacks at once (REVIEWER_ATTACK_NEURIPS2026.md):
  - R2-W2 "perplexity + author-made probes only, no standard benchmark":
    passkey retrieval is the field-standard long-context task (Mohtashami &
    Jaggi 2023; used by LongRoPE, YaRN, Landmark Attention evaluations);
  - R2-W5 "probes cap at T<=512 while the model sees 2048": this eval runs
    at the FULL block_size by construction.

Task (token-level formulation, no instruction-following required):
  [filler ...] [M] [k1..kL] [filler ...] [M] -> greedy-decode L tokens.
  The passkey k1..kL is a random L-token string from a reserved range; the
  marker M is a fixed reserved token. Exact-match requires the model to
  copy an arbitrary string from an arbitrary depth - pure retrieval, the
  induction-head mechanism at its hardest.

Reported per depth in {0.1,...,0.9} at T = block_size:
  passkey_acc_XX   : exact-match rate (all L tokens correct, greedy)
  passkey_token_XX : mean per-token accuracy (partial credit)
Plus scalars: passkey_acc_mean, passkey_worst_depth, passkey_middle_vs_edge.

Chance level: (1/range)^L ~ 0 - any nonzero exact-match is signal. At
random init the score must be 0.000 (calibration-tested).

Usage:
    python scripts/eval_passkey.py --checkpoint runs/.../best.pt \
        --config configs/kaggle_200m_hla_9h_s42.json --out passkey.json \
        [--n-trials 32] [--passkey-len 5] [--seed 42]
CPU-friendly: one forward per generated token; 32 trials x 9 depths on a
200m model ~ minutes.
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
from src.eval import _mask_padded_logits  # noqa: E402  (finding #19)

DEPTHS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MARKER_TOK = 48000          # reserved: outside filler and passkey ranges
PASSKEY_LO, PASSKEY_HI = 41000, 43000   # reserved range for passkey tokens
FILLER_LO, FILLER_HI = 100, 20000


@torch.no_grad()
def eval_passkey(model: GPT, *, n_trials: int = 32, passkey_len: int = 5,
                 seed: int = 42, device: str = "cpu") -> Dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        cfg = model.config
        T = int(cfg.block_size)
        if cfg.vocab_size < MARKER_TOK + 1 or T < 8 * (passkey_len + 2):
            return {"passkey_acc_mean": float("nan"),
                    "note": "vocab/block too small for passkey layout"}
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        out: Dict[str, float] = {}
        accs = {}
        for depth in DEPTHS:
            exact = 0
            token_hits = 0
            for _ in range(n_trials):
                toks = torch.randint(FILLER_LO, FILLER_HI, (T,), generator=g)
                key = torch.randint(PASSKEY_LO, PASSKEY_HI, (passkey_len,),
                                    generator=g)
                # plant [M][key] at depth; query [M] sits right before the
                # decode region at the very end of the window
                lo = 1
                hi = T - 2 * (passkey_len + 2)
                pos = max(lo, min(int(depth * hi), hi))
                toks[pos] = MARKER_TOK
                toks[pos + 1:pos + 1 + passkey_len] = key
                q = T - passkey_len - 1
                toks[q] = MARKER_TOK
                # greedy decode L tokens after the query marker
                seq = toks[:q + 1][None, :].to(device)
                ok = True
                for j in range(passkey_len):
                    logits, _ = model(seq[:, -T:])
                    nxt = int(_mask_padded_logits(model, logits[0, -1].float()).argmax())
                    token_hits += int(nxt == int(key[j]))
                    ok = ok and (nxt == int(key[j]))
                    seq = torch.cat(
                        [seq, torch.tensor([[nxt]], device=device)], dim=1)
                exact += int(ok)
            d = int(depth * 100)
            out[f"passkey_acc_{d:02d}"] = exact / n_trials
            out[f"passkey_token_{d:02d}"] = token_hits / (n_trials * passkey_len)
            accs[depth] = exact / n_trials
        vals = list(accs.values())
        out["passkey_acc_mean"] = sum(vals) / len(vals)
        out["passkey_worst_depth"] = min(vals)
        edge = 0.5 * (accs[0.1] + accs[0.9])
        mid = min(accs[d] for d in (0.4, 0.5, 0.6))
        out["passkey_middle_vs_edge"] = mid - edge   # negative = LITM sag
        out["context_length"] = float(T)
        out["passkey_len"] = float(passkey_len)
        out["n_trials"] = float(n_trials)
        return out
    finally:
        model.train(was_training)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Passkey retrieval at the full context window")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=32)
    ap.add_argument("--passkey-len", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["model"]
        # FIX #32 (attack G1): analysis must be bit-exact and CPU-runnable;
    # speed configs may declare sdpa_fold/pallas. Pin manual for analysis.
    cfg = dict(cfg, attention_backend="manual")
    model = GPT(GPTConfig(**cfg)).eval()
    try:
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=True)
    except Exception:
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v)
                 else v) for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(args.device)

    res = eval_passkey(model, n_trials=args.n_trials,
                       passkey_len=args.passkey_len, seed=args.seed,
                       device=args.device)
    res["meta"] = {"checkpoint": args.checkpoint, "config": args.config,
                   "seed": args.seed}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    acc = res.get("passkey_acc_mean")
    print(f"passkey@{int(res.get('context_length', 0))}: "
          f"mean_acc={acc if acc != acc else round(acc, 4)} "
          f"worst_depth={res.get('passkey_worst_depth')} "
          f"middle_vs_edge={res.get('passkey_middle_vs_edge')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
