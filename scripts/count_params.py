from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402

HLA_MARKERS = ("W_phase_q", "W_phase_k", "W_range_k", "W_range_v", "W_gate_k", "W_gate_v")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = GPT(GPTConfig(**cfg["model"]))
    groups = {"hla": 0, "embedding": 0, "attention_base": 0, "mlp": 0, "norm": 0, "other": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        lname = name.lower()
        if any(m in name for m in HLA_MARKERS):
            groups["hla"] += n
        elif "wte" in lname or "wpe" in lname:
            groups["embedding"] += n
        elif ".attn." in lname or "c_attn" in lname or "c_proj" in lname:
            groups["attention_base"] += n
        elif ".mlp." in lname:
            groups["mlp"] += n
        elif "ln_" in lname or "ln_f" in lname or "norm" in lname:
            groups["norm"] += n
        else:
            groups["other"] += n
    groups["total"] = sum(groups.values())
    print(json.dumps(groups, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
