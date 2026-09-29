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


"""Micro-benchmark: fwd/bwd wall-clock for a base/HLA pair (nanoGPT bench.py analog).

Measures the REAL mechanism overhead on this host - the honest companion to
scripts/profile_flops.py (analytic MACs). Reports median step time and
tokens/sec for both configs plus the ratio. CPU by default; small shapes so
it runs anywhere (the TPU number comes from the trainer's own tokens_per_sec).

Usage:
    python scripts/bench.py --base configs/200m_base_s42.json \
        --hla configs/200m_hla_s42.json --steps 10 --seq-len 256
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402


def bench_config(path: str, *, steps: int, batch: int, seq_len: int,
                 overrides: dict) -> dict:
    cfg = json.load(open(path, encoding="utf-8"))["model"]
    cfg.update(overrides)
    model = GPT(GPTConfig(**cfg)).train()
    T = min(seq_len, cfg["block_size"])
    x = torch.randint(0, cfg["vocab_size"], (batch, T))
    times = []
    for i in range(steps + 2):  # 2 warmup
        t0 = time.perf_counter()
        _, loss = model(x, x)
        loss.backward()
        model.zero_grad(set_to_none=True)
        if i >= 2:
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"median_step_sec": med, "tokens_per_sec": batch * T / med,
            "params": model.parameter_count(),
            # FIX #33 (attack H2): the speed table must carry its own
            # provenance. Without recording WHICH backend was actually
            # benchmarked, a reviewer can claim the numbers came from the
            # wrong path. Also record steps/batch/seq for reproducibility.
            "attention_backend": cfg.get("attention_backend", "manual"),
            "config_path": os.path.abspath(path),
            "steps": steps, "batch": batch, "seq_len": T,
            "torch_version": torch.__version__}


def main() -> None:
    ap = argparse.ArgumentParser(description="fwd/bwd micro-benchmark for a config pair")
    ap.add_argument("--base", required=True)
    ap.add_argument("--hla", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=None,
                    help="override for quick runs on small hosts (keeps fairness: same for both)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    overrides = {"gradient_checkpointing": False}
    if args.n_layer:
        overrides["n_layer"] = args.n_layer

    results = {}
    for name, path in (("base", args.base), ("hla", args.hla)):
        r = bench_config(path, steps=args.steps, batch=args.batch,
                         seq_len=args.seq_len, overrides=overrides)
        results[name] = r
        print(f"{name:5s}: {r['median_step_sec'] * 1e3:8.1f} ms/step  "
              f"{r['tokens_per_sec']:10.0f} tok/s  ({r['params']:,} params)")

    ratio = results["hla"]["median_step_sec"] / results["base"]["median_step_sec"]
    results["hla_over_base_ratio"] = ratio
    print(f"\nHLA/base wall-clock ratio: {ratio:.3f}  "
          f"(analytic FLOPs ratio: see scripts/profile_flops.py)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
