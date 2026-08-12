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


"""Sample text from a trained checkpoint (nanoGPT-style qualitative check).

The quantitative story lives in the CSV metrics; this is the 30-second sanity
look at what the model actually says. Works on CPU.

Usage:
    python scripts/sample.py --checkpoint runs/200m_hla_s42/best_....pt \
        --config configs/200m_hla_s42.json --start "The meaning of life is" \
        --num-samples 3 --max-new-tokens 100
    # token-id prompt for models trained on dummy/custom data:
    python scripts/sample.py --checkpoint init.pt --start-ids 1,2,3 --greedy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402


def load_model(ckpt_path: str, config_path: str | None):
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if config_path:
        model_cfg = json.load(open(config_path, encoding="utf-8"))["model"]
    elif isinstance(payload, dict) and "config" in payload:
        model_cfg = payload["config"]["model"]
    else:
        raise SystemExit("No config: pass --config or use a checkpoint that embeds one")
    model = GPT(GPTConfig(**model_cfg))
    state = {k: v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample from a checkpoint (CPU-friendly)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="run config JSON (else read from checkpoint)")
    ap.add_argument("--start", default=None, help="text prompt (needs tiktoken)")
    ap.add_argument("--start-ids", default=None, help="comma-separated token ids (no tokenizer needed)")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.config)

    enc = None
    if args.start_ids is not None:
        ids = [int(t) for t in args.start_ids.split(",")]
    elif args.start is not None:
        try:
            import tiktoken
        except ImportError:
            raise SystemExit("text prompts need tiktoken (pip install tiktoken); "
                             "or use --start-ids for raw token ids") from None
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode_ordinary(args.start)
    else:
        ids = [0]

    x = torch.tensor(ids, dtype=torch.long)[None, ...]
    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                           top_k=args.top_k, greedy=args.greedy)
        out = y[0].tolist()
        print(f"--- sample {i + 1} ---")
        print(enc.decode(out) if enc is not None else out)


if __name__ == "__main__":
    main()
