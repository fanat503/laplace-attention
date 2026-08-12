"""Linear subspace interaction analysis for attention heads and MLPs.

This implements post-run mechanistic diagnostics inspired by head interference /
virtual circuit analysis:

- head write column space: column space of W_O head block;
- head read row spaces: row spaces of W_Q/W_K/W_V head blocks;
- MLP read row space: top right singular vectors of input projections;
- MLP write column space: top left singular vectors of W3.

Outputs per-layer head-head overlap matrices and summary statistics. Run on CPU;
this is offline analysis, not training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402

HLA_MARKERS = ("W_phase_q", "W_phase_k", "W_range_k", "W_range_v", "W_gate_k", "W_gate_v")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(ckpt: str, config: str | None) -> tuple[GPT, Dict[str, Any]]:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = load_json(config) if config else payload["config"]
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = GPT(GPTConfig(**cfg["model"]))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def orth_basis_columns(A: torch.Tensor, k: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Basis for column space of A [d_out, d_in]."""
    A = A.float()
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    rank = int((eps * S.max().clamp_min(eps) < S).sum().item()) if S.numel() else 0
    if k is not None:
        rank = min(rank, k)
    return U[:, :rank].contiguous()


def orth_basis_rows(A: torch.Tensor, k: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Basis for row space of A [d_out, d_in] as vectors in R^d_in."""
    A = A.float()
    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    rank = int((eps * S.max().clamp_min(eps) < S).sum().item()) if S.numel() else 0
    if k is not None:
        rank = min(rank, k)
    return Vh[:rank, :].T.contiguous()


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Normalized overlap in [0,1] for two orthonormal bases in same ambient space."""
    if A.numel() == 0 or B.numel() == 0:
        return float("nan")
    denom = float(min(A.shape[1], B.shape[1]))
    if denom <= 0:
        return float("nan")
    return float(((A.T @ B).pow(2).sum() / denom).item())


def layer_bases(block, cfg: Dict[str, Any], topk_mlp: int) -> Dict[str, Any]:
    d = cfg["model"]["n_embd"]
    H = cfg["model"]["n_head"]
    hd = d // H
    W = block.attn.c_attn.weight.detach().cpu()  # [3d, d]
    Wq = W[0:d]
    Wk = W[d : 2 * d]
    Wv = W[2 * d : 3 * d]
    Wo = block.attn.c_proj.weight.detach().cpu()  # [d, d], output rows, input concat-head cols

    head_write = []
    q_read = []
    k_read = []
    v_read = []
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        head_write.append(orth_basis_columns(Wo[:, sl]))
        q_read.append(orth_basis_rows(Wq[sl, :]))
        k_read.append(orth_basis_rows(Wk[sl, :]))
        v_read.append(orth_basis_rows(Wv[sl, :]))

    mlp = block.mlp
    if getattr(mlp, "fused", False):
        Wgh = mlp.w_gate_hidden.weight.detach().cpu()  # [2hidden, d]
        mlp_read = orth_basis_rows(Wgh, k=topk_mlp)
    else:
        W1 = mlp.w1.weight.detach().cpu()
        W2 = mlp.w2.weight.detach().cpu()
        mlp_read = orth_basis_rows(torch.cat([W1, W2], dim=0), k=topk_mlp)
    W3 = mlp.w3.weight.detach().cpu()  # [d, hidden]
    mlp_write = orth_basis_columns(W3, k=topk_mlp)

    return {
        "head_write": head_write,
        "q_read": q_read,
        "k_read": k_read,
        "v_read": v_read,
        "mlp_read": mlp_read,
        "mlp_write": mlp_write,
    }


def matrix_overlap(writes, reads) -> list[list[float]]:
    return [[subspace_overlap(w, r) for r in reads] for w in writes]


def summarize_matrix(M: list[list[float]]) -> Dict[str, float]:
    flat = [x for row in M for x in row if x == x]
    if not flat:
        return {"mean": float("nan"), "max": float("nan"), "min": float("nan")}
    vals = torch.tensor(flat, dtype=torch.float32)
    return {"mean": float(vals.mean().item()), "max": float(vals.max().item()), "min": float(vals.min().item())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk-mlp", type=int, default=64)
    args = ap.parse_args()

    model, cfg = load_model(args.checkpoint, args.config)
    layers = []
    for li, block in enumerate(model.transformer.h):
        b = layer_bases(block, cfg, topk_mlp=args.topk_mlp)
        W = b["head_write"]
        qM = matrix_overlap(W, b["q_read"])
        kM = matrix_overlap(W, b["k_read"])
        vM = matrix_overlap(W, b["v_read"])
        mlp_read = [subspace_overlap(w, b["mlp_read"]) for w in W]
        mlp_write_to_q = [subspace_overlap(b["mlp_write"], r) for r in b["q_read"]]
        layers.append(
            {
                "layer": li,
                "write_to_q_read": qM,
                "write_to_k_read": kM,
                "write_to_v_read": vM,
                "write_to_mlp_read": mlp_read,
                "mlp_write_to_q_read": mlp_write_to_q,
                "summary": {
                    "write_to_q": summarize_matrix(qM),
                    "write_to_k": summarize_matrix(kM),
                    "write_to_v": summarize_matrix(vM),
                    "write_to_mlp_mean": float(torch.tensor(mlp_read).mean().item()),
                    "mlp_write_to_q_mean": float(torch.tensor(mlp_write_to_q).mean().item()),
                },
            }
        )
    out = {"checkpoint": args.checkpoint, "topk_mlp": args.topk_mlp, "layers": layers}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
