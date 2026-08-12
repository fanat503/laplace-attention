"""Post-run diagnostics for one trained checkpoint.

Computes paper/appendix metrics on a fixed validation batch subset:
  - val loss / perplexity;
  - attention entropy [layers, heads] and collapse rate;
  - induction score by distance;
  - HLA gate/scale summaries;
  - MLP sparsity;
  - attention/MLP output norms;
  - optional noisy validation degradation.

This is intentionally a post-run analysis script. It is not used in the hot
training loop, so it may enable attention capture and hooks safely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import FixedDataset  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(name: str):
    # Final polish: default "auto" - use XLA when available, else CPU. The old
    # default "xla" crashed with ModuleNotFoundError on any CPU host, which is
    # exactly where post-hoc analysis usually runs.
    if name == "auto":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            return torch.device("cpu")
    if name == "xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    return torch.device(name)


def load_checkpoint_model(ckpt_path: str, config_path: str | None, device: str) -> tuple[GPT, Dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if config_path:
        cfg = load_json(config_path)
    elif isinstance(payload, dict) and "config" in payload:
        cfg = payload["config"]
    else:
        raise ValueError("Provide --config when checkpoint has no embedded config")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = GPT(GPTConfig(**cfg["model"]))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def iter_val_batches(cfg: Dict[str, Any], *, seq_len: int, batch_size: int, max_batches: int, device: str):
    # Clamp to the dataset block: requesting more than block_size would
    # silently yield shorter chunks and desync downstream shape assumptions.
    seq_len = min(int(seq_len), int(cfg["model"]["block_size"]))
    ds = FixedDataset(
        cfg["val_path"],
        cfg["model"]["block_size"],
        expected_vocab_size=cfg.get("expected_vocab_size", cfg["model"].get("vocab_size")),
    )
    for i in range(min(max_batches, len(ds) // batch_size)):
        chunks = [ds[i * batch_size + j]["input_ids"][: seq_len + 1] for j in range(batch_size)]
        ids = torch.stack(chunks, dim=0).to(device)
        yield ids[:, :-1], ids[:, 1:]


def safe_ppl(loss: float) -> float:
    return math.exp(loss) if math.isfinite(loss) and loss < 20 else float("inf")


@torch.no_grad()
def validation_loss(model: GPT, cfg: Dict[str, Any], *, seq_len: int, batch_size: int, batches: int, device: str) -> Dict[str, float]:
    total = 0.0
    toks = 0
    for x, y in iter_val_batches(cfg, seq_len=seq_len, batch_size=batch_size, max_batches=batches, device=device):
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        total += float(loss.detach().cpu().item()) * y.numel()
        toks += y.numel()
    loss = total / max(1, toks)
    return {"val_loss": loss, "val_ppl": safe_ppl(loss), "val_tokens": toks}


@torch.no_grad()
def noisy_validation_loss(model: GPT, cfg: Dict[str, Any], *, seq_len: int, batch_size: int, batches: int, device: str, noise: float) -> Dict[str, float]:
    vocab = int(cfg["model"]["vocab_size"])
    total = 0.0
    toks = 0
    g = torch.Generator(device="cpu")
    g.manual_seed(12345)
    for x, y in iter_val_batches(cfg, seq_len=seq_len, batch_size=batch_size, max_batches=batches, device=device):
        mask = (torch.rand(x.shape, generator=g) < noise).to(device)
        repl = torch.randint(0, vocab, x.shape, generator=g, dtype=torch.long).to(device)
        x_noisy = torch.where(mask, repl, x)
        _, loss = model(x_noisy, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite noisy validation loss")
        total += float(loss.detach().cpu().item()) * y.numel()
        toks += y.numel()
    loss = total / max(1, toks)
    return {"noisy_val_loss": loss, "noisy_val_ppl": safe_ppl(loss), "noise_level": noise}


@torch.no_grad()
def attention_and_hook_metrics(model: GPT, cfg: Dict[str, Any], *, seq_len: int, batch_size: int, batches: int, device: str) -> Dict[str, Any]:
    L, H = cfg["model"]["n_layer"], cfg["model"]["n_head"]
    entropy_sum = torch.zeros(L, H, dtype=torch.float64)
    entropy_norm_sum = torch.zeros(L, H, dtype=torch.float64)
    gate_k_vals: List[List[torch.Tensor]] = [[] for _ in range(L)]
    gate_v_vals: List[List[torch.Tensor]] = [[] for _ in range(L)]
    mix_k_vals: List[List[torch.Tensor]] = [[] for _ in range(L)]
    mix_v_vals: List[List[torch.Tensor]] = [[] for _ in range(L)]
    mlp_sparsity = torch.zeros(L, dtype=torch.float64)
    attn_out_norm = torch.zeros(L, dtype=torch.float64)
    mlp_out_norm = torch.zeros(L, dtype=torch.float64)
    count = 0

    model.set_diagnostics(enabled=True, capture_attention=True)
    try:
        for x, _ in iter_val_batches(cfg, seq_len=seq_len, batch_size=batch_size, max_batches=batches, device=device):
            _ = model(x)
            count += 1
            # Use the ACTUAL sequence length of this batch: when the dataset
            # block is shorter than --seq-len, entropy tensors are (B,H,T_actual)
            # and a denom built from the requested length shape-mismatches
            # (silent-shape bug found in the final review sweep).
            t_actual = x.size(1)
            denom = torch.log(torch.arange(1, t_actual + 1, device=device, dtype=torch.float32).clamp_min(2))
            for li, block in enumerate(model.transformer.h):
                attn = block.attn.last_attn
                if attn is None:
                    continue
                p = attn.float().clamp_min(1e-10)
                ent = -(p * p.log()).sum(dim=-1)  # [B,H,T]
                entropy_sum[li] += ent.mean(dim=(0, 2)).detach().cpu().double()
                entropy_norm_sum[li] += (ent / denom[None, None, :]).mean(dim=(0, 2)).detach().cpu().double()
                if block.attn.last_gate_k is not None:
                    gate_k_vals[li].append(block.attn.last_gate_k.detach().float().flatten().cpu())
                    gate_v_vals[li].append(block.attn.last_gate_v.detach().float().flatten().cpu())
                    mix_k_vals[li].append(block.attn.last_mix_k.detach().float().flatten().cpu())
                    mix_v_vals[li].append(block.attn.last_mix_v.detach().float().flatten().cpu())
                if block.mlp.last_hidden is not None:
                    h = block.mlp.last_hidden.detach().float()
                    mlp_sparsity[li] += (h.abs() > 0.1).float().mean().detach().cpu().double()
                if block.attn.last_attn_out_norm is not None:
                    attn_out_norm[li] += block.attn.last_attn_out_norm.detach().cpu().double()
                if block.mlp.last_mlp_out_norm is not None:
                    mlp_out_norm[li] += block.mlp.last_mlp_out_norm.detach().cpu().double()
    finally:
        model.set_diagnostics(enabled=False, capture_attention=False)

    count = max(1, count)
    entropy = (entropy_sum / count).numpy()
    entropy_norm = (entropy_norm_sum / count).numpy()
    collapse = float((entropy_norm < 0.1).mean())

    def summarize(xs: List[torch.Tensor]) -> Dict[str, float]:
        if not xs:
            return {"mean": float("nan"), "std": float("nan"), "p01": float("nan"), "p50": float("nan"), "p99": float("nan"), "saturation_abs_gt_0_9": float("nan"), "frac_positive": float("nan")}
        x = torch.cat(xs).float()
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "p01": float(torch.quantile(x, 0.01).item()),
            "p50": float(torch.quantile(x, 0.50).item()),
            "p99": float(torch.quantile(x, 0.99).item()),
            "saturation_abs_gt_0_9": float((x.abs() > 0.9).float().mean().item()),
            "frac_positive": float((x > 0).float().mean().item()),
        }

    per_layer = []
    for li in range(L):
        per_layer.append(
            {
                "layer": li,
                "layer_gate_multiplier": float(getattr(model.transformer.h[li].attn, "layer_gate_multiplier", 1.0)),
                "gate_k": summarize(gate_k_vals[li]),
                "gate_v": summarize(gate_v_vals[li]),
                "mix_k": summarize(mix_k_vals[li]),
                "mix_v": summarize(mix_v_vals[li]),
                "mlp_sparsity_abs_gt_0_1": float((mlp_sparsity[li] / count).item()),
                "attn_out_norm": float((attn_out_norm[li] / count).item()),
                "mlp_out_norm": float((mlp_out_norm[li] / count).item()),
                "distance_bias_mean": float(getattr(model.transformer.h[li].attn, "last_distance_bias_mean", torch.tensor(float("nan"))).detach().cpu().item()) if getattr(model.transformer.h[li].attn, "last_distance_bias_mean", None) is not None else float("nan"),
                "distance_bias_abs_mean": float(getattr(model.transformer.h[li].attn, "last_distance_bias_abs_mean", torch.tensor(float("nan"))).detach().cpu().item()) if getattr(model.transformer.h[li].attn, "last_distance_bias_abs_mean", None) is not None else float("nan"),
            }
        )
    return {
        "attention_entropy": entropy.tolist(),
        "attention_entropy_normalized": entropy_norm.tolist(),
        "attention_collapse_fraction_norm_lt_0_1": collapse,
        "layer_metrics": per_layer,
    }


@torch.no_grad()
def induction_by_distance(model: GPT, cfg: Dict[str, Any], *, distances: List[int], batch_size: int, device: str) -> Dict[str, float]:
    vocab = int(cfg["model"]["vocab_size"])
    out: Dict[str, float] = {}
    if vocab <= 47000 + batch_size:
        return {str(d): float("nan") for d in distances}
    g = torch.Generator(device="cpu")
    g.manual_seed(999)
    for dist in distances:
        T = min(max(dist + 16, 64), int(cfg["model"]["block_size"]))
        pos_a1 = 4
        pos_b1 = 5
        pos_a2 = min(pos_b1 + dist, T - 2)
        toks = torch.randint(100, min(20000, vocab - 1), (batch_size, T), generator=g).to(device)
        targets = []
        for i in range(batch_size):
            a = 45000 + i
            b = 46000 + i
            toks[i, pos_a1] = a
            toks[i, pos_b1] = b
            toks[i, pos_a2] = a
            targets.append(b)
        target = torch.tensor(targets, device=device)
        # Memory fix (final polish): a single forward at batch=16 materializes
        # logits (16, T, vocab) fp32 ~= 0.4-0.7 GB and OOMs small hosts. The
        # probe only needs ONE position per row - run in micro-chunks of 4 and
        # keep just that column. Peak drops ~4x with identical results.
        chunk = 4
        vals = []
        for s in range(0, batch_size, chunk):
            logits, _ = model(toks[s:s + chunk])
            from src.eval import _mask_padded_logits  # finding #19
            probs = torch.softmax(_mask_padded_logits(model, logits[:, pos_a2, :].float()), dim=-1)
            vals.append(probs.gather(1, target[s:s + chunk, None]))
            del logits, probs
        out[str(dist)] = float(torch.cat(vals).mean().detach().cpu().item())
    return out


@torch.no_grad()
def residual_geometry_metrics(model: GPT, cfg: Dict[str, Any], *, seq_len: int, batch_size: int, batches: int, device: str, max_vectors: int = 4096) -> Dict[str, Any]:
    """M18-M20: anisotropy, participation ratio, cosine similarity by layer."""
    L = cfg["model"]["n_layer"]
    buckets: List[List[torch.Tensor]] = [[] for _ in range(L)]
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            x = out.detach().float().reshape(-1, out.shape[-1]).cpu()
            if x.shape[0] > 512:
                x = x[:512]
            buckets[layer_idx].append(x)
        return hook

    for li, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(li)))
    try:
        for x, _ in iter_val_batches(cfg, seq_len=seq_len, batch_size=batch_size, max_batches=batches, device=device):
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    layers = []
    for li in range(L):
        if not buckets[li]:
            layers.append({"layer": li, "participation_ratio": float("nan"), "top10_variance_fraction": float("nan"), "mean_abs_cosine": float("nan")})
            continue
        X = torch.cat(buckets[li], dim=0)[:max_vectors]
        X = X - X.mean(dim=0, keepdim=True)
        # covariance eigenspectrum in float64 for stable diagnostics
        cov = (X.double().T @ X.double()) / max(1, X.shape[0] - 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0)
        total = eig.sum().clamp_min(1e-12)
        pr = (total * total / eig.pow(2).sum().clamp_min(1e-12)).item()
        top10 = eig[-10:].sum().item() / total.item() if eig.numel() >= 10 else eig.sum().item() / total.item()
        # mean absolute off-diagonal cosine on a subset
        Y = F.normalize(X.float()[: min(512, X.shape[0])], dim=-1)
        sim = (Y @ Y.T).abs()
        if sim.shape[0] > 1:
            mean_abs_cos = (sim.sum() - sim.diag().sum()) / (sim.numel() - sim.shape[0])
            mac = float(mean_abs_cos.item())
        else:
            mac = float("nan")
        layers.append({"layer": li, "participation_ratio": float(pr), "top10_variance_fraction": float(top10), "mean_abs_cosine": mac})
    return {"residual_geometry": layers}


@torch.no_grad()
def knockout_by_context_length(model: GPT, cfg: Dict[str, Any], *, lengths: List[int], device: str) -> Dict[str, Dict[str, float]]:
    """Causal knockout (src.eval.mechanism_knockout) at several context lengths.

    THE long-context figure: for each mechanism, delta-loss from silencing it,
    as a function of how much context the batch has. A mechanism whose delta
    GROWS with length is causally carrying long-range structure - this curve
    is the quantitative evidence behind the long-context claim (knockout is
    measurement-only; the mechanism stays on in deployment).
    """
    from src.eval import mechanism_knockout
    out: Dict[str, Dict[str, float]] = {}
    block = int(cfg["model"]["block_size"])
    for T in lengths:
        T = max(8, min(int(T), block))
        got = None
        for x, y in iter_val_batches(cfg, seq_len=T, batch_size=1, max_batches=1, device=device):
            got = mechanism_knockout(model, x, y)
        if got is not None:
            out[str(T)] = {k: float(v) for k, v in got.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", help="auto|xla|cpu|cuda (auto = xla if available, else cpu)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.01)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, cfg = load_checkpoint_model(args.checkpoint, args.config, device)
    result: Dict[str, Any] = {"checkpoint": args.checkpoint, "config_variant": cfg.get("variant"), "device": str(device)}
    result.update(validation_loss(model, cfg, seq_len=args.seq_len, batch_size=args.batch_size, batches=args.batches, device=device))
    result.update(noisy_validation_loss(model, cfg, seq_len=args.seq_len, batch_size=args.batch_size, batches=max(1, args.batches // 2), device=device, noise=args.noise))
    result["induction_by_distance"] = induction_by_distance(model, cfg, distances=[10, 25, 50, 100, 200], batch_size=16, device=device)
    result.update(attention_and_hook_metrics(model, cfg, seq_len=args.seq_len, batch_size=args.batch_size, batches=args.batches, device=device))
    result.update(residual_geometry_metrics(model, cfg, seq_len=args.seq_len, batch_size=args.batch_size, batches=max(1, args.batches // 2), device=device))
    result["knockout_by_context_length"] = knockout_by_context_length(
        model, cfg, lengths=[args.seq_len // 4, args.seq_len // 2, args.seq_len], device=device)
    try:
        from src.eval import prefix_matching_score
        result["prefix_matching"] = prefix_matching_score(model, device=device, batch_size=1)
    except Exception as e:  # vocab too small etc.
        result["prefix_matching"] = {"error": str(e)}
    try:
        from src.eval import gate_redundancy_statistics
        result["gate_redundancy"] = gate_redundancy_statistics(model, device=device)
    except Exception as e:
        result["gate_redundancy"] = {"error": str(e)}
    try:
        from src.eval import positional_recall_curve
        result["positional_recall"] = positional_recall_curve(model, device=device)
    except Exception as e:
        result["positional_recall"] = {"error": str(e)}
    try:
        from src.eval import per_position_loss_curve
        ds2 = FixedDataset(
            cfg["val_path"],
            cfg["model"]["block_size"],
            expected_vocab_size=cfg.get("expected_vocab_size", cfg["model"].get("vocab_size")),
        )
        n = min(len(ds2), 32)
        toks = torch.stack([ds2[i]["input_ids"] for i in range(n)])
        result["per_position_loss"] = per_position_loss_curve(model, toks, device=device)
    except Exception as e:
        result["per_position_loss"] = {"error": str(e)}
    try:
        from src.eval import attention_sink_stats, activation_outlier_stats
        result["attention_sink"] = attention_sink_stats(model, device=device)
        result["activation_outliers"] = activation_outlier_stats(model, device=device)
    except Exception as e:
        result["attention_sink"] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
