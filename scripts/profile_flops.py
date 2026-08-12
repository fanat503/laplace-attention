"""Profile FLOPs for both base and HLA models to compute matched compute."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig


def count_forward_flops(model: GPT, batch_size: int = 1, seq_len: int = 2048) -> int:
    """Count forward FLOPs per call (excluding backward).

    Uses 6N rule for total training: forward + backward = 6N where N = params * tokens.
    Here we count forward only (N = 2 * params * tokens).
    """
    config = model.config
    B = batch_size
    T = seq_len
    C = config.n_embd
    H = config.n_head
    hs = config.n_embd // config.n_head
    L = config.n_layer

    # Attention QKV projection.
    qkv_flops = L * (2 * B * T * C * 3 * C)

    # Holographic phase rotation.
    if config.phase_mult != 0:
        # W_phase projection: einsum over C dim.
        phase_flops = L * (4 * B * T * C * H * (hs // 2))  # Q + K
        # rotate_pairwise: 4 mul-add per element.
        rotate_flops = L * (B * T * H * hs * 4)
    else:
        phase_flops = 0
        rotate_flops = 0

    # Laplace gating.
    if config.use_laplace and config.laplace_alpha != 0:
        # W_gate: 2 * C * H per layer (no batch dim).
        gate_flops = L * (4 * B * T * C * H)
        # mix computation: per-element mul + add.
        mix_flops = L * (B * T * H * hs * 4)
    else:
        gate_flops = 0
        mix_flops = 0

    # Attention scores (Q @ K^T).
    attn_score_flops = L * (2 * B * H * T * T * hs)
    # Attention values (attn @ V).
    attn_value_flops = L * (2 * B * H * T * T * hs)

    # Distance-Laplace bias.
    if config.use_distance_laplace and config.distance_laplace_alpha != 0:
        # Since the deconfound fix distance has its OWN gate W_gate_d (C->H
        # projection per key) - count it, plus the T*T bias computation.
        dist_flops = L * (2 * B * T * C * H + 2 * B * H * T * T)
    else:
        dist_flops = 0

    # Salience bias (per-key projection C->H + add to scores).
    if getattr(config, "use_salience_bias", False) and getattr(config, "salience_alpha", 0.0) != 0:
        salience_flops = L * (2 * B * T * C * H + B * H * T * T)
    else:
        salience_flops = 0

    # Forget gate (projection C->H, cumsum T, pairwise bias add T*T).
    if getattr(config, "use_forget_gate", False) and getattr(config, "forget_alpha", 0.0) != 0:
        forget_flops = L * (2 * B * T * C * H + B * H * T + B * H * T * T)
    else:
        forget_flops = 0

    # Q-side temperature (SSA-family): W_qtemp projection C->H + tanh/exp
    # scaling of q (per-element mul).
    if getattr(config, "use_qtemp", False) and getattr(config, "qtemp_alpha", 0.0) != 0:
        qtemp_flops = L * (2 * B * T * C * H + B * T * H * hs)
    else:
        qtemp_flops = 0

    # Output projection.
    proj_flops = L * (2 * B * T * C * C)

    # MLP (SwiGLU).
    # BUG FIX (final review sweep): hidden was previously multiplied by L
    # (hidden = L*(8C//3)), inflating MLP FLOPs by a factor of n_layer and
    # distorting every per-component share. hidden_dim must match model.py:
    hidden = int(8 * C / 3)
    hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64
    if config.fused_swiglu:
        mlp_flops = L * (2 * B * T * C * (2 * hidden) + 2 * B * T * hidden * C)
    else:
        mlp_flops = L * (4 * B * T * C * hidden + 2 * B * T * hidden * C)

    # LayerNorm (negligible but include for completeness).
    ln_flops = L * (4 * B * T * C) * 2  # 2 LN per block

    # Final LN.
    final_ln_flops = 4 * B * T * C

    total = (
            qkv_flops + phase_flops + rotate_flops +
            gate_flops + mix_flops +
            attn_score_flops + attn_value_flops + dist_flops +
            salience_flops + forget_flops + qtemp_flops +
            proj_flops + mlp_flops + ln_flops + final_ln_flops
    )

    return {
        "qkv": qkv_flops,
        "phase": phase_flops,
        "rotate": rotate_flops,
        "gate": gate_flops,
        "mix": mix_flops,
        "attn_score": attn_score_flops,
        "attn_value": attn_value_flops,
        "distance_bias": dist_flops,
        "salience": salience_flops,
        "forget": forget_flops,
        "qtemp": qtemp_flops,
        "proj": proj_flops,
        "mlp": mlp_flops,
        "ln": ln_flops + final_ln_flops,
        "total": total,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Per-mechanism FLOPs breakdown for a base/HLA pair")
    ap.add_argument("--base", default="configs/200m_base_s42.json")
    ap.add_argument("--hla", default="configs/200m_hla_s42.json")
    ap.add_argument("--seq-len", type=int, default=None, help="default: model block_size")
    ap.add_argument("--out", default="flops_profile.json")
    args = ap.parse_args()
    base_path, hla_path = args.base, args.hla

    with open(base_path) as f:
        base_config = json.load(f)
    with open(hla_path) as f:
        hla_config = json.load(f)

    # FLOPs are computed ANALYTICALLY from the config - instantiating full
    # 200M+ models here only to read their .config caused OOM on small hosts
    # (found in the final review sweep). Build configs, not weights.
    class _Cfg:
        def __init__(self, d): self.__dict__.update(GPTConfig(**d).__dict__)
    base_model = type("M", (), {"config": GPTConfig(**base_config["model"])})()
    hla_model = type("M", (), {"config": GPTConfig(**hla_config["model"])})()

    print("Profiling FLOPs (analytic, no weights allocated)...")
    T = args.seq_len or int(base_config["model"]["block_size"])
    base_flops = count_forward_flops(base_model, batch_size=1, seq_len=T)
    hla_flops = count_forward_flops(hla_model, batch_size=1, seq_len=T)

    # Print breakdown.
    print(f"\n{'Component':25s}  {'Base':>15s}  {'HLA':>15s}  {'Ratio':>10s}")
    print(f"{'-' * 70}")
    for k in base_flops:
        ratio = hla_flops[k] / base_flops[k] if base_flops[k] > 0 else float("inf")
        print(f"{k:25s}  {base_flops[k]:>15,}  {hla_flops[k]:>15,}  {ratio:>10.3f}")

    # Total ratio.
    total_ratio = hla_flops["total"] / base_flops["total"]
    print(f"\nTotal FLOPs ratio (HLA/Base): {total_ratio:.4f}")
    print(f"HLA overhead: {(total_ratio - 1) * 100:.2f}%")

    # Recommendation.
    print("\n--- Recommendation ---")
    base_max_steps = base_config["max_steps"]
    print(f"Base max_steps: {base_max_steps}")

    # For matched training FLOPs: multiply base steps by inverse ratio.
    # Training FLOPs ≈ 6 × params × tokens = 6 × params × (steps × batch × seq_len).
    # If we want same total compute, HLA needs fewer steps.
    matched_hla_steps = int(base_max_steps / total_ratio)
    print(f"For matched compute: HLA max_steps = {matched_hla_steps}")
    print(f"  (vs current {hla_config['max_steps']})")
    print(
        f"  Difference: {hla_config['max_steps'] - matched_hla_steps} steps ({100 * (hla_config['max_steps'] - matched_hla_steps) / base_max_steps:.2f}%)")

    # Save profile results.
    results = {
        "base_flops": base_flops,
        "hla_flops": hla_flops,
        "total_ratio": total_ratio,
        "base_max_steps": base_max_steps,
        "matched_hla_max_steps": matched_hla_steps,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
