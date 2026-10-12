# Copyright 2026 Slyatski Ilya
"""O1 draft (r135, finding #83/H1): zero-padding of folded q'/k' head dims.

Zero columns contribute exactly zero to every q'k' dot product, so padded
and unpadded folds must produce the same attention output; padding must
also stay OFF by default so the frozen science path is untouched."""
import json
import glob
import os

import pytest
import torch

from src.model import GPT, GPTConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = dict(block_size=64, vocab_size=257, n_layer=2, n_head=4, n_embd=128,
            use_rope=True, use_wpe=False,
            phase_mult=0.15, laplace_alpha=1.0, salience_alpha=1.0,
            distance_laplace_alpha=0.5, use_salience_bias=True,
            use_distance_laplace=True)


def _logits(backend, pad):
    torch.manual_seed(7)
    cfg = GPTConfig(**BASE, attention_backend=backend, fold_pad_multiple=pad)
    m = GPT(cfg); m.eval()
    sd = m.state_dict()
    # одинаковые веса для всех вариантов
    torch.manual_seed(11)
    for k in sd:
        if sd[k].is_floating_point():
            sd[k] = torch.randn_like(sd[k]) * 0.3
    m.load_state_dict(sd)
    x = torch.randint(0, 257, (2, 48), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        out = m(x)
    return (out[0] if isinstance(out, tuple) else out)


def test_padding_is_exact_vs_unpadded_fold():
    a = _logits("sdpa_fold", 0)
    b = _logits("sdpa_fold", 8)
    assert torch.allclose(a, b, atol=1e-5, rtol=0), (a - b).abs().max().item()


def test_padded_fold_matches_manual():
    m = _logits("manual", 0)
    b = _logits("sdpa_fold", 8)
    assert torch.allclose(m, b, atol=5e-5, rtol=0), (m - b).abs().max().item()


def test_padded_dim_math():
    # 4 головы, hs=32; extras: distance(+2)+salience(+1) -> 35 -> pad до 40
    assert (35 + 7) // 8 * 8 == 40


def test_default_off_everywhere():
    """Наука и все шипнутые конфиги не несут флаг: дефолт 0 = поведение
    бит-в-бит как до драфта."""
    for p in glob.glob(os.path.join(REPO, "configs", "*.json")):
        c = json.load(open(p))
        assert "fold_pad_multiple" not in c.get("model", {}), p
    assert GPTConfig(block_size=8, vocab_size=16).fold_pad_multiple == 0


# --- r136: mech_fast_math (один GEMM на гейты + стекнутый phase-einsum) ---

def test_fast_math_matches_slow_fp32():
    """На CPU/fp32 fast-math отличается только порядком редукций:
    допуск жёсткий. (bf16-поведение меряется на устройстве бисектом v2.)"""
    a = _logits("sdpa_fold", 0)
    cfg = dict(BASE)
    torch.manual_seed(7)
    m = GPT(GPTConfig(**cfg, attention_backend="sdpa_fold",
                      fold_pad_multiple=8, mech_fast_math=True))
    m.eval()
    sd = m.state_dict()
    torch.manual_seed(11)
    for k in sd:
        if sd[k].is_floating_point():
            sd[k] = torch.randn_like(sd[k]) * 0.3
    m.load_state_dict(sd)
    x = torch.randint(0, 257, (2, 48), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        out = m(x)
    b = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(a, b, atol=2e-4, rtol=0), (a - b).abs().max().item()


def test_fast_math_matches_manual_fp32():
    mref = _logits("manual", 0)
    cfg = dict(BASE)
    torch.manual_seed(7)
    m = GPT(GPTConfig(**cfg, attention_backend="sdpa_fold",
                      fold_pad_multiple=8, mech_fast_math=True))
    m.eval()
    sd = m.state_dict()
    torch.manual_seed(11)
    for k in sd:
        if sd[k].is_floating_point():
            sd[k] = torch.randn_like(sd[k]) * 0.3
    m.load_state_dict(sd)
    x = torch.randint(0, 257, (2, 48), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        out = m(x)
    b = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(mref, b, atol=3e-4, rtol=0), (mref - b).abs().max().item()


def test_fast_math_grads_flow_and_finite():
    cfg = dict(BASE)
    torch.manual_seed(7)
    m = GPT(GPTConfig(**cfg, attention_backend="sdpa_fold",
                      fold_pad_multiple=8, mech_fast_math=True))
    x = torch.randint(0, 257, (2, 32))
    out = m(x, x)
    loss = out[1] if isinstance(out, tuple) else out.mean()
    loss.backward()
    for name in ("W_phase_q", "W_gate_k", "W_gate_sal", "W_gate_d"):
        p = dict(m.named_parameters())[f"transformer.h.0.attn.{name}"
                                       if not name.startswith("W_gate") else
                                       f"transformer.h.0.attn.{name}.weight"
                                       ] if name != "W_phase_q" else \
            dict(m.named_parameters())["transformer.h.0.attn.W_phase_q"]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_fast_math_default_off_everywhere():
    for p in glob.glob(os.path.join(REPO, "configs", "*.json")):
        c = json.load(open(p))
        assert "mech_fast_math" not in c.get("model", {}), p
    assert GPTConfig(block_size=8, vocab_size=16).mech_fast_math is False
