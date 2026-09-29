# HLA Speed Backends: how to run HLA at flash-attention speed on ANY stack

## TL;DR for adopters

All HLA additive attention biases are per-key or rank-1. Therefore they fold
INTO the q/k dot product via 3 augmented head dimensions, after which **any
flash-attention kernel is legal without modification**:

```
d  = head_size;  sd = sqrt(d);  p_i = pos_i / (block_size - 1)
c_j = alpha_d * range_d * layer_mult * tanh(W_gate_d x_j)     # distance gate
s_j = alpha_s * range_s * tanh(W_gate_sal x_j)                # salience gate

q'_i = [q_i,  sd * p_i,  sd,  sd ]          # d+3 dims
k'_j = [k_j,  c_j,  -p_j * c_j,  s_j ]      # d+3 dims

softmax(q' k'^T / sd) == softmax(q k^T / sd  +  (p_i - p_j) * c_j  +  s_j)   EXACT
```

`layer_mult` (FIX #35, attack J2 - it was used above but never defined;
a copy-pasting adopter could not have computed c_j):

```
layer_mult = 1.0                                          # default
# if layer_dependent_gate:   static depth heuristic (see model.py __init__)
# if additionally learnable_layer_temp:
layer_mult = 1.0 + (layer_idx / n_layer) * softplus(W_layer_temp) / log(2)
#            (softplus(0)=log 2  =>  exactly the static heuristic at init)
```

Call your kernel with an **explicit scale = 1/sqrt(d)** (NOT 1/sqrt(d+3)).
The multiplicative mechanisms (K-mix, V-mix, Q-temp, phase rotation) modify
q/k/v BEFORE attention and need nothing at all.

Exactness condition (guard #29): |alpha * range| <= clip for each folded
bias, so the clamp is provably inactive on the causal region. Our shipped
configs satisfy it; configs that do not automatically fall back to the exact
clamped manual path.

## Backends in this repo

| backend      | kernel                                             | device    | use case |
|--------------|----------------------------------------------------|-----------|----------|
| `manual`     | explicit att matrix, bit-exact                     | any       | the frozen science pair (V7/V8), diagnostics, sterility proofs |
| `sdpa`       | `F.scaled_dot_product_attention`                   | GPU/TPU   | speed-base (dispatches to FlashAttention-2 on CUDA, XLA fused attention on TPU) |
| `sdpa_fold`  | same kernel, HLA biases folded into q'/k' (#30)    | GPU/TPU   | production HLA |
| `pallas`     | folded q'/k' -> `torch_xla ... flash_attention`    | TPU only  | max TPU speed; fails LOUD without torch_xla |

Note on names: **Triton does not exist on TPU** (it is a CUDA/NVIDIA kernel
compiler). Pallas is the ecosystem's answer: it lowers **through Triton on
GPU and through Mosaic on TPU** - so "Pallas" is the honest cross-platform
label, and `attention_backend="triton"` is rejected with ValueError to
prevent accidental cargo-cult configs.

## How the best labs integrate Pallas (and how we mirror it)

Documented practice from the JAX/PyTorch-XLA ecosystem:

1. **Reuse reference kernels, do not rewrite them.** The JAX repo ships
   maintained Pallas kernels (flash attention among them); torch_xla adopts
   them via `torch_xla.experimental.custom_kernel` (that is exactly what our
   `pallas` backend calls). Writing kernels from scratch is reserved for
   cases the compiler cannot fuse.
2. **Trust XLA fusion first, drop to Pallas only when profiling says so.**
   XLA already fuses attention well below ~8K sequence; hand kernels win at
   long sequence / sparse patterns (Splash Attention). Our design follows
   this: `sdpa` (XLA-fused) is the default speed path; `pallas` is the
   opt-in maximum.
3. **Never let a fast path silently change semantics.** Production stacks
   gate kernel dispatch on exact applicability. Ours: fold is only taken
   when the clamp is provably inactive (#29 guard); diagnostics force the
   bit-exact manual path; pallas without torch_xla raises ImportError
   instead of silently benchmarking the wrong kernel.

## Config matrix (all validated by GPTConfig at test time)

TPU (Kaggle v5e-8):
- `kaggle_200m_base_speed_sdpa_s42.json`  - speed-base
- `kaggle_200m_hla_speed_fold_s42.json`   - HLA fold
- `kaggle_200m_hla_speed_pallas_s42.json` - HLA Pallas
- `700m_base_14b_sdpa_s42.json`, `700m_hla_14b_fold_s42.json`, `700m_hla_14b_pallas_s42.json`

GPU (any CUDA box with PyTorch >= 2.x):
- `gpu_200m_base_flash_s42.json`      - base on FlashAttention-2 via SDPA
- `gpu_200m_hla_flash_fold_s42.json`  - HLA fold on FlashAttention-2
- `gpu_700m_base_flash_s42.json`, `gpu_700m_hla_flash_fold_s42.json`

## Verified equivalence (CPU, two fresh clones)

- fold vs manual logits (all gates live): 5.96e-07; grads 4.84e-08; loss 0.0
- weights after one AdamW step: 4.1e-05 (fp32 reduction-order scale)
- binding-clip configs: bit-exact fallback to manual (torch.equal)
- diagnostics on: bit-exact manual (torch.equal)
- sterility caveat: flash reduction order != manual order, so fold@Theta0 vs
  base is ~5e-07, NOT 0.0 - this is why the SCIENCE pair stays on `manual`
  and speed numbers are reported from the speed configs separately.

## What is still TPU-only (honest list)

- wall-clock numbers for sdpa/fold/pallas on v5e-8 (smoke planned: 200 steps
  each, one session, after the science pair finishes);
- pallas kernel gradient path on real hardware (torch_xla custom kernels
  have their own backward; smoke includes a backward step);
- bf16 behavior of the folded extra dims under Mosaic lowering.
