# HLA: unified formulation and formal guarantees

This document states the single attention formula that subsumes every HLA
mechanism, then proves exactly what can be proven — and states honestly what
cannot. Every numbered claim marked ✓ is additionally verified numerically in
`tests/test_theory.py`.

---

## 0. On the name "holographic"

The name is a structural analogy, claimed precisely and nothing more. In
optical holography the recorded interference pattern stores information in
relative phase, while magnitude carries the illumination. Retrieval information is thus encoded in learned relative
phase; content volume travels in magnitudes through u_j v_j.

What we do NOT claim: no wave optics, no complex-valued network (the
implementation is real-valued rotation pairs), no connection to holographic
reduced representations beyond the shared phase-binding intuition (Plate,
1995, is acknowledged as the closest conceptual ancestor: binding via
circular convolution is phase addition in Fourier space, and our
content-conditioned rotations perform learned phase addition per plane).
If the reader prefers, every occurrence of "holographic" may be mentally
replaced by "phase-magnitude decoupled" with zero loss of technical content.

## 1. The unified HLA attention formula

Standard attention (per head, causal):

```
Att(x)_i = Σ_{j≤i} softmax_j( q_i·k_j / √d ) · v_j
```

**HLA replaces q, k, v by gated/rotated versions and adds a structured score
bias.** One formula, all seven mechanisms:

```
             ┌──── retrieval geometry ────┐   ┌── key salience ──┐
score_ij  =  ( τ_i · R(θ_i) q_i )  ·  ( m_j · R(φ_j) k_j )  / √d   +   B_ij
                                                    
out_i     =  Σ_{j≤i}  softmax_j(score_ij)  ·  ( u_j · v_j )
                                              └ content volume ┘
```

with the components (all zero / identity at initialization):

| Symbol | Definition | Mechanism | Init value |
|---|---|---|---|
| R(θ_i) | pairwise rotation by θ_i = π·μ_φ·ρ_h·λ_l·tanh(W_φq x_i) | phase (query side) | R(0) = **I** |
| R(φ_j) | pairwise rotation by φ_j = π·μ_φ·ρ_h·λ_l·tanh(W_φk x_j) | phase (key side) | R(0) = **I** |
| m_j | (1−β_k) + β_k·exp(clamp(α·tanh(W_gk x_j)·r_k·λ_l, ±c_k·λ_l)) | K-gate | **1** |
| u_j | (1−β_v) + β_v·exp(clamp(α·tanh(W_gv x_j)·r_v·λ_l, ±c_v·λ_l)) | V-gate | **1** |
| B_ij | b^sal_j + b^dist_ij + (S_i − S_j) | additive biases | **0** |
| b^sal_j | clamp(α_s·r_s·tanh(W_s x_j), ±c_s) | salience | 0 |
| b^dist_ij | clamp(α_d·r_d·λ_l·d(i,j)·tanh(W_gd x_j), ±c_d) — W_gd is distance's OWN gate (deconfounded from the K-gate); d(i,j) = (i−j)/(block_size−1) ∈ [0,1], normalized by the MODEL constant (not the batch length) so the bias for a fixed pair is independent of how much context is in the batch — prefix-stable, hence KV-cache-exact (tested) | distance | 0 |
| τ_i | exp(clamp(α_q·r_q·tanh(W_qt x_i), ±c_q)) — per-query softmax temperature (SSA/SSMax family; the only non-no-op Q-side score form, since an additive per-query bias cancels in softmax) | Q-temp | **1** |
| S_i − S_j | S_t = Σ_{τ≤t} α_f·r_f·tanh(W_f x_τ), clamped | forget (FoX-family) | 0 |
| ρ_h | 1 + tanh(s_h) — per-head phase budget | head adaptivity | 1 |
| μ_φ | `phase_mult` config constant (0.15 in all runs) — global phase budget; max attainable angle = π·μ_φ·2·2 = 0.6π with head+layer budgets maxed | phase scale | 0.15 (fixed) |
| λ_l | 1 + (l/L)·softplus(θ_l)/softplus(0) — depth profile; applies to phase, K/V-gates (range AND clamp) and distance; NOT to salience/q-temp/forget (code-verified r85) | layer adaptivity | 1 + l/L (static) |

Where the base model is recovered by ρ, λ multipliers irrelevant because every
learned input to them is zero. **Reading the formula**: R controls *where
matching happens* (geometry), m and b^sal control *how loud each key is*
(multiplicative with floor 1−β_k vs additive with no floor), B's distance and
forget terms control *reach over positions*, τ controls *how sharp each query
listens* (per-query temperature), u controls *how loud each message is*.
Retrieval (everything inside softmax) and transmission (u_j·v_j) share no
learned scalars — that is the decoupling, stated syntactically.

---

## 2. What is PROVEN (with proofs)

### Theorem 1 (Exact identity at initialization). ✓ tested bit-exact
Let Θ₀ = {W_φq, W_φk, W_gk, W_gv, W_s, W_f, W_rk, W_rv, W_rf, s, θ} = 0.
Then HLA(x; Θ_backbone, Θ₀) = GPT(x; Θ_backbone) for every input x, exactly
(not approximately).

*Proof.* tanh(0)=0 pointwise ⇒ every angle is 0 ⇒ R(0)=I (rotation by zero is
the identity map); every gate exponent is 0 ⇒ exp(0)=1 ⇒ m_j=u_j=(1−β)+β=1;
every bias term has a tanh(0)=0 factor ⇒ B_ij=0; cumsum of zeros is zero.
Substituting I, 1, 1, 0 into the unified formula yields the standard formula
symbol-for-symbol. Clamps are inactive at 0 (0 is interior to every clamp
interval), so no non-smoothness is touched. ∎

Corollary: any final-loss difference between HLA and the parameter-matched
base trained from the shared init is attributable to the *training dynamics*
of the mechanisms, not to initialization.

### Theorem 2 (Strict expressivity inclusion). ✓ tested constructively
Let F_base and F_HLA be the function classes of one attention layer. Then
F_base ⊊ F_HLA (strict).

*Proof.* (⊆) By Theorem 1 any base function is realized in F_HLA at Θ₀.
(strictness) Exhibit f ∈ F_HLA \ F_base. Set W_Q = 0 in both models. In base,
score_ij = 0 for all j ⇒ softmax is exactly uniform on {j ≤ i} for every
input: attention weights cannot depend on key content. In HLA with the same
W_Q = 0 and any W_s ≠ 0, score_ij = b^sal_j varies with key content ⇒
non-uniform, key-dependent attention. No base parameterization with W_Q = 0
reproduces this, and the argument extends to W_Q ≠ 0 by a dimension count:
base scores are bilinear forms q_i^T M k_j (rank ≤ d), while B_ij adds terms
outside the bilinear image (b^sal_j is independent of i; S_i − S_j is additively
separable — neither is expressible as a bilinear form in general). ∎

*(Numerical witness: with W_Q=0, base attention row = uniform to machine
precision; activating salience makes the row non-uniform with spread
max−min of order 10^-1 — e.g. 0.23 and 0.13 for two random draws of
W_s (seed-dependent, as expected: the CLAIM is non-uniformity, the
magnitude depends on the draw). Locked by the salience-causality test.)*

### Theorem 3 (Retrieval isometry). ✓ tested
R(θ) preserves norms: ‖R(θ)z‖₂ = ‖z‖₂ for every θ, z, and composes as a group
action: R(a)R(b) = R(a+b).

*Proof.* R acts on ⌊d/2⌋ independent 2-D planes as a standard rotation matrix
[[cosθ,−sinθ],[sinθ,cosθ]], which is orthogonal; block-diagonal orthogonal
matrices are orthogonal; the composition rule is the angle-addition identity
applied per plane. ∎ Consequence: the phase channel cannot inflate or shrink
logit *scale* — it changes only matching geometry. Score magnitudes remain
governed by ‖q‖‖k‖·m_j/√d, so phase cannot cause score explosions.

### Theorem 4 (Bounded modulation envelopes). ✓ tested under saturation
For all inputs and parameter values:
m_j ∈ [(1−β_k)+β_k e^{−c_k·λ_l}, (1−β_k)+β_k e^{+c_k·λ_l}], analogously u_j;
τ_i ∈ [e^{−c_q}, e^{+c_q}] (Q-temp clamp acts in log-space before exp);
|b^sal| ≤ c_s; |b^dist| ≤ c_d·λ_l; |S_i−S_j| ≤ c_f.
Total additive score perturbation is bounded by c_s + c_d·λ_l + c_f
independent of sequence length.

*Proof.* Each factor passes through tanh (range (−1,1)) then an explicit
clamp; exp is monotone, so the envelope endpoints are attained only at clamp
boundaries. Sums of clamped terms are bounded by sums of the clamp radii. ∎
Consequence: no input can drive HLA logits unboundedly far from base logits —
finite-Lipschitz deviation with explicit constants (used in the NaN-robustness
test with weights ×100). A useful reading: since m_j rescales key norms, the
K-gate acts as a learned PER-KEY softmax temperature (bounded by the envelope);
attention entropy, already logged every eval, is the direct observable of this
effect.

### Theorem 6 (Gauge invariance of phase matching). ✓ tested
For any common phase shift c applied per rotation plane,
score(θ_i + c, φ_j + c) = score(θ_i, φ_j): the retrieval score depends only on
the RELATIVE phase φ_j − θ_i.

*Proof.* Per plane, R(θ)q · R(φ)k = q^T R(θ)^T R(φ) k = q^T R(φ−θ) k by the
group action (Theorem 3); adding c to both angles leaves φ−θ unchanged. ∎

Consequence (canonical reading of the mechanism): HLA learns a per-token,
per-head *relative phase code* — the absolute phase is unobservable, exactly
as in physical holography where only phase DIFFERENCES against the reference
beam carry information. This also means the mechanism has a gauge freedom:
solutions differing by a common phase field are functionally identical, which
should be remembered when comparing learned W_phase across seeds (compare
score-level behavior, not raw phase weights).

### Theorem 1b (Gradient identity at initialization). ✓ tested bit-exact
At Θ₀ not only the outputs but the BACKBONE GRADIENTS of HLA and base are
bit-identical: ∂L/∂θ_backbone(HLA, Θ₀) = ∂L/∂θ_backbone(base) exactly.

*Proof sketch.* Every mechanism enters the computation graph multiplied by a
function vanishing at Θ₀ with the mechanism weights (angles = tanh(Wx)·c with
W=0 ⇒ angles ≡ 0 AND ∂angles/∂x = W^T·(...) = 0; the same argument for every
gate). Hence the backward pass through the backbone is the identity graph's
backward pass. ∎ Consequence: the first optimizer step of an HLA run moves the
backbone EXACTLY as the base run does — divergence is driven solely by the
mechanisms' own learning, never by a perturbed backbone signal. This is the
strongest possible form of the "no cold start" guarantee.

### Theorem 7 (Phase as a content-conditioned position shift). ✓ tested
When RoPE and the content phase act on the same rotation planes,
R_rope(w_c * p) . R_phase(theta) = R(w_c * p + theta) per plane c: the learned
phase is algebraically EQUIVALENT to displacing the token's RoPE position by
delta_c = theta_c / w_c (a per-plane, per-token, per-head learned offset).

*Proof.* Immediate from the group action (Theorem 3): rotations about the same
plane commute and compose additively. ∎

Consequence: HLA-phase subsumes "content-dependent position" mechanisms (cf.
CoPE) as a special case while remaining strictly more general - the shift
delta_c may differ per plane (multi-scale, since RoPE frequencies w_c span
octaves), which a scalar position shift cannot express. For the paper this
gives a second, positional reading of the same mechanism: retrieval geometry
(holographic reading, Theorem 6) == learned positional displacement field
(positional reading, this theorem). Both are exact, not analogies.

#### Corollary 7.1 (RoPE and content phase commute). ✓ tested to fp32 eps
Both RoPE and the content phase act as pairwise rotations in the SAME 2D
planes (the chunk pairing of Theorem 8). Rotations within one plane form the
abelian group SO(2), so for every plane c:

    R(w_c * t) . R(theta_c(x)) = R(theta_c(x)) . R(w_c * t) = R(w_c * t + theta_c(x))

Consequence: the "phase before RoPE vs after RoPE" ordering question - a
natural reviewer ablation request - is settled by algebra, not by experiment:
both orders produce bit-identical scores (verified numerically to fp32
rounding, ~1e-6). The only meaningful object is the SUM of angles, which is
exactly the positional-displacement reading of Theorem 7. No ablation arm is
needed, and none is run. ∎

### Theorem 8 (Pairing-scheme equivalence). ✓ tested to 0.0
The chunk pairing used here (plane i = coordinates (x_i, x_{i+d/2})) and the
interleaved pairing of the original RoPE paper (plane i = (x_{2i}, x_{2i+1}))
define ISOMORPHIC model classes: for the fixed permutation P mapping one
layout to the other, rotate_chunk(x, θ) = P^{-1} rotate_inter(P x, θ) for all
x, θ. Any interleaved-parameterized model equals a chunk-parameterized model
whose adjacent weight matrices absorb P (a relabeling of rows/columns), and
vice versa.

*Proof.* Both schemes apply the same 2x2 rotation to disjoint coordinate
pairs; P is exactly the bijection between the two pair layouts, and
permutations commute with per-pair block-diagonal action up to relabeling. ∎

Consequence: the choice is a pure implementation detail with ZERO effect on
the function class, optimization landscape (AdamW is permutation-equivariant),
or any result in this repository. We keep chunk pairing (contiguous-slice
friendly on TPU; no checkpoint migration) and cite this theorem when asked
"why not interleaved like the RoPE paper?".

### Theorem 5 (Non-vanishing first-order signal). ✓ tested
At Θ₀ the loss gradient w.r.t. the PRIMARY mechanism parameters (W_φq, W_φk,
W_gk, W_gv, W_s, W_gd - each entering through tanh with tanh'(0)=1) is
generically non-zero: the identity point is NOT a saddle in these coordinates.
SECONDARY multipliers wake later, in a provable ORDER (see 5.1). Gradient
descent from the shared init satisfies, to first order in the learning rate,

  L_HLA(after step) = L_base(after step) − η‖∇_Θ₀ L‖² + O(η²) ≤ L_base(after step).

*Proof sketch.* The mechanism branch enters every score linearly through
tanh at 0, whose derivative is 1, so ∂L/∂Θ₀ is a non-degenerate linear image
of ∂L/∂scores (zero only on a measure-zero set of data/backbones). The joint
descent direction includes the base direction as a projection; descent along
a superset of coordinates decreases L at least as much at first order. ∎
*(Numerical witness: ‖∇_mech‖² ≈ 2·10⁻⁵ > 0 at init on random data — small,
as expected at the identity point, but strictly positive.)*

#### Theorem 5.1 (Mechanism wake order). ✓ tested (exact zeros verified)
At Θ₀ the gradients of the SECONDARY parameters vanish EXACTLY, by the chain
rule, in a fixed order:
- ∂L/∂r_k = ∂L/∂r_v = 0 (each enters as exp(α·r·tanh(Wx)) with tanh(0)=0:
  the derivative carries a tanh(Wx) factor that is identically zero at Θ₀);
- ∂L/∂s_h = 0 (phase budget multiplies angles that are zero);
- ∂L/∂r_f = 0 while the forget GATE is closed (the derivative carries a
  tanh(W_f x) factor, identically zero at Θ₀). NOTE the converse of the
  naive reading: r_f = forget_range·(1+flex·tanh(W_range_f)) equals 0.1,
  NOT 0, at Θ₀ — identity-at-init is carried by the gate, so ∂L/∂W_f is
  a small NONZERO PRIMARY gradient (measured 9.1e-4 at tiny-Θ₀): W_f
  wakes FIRST, r_f wakes only after the gate opens (code-verified r86);
- ∂L/∂θ_l = 0 (depth profile multiplies zero-valued mechanism outputs).
Measured at Θ₀: primary gradients 1e-5–3e-3; ALL secondary gradients 0.0
exactly (bit-zero, not small). Consequence - a falsifiable training-dynamics
prediction, pre-registered for fig5: gate/phase trajectories move FIRST;
range/budget/depth trajectories stay at zero until their gates leave zero,
then wake. If a secondary parameter moves while its gate is still ~0, our
reading of the parameterization is wrong and must be reported as an anomaly.
(Locked by TestMechanismWakeOrder; discovered in audit round 16.)

---

## 3. What CANNOT be proven — read before skipping intermediate runs

You asked whether theory can license jumping straight to large scale. **It
cannot, and no honest theory can.** Here is precisely where mathematics stops:

1. **Theorems 2 & 5 are about capacity and the first optimization step — not
   about the trained optimum.** Larger function classes can generalize worse
   (extra capacity can be spent on fitting noise), and first-order descent
   says nothing about where non-convex training converges after 50k steps.
   *There is no known technique that predicts the final-loss ordering of two
   Transformer variants from architecture alone* — if there were, nobody
   would run ablations.
2. **Scaling non-transfer is an empirical fact of the field.** Mechanisms
   that help at 100M routinely flatten at 1B+ (and vice versa); this is why
   scaling-law papers exist. A gap measured at one scale is a data point, not
   a law. The *shape* of the gap across 200M → 300M → 700M is the evidence
   that licenses extrapolation — three cheap points buy you the right to
   claim a trend; zero points buy nothing.
3. **What the theory DOES license** (and this is genuinely valuable):
   - Theorem 1 ⇒ HLA runs cannot be *worse than base at step 0*; there is no
     "cold start" risk, at any scale.
   - Theorem 4 ⇒ no stability cliff: HLA cannot NaN where base would not
     (bounded perturbation), so a big run will not be *wasted* by the
     mechanisms exploding.
   - Theorem 5 ⇒ the mechanisms are guaranteed to receive learning signal —
     a large run cannot silently train with dead branches.
   
   Together: **the downside at scale is bounded (≈ compute overhead ~5–7%),
   while the upside is open.** That is a rational basis for *risking* a large
   run — it is not a proof the run will win. Recommended minimal ladder
   before the main run: smoke → 200M (1 seed) → 200M (3 seeds) — that is
   enough to estimate the seed-noise band and the sign of the effect; skipping
   the 300M/700M points trades scientific strength for time, which reviewers
   will notice but which you may decide is worth it.

## 4. Design non-goals (attacks we reject on purpose, with reasons)

Recurring reviewer suggestions we decline - each violates identity-init,
sterility, or the pre-registered protocol. Recorded here so the rebuttal is
a citation, not an improvisation.

**Learned floor beta = sigmoid(W_b x).** The floor (1-beta) is a bounded
GUARANTEE (Theorem 4's envelope), not a modeling choice: making it
input-dependent destroys the analytic bound exactly where it matters (a
learned beta -> 1 removes the floor mid-training, silently). Identity init is
also lost: sigmoid(0) = 0.5 != the shipped beta values, so a zero-initialized
W_b CHANGES the function at step 0. The unbounded-suppression role already
has a dedicated, floor-free channel: the additive salience bias. This
division (bounded multiplicative whisper vs unbounded additive silence) is a
feature, stated syntactically in the unified formula - not an oversight.

**Learned clip bounds.** Clips are numerical guards proven never to bind in
shipped configs (audit E3: slack >= 2x). A learnable guard can learn to bind
- turning a safety mechanism into an uninstrumented modeling knob and
invalidating the E3 audit and the Theorem-4 envelope. The learnable degrees
of freedom already exist one level down (W_range_* with range_flex).

**Auxiliary losses (gate entropy, head orthogonality, disentanglement).**
Any aux term changes the objective, so base-vs-HLA would no longer compare
architectures under the SAME loss - the core sterility invariant. Each term
also ships loss-weight hyperparameters that would need per-arm tuning,
violating the fairness invariant (hparams tuned on base only). We measure
what these losses would enforce (gate_redundancy_statistics, gate_erank,
attention_head_similarity, sat_frac) and let the ablation matrix decide;
pre-registered rules R-A/R-B in EXPERIMENT_CARD convert measurements into
architecture changes for v6.

**Sigmoid mixing instead of exponential.** sigmoid caps mix at 1.0 -
amplification becomes impossible, killing half the mechanism (shipped v2
envelope reaches x2.20 amplification; measured, audit E1). Log-space exp is
symmetric in suppress/amplify and composes additively with the score-space
biases. The floor handles the "smooth near identity" concern.

**Full rotation matrices instead of pairwise.** An unconstrained d x d map
is not an isometry - it can rescale norms, which re-entangles retrieval
geometry with magnitude and destroys Theorem 3 (and the gauge reading of
Theorem 6). Constraining to SO(d) requires expm/Cayley (O(d^2) params,
numerically fragile on TPU bf16). Commuting 2D planes are the standard,
RoPE-compatible parameterization (Corollary 7.1 depends on it); per-plane
budgets already relax uniformity.

**Cross-layer gate conditioning (gate_l = f(gate_{l-1}, x_l)).** Introduces
a second recurrent pathway outside the residual stream: breaks per-layer
identity-init locality (a layer's mechanism is no longer zero-parameter
silent if its input gate is nonzero), complicates KV-cache exactness, and
duplicates what the residual stream already provides - layer l's x ALREADY
contains layer l-1's outputs. The depth profile (lambda_l, learnable temp)
is the sanctioned cross-layer axis: monotone, bounded, identity at init.

**Non-zero / per-layer-scaled gate init.** Zero init IS the sterility
guarantee (bit-exact base at step 0, Theorem 1); any symmetry-breaking init
trades the central claim for an unproven optimization hunch. Symmetry breaks
through the input projections (W_gate rows see different x-statistics from
step 1); mech_grad_* logging verifies gradients are alive (Theorem 5).

**Warmup scheduling of alphas.** Alpha ramps change the effective
architecture during training, confounding "mechanism helps" with "curriculum
helps" - a new axis the ablation matrix cannot isolate. The identity init
already provides a natural learned ramp: mechanisms grow from exact zero at
whatever rate the data demands (observable in gate_abs_mean / angle_std).

## 4b. Relation to gated linear attention, adaptive rotations, and gated softmax

Reviewers who know the linear-attention line will ask whether HLA's phase and
gates re-invent Selective-RoPE-style adaptive rotation or GLA-style gating.
They do not, and the difference is structural, not cosmetic. One sentence per
neighbor (functional differentiation, not citation listing):

| Neighbor | What is modulated | HLA counterpart | Structural difference |
|---|---|---|---|
| GLA / Mamba2 / Gated DeltaNet | recurrent state decay: S_t = G_t (elementwise) S_(t-1) + ... — gating acts on the TIME axis of a linear-attention state | forget arm (FoX-family, separate baseline arm) | HLA's K/V gates have no time axis at all: mix(x) scales a token's key/value amplitude; competition is resolved by softmax, not by state decay |
| Selective/adaptive-RoPE class | positional angle, content-scaled: angle ~ g(x) * f(t) — the kernel of the rotation is still position t | content phase: angle = f(x) only | at t = const their rotation carries zero information; ours is position-free by construction and COMMUTES with standard RoPE (Corollary 7.1): total rotation = f_pos(t) + f_content(x), a clean superposition of "where" and "what" |
| Quantizable Transformers (Bondarenko et al., 2023), gated softmax G(x) | post-softmax output scaling: one gate lets a head "do nothing" (kills outlier workaround) | V-gate (transmission channel) | closest single relative of our V-gate — both act after the softmax on what is transmitted. Differences: (i) purpose — theirs suppresses no-op-head outliers for quantization, ours modulates transmission content per token; (ii) HLA pairs it with a separate PRE-softmax K-gate, and the retrieval/transmission split is exactly what the franken-transplant measures; (iii) identity-init: our gates start bit-invisible (sterile twin), theirs do not |

The load-bearing distinction across all three rows is the same: **HLA
decomposes** — a retrieval-side gate (K, pre-softmax) and a transmission-side
gate (V, post-softmax) are separate learnable channels with a measurable
boundary (the franken experiment transplants one side without the other).
Each neighbor above has a single mechanism doing one job on one axis; none
of them can express, let alone measure, the retrieval/transmission split
that is HLA's central claim. Empirical corollary worth reporting either way:
if Bondarenko-style outlier suppression is a side effect of a trained V-gate,
our activation_outlier_stats (kurtosis / max-over-rms) on the twins will show
it — a free secondary result connecting to the quantization literature.

## 5. Why the FoX-family gate is in the codebase (and why it is not "stealing")

The forget gate is implemented as a **baseline arm**, clearly labeled
FoX-family with citation, OFF by default in every HLA config. Reproducing a
competitor's mechanism inside your own controlled harness is standard,
expected practice (every strong paper re-implements its baselines), and the
opposite of theft: theft is presenting someone's idea as yours; this is
crediting someone's idea and *comparing against it fairly* — the same data,
the same init, the same trainer. It also enables the composition experiment
(HLA + forget), which no one has run. The paper text should say: "as a strong
data-dependent-decay baseline we re-implement a Forgetting-Transformer-style
cumulative gate (Lin et al., 2025) in our sterile harness."
