# Sterility protocol — formal statement

This document states precisely what "sterile comparison" means in this
repository, what is guaranteed by construction, what is verified by tests,
and what remains the experimenter's responsibility. It is written to be
citable from the paper's methodology section.

## Definition

Two runs A (baseline) and B (treatment) are **sterile** iff every difference
in their outcome is attributable to the treatment mechanisms alone. We
decompose this into five invariants:

### I1 — Weight identity at initialization
B's parameter vector at step 0 equals A's on the shared backbone, and B's
mechanism parameters are exactly zero, making B's function bit-identical to A.

*Enforced by*: `make_init.py --shared-backbone` (copies every non-mechanism
tensor, verifies `torch.equal`); `reset_hla_identity()` called after generic
init (generic `apply(_init_weights)` would otherwise re-randomize the
`nn.Linear` gate weights — this exact bug existed in HLA-v4 and was fixed
retroactively); `hla_identity_error() == 0.0` asserted at checkpoint creation
and again at trainer load (`check_init_hla_identity`).
*Verified by tests*: bit-exact logit equality (fp32 and bf16), corrupted
identity rejection.

### I2 — Parameter-count identity
A and B have exactly equal parameter counts: every mechanism module exists in
A as well, disabled via its alpha and excluded from gradient flow.

*Enforced by*: unconditional module creation in `CausalSelfAttention`;
alphas = 0 in A-configs.
*Verified by tests*: equal counts across all config pairs; inactive
parameters receive no gradient (no silent training of "disabled" branches);
inactive parameters excluded from weight decay (decay toward zero IS the
identity state, so decaying mechanism parameters would bias B toward A —
see optimizer note below).

### I3 — Data identity
A and B consume identical token sequences in identical order on every rank.

*Enforced by*: `FixedDataset` (deterministic non-overlapping slices, no
shuffling, integer-dtype validation, content fingerprints);
`EvenShardedSequentialSampler` (equal per-rank lengths, no duplicates,
O(1) exact-suffix resume); loud failure when the dataset is too small for
the sharding configuration.
*Verified by tests*: determinism, non-overlap, resume-suffix equality,
duplicate-free sharding.

### I4 — Configuration discipline
A and B configs differ only in an explicit allow-list of keys
(mechanism alphas, run names/paths, docs).

*Enforced by*: `validate_configs.py` (CI runs it for every pair on every
push). FLOPs-matched pairs may additionally differ in `max_steps`, in the
direction B <= A only.

### I5 — Optimization identity
Same optimizer, schedule, clipping, and hyperparameters; all tuned on A only.
Mechanism parameters are excluded from weight decay (see I2). Learning-rate
and warmup values sit inside standard-practice bands, checked numerically by
`audit_config_values.py` (E9/E10).

## What sterility does NOT cover (experimenter's duty)

- **Compute parity**: B's mechanisms add ~5-7% MACs. Parameter-matched pairs
  hold steps equal (B gets more compute); FLOPs-matched pairs reduce B's
  steps. Report both where possible, plus wall-clock tokens/sec.
- **Statistical power**: run >= 3 seeds (5 if the observed gap is < ~5x the
  seed-noise band); report mean +- std and a paired test (same seeds for A
  and B make the design paired by construction).
- **Hyperparameter fairness over time**: if any knob is later tuned on B,
  the tuning run must be excluded from headline numbers and disclosed.
- **Selection effects**: pre-register the primary configuration before
  launching headline runs; everything else is labeled ablation.

## Threat model (what could still go wrong, and the mitigation)

| Threat | Mitigation |
|---|---|
| Silent init mismatch via `init_strict=false` | init-compat validator rejects shape AND structural-flag mismatches (positional scheme, padded vocab, FFN geometry) |
| Non-determinism from dataloader workers | deterministic sampler; worker seeding; order independent of num_workers |
| bf16 breaking bit-identity on TPU | verified by test (identity holds exactly in bf16) |
| Metric-induced training difference | diagnostics are read-only (`no_grad`, `detach`); entropy computation gated out of training forwards; SVD/interference run master-only at low cadence |
| Crash-recovery divergence | resume restores model+optimizer+step; sampler resumes at the exact sample offset; config-hash compatibility check on resume |

## The seven proven levels (audit rounds 7-21; each is a number, not a claim)

Sterility here is not one check but a LADDER - each level subsumes weaker
failure modes the previous levels cannot see. All seven measured bit-exact
(0.000e+00) in a single session (round 21):

| # | Level | What silent failure it excludes |
|---|---|---|
| 1 | Logits at identity init | mechanism leaks into the forward pass |
| 2 | Backbone gradients | hidden gradient path splits the twins from step 1 |
| 3 | Weights after a real AdamW step | optimizer-state asymmetry (the step-0 contract) |
| 4 | Logits after a bf16 save/load cycle | checkpoint format perturbs the identity |
| 5 | Franken(identity) vs base | the transplant machinery itself is non-sterile |
| 6 | Twin probes (induction, LITM) | eval pipeline treats the twins differently |
| 7 | Knockout deltas at identity | knockout harness has side effects |

Levels 3-5 are, to our knowledge, not reported in comparable architecture
papers; we consider them the minimum bar for any "sterile twin" claim,
because training equals (gradients -> optimizer step)^N and evaluation
equals (checkpoint IO -> probes) - a comparison is only as sterile as its
weakest stage. Regression tests: TestGradientSterility (2, 3),
TestPipelineSterilityEndToEnd (4, 5, 6), knockout zeros in the eval suite (7);
bit-exact resume (TestBitExactResume) extends level 3 across session
boundaries: an interrupted+resumed run equals the continuous run weight-for-
weight and row-for-row in the training log.
