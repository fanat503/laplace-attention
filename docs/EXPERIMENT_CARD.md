# Experiment card — pre-registration

This file is the pre-registered experimental plan. It is committed BEFORE the
headline runs; any deviation must be documented in the "Deviations" section
with a date and reason. Purpose: protection against post-hoc configuration
selection (see docs/STERILITY.md, "Selection effects").

## Primary hypothesis
H1: The full HLA mechanism set (phase + K/V gates + salience + distance),
trained from a sterile shared init against a parameter-matched ablated base,
achieves lower validation loss at matched tokens, and the gap does not shrink
with model scale over 200M -> 300M -> 700M.
(The Q-temperature mechanism ships OFF in the primary HLA recipe; it is
evaluated as its own single-factor arm `qtemp` in the ablation matrix and
joins the full recipe only if that arm shows a positive effect — a
pre-registered decision, not post-hoc selection.)

Secondary (mechanistic) hypotheses:
H2: qk_interference decreases vs base while ov_interference is preserved
    (qk_ov_separation increases).
H3: distractor_margin improves faster than base during training.
H4: the positional recall curve (positional_recall_curve probe) is FLATTER
    for HLA than for base at matched tokens (litm_middle_drop lower,
    litm_worst_frac closer to 1) - the Lost-in-the-Middle reading of the
    distance/salience mechanisms. Pre-registered BEFORE any run has ever
    produced this curve.
H5 (CAUSAL, the Oral experiment): transplanting the trained HLA retrieval
    side into the trained base twin (scripts/causal_patch.py, transplant=
    retrieval: Q/K rows + phase + K-gate + score biases + q-temp; V-side and
    MLPs stay base) recovers a majority of the HLA-over-base gap on retrieval
    probes (induction, distractor_margin). Decision rule: franken closes
    >50% of the gap => the retrieval-geometry mechanism CAUSES the gain;
    <20% => the gain lives in the transmission path and the mechanistic
    narrative must be revised (reported either way). Ladder position: run
    immediately after the first trained 200M pair.

    H5-A (reverse / necessity, pre-registered BEFORE any pair was trained):
    the mirror transplant (HLA body + base retrieval, --direction reverse)
    must COLLAPSE the gain: reverse closure <50% (franken_rev falls toward
    base) if retrieval geometry is necessary. Forward >50% AND reverse ALSO
    >50% would mean the V/MLP path compensates - the boundary story is then
    revised and reported honestly. Both directions ship in one run
    (--direction both).

    H4-P (linear-access probe, pre-registered before any trained pair):
    scripts/train_probe.py decodes the needle identity from the residual
    stream at the query position across depths. Reading: HLA's
    probe_litm_gap (edge - middle accuracy at best layer) is closer to 0
    than base's => mid-context info is genuinely more linearly accessible.
    Runs post-hoc on any checkpoint; report alongside pos_XX curves.

    H5-B (two-level evidence, pre-registered likewise): gap closure is
    computed on FIVE metrics - the four behavioral probes plus
    snr_needle_last (activation-level attention SNR). P(B)-style probes can
    in principle be moved by the transmission path; snr_needle moves only
    if the score geometry itself concentrates on the target. A franken that
    inherits both the behavioral gap AND the SNR gap closes the causal
    chain on two independent measurement levels.

## Active mechanism sets per config (B6: capacity vs default)

The codebase implements SEVEN mechanisms; shipped training configs activate a
deliberate SUBSET. "Seven mechanisms" describes model capacity, not the
default treatment arm - stated here explicitly so no reader infers that the
headline number used everything at once.

| Config family | Active in HLA arm | Off (identity, parameter-matched) |
|---|---|---|
| 200m v1 (`200m_hla_s42`) | phase, K/V gates, distance | salience, forget, qtemp, adaptivity extras |
| 200m v2 (`200m_hla_v2_s42`, primary) | phase, K/V gates, distance, salience | forget, qtemp, adaptivity extras |
| Ablation matrix arms | exactly one mechanism per single-factor arm | everything else |
| `forget` arm | forget only (FoX baseline) | all HLA mechanisms |
| `qtemp` arm | qtemp only | all others |

## Two comparison axes (pre-registered; both reported)

| Axis | Pair | Question it answers | Status |
|---|---|---|---|
| **Token-matched (PRIMARY for attribution)** | base@17900 vs hla@17900 (Kaggle); base@20000 vs hla@20000 (TPU) | "Is the gain caused by the mechanisms?" - everything except mechanisms is identical (same tokens, same order, same init) | primary |
| **Iso-FLOPs (PRIMARY for efficiency)** | Kaggle: base@17900 vs hla@16740 (compute matched to 0.006%); TPU strongest control: base@21386 vs hla@20000 (base-longer: base receives HLA's +6.93% compute as extra steps) | "Does HLA beat a base given equal compute?" | secondary |

Neither axis alone survives review: token-matched leaves the equal-compute
question open; iso-FLOPs pairs see different data, so attribution is
confounded by construction. The TPU base-longer run additionally yields a
3-point base compute curve (17900-eq / 20000 / 21386) against hla@20000 -
a mini Accuracy-vs-Compute figure.

Dataset-size note (pre-registered): the Kaggle working-dir limit (19.5 GB)
caps the Kaggle dataset at 4.7B stored tokens => 200M Kaggle runs use
max_steps 17900 (4.69B tokens). The month-TPU dataset (28B+) removes this
cap; TPU 200M runs use the original 20000 steps. Both are reported as-is.

**Budget amendment (2026-08, pre-run, after v5e-8 speed measurement):** the
measured steady-state speed (V4 smoke: 0.53 steps/s at b=1/accum=16,
262,144 tokens/update) does not fit 17900 steps into one 9h batch session.
The headline Kaggle pair therefore uses **kaggle_200m_{base,hla}_9h_s42:
max_steps 15000 = 3.93B tokens = 18.1 tok/param** (still Chinchilla-range),
identical for both twins; autoresume (resume_every=500) covers slowdowns.
Pre-registered fallback if V6 measures < 0.48 steps/s: cut BOTH twins to
13000 steps (3.4B tokens) — never one twin alone. The 17900/20000 plans
remain for the month-TPU stage.

## Pre-registered decision rules for architecture simplification

The mechanism set is the UNION of hypotheses, not the claim that every part
is necessary. Two rules, committed before any headline run, convert the
ablation matrix into pruning decisions (protection against both kitchen-sink
criticism and post-hoc cherry-picking):

- **R-A (arm pruning)**: any mechanism whose single-factor arm shows no gain
  AND whose leave-one-out arm shows no loss (both within 2x seed std at 200M)
  is dropped from the recommended recipe; the paper reports the minimal set.
- **R-B (gate merging)**: `gate_redundancy_statistics` (pairwise Pearson
  between the K/V/salience/distance gates, logged by checkpoint analysis) is
  the pre-registered merge criterion: any gate pair with |corr| > 0.9 across
  seeds at 200M is merged into a shared projection in v6. We measure
  redundancy instead of penalizing it - an auxiliary orthogonality loss
  would change the training objective and break the sterile base-vs-HLA
  comparison (same-objective invariant I5).

## Pre-registered decisions (locked before headline runs)
| Decision | Value | Rationale |
|---|---|---|
| Primary config | `200m_hla_v2_s42` recipe (aggressive envelope + salience) | v1 recipe's multiplicative floor caps suppression at x0.77 (see CONFIG_AUDIT) |
| Primary metric | token-weighted final val loss; tie-breaker: val loss at matched wall-clock | |
| Seeds | 42, 43, 44 (add 45, 46 if gap < 5x seed std) | seed-noise band measured at ~0.002 init loss |
| Statistical test | paired t-test across seeds (same seed = same init pair) | paired by construction |
| Ablation matrix | 14 arms x 3 seeds via `make_ablation_configs.py` | single-factor discipline enforced by tests |
| Hyperparameters | tuned on BASE only (standard recipes); never adjusted per-arm | fairness invariant I5 |
| Exclusion rule | a run is excluded only for infrastructure failure (crash, data corruption), never for its result; exclusions logged here | |

## Run ladder (in order; each gates the next)
1. smoke (10 steps, both variants) — infrastructure alive
2. pilot (1000 steps) — loss curves sane, diagnostics populated
3. 200M v1 pair, seed 42 — reproduce the historical v3/v4 gap in the sterile harness
4. 200M v2 pair, seed 42 — primary recipe first reading
5. 200M ablation matrix (14 arms x 3 seeds)
6. 300M FLOPs-matched pair (3 seeds)
7. 700M pair (>= 2 seeds, budget permitting)
8. Downstream evals (lm-eval-harness) on best checkpoints
9. Headline scale (compute-dependent; only after 5-7 confirm the trend)

## Reporting commitments
- Report parameter-matched AND FLOPs-matched comparisons where both exist.
- Report wall-clock tokens/sec for both variants (mechanism overhead honesty).
- Report all seeds (no seed selection), mean +- std, and the paired test.
- Negative/flat results at any rung are reported, not hidden.

## Deviations
| Date | Deviation | Reason |
|---|---|---|
| (none yet) | | |

## Post-registration additions (2026-08-11, before any headline run)

Added to the evaluation suite BEFORE the first 200m pair exists (so these
are pre-registered too, with calibration tests where applicable):

- **Passkey retrieval at full window** (`scripts/eval_passkey.py`): the
  external-anchor task (Mohtashami & Jaggi). Pre-registered reading: HLA's
  passkey_middle_vs_edge sag is no worse than base's, and mean exact-match
  is >= base's at equal tokens. Calibration: 0.000 at random init (tested).
- **H5 power fields**: every gap-closure record carries
  min_detectable_gap_z3 / gap_over_noise_z / powered. Pre-registered rule:
  closures with powered = 0 are reported but carry NO causal claim.
- **Per-head induction census** (fig10): descriptive, no threshold —
  reported for both twins at the final checkpoint.
- **Mechanism wake order**: analytical property (∂mix/∂range = 0 at
  gate = 0). Pre-registered check on fig5: range-parameter trajectories
  lag gate trajectories; if they do NOT (ranges move while gates are
  ~0), that falsifies our reading of the parameterization and must be
  reported as an anomaly.
- **Second seed pair** (kaggle_200m_{base,hla}_9h_s43): same protocol,
  seed 43; pre-registered use: sign-stability check of the headline
  deltas, not a new hypothesis.
