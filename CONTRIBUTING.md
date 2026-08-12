# Contributing

## Ground rules (they exist for scientific validity, not bureaucracy)

1. **No commit without green tests.** `python -m pytest tests/ -q -W error::UserWarning` must print `349 passed` (or more) before any push. CI enforces this, but run locally first.
2. **Sterility is inviolable.** Any change touching `src/model.py`, `src/make_init.py`, config pairs, or the sampler must keep the identity/parameter-matching/data-order invariants (see `docs/STERILITY.md`). The relevant tests (`test_theory.py::TestTheorem1Identity`, `TestSterility`, `test_make_init.py`) are the gate.
3. **New mechanisms follow the pattern**: identity-initialized (zero params ⇒ exact no-op, add to `reset_hla_identity` + `hla_identity_error` + `HLA_KEY_PATTERNS` + `HLA_NODECAY_MARKERS`), always-created for parameter matching, alpha-gated, clamped, with saturation diagnostics, an SDPA guard if it biases scores, and tests for identity/causality/gradient-flow/clip.
4. **Configs change via the validators.** `validate_configs.py` (pair discipline) and `audit_config_values.py` (numeric sanity) must pass; new allowed diffs require justification in the PR description.
5. **Branch + PR always**, even for maintainers. Patch files are never committed (`*.patch` is gitignored).

## Quick dev setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # full suite, CPU, ~10 s
```

## Where things live
- Architecture: `src/model.py` (single file, deliberately)
- Metric math: `docs/METRICS.md` · Theory: `docs/THEORY.md`
- Experimental pre-registration: `docs/EXPERIMENT_CARD.md`
- HLA-v4 is a fixed archive for provenance; do not develop against it.
