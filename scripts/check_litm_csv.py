# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""CI gate: the trainer CSV must contain a REAL (non-NaN) LITM trajectory.

The H4 claim needs pos_10..pos_90 logged DURING training. A silent wiring
regression (import dropped, cadence gated wrong, column renamed) would
produce an all-NaN column - plausible-looking CSV, dead paper figure.

Usage: python scripts/check_litm_csv.py --csv runs/.../train_log_*.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys

REQUIRED = ("pos_10", "pos_30", "pos_50", "pos_70", "pos_90",
            "litm_middle_drop", "litm_worst_frac")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    rows = [r for r in csv.reader(open(args.csv, encoding="utf-8"))
            if r and r[0] and not r[0].startswith("#")]
    if not rows:
        sys.exit(f"FAIL: no rows in {args.csv}")
    header, data = rows[0], rows[1:]
    if not data:
        sys.exit("FAIL: no eval rows logged")
    idx = {c: i for i, c in enumerate(header)}
    missing = [c for c in REQUIRED if c not in idx]
    if missing:
        sys.exit(f"FAIL: CSV missing LITM columns: {missing}")
    # Truncated crash rows are shorter than the header (finding #14): guard
    # every cell access or a single crash fossil turns the CI gate into an
    # IndexError instead of a verdict.
    def cell(row, col):
        i = idx[col]
        return row[i] if i < len(row) else ""

    bad = []
    for col in REQUIRED:
        vals = [float(cell(d, col)) for d in data if cell(d, col) != ""]
        if not any(not math.isnan(v) for v in vals):
            bad.append(col)
    if bad:
        sys.exit(f"FAIL: LITM columns all-NaN (probe never ran?): {bad}")
    last = next((d for d in reversed(data) if cell(d, "pos_10") != ""), data[-1])
    print("LITM trajectory OK:",
          {c: cell(last, c) for c in ("pos_10", "pos_50", "pos_90", "litm_middle_drop")})


if __name__ == "__main__":
    main()
