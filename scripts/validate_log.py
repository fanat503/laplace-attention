"""Validate a training CSV log for monotonicity and finite critical metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_float(x: str) -> float:
    if x in {"", "nan", "NaN", "inf", "-inf"}:
        return float(x) if x else float("nan")
    return float(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--allow-nan-val", action="store_true",
                    help="legacy no-op (nan val on non-eval rows is now the default contract)")
    ap.add_argument("--require-val-every-row", action="store_true",
                    help="strict mode: every CSV row must carry a finite val_loss")
    args = ap.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    header = None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
            else:
                rows.append(dict(zip(header, row, strict=False)))  # ragged rows allowed pre-crash
    if header is None:
        raise RuntimeError("no CSV header found")
    if not rows:
        raise RuntimeError("no data rows found")

    # Finding #14 (autoresume contract): a resumed run re-logs steps from the
    # last resume checkpoint, so a REAL healthy 200m log contains duplicated
    # step ranges around each "# resumed" seam. Semantics: keep-LAST per step
    # (the post-resume row is the surviving history), then require the
    # deduplicated sequence to be strictly monotonic. Truncated crash rows
    # (missing step/train_loss cells) are tolerated and counted, not fatal -
    # they are the expected fossil of the crash the resume recovered from.
    n_truncated = 0
    n_reseam = 0
    dedup: dict = {}
    order: list = []
    for r in rows:
        if "step" not in r or "tokens_seen" not in r or "train_loss" not in r:
            n_truncated += 1
            continue
        try:
            step = int(r["step"])
        except ValueError:
            n_truncated += 1
            continue
        if step in dedup:
            n_reseam += 1
        else:
            order.append(step)
        dedup[step] = r
    if not dedup:
        raise RuntimeError("no complete data rows found")
    rows = [dedup[s] for s in order]

    prev_step = -1
    prev_tokens = -1
    n_finite_val = 0
    for i, r in enumerate(rows):
        step = int(r["step"])
        tokens = int(r["tokens_seen"])
        if step <= prev_step:
            raise RuntimeError(f"non-monotonic step at row {i}: {step} <= {prev_step}")
        if tokens <= prev_tokens:
            raise RuntimeError(f"non-monotonic tokens at row {i}: {tokens} <= {prev_tokens}")
        prev_step, prev_tokens = step, tokens
        train_loss = parse_float(r["train_loss"])
        if not math.isfinite(train_loss):
            raise RuntimeError(f"non-finite train_loss at row {i}: {r['train_loss']}")
        val_loss = parse_float(r.get("val_loss", "nan"))
        # Contract fix (pre-pilot audit): the trainer logs every `log_every`
        # steps but evaluates every `val_every` steps, so nan val_loss on
        # non-eval rows is NORMAL (a real pilot log has ~4 nan rows per eval
        # row). The old default rejected every real log. Now: nan rows are
        # fine, but every PRESENT val_loss must be finite, and at least one
        # eval row must exist.
        if math.isfinite(val_loss):
            n_finite_val += 1
        elif r.get("val_loss", "") not in ("", "nan", "NaN"):
            raise RuntimeError(f"non-finite val_loss at row {i}: {r.get('val_loss')}")
        elif args.require_val_every_row:
            raise RuntimeError(f"missing val_loss at row {i} (strict mode)")
    if n_finite_val == 0:
        raise RuntimeError("no finite val_loss anywhere - eval never ran")
    print(f"LOG VALID: rows={len(rows)} eval_rows={n_finite_val} "
          f"final_step={prev_step} final_tokens={prev_tokens} "
          f"resume_dups={n_reseam} truncated_rows={n_truncated}")


if __name__ == "__main__":
    main()
