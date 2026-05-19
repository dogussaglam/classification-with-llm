"""Merge labeled batch CSVs into the final labeled.parquet.

Globs every ``batch_*_labeled.csv`` (including supplemental batches added via
``scripts/generate_supplemental_batch.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import ERLabeledRow, project_root, read_parquet, write_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402

VALID_LABELS = {"remote", "not_remote", "not_mentioned"}
GATE_MIN_TOTAL = 180
GATE_MIN_PER_CLASS = 40


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge labeled batch CSVs into labeled.parquet")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "to_label",
    )
    parser.add_argument(
        "--unlabeled",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "unlabeled.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "labeled.parquet",
    )
    args = parser.parse_args()

    log = get_logger("merge_labeled", phase="phase1")

    batch_paths = sorted(args.in_dir.glob("batch_*_labeled.csv"))
    if not batch_paths:
        raise FileNotFoundError(
            f"No batch_*_labeled.csv files in {args.in_dir}. "
            "Label the batch_*.csv files first (see LABELING_INSTRUCTIONS.md)."
        )

    parts: list[pd.DataFrame] = []
    for p in batch_paths:
        sub = pd.read_csv(p)
        missing_cols = {"id", "text", "label"} - set(sub.columns)
        if missing_cols:
            raise ValueError(f"{p} missing columns: {sorted(missing_cols)}")
        sub["label"] = sub["label"].astype(str).str.strip()
        blank = sub["label"].isin(["", "nan", "None"]) | sub["label"].isna()
        if blank.any():
            bad_idx = sub.index[blank].tolist()
            raise ValueError(f"{p}: empty/NaN label at rows {bad_idx}")
        invalid = ~sub["label"].isin(VALID_LABELS)
        if invalid.any():
            bad = sub.loc[invalid, ["id", "label"]].to_dict(orient="records")
            raise ValueError(f"{p}: invalid labels (must be one of {sorted(VALID_LABELS)}): {bad}")
        parts.append(sub[["id", "label"]])
        log.warning("Loaded %s (%d rows)", p, len(sub))

    labels = pd.concat(parts, ignore_index=True)
    dup_ids = labels[labels.duplicated(subset=["id"], keep=False)]
    if not dup_ids.empty:
        log.warning("Duplicate ids across batches; keeping first: %s", dup_ids["id"].tolist())
        labels = labels.drop_duplicates(subset=["id"], keep="first")

    unlabeled = read_parquet(args.unlabeled)
    merged = unlabeled.merge(labels, on="id", how="inner", validate="1:1")

    out = merged[["id", "text", "label", "token_count", "held_out_fs"]].copy()
    out["id"] = out["id"].astype(str)
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(str)
    out["token_count"] = out["token_count"].astype("int64")
    out["held_out_fs"] = out["held_out_fs"].astype(bool)

    total = len(out)
    counts = out["label"].value_counts().to_dict()
    log.warning("Merged: %d rows, per-class counts: %s", total, counts)

    if total < GATE_MIN_TOTAL:
        deficit = GATE_MIN_TOTAL - total
        raise RuntimeError(
            f"Validation gate FAILED: total {total} < {GATE_MIN_TOTAL} (need {deficit} more rows). "
            "Run: python scripts/generate_supplemental_batch.py --class <name> --n <count>"
        )
    deficits = {
        cls: GATE_MIN_PER_CLASS - counts.get(cls, 0)
        for cls in VALID_LABELS
        if counts.get(cls, 0) < GATE_MIN_PER_CLASS
    }
    if deficits:
        recovery_cmds = "; ".join(
            f"python scripts/generate_supplemental_batch.py --class {cls} --n {n}"
            for cls, n in deficits.items()
        )
        raise RuntimeError(
            f"Validation gate FAILED: per-class minimum {GATE_MIN_PER_CLASS} violated. "
            f"Deficits: {deficits}. Recovery: {recovery_cmds}"
        )

    write_parquet(out, args.out, ERLabeledRow)
    log.warning("Wrote %s (%d rows, gate PASSED)", args.out, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
