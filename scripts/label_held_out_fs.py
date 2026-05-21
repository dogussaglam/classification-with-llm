"""Two-mode script for labeling the 6 ER held-out few-shot rows.

First run (no flag): write `held_out_to_label.csv` for Doğuş to fill in.
Second run (`--merge`): validate filled CSV, write `held_out_fs_labeled.parquet`.

Phase 1 OD-1 deferred this labeling. Phase 3 needs it before FS prompts run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import ERLabeledRow, project_root, read_parquet, write_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402

VALID_ER_LABELS = ("remote", "not_remote", "not_mentioned")


def _write_to_label_csv(held_path: Path, csv_path: Path) -> None:
    df = read_parquet(held_path)
    if len(df) != 6:
        raise RuntimeError(f"Expected 6 held-out rows, got {len(df)} in {held_path}")
    out = df[["id", "text"]].copy()
    out["label"] = ""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False, encoding="utf-8")
    instructions = f"""
Action required (~5 minutes):
1. Open {csv_path} in Excel/Sheets.
2. Fill the `label` column for each of the 6 rows using the rubric in
   data/employee_reviews/LABELING_INSTRUCTIONS.md (same rubric as Phase 1).
   Valid labels: {", ".join(VALID_ER_LABELS)}
3. Save as: data/employee_reviews/held_out_labeled.csv
4. Re-run: python scripts/label_held_out_fs.py --merge

All 3 classes must be represented at least once across the 6 rows;
otherwise the few-shot prompt cannot build one example per class.
"""
    print(instructions.strip())


def _merge_labeled(
    held_path: Path,
    filled_csv: Path,
    out_parquet: Path,
    log,
) -> int:
    if not filled_csv.exists():
        log.error("Filled CSV not found: %s", filled_csv)
        print(f"ERROR: {filled_csv} not found. Did you fill and save the CSV?")
        return 2

    filled = pd.read_csv(filled_csv, dtype={"id": str, "label": str})
    required = {"id", "label"}
    missing_cols = required - set(filled.columns)
    if missing_cols:
        print(f"ERROR: filled CSV missing columns: {sorted(missing_cols)}")
        return 2

    filled["label"] = filled["label"].fillna("").astype(str).str.strip()
    if (filled["label"].str.len() == 0).any():
        blanks = filled[filled["label"].str.len() == 0]["id"].tolist()
        print(f"ERROR: {len(blanks)} rows have empty `label`: {blanks}")
        return 2

    bad = filled[~filled["label"].isin(VALID_ER_LABELS)]
    if not bad.empty:
        print(
            f"ERROR: {len(bad)} rows have invalid labels: {bad[['id', 'label']].to_dict('records')}"
        )
        print(f"Valid labels: {VALID_ER_LABELS}")
        return 2

    present = set(filled["label"].unique())
    missing = set(VALID_ER_LABELS) - present
    if missing:
        counts = filled["label"].value_counts().to_dict()
        print(
            "ERROR: held-out labels missing class(es) needed for few-shot: "
            f"{sorted(missing)}.\nCurrent counts: {counts}.\n"
            "Edit data/employee_reviews/held_out_labeled.csv so that all 3 "
            f"classes ({', '.join(VALID_ER_LABELS)}) are represented at least "
            "once, then rerun --merge."
        )
        return 2

    held = read_parquet(held_path)
    if len(filled) != len(held):
        print(f"ERROR: CSV has {len(filled)} rows, parquet has {len(held)}.")
        return 2

    joined = filled[["id", "label"]].merge(held, on="id", how="inner", validate="1:1")
    if len(joined) != len(held):
        missing_ids = set(held["id"]) - set(filled["id"])
        print(f"ERROR: ids in CSV don't match parquet. Missing: {sorted(missing_ids)}")
        return 2

    out = joined[["id", "text", "label", "token_count", "held_out_fs"]].copy()
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(str)
    out["token_count"] = out["token_count"].astype("int64")
    out["held_out_fs"] = out["held_out_fs"].astype(bool)
    write_parquet(out, out_parquet, ERLabeledRow)
    log.warning(
        "Wrote %s (%d rows, classes=%s)",
        out_parquet,
        len(out),
        out["label"].value_counts().to_dict(),
    )
    print(f"OK: wrote {out_parquet} with {len(out)} rows.")
    print(f"Per-class counts: {out['label'].value_counts().to_dict()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Label the 6 ER held-out FS rows.")
    parser.add_argument("--merge", action="store_true", help="Merge filled CSV into parquet.")
    parser.add_argument(
        "--held",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "held_out_fs.parquet",
    )
    parser.add_argument(
        "--to-label-csv",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "held_out_to_label.csv",
    )
    parser.add_argument(
        "--filled-csv",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "held_out_labeled.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "held_out_fs_labeled.parquet",
    )
    args = parser.parse_args()

    log = get_logger("label_held_out_fs", phase="phase3")

    if not args.merge:
        _write_to_label_csv(args.held, args.to_label_csv)
        log.warning("Emitted %s for manual labeling.", args.to_label_csv)
        return 0
    return _merge_labeled(args.held, args.filled_csv, args.out, log)


if __name__ == "__main__":
    raise SystemExit(main())
