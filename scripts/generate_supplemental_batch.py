"""Generate a supplemental labeling batch when a class falls below the gate.

Reads the full Glassdoor CSV, re-runs keyword classification, and samples N
additional rows of the requested keyword_class that are NOT already present in
any existing batch_*.csv or batch_*_labeled.csv. Writes
``data/employee_reviews/to_label/batch_supplemental_<class>.csv``.

merge_labeled.py globs ``batch_*_labeled.csv``, so the supplemental batch is
included automatically on the next merge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.data import employee_reviews as er  # noqa: E402
from llm_textcls.io import project_root  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402

VALID_CLASSES = {"remote", "not_remote", "not_mentioned"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a supplemental labeling batch")
    parser.add_argument("--class", dest="cls", required=True, choices=sorted(VALID_CLASSES))
    parser.add_argument(
        "--n", type=int, required=True, help="Number of supplemental rows to sample"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=project_root() / "data" / "raw" / "glassdoor"
    )
    parser.add_argument(
        "--to-label-dir",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "to_label",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = get_logger("generate_supplemental_batch", phase="phase1")

    args.to_label_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = er.existing_ids_in_batches(args.to_label_dir, logger=log)
    for parquet_name in ("unlabeled.parquet", "held_out_fs.parquet"):
        p = args.to_label_dir.parent / parquet_name
        if p.exists():
            existing_ids |= set(pd.read_parquet(p, columns=["id"])["id"].astype(str))
    log.warning("Existing ids across all batches + parquets: %d", len(existing_ids))

    log.warning("Loading raw CSV from %s", args.raw_dir)
    df = er.load_raw(args.raw_dir)
    df = er.filter_by_keywords(df)
    df = df[df["keyword_class"] == args.cls].copy()
    df = df[~df["id"].isin(existing_ids)].reset_index(drop=True)
    log.warning("Available '%s' rows after exclusion: %d", args.cls, len(df))

    df = er.filter_by_length(df, max_tokens=args.max_tokens)
    if len(df) < args.n:
        raise RuntimeError(
            f"Only {len(df)} '{args.cls}' rows available after exclusion + length filter; "
            f"requested {args.n}."
        )

    sample = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    out = sample[["id", "text"]].copy()
    out["label"] = ""
    out_path = args.to_label_dir / f"batch_supplemental_{args.cls}.csv"
    out.to_csv(out_path, index=False)
    log.warning(
        "Wrote %s (%d rows). Label it then re-run scripts/merge_labeled.py", out_path, len(out)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
