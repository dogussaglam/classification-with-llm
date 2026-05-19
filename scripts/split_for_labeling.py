"""Split unlabeled.parquet into N CSV batches for parallel manual labeling."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import project_root, read_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Split unlabeled.parquet into batch CSVs")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "unlabeled.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "to_label",
    )
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = get_logger("split_for_labeling", phase="phase1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_parquet(args.in_path)
    if "held_out_fs" in df.columns and df["held_out_fs"].any():
        log.warning(
            "Input has %d held_out_fs rows; these will be excluded from labeling batches.",
            int(df["held_out_fs"].sum()),
        )
        df = df[~df["held_out_fs"]].copy()

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    chunk_size = math.ceil(n / args.n_batches)
    for i in range(args.n_batches):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        chunk = df.iloc[start:end]
        out = chunk[["id", "text"]].copy()
        out["label"] = ""
        out_path = args.out_dir / f"batch_{i + 1}.csv"
        out.to_csv(out_path, index=False)
        log.warning("Wrote %s (%d rows)", out_path, len(out))

    log.warning("Split complete: %d batches, %d rows total", args.n_batches, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
