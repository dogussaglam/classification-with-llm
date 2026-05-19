"""Build Employee Reviews unlabeled candidate set + held-out FS rows.

Loads Glassdoor CSV, applies keyword pre-classification, samples balanced
candidates, filters by token length, reserves 6 rows for Few-Shot, and writes
two Parquets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.data import employee_reviews as er  # noqa: E402
from llm_textcls.io import ERUnlabeledRow, load_env, project_root, write_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Employee Reviews unlabeled set")
    parser.add_argument(
        "--raw-dir", type=Path, default=project_root() / "data" / "raw" / "glassdoor"
    )
    parser.add_argument(
        "--out-unlabeled",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "unlabeled.parquet",
    )
    parser.add_argument(
        "--out-held-out",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "held_out_fs.parquet",
    )
    parser.add_argument("--n-per-class", type=int, default=70)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()
    log = get_logger("build_employee_reviews", phase="phase1")

    log.warning("ER build: loading raw CSV from %s", args.raw_dir)
    df = er.load_raw(args.raw_dir)
    log.warning("Raw after dedup + len>=20: %d rows", len(df))

    df = er.filter_by_keywords(df)
    pre_counts = df["keyword_class"].value_counts().to_dict()
    log.warning("After keyword pass: %s", pre_counts)

    cand = er.sample_candidates(df, n_per_class=args.n_per_class, seed=args.seed)
    log.warning("Candidate pool: %d rows", len(cand))

    cand = er.filter_by_length(cand, max_tokens=args.max_tokens)
    post_counts = cand["keyword_class"].value_counts().to_dict()
    log.warning("After length filter (<=%d tokens): %s", args.max_tokens, post_counts)
    for cls in ("remote", "not_remote", "not_mentioned"):
        if post_counts.get(cls, 0) < 50:
            raise RuntimeError(
                f"After length filter, keyword_class '{cls}' has {post_counts.get(cls, 0)} rows (<50). "
                "Increase --n-per-class or relax filters."
            )

    cand = er.mark_held_out_fs(cand, n_per_class=2)
    cand["id"] = cand["id"].astype(str)
    cand["text"] = cand["text"].astype(str)
    cand["keyword_class"] = cand["keyword_class"].astype(str)
    cand["keyword_match_count"] = cand["keyword_match_count"].astype("int64")
    cand["token_count"] = cand["token_count"].astype("int64")
    cand["held_out_fs"] = cand["held_out_fs"].astype(bool)

    held_out = cand[cand["held_out_fs"]].reset_index(drop=True)
    unlabeled = cand[~cand["held_out_fs"]].reset_index(drop=True)

    cols = ["id", "text", "keyword_class", "keyword_match_count", "token_count", "held_out_fs"]
    write_parquet(unlabeled[cols], args.out_unlabeled, ERUnlabeledRow)
    write_parquet(held_out[cols], args.out_held_out, ERUnlabeledRow)

    log.warning(
        "Wrote unlabeled=%d to %s; held_out=%d to %s",
        len(unlabeled),
        args.out_unlabeled,
        len(held_out),
        args.out_held_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
