"""Build the FakeNewsNet 214-row balanced sample.

Tries HuggingFace first, falls back to CSV+scrape on ambiguous labels or load failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/build_fakenewsnet.py` without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.data import fakenewsnet as fnn  # noqa: E402
from llm_textcls.io import FNNRow, load_env, project_root, write_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FakeNewsNet sample")
    parser.add_argument("--hf-dataset-id", default="Jinyan1/PolitiFact")
    parser.add_argument(
        "--raw-dir", type=Path, default=project_root() / "data" / "raw" / "fakenewsnet"
    )
    parser.add_argument(
        "--out", type=Path, default=project_root() / "data" / "fakenewsnet" / "sample.parquet"
    )
    parser.add_argument("--n-per-class", type=int, default=107)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()
    log = get_logger("build_fakenewsnet", phase="phase1")

    log.warning("FNN build: HF route first (%s)", args.hf_dataset_id)
    df = fnn.load_from_huggingface(args.hf_dataset_id, logger=log)
    route = "huggingface"

    if len(df) > 0:
        filtered = fnn.filter_by_length(df, max_tokens=args.max_tokens)
        per_label = filtered["label"].value_counts().to_dict()
        log.warning("HF after length filter: %s", per_label)
        if any(per_label.get(lbl, 0) < args.n_per_class for lbl in ("fake", "real")):
            log.warning(
                "HF route insufficient (need %d/class). Switching to CSV fallback.",
                args.n_per_class,
            )
            df = fnn.load_from_csv(args.raw_dir, logger=log)
            route = "csv_scrape"

    if len(df) == 0:
        log.warning("HF route returned empty. Switching to CSV fallback.")
        df = fnn.load_from_csv(args.raw_dir, logger=log)
        route = "csv_scrape"

    log.warning("FNN route taken: %s (rows=%d)", route, len(df))
    filtered = fnn.filter_by_length(df, max_tokens=args.max_tokens)
    per_label = filtered["label"].value_counts().to_dict()
    log.warning("After length filter: %s", per_label)

    sample = fnn.stratified_sample(filtered, n_per_class=args.n_per_class, seed=args.seed)
    sample = fnn.mark_held_out_fs(sample, n_per_class=2)

    sample = sample[["id", "label", "title", "text", "url", "token_count", "held_out_fs"]]
    sample["id"] = sample["id"].astype(str)
    sample["label"] = sample["label"].astype(str)
    sample["title"] = sample["title"].fillna("").astype(str)
    sample["text"] = sample["text"].astype(str)
    sample["url"] = sample["url"].fillna("").astype(str)
    sample["token_count"] = sample["token_count"].astype("int64")
    sample["held_out_fs"] = sample["held_out_fs"].astype(bool)

    if len(sample) != args.n_per_class * 2:
        raise RuntimeError(f"Expected {args.n_per_class * 2} rows, got {len(sample)}")
    if (sample["token_count"] > args.max_tokens).any():
        raise RuntimeError("token_count > max_tokens after filter — bug")

    write_parquet(sample, args.out, FNNRow)
    log.warning(
        "Wrote %s (%d rows, %d held-out, route=%s)",
        args.out,
        len(sample),
        int(sample["held_out_fs"].sum()),
        route,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
