"""Sanity check labeled.parquet: class balance, token stats, sample rows, histogram."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import project_root, read_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check labeled.parquet")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "labeled.parquet",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=project_root() / "reports" / "figures" / "er_token_lengths.png",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = get_logger("sanity_check_labels", phase="phase1")
    df = read_parquet(args.in_path)
    log.warning("Loaded %s (%d rows)", args.in_path, len(df))

    counts = df["label"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print("Class balance:")
    for cls in counts.index:
        print(f"  {cls:15s} {counts[cls]:5d}  ({pct[cls]:.1f}%)")

    tokens = df["token_count"]
    print("\nToken length stats:")
    print(f"  mean   = {tokens.mean():.1f}")
    print(f"  median = {tokens.median():.1f}")
    print(f"  p95    = {tokens.quantile(0.95):.1f}")
    print(f"  max    = {tokens.max()}")

    print("\nSamples per class (5 each, text truncated to 200 chars):")
    for cls, group in df.groupby("label", sort=True):
        print(f"\n--- {cls} ---")
        sub = group.sample(n=min(5, len(group)), random_state=args.seed)
        for _, row in sub.iterrows():
            print(f"  [{row['id']}] {row['text'][:200]!r}")

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(tokens, bins=30, edgecolor="black")
    plt.xlabel("Token count")
    plt.ylabel("Number of reviews")
    plt.title("Employee Reviews — token length distribution")
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=120)
    plt.close()
    log.warning("Wrote %s", args.out_fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
