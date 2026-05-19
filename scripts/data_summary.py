"""Write reports/data_summary.md with class balance, token histograms, and samples."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import project_root, read_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402


def _hist(df: pd.DataFrame, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(df["token_count"], bins=30, edgecolor="black")
    plt.xlabel("Token count (cl100k_base)")
    plt.ylabel("Rows")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _section(df: pd.DataFrame, name: str, label_col: str, seed: int) -> list[str]:
    out: list[str] = []
    out.append(f"## {name}")
    out.append(f"- Rows: {len(df)}")
    out.append("- Class balance:")
    counts = df[label_col].value_counts()
    for cls in counts.index:
        out.append(f"  - `{cls}`: {counts[cls]} ({counts[cls] / len(df) * 100:.1f}%)")
    out.append(
        f"- Token length: mean={df['token_count'].mean():.1f}, "
        f"median={df['token_count'].median():.1f}, "
        f"p95={df['token_count'].quantile(0.95):.1f}, "
        f"max={int(df['token_count'].max())}"
    )
    out.append("- Random sample (5 rows, text truncated 200 chars):")
    sub = df.sample(n=min(5, len(df)), random_state=seed)
    for _, row in sub.iterrows():
        text = (row.get("text") or "")[:200].replace("\n", " ")
        out.append(f"  - `{row['id']}` [{row[label_col]}]: {text!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Write reports/data_summary.md")
    parser.add_argument(
        "--fnn",
        type=Path,
        default=project_root() / "data" / "fakenewsnet" / "sample.parquet",
    )
    parser.add_argument(
        "--er",
        type=Path,
        default=project_root() / "data" / "employee_reviews" / "labeled.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root() / "reports" / "data_summary.md",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = get_logger("data_summary", phase="phase1")
    figures_dir = args.out.parent / "figures"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Phase 1 — Data summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")

    if args.fnn.exists():
        fnn_df = read_parquet(args.fnn)
        fnn_fig = figures_dir / "fnn_token_lengths.png"
        _hist(fnn_df, "FakeNewsNet — token length distribution", fnn_fig)
        lines.extend(_section(fnn_df, "FakeNewsNet (sample.parquet)", "label", args.seed))
        lines.append("")
        lines.append(f"![FNN token length histogram](figures/{fnn_fig.name})")
        lines.append("")
        lines.append(
            "- Source: HuggingFace `Jinyan1/PolitiFact` (primary) or KaiDMML/FakeNewsNet CSVs (fallback)."
        )
        lines.append("")
    else:
        log.warning("FNN sample missing at %s; skipping section", args.fnn)

    if args.er.exists():
        er_df = read_parquet(args.er)
        er_fig = figures_dir / "er_token_lengths.png"
        _hist(er_df, "Employee Reviews — token length distribution", er_fig)
        lines.extend(_section(er_df, "Employee Reviews (labeled.parquet)", "label", args.seed))
        lines.append("")
        lines.append(f"![ER token length histogram](figures/{er_fig.name})")
        lines.append("")
        lines.append(
            "- Source: Kaggle `davidgauthier/glassdoor-job-reviews` v1 (manually downloaded)."
        )
        lines.append("")
    else:
        log.warning("ER labeled missing at %s; skipping section", args.er)

    args.out.write_text("\n".join(lines), encoding="utf-8")
    log.warning("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
