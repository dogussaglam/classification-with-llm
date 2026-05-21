"""Glob results/llms/*.jsonl into one DataFrame; write parquet + status report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.evaluation.llm_metrics import summarise_all  # noqa: E402
from llm_textcls.io import project_root  # noqa: E402
from llm_textcls.llms.runner import LLMCallResult  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402

EXPECTED_PER_DATASET = {"fakenewsnet": 210, "employee_reviews": 204}
EXPECTED_TOTAL = 4554


def _load_jsonl_files(jsonl_dir: Path, log) -> pd.DataFrame:
    rows: list[dict] = []
    paths = sorted(jsonl_dir.glob("*.jsonl"))
    log.warning("Found %d JSONL files under %s", len(paths), jsonl_dir)
    schema_cols = [f.name for f in fields(LLMCallResult)]
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for ln_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Bad JSON in %s:%d: %s", p, ln_no, e)
                    continue
                for col in schema_cols:
                    obj.setdefault(col, "" if col != "latency_ms" else 0)
                rows.append(obj)
    return pd.DataFrame(rows, columns=schema_cols)


def _write_status_report(
    summary_df: pd.DataFrame,
    df_all: pd.DataFrame,
    out_path: Path,
    log,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 3 — LLM benchmark status\n")
    lines.append(f"\nTotal rows aggregated: {len(df_all)}")
    lines.append(f"Expected: ~{EXPECTED_TOTAL} (≥ {int(EXPECTED_TOTAL * 0.92)} acceptable)\n")

    if summary_df.empty:
        lines.append("\n**No combo rows.** Did the runner write any JSONL?\n")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("\n## Per-combination summary\n")
    lines.append(
        "| Logical name | Prompt | Dataset | Total | OK | Parse fail | "
        "API fail | F1 weighted | Accuracy | Median ms | Reasoning tokens |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    flagged: list[str] = []
    for _, r in summary_df.sort_values(["logical_name", "prompt_code", "dataset"]).iterrows():
        expected = EXPECTED_PER_DATASET.get(str(r["dataset"]), 0)
        success_rate = (
            (r["n_completed"] - r["n_parse_fail"]) / r["n_completed"] if r["n_completed"] else 0.0
        )
        coverage = r["n_total"] / expected if expected else 0.0
        f1 = r["f1_weighted"]
        acc = r["accuracy"]
        lines.append(
            f"| {r['logical_name']} | {r['prompt_code']} | {r['dataset']} | "
            f"{r['n_total']} | {r['n_completed']} | {r['n_parse_fail']} | "
            f"{r['n_api_fail']} | {f1:.4f} | {acc:.4f} | "
            f"{int(r['median_latency_ms'])} | {int(r['total_reasoning_tokens'])} |"
        )
        if coverage < 0.95:
            flagged.append(
                f"- coverage {coverage:.1%} (<95%) on "
                f"{r['logical_name']} / {r['prompt_code']} / {r['dataset']}"
            )
        if success_rate < 0.95 and r["n_completed"] > 0:
            flagged.append(
                f"- parse-success {success_rate:.1%} (<95%) on "
                f"{r['logical_name']} / {r['prompt_code']} / {r['dataset']}"
            )

    if flagged:
        lines.append("\n## Flagged combos\n")
        lines.extend(flagged)
    else:
        lines.append("\nAll combos ≥95% coverage and parse-success.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.warning("Wrote status report: %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Phase 3 LLM JSONL results.")
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=project_root() / "results" / "llms",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=project_root() / "results" / "llms" / "all_results.parquet",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=project_root() / "reports" / "benchmark_status.md",
    )
    args = parser.parse_args()

    log = get_logger("aggregate_llm_results", phase="phase3")
    df = _load_jsonl_files(args.jsonl_dir, log)
    if df.empty:
        print("ERROR: no rows aggregated. Did the runner write JSONLs?")
        return 2

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    log.warning("Wrote %s (%d rows)", args.out_parquet, len(df))

    summary_df = summarise_all(df)
    _write_status_report(summary_df, df, args.out_report, log)
    print(f"OK aggregated {len(df)} rows from {args.jsonl_dir}")
    print(f"Combos: {len(summary_df)}; report: {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
