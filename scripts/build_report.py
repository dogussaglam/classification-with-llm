"""Phase 4 — build master parquets, render figures, write results.md/.tex.

Idempotent. Re-run after any data change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import tabulate  # noqa: F401  # required by pandas.DataFrame.to_markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import project_root  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402
from llm_textcls.reporting import aggregation, figures, markdown  # noqa: E402

EXPECTED_LLM_ROWS_MIN = 4200
EXPECTED_BASELINE_ROWS = 10
EXPECTED_COMBO_ROWS = 22


def _load_inputs(root: Path, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all_results.parquet + summary.parquet with runtime gates."""
    llm_path = root / "results" / "llms" / "all_results.parquet"
    baseline_path = root / "results" / "baselines" / "summary.parquet"
    if not llm_path.exists():
        raise SystemExit(f"missing: {llm_path}")
    if not baseline_path.exists():
        raise SystemExit(f"missing: {baseline_path}")
    df_llm = pd.read_parquet(llm_path)
    df_base = pd.read_parquet(baseline_path)
    if len(df_llm) < EXPECTED_LLM_ROWS_MIN:
        raise SystemExit(
            f"all_results.parquet has {len(df_llm)} rows; expected at least {EXPECTED_LLM_ROWS_MIN}"
        )
    if len(df_base) != EXPECTED_BASELINE_ROWS:
        raise SystemExit(
            f"summary.parquet has {len(df_base)} rows; expected {EXPECTED_BASELINE_ROWS}"
        )
    log.warning("Loaded LLM rows=%d, baseline rows=%d", len(df_llm), len(df_base))
    return df_llm, df_base


def _write_parquets(
    combined: pd.DataFrame,
    by_model: pd.DataFrame,
    by_prompt: pd.DataFrame,
    master_wide: pd.DataFrame,
    out_dir: Path,
    log,
) -> None:
    """Persist the 4 master parquets under results/master/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_dir / "combined_long.parquet", index=False)
    by_model.to_parquet(out_dir / "by_model.parquet", index=False)
    by_prompt.to_parquet(out_dir / "by_prompt.parquet", index=False)
    master_wide.to_parquet(out_dir / "master_table.parquet", index=False)
    log.warning(
        "Wrote parquets: combined_long=%d, by_model=%d, by_prompt=%d, master_table=%d",
        len(combined),
        len(by_model),
        len(by_prompt),
        len(master_wide),
    )


def _render_figures(
    by_model: pd.DataFrame,
    combined: pd.DataFrame,
    llm_summary_df: pd.DataFrame,
    fig_dir: Path,
    log,
) -> dict[str, Path]:
    """Render all six figures and return their paths."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fig1": fig_dir / "fig1_f1_master.png",
        "fig2": fig_dir / "fig2_pareto.png",
        "fig3": fig_dir / "fig3_axis_a_scaling.png",
        "fig4": fig_dir / "fig4_axis_b_reasoning.png",
        "fig5": fig_dir / "fig5_prompt_effect.png",
        "fig6": fig_dir / "fig6_reasoning_cost.png",
    }
    figures.fig1_f1_master(by_model, paths["fig1"])
    log.warning("Rendered %s", paths["fig1"].name)
    figures.fig2_pareto(combined, paths["fig2"])
    log.warning("Rendered %s", paths["fig2"].name)
    figures.fig3_axis_a_scaling(llm_summary_df, paths["fig3"])
    log.warning("Rendered %s", paths["fig3"].name)
    figures.fig4_axis_b_reasoning(llm_summary_df, paths["fig4"])
    log.warning("Rendered %s", paths["fig4"].name)
    figures.fig5_prompt_effect(llm_summary_df, paths["fig5"])
    log.warning("Rendered %s", paths["fig5"].name)
    figures.fig6_reasoning_cost(llm_summary_df, paths["fig6"])
    log.warning("Rendered %s", paths["fig6"].name)
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing figures after render: {missing}")
    return paths


def _validate_results_md(text: str) -> None:
    """Sanity-check that results.md contains the load-bearing references."""
    required = [
        "Master F1 table",
        "Replication validation",
        "figures/fig1_f1_master.png",
        "figures/fig2_pareto.png",
        "figures/fig3_axis_a_scaling.png",
        "figures/fig4_axis_b_reasoning.png",
        "figures/fig5_prompt_effect.png",
        "figures/fig6_reasoning_cost.png",
    ]
    missing = [r for r in required if r not in text]
    if missing:
        raise SystemExit(f"results.md missing required sections/links: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 — build master report.")
    parser.add_argument(
        "--skip-figs",
        action="store_true",
        help="Skip PNG re-render (faster iteration on markdown).",
    )
    args = parser.parse_args()

    log = get_logger("build_report", phase="phase4")
    root = project_root()

    print("Phase 4 — building report...")
    df_llm, df_base = _load_inputs(root, log)
    print(f"  loaded all_results.parquet ({len(df_llm)} rows)")
    print(f"  loaded baseline summary.parquet ({len(df_base)} rows)")

    llm_summary_df = aggregation.llm_summary(df_llm)
    if len(llm_summary_df) != EXPECTED_COMBO_ROWS:
        raise SystemExit(
            f"llm_summary has {len(llm_summary_df)} rows; expected {EXPECTED_COMBO_ROWS}"
        )
    log.warning("Built llm_summary: %d rows", len(llm_summary_df))
    print(f"  llm_summary: {len(llm_summary_df)} combos")

    base_long = aggregation.baseline_long(df_base)
    llm_long_df = aggregation.llm_long(llm_summary_df)
    combined = aggregation.combine(base_long, llm_long_df)
    expected_combined = EXPECTED_BASELINE_ROWS + EXPECTED_COMBO_ROWS
    if len(combined) != expected_combined:
        raise SystemExit(f"combined_long has {len(combined)} rows; expected {expected_combined}")
    print(f"  combined_long: {len(combined)} rows")

    by_model = aggregation.best_per_model(combined)
    by_prompt = aggregation.by_prompt_impact(llm_summary_df)
    master_wide = aggregation.master_table_wide(by_model)
    print(
        f"  by_model: {len(by_model)} rows, "
        f"by_prompt: {len(by_prompt)} rows, "
        f"master_table: {len(master_wide)} rows"
    )

    _write_parquets(combined, by_model, by_prompt, master_wide, root / "results" / "master", log)
    print("  wrote 4 parquets to results/master/")

    fig_dir = root / "reports" / "figures"
    if args.skip_figs:
        paths = {
            "fig1": fig_dir / "fig1_f1_master.png",
            "fig2": fig_dir / "fig2_pareto.png",
            "fig3": fig_dir / "fig3_axis_a_scaling.png",
            "fig4": fig_dir / "fig4_axis_b_reasoning.png",
            "fig5": fig_dir / "fig5_prompt_effect.png",
            "fig6": fig_dir / "fig6_reasoning_cost.png",
        }
        print("  skipped figure rendering (--skip-figs)")
    else:
        paths = _render_figures(by_model, combined, llm_summary_df, fig_dir, log)
        print(f"  rendered 6 figures to {fig_dir}")

    reports_dir = root / "reports"
    markdown.write_results_table_md(master_wide, reports_dir / "results_table.md")
    markdown.write_results_table_tex(master_wide, reports_dir / "results_table.tex")
    markdown.write_results_md(
        by_model=by_model,
        by_prompt=by_prompt,
        combined=combined,
        master_wide=master_wide,
        figure_paths=paths,
        output_path=reports_dir / "results.md",
    )
    text = (reports_dir / "results.md").read_text(encoding="utf-8")
    _validate_results_md(text)
    log.warning(
        "Wrote results.md (%d chars), results_table.md, results_table.tex",
        len(text),
    )
    print("  wrote results.md, results_table.md, results_table.tex")

    fnn_winner = (
        by_model[by_model["dataset"] == "fakenewsnet"]
        .sort_values("f1_weighted", ascending=False)
        .iloc[0]
    )
    er_winner = (
        by_model[by_model["dataset"] == "employee_reviews"]
        .sort_values("f1_weighted", ascending=False)
        .iloc[0]
    )
    print("\nSummary:")
    print(
        f"  FNN winner: {fnn_winner['model']} ({fnn_winner['prompt'] or '—'}) "
        f"F1={fnn_winner['f1_weighted']:.3f}"
    )
    print(
        f"  ER  winner: {er_winner['model']} ({er_winner['prompt'] or '—'}) "
        f"F1={er_winner['f1_weighted']:.3f}"
    )
    print(f"  parquets at: {root / 'results' / 'master'}")
    print(f"  figures at:  {fig_dir}")
    print(f"  report at:   {reports_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
