"""Phase 4 aggregation tests — fixture-based, no real data dependency."""

from __future__ import annotations

import pandas as pd

from llm_textcls.reporting.aggregation import (
    CombinedRow,
    baseline_long,
    best_per_model,
    by_prompt_impact,
    combine,
    llm_long,
    master_table_wide,
)


def _baseline_fixture() -> pd.DataFrame:
    """Build a 10-row baseline summary fixture (5 models x 2 datasets)."""
    rows: list[dict] = []
    for model in ["nb", "svm", "rf", "xgboost", "roberta"]:
        for dataset, f1 in [("fakenewsnet", 0.90), ("employee_reviews", 0.65)]:
            rows.append(
                {
                    "model_name": model,
                    "dataset": dataset,
                    "f1_mean": f1,
                    "f1_std": 0.02,
                    "accuracy_mean": f1,
                    "accuracy_std": 0.02,
                    "fit_time_mean": 0.1,
                    "inference_time_mean": 0.05,
                    "n_folds": 5,
                }
            )
    return pd.DataFrame(rows)


def _llm_summary_fixture() -> pd.DataFrame:
    """Build a 22-row mock llm_summary fixture matching expected combos."""
    rows: list[dict] = []
    combos = [
        ("llama-3.1-8b-instant", "", ["ZS", "ZS_CoT", "FS_CoT_RP_NA"]),
        ("llama-3.3-70b-versatile", "", ["ZS", "ZS_CoT", "FS_CoT_RP_NA"]),
        ("openai-gpt-oss-120b", "medium", ["ZS", "ZS_CoT", "FS_CoT_RP_NA"]),
        ("qwen3-32b-reasoning-none", "none", ["ZS"]),
        ("qwen3-32b-reasoning-default", "default", ["ZS"]),
    ]
    for model, effort, prompts in combos:
        for prompt in prompts:
            for dataset, f1 in [("fakenewsnet", 0.80), ("employee_reviews", 0.75)]:
                rows.append(
                    {
                        "logical_name": model,
                        "model_slug": model,
                        "reasoning_effort": effort,
                        "prompt_code": prompt,
                        "dataset": dataset,
                        "n_total": 210 if dataset == "fakenewsnet" else 204,
                        "n_completed": 210 if dataset == "fakenewsnet" else 204,
                        "n_parse_fail": 0,
                        "n_api_fail": 0,
                        "f1_weighted": f1,
                        "accuracy": f1,
                        "median_latency_ms": 1000,
                        "total_input_tokens": 1000,
                        "total_output_tokens": 500,
                        "total_reasoning_tokens": (5000 if effort in {"medium", "default"} else 0),
                        "median_reasoning_tokens": (20 if effort in {"medium", "default"} else 0),
                    }
                )
    return pd.DataFrame(rows)


def test_llm_long_schema() -> None:
    """llm_long emits exactly the CombinedRow columns with correct dtypes."""
    df = llm_long(_llm_summary_fixture())
    assert len(df) == 22
    expected_cols = set(CombinedRow.__dataclass_fields__.keys())
    assert set(df.columns) == expected_cols
    assert (df["source"] == "llm").all()
    assert df["f1_std"].eq(0.0).all()
    assert df["n_records"].dtype.kind in {"i", "u"}


def test_baseline_long_latency_normalisation() -> None:
    """Baseline latency normalisation uses 42 / 41 per-fold test sizes."""
    summary = _baseline_fixture()
    df = baseline_long(summary)
    nb_fnn = df[(df["model"] == "Naive Bayes (TF-IDF)") & (df["dataset"] == "fakenewsnet")].iloc[0]
    nb_er = df[
        (df["model"] == "Naive Bayes (TF-IDF)") & (df["dataset"] == "employee_reviews")
    ].iloc[0]
    # inference_time_mean=0.05s, n_test=42 → 0.05/42*1000 ≈ 1.190 ms
    assert abs(nb_fnn["median_latency_ms"] - (0.05 / 42 * 1000)) < 1e-6
    # n_test=41 → 0.05/41*1000 ≈ 1.219 ms
    assert abs(nb_er["median_latency_ms"] - (0.05 / 41 * 1000)) < 1e-6


def test_combine_row_count() -> None:
    """combine produces exactly 10 baseline + 22 LLM = 32 rows."""
    base = baseline_long(_baseline_fixture())
    llm = llm_long(_llm_summary_fixture())
    combined = combine(base, llm)
    assert len(combined) == 32
    assert (combined["source"] == "baseline").sum() == 10
    assert (combined["source"] == "llm").sum() == 22


def test_best_per_model_picks_max() -> None:
    """best_per_model returns the highest-F1 row per (source, model, dataset)."""
    base = baseline_long(_baseline_fixture())
    llm = llm_long(_llm_summary_fixture())
    # Inject a clear winner for Llama 70B on FNN ZS_CoT.
    llm.loc[
        (llm["model"] == "Llama 3.3 70B")
        & (llm["dataset"] == "fakenewsnet")
        & (llm["prompt"] == "ZS_CoT"),
        "f1_weighted",
    ] = 0.95
    combined = combine(base, llm)
    best = best_per_model(combined)
    llama_fnn = best[(best["model"] == "Llama 3.3 70B") & (best["dataset"] == "fakenewsnet")]
    assert len(llama_fnn) == 1
    assert llama_fnn["prompt"].iloc[0] == "ZS_CoT"
    assert abs(llama_fnn["f1_weighted"].iloc[0] - 0.95) < 1e-9
    # 5 baseline + 5 LLM logical models (incl. 2 Qwen3 variants) x 2 datasets = 20 rows.
    assert len(best) == 20


def test_by_prompt_excludes_qwen() -> None:
    """by_prompt_impact drops Qwen3 variants from the mean."""
    summary = _llm_summary_fixture()
    # Push Qwen3 values way up to detect leakage if Qwen3 were included.
    summary.loc[
        summary["logical_name"].isin({"qwen3-32b-reasoning-none", "qwen3-32b-reasoning-default"}),
        "f1_weighted",
    ] = 0.99
    g = by_prompt_impact(summary)
    # 3 prompts x 2 datasets = 6 rows.
    assert len(g) == 6
    # Mean across non-Qwen LLMs is 0.80 / 0.75 — Qwen3 0.99 must not bias it.
    zs_fnn = g[(g["prompt_code"] == "ZS") & (g["dataset"] == "fakenewsnet")]["f1_mean"].iloc[0]
    assert abs(zs_fnn - 0.80) < 1e-9
    # delta_vs_zs for ZS itself is 0.
    zs_delta = g[(g["prompt_code"] == "ZS") & (g["dataset"] == "fakenewsnet")]["delta_vs_zs"].iloc[
        0
    ]
    assert abs(zs_delta) < 1e-9


def test_master_table_wide_shape() -> None:
    """master_table_wide has the column schema and row count from §6.6."""
    base = baseline_long(_baseline_fixture())
    llm = llm_long(_llm_summary_fixture())
    combined = combine(base, llm)
    best = best_per_model(combined)
    wide = master_table_wide(best)
    expected_cols = [
        "Model",
        "Family",
        "FNN F1",
        "FNN Best Prompt",
        "FNN Latency (ms)",
        "ER F1",
        "ER Best Prompt",
        "ER Latency (ms)",
    ]
    assert list(wide.columns) == expected_cols
    # 5 baselines + 5 LLM logical models (Qwen3-none and Qwen3-default both counted
    # per OD-4) = 10 unique (source, model) wide rows.
    assert len(wide) == 10
