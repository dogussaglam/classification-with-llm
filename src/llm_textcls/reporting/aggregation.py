"""Phase 4 aggregation — join baselines + LLMs into long/wide reporting tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from llm_textcls.evaluation.llm_metrics import summarise_all

# Paper reference F1 scores (Kostina et al. 2025) for the replication-validation column.
PAPER_BASELINES: dict[tuple[str, str], float] = {
    ("nb", "fakenewsnet"): 0.900,
    ("nb", "employee_reviews"): 0.613,
    ("svm", "fakenewsnet"): 0.888,
    ("svm", "employee_reviews"): 0.687,
    ("roberta", "fakenewsnet"): 0.930,
    ("roberta", "employee_reviews"): 0.838,
}

MODEL_FAMILY: dict[str, str] = {
    "nb": "classical_ml",
    "svm": "classical_ml",
    "rf": "classical_ml",
    "xgboost": "classical_ml",
    "roberta": "roberta",
    "llama-3.1-8b-instant": "llama",
    "llama-3.3-70b-versatile": "llama",
    "qwen3-32b-reasoning-none": "qwen",
    "qwen3-32b-reasoning-default": "qwen",
    "openai-gpt-oss-120b": "gpt_oss",
}

DISPLAY_NAME: dict[str, str] = {
    "nb": "Naive Bayes (TF-IDF)",
    "svm": "LinearSVC (TF-IDF)",
    "rf": "Random Forest (TF-IDF)",
    "xgboost": "XGBoost (TF-IDF)",
    "roberta": "RoBERTa-base (fine-tuned)",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "qwen3-32b-reasoning-none": "Qwen3 32B (reasoning=none)",
    "qwen3-32b-reasoning-default": "Qwen3 32B (reasoning=default)",
    "openai-gpt-oss-120b": "GPT-OSS 120B (reasoning=medium)",
}

# Per-fold test-set sizes used to convert baseline `inference_time_mean` into a
# per-record millisecond proxy for the Pareto axis.
_BASELINE_N_TEST: dict[str, int] = {
    "fakenewsnet": 42,
    "employee_reviews": 41,
}


@dataclass(frozen=True)
class CombinedRow:
    """One row in combined_long.parquet — covers both baselines and LLMs.

    Attributes:
        source: "baseline" or "llm".
        model_family: One of classical_ml, roberta, llama, qwen, gpt_oss.
        model: Display name (see DISPLAY_NAME).
        dataset: "fakenewsnet" or "employee_reviews".
        prompt: "" for baselines (rendered as em dash in tables); otherwise the
            prompt code (ZS, ZS_CoT, FS_CoT_RP_NA).
        reasoning_effort: "" for non-reasoning, else "none" / "default" / "medium".
        f1_weighted: Weighted F1 score.
        f1_std: CV std for baselines, 0.0 for LLMs (single deterministic pass).
        accuracy: Top-1 accuracy.
        median_latency_ms: Per-call (LLM) or per-record (baseline) millisecond
            latency. Baselines are an approximation; see §6.2 of the design.
        n_records: Total scored rows (LLM = dataset size; baseline = CV total
            rows across all folds).
        n_success: Successfully scored rows.
        total_reasoning_tokens: Sum across the combination, 0 for non-reasoning.
        median_reasoning_tokens: Per-call median; 0 for non-reasoning.
    """

    source: str
    model_family: str
    model: str
    dataset: str
    prompt: str
    reasoning_effort: str
    f1_weighted: float
    f1_std: float
    accuracy: float
    median_latency_ms: float
    n_records: int
    n_success: int
    total_reasoning_tokens: int
    median_reasoning_tokens: int


def llm_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    """Wrap ``evaluation.llm_metrics.summarise_all`` and add per-call medians.

    Args:
        all_results: Concatenated per-call DataFrame (results/llms/all_results.parquet).

    Returns:
        DataFrame with one row per (logical_name, reasoning_effort, prompt_code,
        dataset) combo plus ``median_reasoning_tokens`` column.
    """
    summary = summarise_all(all_results)
    if summary.empty:
        summary = summary.assign(median_reasoning_tokens=pd.Series(dtype=int))
        return summary
    med_rt = (
        all_results.groupby(["model", "reasoning_effort", "prompt_code", "dataset"], dropna=False)[
            "reasoning_tokens"
        ]
        .median()
        .astype(int)
        .reset_index()
        .rename(
            columns={
                "reasoning_tokens": "median_reasoning_tokens",
                "model": "logical_name",
            }
        )
    )
    merged = summary.merge(
        med_rt,
        on=["logical_name", "reasoning_effort", "prompt_code", "dataset"],
        how="left",
    )
    merged["median_reasoning_tokens"] = merged["median_reasoning_tokens"].fillna(0).astype(int)
    return merged


def baseline_long(summary: pd.DataFrame) -> pd.DataFrame:
    """Project ``results/baselines/summary.parquet`` onto ``CombinedRow``.

    Args:
        summary: 10-row baseline summary DataFrame.

    Returns:
        DataFrame with CombinedRow columns, one row per (model, dataset).
    """
    rows: list[CombinedRow] = []
    for r in summary.itertuples(index=False):
        n_test = _BASELINE_N_TEST.get(str(r.dataset), 41)
        median_ms = float(r.inference_time_mean) / float(n_test) * 1000.0
        model_name = str(r.model_name)
        rows.append(
            CombinedRow(
                source="baseline",
                model_family=MODEL_FAMILY[model_name],
                model=DISPLAY_NAME[model_name],
                dataset=str(r.dataset),
                prompt="",
                reasoning_effort="",
                f1_weighted=float(r.f1_mean),
                f1_std=float(r.f1_std),
                accuracy=float(r.accuracy_mean),
                median_latency_ms=median_ms,
                n_records=int(n_test * r.n_folds),
                n_success=int(n_test * r.n_folds),
                total_reasoning_tokens=0,
                median_reasoning_tokens=0,
            )
        )
    return pd.DataFrame([asdict(x) for x in rows])


def llm_long(llm_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Project ``llm_summary`` output onto ``CombinedRow``.

    Args:
        llm_summary_df: Output of :func:`llm_summary` — 22 rows.

    Returns:
        DataFrame with CombinedRow columns, one row per LLM combo.
    """
    rows: list[CombinedRow] = []
    for r in llm_summary_df.itertuples(index=False):
        logical = str(r.logical_name)
        rows.append(
            CombinedRow(
                source="llm",
                model_family=MODEL_FAMILY[logical],
                model=DISPLAY_NAME[logical],
                dataset=str(r.dataset),
                prompt=str(r.prompt_code),
                reasoning_effort=str(r.reasoning_effort),
                f1_weighted=float(r.f1_weighted),
                f1_std=0.0,
                accuracy=float(r.accuracy),
                median_latency_ms=float(r.median_latency_ms),
                n_records=int(r.n_total),
                n_success=int(r.n_completed),
                total_reasoning_tokens=int(r.total_reasoning_tokens),
                median_reasoning_tokens=int(r.median_reasoning_tokens),
            )
        )
    return pd.DataFrame([asdict(x) for x in rows])


def combine(baseline_long_df: pd.DataFrame, llm_long_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate baseline + LLM long rows; sort by dataset then F1 desc.

    Args:
        baseline_long_df: Output of :func:`baseline_long`.
        llm_long_df: Output of :func:`llm_long`.

    Returns:
        DataFrame of all combined rows sorted by (dataset asc, f1_weighted desc).
    """
    combined = pd.concat([baseline_long_df, llm_long_df], ignore_index=True)
    combined = combined.sort_values(
        ["dataset", "f1_weighted"], ascending=[True, False]
    ).reset_index(drop=True)
    return combined


def best_per_model(combined: pd.DataFrame) -> pd.DataFrame:
    """Pick the highest-F1 row per (source, model, dataset).

    Used by the master F1 figure (fig1) and the master wide table. NOT used by
    the Pareto plot, which consumes ``combined_long`` directly to retain
    within-model prompt variance.

    Args:
        combined: Output of :func:`combine` (32 rows).

    Returns:
        DataFrame with one row per (source, model, dataset) — 21 rows total
        (10 baseline + 11 LLM).
    """
    idx = combined.groupby(["source", "model", "dataset"])["f1_weighted"].idxmax()
    best = combined.loc[idx].copy()
    best = best.sort_values(
        ["dataset", "source", "f1_weighted"], ascending=[True, True, False]
    ).reset_index(drop=True)
    return best


def by_prompt_impact(llm_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Mean LLM F1 per (prompt, dataset) plus delta vs ZS.

    Qwen3 variants are excluded from the mean because they only have ZS, which
    would bias the ZS average upward.

    Args:
        llm_summary_df: Output of :func:`llm_summary`.

    Returns:
        DataFrame with columns ``prompt_code, dataset, f1_mean, delta_vs_zs``.
        6 rows = 3 prompts x 2 datasets.
    """
    qwen_names = {"qwen3-32b-reasoning-none", "qwen3-32b-reasoning-default"}
    m = llm_summary_df[~llm_summary_df["logical_name"].isin(qwen_names)].copy()
    g = (
        m.groupby(["prompt_code", "dataset"], as_index=False)["f1_weighted"]
        .mean()
        .rename(columns={"f1_weighted": "f1_mean"})
    )
    zs = g[g["prompt_code"] == "ZS"].set_index("dataset")["f1_mean"]
    g["delta_vs_zs"] = g.apply(lambda r: r["f1_mean"] - zs.get(r["dataset"], float("nan")), axis=1)
    return g.sort_values(["dataset", "prompt_code"]).reset_index(drop=True)


def master_table_wide(by_model_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot best-per-(model, dataset) rows into a slide-ready wide table.

    Columns: Model, Family, FNN F1, FNN Best Prompt, FNN Latency (ms),
    ER F1, ER Best Prompt, ER Latency (ms). Sorted with baselines first
    (by max(FNN F1, ER F1) desc), then LLMs (same sort).

    Args:
        by_model_df: Output of :func:`best_per_model`.

    Returns:
        Wide DataFrame, one row per (source, model) — 5 baseline + 6 LLM = 11 rows.
    """
    rows: list[dict] = []
    grouped = by_model_df.groupby(["source", "model_family", "model"], sort=False)
    for (source, family, model), block in grouped:
        fnn = block[block["dataset"] == "fakenewsnet"]
        er = block[block["dataset"] == "employee_reviews"]
        rows.append(
            {
                "source": source,
                "Model": model,
                "Family": family,
                "FNN F1": float(fnn["f1_weighted"].iloc[0]) if not fnn.empty else float("nan"),
                "FNN Best Prompt": (str(fnn["prompt"].iloc[0]) if not fnn.empty else ""),
                "FNN Latency (ms)": (
                    float(fnn["median_latency_ms"].iloc[0]) if not fnn.empty else float("nan")
                ),
                "ER F1": float(er["f1_weighted"].iloc[0]) if not er.empty else float("nan"),
                "ER Best Prompt": (str(er["prompt"].iloc[0]) if not er.empty else ""),
                "ER Latency (ms)": (
                    float(er["median_latency_ms"].iloc[0]) if not er.empty else float("nan")
                ),
            }
        )
    wide = pd.DataFrame(rows)
    wide["_best"] = wide[["FNN F1", "ER F1"]].max(axis=1)
    wide["_source_rank"] = wide["source"].map({"baseline": 0, "llm": 1}).fillna(2)
    wide = wide.sort_values(["_source_rank", "_best"], ascending=[True, False]).reset_index(
        drop=True
    )
    wide = wide.drop(columns=["source", "_best", "_source_rank"])
    return wide
