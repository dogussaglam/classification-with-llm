"""Aggregate per-call LLM results into per-combination summaries."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class LLMCombinationSummary:
    """One row per (logical_model, reasoning_effort, prompt_code, dataset)."""

    logical_name: str
    model_slug: str
    reasoning_effort: str
    prompt_code: str
    dataset: str
    n_total: int
    n_completed: int
    n_parse_fail: int
    n_api_fail: int
    f1_weighted: float
    accuracy: float
    median_latency_ms: int
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int


def _safe_median_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(statistics.median(values))


def summarise_combination(df_combo: pd.DataFrame) -> LLMCombinationSummary:
    """Aggregate one combo's per-call rows into a single summary.

    Args:
        df_combo: DataFrame rows from one (model, reasoning_effort, prompt,
            dataset) combo.

    Returns:
        :class:`LLMCombinationSummary` with counts, F1, accuracy, median
        latency, and token sums. F1 and accuracy are NaN when no row has a
        non-empty predicted label.

    Raises:
        ValueError: if the input DataFrame is empty.
    """
    if df_combo.empty:
        raise ValueError("summarise_combination requires at least one row")
    row0 = df_combo.iloc[0]
    parsed = df_combo[df_combo["predicted_label"] != ""]
    n_total = len(df_combo)
    n_completed = int((df_combo["error"] == "").sum())
    n_parse_fail = int((df_combo["error"] == "PARSE_FAIL").sum())
    n_api_fail = int(df_combo["error"].str.startswith("API_ERROR").sum())
    if not parsed.empty:
        y_true = parsed["true_label"].tolist()
        y_pred = parsed["predicted_label"].tolist()
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
    else:
        f1 = float("nan")
        acc = float("nan")
    latencies = [int(x) for x in df_combo["latency_ms"].tolist() if x and x > 0]
    return LLMCombinationSummary(
        logical_name=str(row0["model"]),
        model_slug=str(row0.get("model_slug", "")),
        reasoning_effort=str(row0.get("reasoning_effort", "")),
        prompt_code=str(row0["prompt_code"]),
        dataset=str(row0["dataset"]),
        n_total=n_total,
        n_completed=n_completed,
        n_parse_fail=n_parse_fail,
        n_api_fail=n_api_fail,
        f1_weighted=f1,
        accuracy=acc,
        median_latency_ms=_safe_median_int(latencies),
        total_input_tokens=int(df_combo["input_tokens"].sum()),
        total_output_tokens=int(df_combo["output_tokens"].sum()),
        total_reasoning_tokens=int(df_combo.get("reasoning_tokens", pd.Series(dtype=int)).sum()),
    )


def summarise_all(df_all: pd.DataFrame) -> pd.DataFrame:
    """Group ``df_all`` by combo and return one summary row per combo.

    Args:
        df_all: Concatenated rows from all 22 JSONLs.

    Returns:
        DataFrame with :class:`LLMCombinationSummary` fields as columns.
    """
    if df_all.empty:
        return pd.DataFrame()
    keys = ["model", "reasoning_effort", "prompt_code", "dataset"]
    rows: list[dict] = []
    for _, g in df_all.groupby(keys, dropna=False):
        s = summarise_combination(g)
        rows.append(
            {
                "logical_name": s.logical_name,
                "model_slug": s.model_slug,
                "reasoning_effort": s.reasoning_effort,
                "prompt_code": s.prompt_code,
                "dataset": s.dataset,
                "n_total": s.n_total,
                "n_completed": s.n_completed,
                "n_parse_fail": s.n_parse_fail,
                "n_api_fail": s.n_api_fail,
                "f1_weighted": s.f1_weighted,
                "accuracy": s.accuracy,
                "median_latency_ms": s.median_latency_ms,
                "total_input_tokens": s.total_input_tokens,
                "total_output_tokens": s.total_output_tokens,
                "total_reasoning_tokens": s.total_reasoning_tokens,
            }
        )
    return pd.DataFrame(rows)
