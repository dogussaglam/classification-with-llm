"""Shared evaluation metrics + BaselineResult schema for Phase 2."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    dataset: str
    fold: int
    f1_weighted: float
    accuracy: float
    fit_time_sec: float
    inference_time_sec: float
    n_train: int
    n_test: int
    classification_report: str


def weighted_f1(y_true, y_pred) -> float:
    """Return sklearn weighted F1.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Weighted F1 score in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def aggregate_cv_results(results: list[BaselineResult]) -> dict:
    """Aggregate fold-level results into mean/std summary statistics.

    Args:
        results: One BaselineResult per fold for a single (model, dataset) pair.

    Returns:
        Dict with f1_mean, f1_std, accuracy_mean, accuracy_std,
        fit_time_mean, inference_time_mean, n_folds.

    Raises:
        ValueError: if results is empty.
    """
    if not results:
        raise ValueError("aggregate_cv_results requires at least one BaselineResult")
    f1s = [r.f1_weighted for r in results]
    accs = [r.accuracy for r in results]
    fit_times = [r.fit_time_sec for r in results]
    inf_times = [r.inference_time_sec for r in results]
    n = len(results)
    return {
        "f1_mean": statistics.fmean(f1s),
        "f1_std": statistics.pstdev(f1s) if n > 1 else 0.0,
        "accuracy_mean": statistics.fmean(accs),
        "accuracy_std": statistics.pstdev(accs) if n > 1 else 0.0,
        "fit_time_mean": statistics.fmean(fit_times),
        "inference_time_mean": statistics.fmean(inf_times),
        "n_folds": n,
    }


def results_to_dataframe(results: list[BaselineResult]) -> pd.DataFrame:
    """Convert a list of BaselineResult into a DataFrame with the exact schema.

    Args:
        results: List of BaselineResult.

    Returns:
        DataFrame with BaselineResult fields as columns and matching dtypes.
    """
    df = pd.DataFrame(
        [
            {
                "model_name": r.model_name,
                "dataset": r.dataset,
                "fold": r.fold,
                "f1_weighted": r.f1_weighted,
                "accuracy": r.accuracy,
                "fit_time_sec": r.fit_time_sec,
                "inference_time_sec": r.inference_time_sec,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "classification_report": r.classification_report,
            }
            for r in results
        ]
    )
    df["model_name"] = df["model_name"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["fold"] = df["fold"].astype("int64")
    df["f1_weighted"] = df["f1_weighted"].astype("float64")
    df["accuracy"] = df["accuracy"].astype("float64")
    df["fit_time_sec"] = df["fit_time_sec"].astype("float64")
    df["inference_time_sec"] = df["inference_time_sec"].astype("float64")
    df["n_train"] = df["n_train"].astype("int64")
    df["n_test"] = df["n_test"].astype("int64")
    df["classification_report"] = df["classification_report"].astype(str)
    return df
