"""Tests for evaluation/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from llm_textcls.evaluation.metrics import (
    BaselineResult,
    aggregate_cv_results,
    results_to_dataframe,
    weighted_f1,
)


def test_weighted_f1_matches_sklearn():
    rng = np.random.default_rng(42)
    y_true = rng.choice(["a", "b", "c"], size=200)
    y_pred = rng.choice(["a", "b", "c"], size=200)
    expected = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    assert weighted_f1(y_true, y_pred) == pytest.approx(expected)


def test_weighted_f1_perfect_prediction():
    y = np.array(["x", "y", "x", "y", "x"])
    assert weighted_f1(y, y) == pytest.approx(1.0)


def _result(fold: int, f1: float, acc: float) -> BaselineResult:
    return BaselineResult(
        model_name="nb",
        dataset="test",
        fold=fold,
        f1_weighted=f1,
        accuracy=acc,
        fit_time_sec=0.1 * fold,
        inference_time_sec=0.01 * fold,
        n_train=80,
        n_test=20,
        classification_report="{}",
    )


def test_aggregate_cv_results_means_and_stds():
    results = [
        _result(i, f1, acc)
        for i, (f1, acc) in enumerate([(0.80, 0.82), (0.84, 0.86), (0.82, 0.84)])
    ]
    agg = aggregate_cv_results(results)
    assert agg["f1_mean"] == pytest.approx((0.80 + 0.84 + 0.82) / 3)
    assert agg["accuracy_mean"] == pytest.approx((0.82 + 0.86 + 0.84) / 3)
    assert agg["n_folds"] == 3
    assert agg["f1_std"] > 0


def test_aggregate_cv_results_single_fold_zero_std():
    agg = aggregate_cv_results([_result(0, 0.9, 0.9)])
    assert agg["f1_std"] == 0.0
    assert agg["n_folds"] == 1


def test_aggregate_cv_results_empty_raises():
    with pytest.raises(ValueError):
        aggregate_cv_results([])


def test_results_to_dataframe_schema():
    results = [_result(0, 0.80, 0.82), _result(1, 0.84, 0.86)]
    df = results_to_dataframe(results)
    assert list(df.columns) == [
        "model_name",
        "dataset",
        "fold",
        "f1_weighted",
        "accuracy",
        "fit_time_sec",
        "inference_time_sec",
        "n_train",
        "n_test",
        "classification_report",
    ]
    assert str(df["fold"].dtype) == "int64"
    assert str(df["f1_weighted"].dtype) == "float64"
    assert len(df) == 2
