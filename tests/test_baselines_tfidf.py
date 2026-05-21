"""Tests for baselines/tfidf_models.py."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from llm_textcls.baselines.tfidf_models import (
    VALID_MODELS,
    build_tfidf_pipeline,
    cross_validate_baseline,
)


def _toy_df(per_class: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(per_class):
        rows.append({"text": f"alpha beta gamma {i} positive sentiment good", "label": "pos"})
        rows.append({"text": f"delta epsilon zeta {i} negative sentiment bad", "label": "neg"})
    return pd.DataFrame(rows)


def test_build_tfidf_pipeline_returns_pipeline_for_all_models():
    for name in VALID_MODELS:
        pipe = build_tfidf_pipeline(name)
        assert isinstance(pipe, Pipeline)
        assert "tfidf" in pipe.named_steps
        assert "clf" in pipe.named_steps


def test_build_tfidf_pipeline_nb_classifier_type():
    pipe = build_tfidf_pipeline("nb")
    assert isinstance(pipe.named_steps["clf"], MultinomialNB)


def test_build_tfidf_pipeline_svm_classifier_type():
    pipe = build_tfidf_pipeline("svm")
    assert isinstance(pipe.named_steps["clf"], LinearSVC)


def test_build_tfidf_pipeline_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown model_name"):
        build_tfidf_pipeline("not_a_model")


def test_xgboost_roundtrips_string_labels():
    df = _toy_df(per_class=20)
    pipe = build_tfidf_pipeline("xgboost")
    pipe.fit(df["text"].to_numpy(), df["label"].to_numpy())
    preds = pipe.predict(df["text"].to_numpy())
    assert set(preds) <= {"pos", "neg"}
    assert all(isinstance(p, str) for p in preds)


def test_cross_validate_baseline_returns_one_result_per_fold():
    df = _toy_df(per_class=20)
    results = cross_validate_baseline("nb", df, "toy", n_splits=2, seed=42)
    assert len(results) == 2
    for r in results:
        assert r.model_name == "nb"
        assert r.dataset == "toy"
        assert 0.0 <= r.f1_weighted <= 1.0
        assert 0.0 <= r.accuracy <= 1.0
        assert r.n_train + r.n_test == len(df)


def test_cross_validate_baseline_deterministic_with_seed():
    df = _toy_df(per_class=20)
    a = cross_validate_baseline("nb", df, "toy", n_splits=2, seed=42)
    b = cross_validate_baseline("nb", df, "toy", n_splits=2, seed=42)
    assert [r.f1_weighted for r in a] == [r.f1_weighted for r in b]
