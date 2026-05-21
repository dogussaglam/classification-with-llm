"""TF-IDF baselines: Naive Bayes, LinearSVC, Random Forest, XGBoost.

All four classifiers share a TfidfVectorizer with identical params and run
through `cross_validate_baseline`.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from llm_textcls.evaluation.metrics import BaselineResult, weighted_f1

TFIDF_PARAMS: dict[str, Any] = dict(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

XGB_PARAMS: dict[str, Any] = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

RF_PARAMS: dict[str, Any] = dict(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

VALID_MODELS = frozenset({"nb", "svm", "rf", "xgboost"})


class _XGBStringWrapper(BaseEstimator, ClassifierMixin):
    """Wrap XGBClassifier so it accepts string labels (round-trip via LabelEncoder).

    XGBoost requires integer labels; this wrapper hides that detail so the
    cross-validation loop can treat all four TF-IDF models uniformly.
    """

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self.le_ = LabelEncoder()
        y_int = self.le_.fit_transform(y)
        self.clf_ = XGBClassifier(**self.xgb_params)
        self.clf_.fit(X, y_int)
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        y_int = self.clf_.predict(X)
        return self.le_.inverse_transform(y_int)


def build_tfidf_pipeline(model_name: str) -> Pipeline:
    """Return a sklearn Pipeline = TfidfVectorizer + classifier for model_name.

    Args:
        model_name: One of "nb", "svm", "rf", "xgboost".

    Returns:
        Unfitted sklearn Pipeline.

    Raises:
        ValueError: if model_name is not recognized.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Expected one of {sorted(VALID_MODELS)}"
        )

    clf: Any
    if model_name == "nb":
        clf = MultinomialNB()
    elif model_name == "svm":
        clf = LinearSVC(random_state=42)
    elif model_name == "rf":
        clf = RandomForestClassifier(**RF_PARAMS)
    else:  # xgboost
        clf = _XGBStringWrapper(**XGB_PARAMS)

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", clf),
        ]
    )


def cross_validate_baseline(
    model_name: str,
    df: pd.DataFrame,
    dataset_name: str,
    text_col: str = "text",
    label_col: str = "label",
    n_splits: int = 5,
    seed: int = 42,
) -> list[BaselineResult]:
    """Run StratifiedKFold CV for a TF-IDF baseline; one BaselineResult per fold.

    Args:
        model_name: One of "nb", "svm", "rf", "xgboost".
        df: DataFrame with text_col and label_col.
        dataset_name: Tag stored on each BaselineResult (e.g. "fakenewsnet").
        text_col: Name of the text column.
        label_col: Name of the label column.
        n_splits: Number of CV folds.
        seed: Random state for the fold splitter.

    Returns:
        List of `n_splits` BaselineResult objects.
    """
    X = df[text_col].astype(str).to_numpy()
    y = df[label_col].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: list[BaselineResult] = []
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        pipe = build_tfidf_pipeline(model_name)
        t0 = time.perf_counter()
        pipe.fit(X[tr], y[tr])
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_hat = pipe.predict(X[te])
        inf_time = time.perf_counter() - t0

        report = classification_report(y[te], y_hat, output_dict=True, zero_division=0)
        results.append(
            BaselineResult(
                model_name=model_name,
                dataset=dataset_name,
                fold=fold_i,
                f1_weighted=weighted_f1(y[te], y_hat),
                accuracy=float(accuracy_score(y[te], y_hat)),
                fit_time_sec=float(fit_time),
                inference_time_sec=float(inf_time),
                n_train=int(len(tr)),
                n_test=int(len(te)),
                classification_report=json.dumps(report),
            )
        )
    return results
