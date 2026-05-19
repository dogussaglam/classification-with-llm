"""Tests for FakeNewsNet sampling + filtering."""

from __future__ import annotations

import pandas as pd

from llm_textcls.data.fakenewsnet import filter_by_length, mark_held_out_fs, stratified_sample


def _synthetic(n_per_label: int = 250) -> pd.DataFrame:
    rows = []
    for label in ("fake", "real"):
        for i in range(n_per_label):
            text = f"{label} article {i} " + "lorem ipsum " * (i % 7)
            rows.append(
                {
                    "id": f"{label}_{i:04d}",
                    "label": label,
                    "title": f"title {i}",
                    "text": text,
                    "url": f"http://example.com/{label}/{i}",
                }
            )
    return pd.DataFrame(rows)


def test_stratified_sample_deterministic_and_balanced():
    df = _synthetic(250)
    a = stratified_sample(df, n_per_class=107, seed=42)
    b = stratified_sample(df, n_per_class=107, seed=42)
    assert len(a) == 214
    assert a["label"].value_counts().to_dict() == {"fake": 107, "real": 107}
    assert a["id"].tolist() == b["id"].tolist()


def test_filter_by_length_adds_token_count_and_filters():
    df = _synthetic(50)
    out = filter_by_length(df, max_tokens=4096)
    assert "token_count" in out.columns
    assert (out["token_count"] <= 4096).all()
    assert len(out) == len(df)


def test_mark_held_out_fs_two_per_label():
    df = _synthetic(120)
    sample = stratified_sample(df, n_per_class=107, seed=42)
    marked = mark_held_out_fs(sample, n_per_class=2)
    assert int(marked["held_out_fs"].sum()) == 4
    per_label = marked[marked["held_out_fs"]].groupby("label").size().to_dict()
    assert per_label == {"fake": 2, "real": 2}
