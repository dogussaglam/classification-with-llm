"""Tests for io.py: schema validation + Parquet round-trip."""

from __future__ import annotations

import pandas as pd
import pytest

from llm_textcls.io import ERLabeledRow, assert_schema, read_parquet, write_parquet


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b"],
            "text": ["foo", "bar"],
            "label": ["remote", "not_remote"],
            "token_count": pd.Series([5, 6], dtype="int64"),
            "held_out_fs": pd.Series([False, True], dtype="bool"),
        }
    )


def test_write_then_read_roundtrip(tmp_path):
    df = _good_df()
    out = tmp_path / "x.parquet"
    write_parquet(df, out, ERLabeledRow)
    loaded = read_parquet(out)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)
    assert loaded.iloc[0]["label"] == "remote"


def test_assert_schema_rejects_missing_column():
    df = _good_df().drop(columns=["label"])
    with pytest.raises(ValueError, match="missing"):
        assert_schema(df, ERLabeledRow)


def test_assert_schema_rejects_extra_column():
    df = _good_df()
    df["junk"] = 1
    with pytest.raises(ValueError, match="extra"):
        assert_schema(df, ERLabeledRow)


def test_assert_schema_rejects_wrong_dtype():
    df = _good_df()
    df["token_count"] = df["token_count"].astype("float64")
    with pytest.raises(ValueError, match="token_count"):
        assert_schema(df, ERLabeledRow)
