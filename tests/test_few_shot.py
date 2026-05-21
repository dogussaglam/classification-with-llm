"""Tests for FNN/ER few-shot builders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llm_textcls.io import project_root, read_parquet
from llm_textcls.llms.few_shot import (
    build_er_few_shot_examples,
    build_fnn_few_shot_examples,
)


def test_fnn_builder_returns_two_examples_ordered_fake_real():
    examples = build_fnn_few_shot_examples()
    assert len(examples) == 2
    assert [e.label for e in examples] == ["fake", "real"]
    for ex in examples:
        assert ex.text
        assert ex.reasoning


def test_fnn_examples_are_held_out_ids_not_in_test_set():
    sample = read_parquet(project_root() / "data" / "fakenewsnet" / "sample.parquet")
    held = sample[sample["held_out_fs"]]["id"].tolist()
    test = sample[~sample["held_out_fs"]]["id"].tolist()

    examples = build_fnn_few_shot_examples()
    held_text = set(sample[sample["held_out_fs"]]["text"].astype(str).tolist())
    for ex in examples:
        assert ex.text in held_text
    assert set(held).isdisjoint(set(test))


def _write_synthetic_er(
    tmp_path: Path, labels: list[str], reasonings: dict[str, str] | None = None
) -> tuple[Path, Path]:
    rows = []
    for i, lbl in enumerate(labels):
        rows.append(
            {
                "id": f"er_synth_{i:02d}",
                "text": f"Text for row {i} ({lbl})",
                "label": lbl,
                "token_count": 10,
                "held_out_fs": True,
            }
        )
    df = pd.DataFrame(rows)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df["id"] = df["id"].astype(str)
    df["token_count"] = df["token_count"].astype("int64")
    df["held_out_fs"] = df["held_out_fs"].astype(bool)
    parquet_path = tmp_path / "synth_labeled.parquet"
    df.to_parquet(parquet_path, index=False)

    if reasonings is None:
        reasonings = {row["id"]: f"reason for {row['id']}" for row in rows}
    reasonings_path = tmp_path / "synth_reasonings.json"
    reasonings_path.write_text(json.dumps(reasonings), encoding="utf-8")
    return parquet_path, reasonings_path


def test_er_builder_returns_three_examples_ordered_by_class(tmp_path: Path):
    parquet, reasonings = _write_synthetic_er(
        tmp_path,
        labels=["not_mentioned", "remote", "not_remote", "remote", "not_mentioned", "not_remote"],
    )
    examples = build_er_few_shot_examples(parquet, reasonings)
    assert len(examples) == 3
    assert [e.label for e in examples] == ["remote", "not_remote", "not_mentioned"]


def test_er_builder_raises_when_class_missing(tmp_path: Path):
    parquet, reasonings = _write_synthetic_er(
        tmp_path,
        labels=["remote", "remote", "remote", "remote", "remote", "remote"],
    )
    with pytest.raises(RuntimeError, match="not_remote"):
        build_er_few_shot_examples(parquet, reasonings)


def test_er_builder_raises_when_reasoning_missing(tmp_path: Path):
    parquet, _ = _write_synthetic_er(
        tmp_path,
        labels=["remote", "not_remote", "not_mentioned", "remote", "remote", "not_remote"],
    )
    empty_reasonings = tmp_path / "empty.json"
    empty_reasonings.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing reasoning"):
        build_er_few_shot_examples(parquet, empty_reasonings)
