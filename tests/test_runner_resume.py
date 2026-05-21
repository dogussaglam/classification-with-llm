"""Test that the runner's idempotent resume skips already-completed records."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from llm_textcls.llms.groq_client import CompletionResult
from llm_textcls.llms.runner import LLMCallResult, run_benchmark


class _MockGroqLabeler:
    """Returns a fixed JSON response; counts calls so the test can assert it."""

    def __init__(self, dataset: str = "fakenewsnet"):
        self.calls = 0
        self.dataset = dataset

    def complete(self, prompt: str) -> CompletionResult:
        self.calls += 1
        if self.dataset == "fakenewsnet":
            text = '{"label": "fake", "reasoning": "test"}'
        else:
            text = '{"label": "remote", "reasoning": "test"}'
        return CompletionResult(
            text=text,
            reasoning_text="",
            latency_ms=12,
            input_tokens=10,
            output_tokens=5,
            reasoning_tokens=0,
            model="mock",
            error="",
        )


def _seed_jsonl(path: Path, ids: list[str], dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rid in ids:
            row = LLMCallResult(
                record_id=rid,
                dataset=dataset,
                model="mock-logical",
                model_slug="mock-slug",
                reasoning_effort="",
                prompt_code="ZS",
                raw_output='{"label": "fake", "reasoning": "seeded"}',
                predicted_label="fake",
                true_label="fake",
                latency_ms=10,
                input_tokens=5,
                output_tokens=3,
                reasoning_tokens=0,
                error="",
                timestamp_utc="2026-01-01T00:00:00Z",
            )
            fh.write(json.dumps(asdict(row)) + "\n")


def _five_row_fnn_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(5)],
            "text": [f"article {i}" for i in range(5)],
            "label": ["fake", "real", "fake", "real", "fake"],
        }
    )


def test_runner_resume_skips_completed(tmp_path: Path):
    out = tmp_path / "results" / "mock__ZS__fakenewsnet.jsonl"
    _seed_jsonl(out, ids=["r0", "r1", "r2"], dataset="fakenewsnet")

    df = _five_row_fnn_df()
    client = _MockGroqLabeler(dataset="fakenewsnet")

    summary = run_benchmark(
        logical_name="mock-logical",
        model_slug="mock-slug",
        reasoning_effort="",
        prompt_code="ZS",
        dataset="fakenewsnet",
        df=df,
        output_path=out,
        groq_client=client,
        few_shot_examples=None,
    )

    assert client.calls == 2, "expected to skip 3 seeded ids and only call for r3, r4"
    assert summary["completed"] == 2
    assert summary["skipped"] == 3
    assert summary["total"] == 5

    written_ids = []
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            written_ids.append(json.loads(line)["record_id"])
    assert sorted(written_ids) == ["r0", "r1", "r2", "r3", "r4"]


def test_runner_fresh_processes_all(tmp_path: Path):
    out = tmp_path / "fresh__ZS__fakenewsnet.jsonl"
    df = _five_row_fnn_df()
    client = _MockGroqLabeler(dataset="fakenewsnet")

    summary = run_benchmark(
        logical_name="fresh-logical",
        model_slug="mock-slug",
        reasoning_effort="",
        prompt_code="ZS",
        dataset="fakenewsnet",
        df=df,
        output_path=out,
        groq_client=client,
        few_shot_examples=None,
    )

    assert client.calls == 5
    assert summary["completed"] == 5
    assert summary["skipped"] == 0
    assert summary["parse_failures"] == 0


def test_runner_records_parse_failure(tmp_path: Path):
    class BadJsonLabeler:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt: str) -> CompletionResult:
            self.calls += 1
            return CompletionResult(
                text="this is not json at all",
                reasoning_text="",
                latency_ms=5,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                model="bad",
                error="",
            )

    out = tmp_path / "bad__ZS__fakenewsnet.jsonl"
    df = pd.DataFrame({"id": ["x"], "text": ["t"], "label": ["fake"]})
    client = BadJsonLabeler()

    summary = run_benchmark(
        logical_name="bad",
        model_slug="bad",
        reasoning_effort="",
        prompt_code="ZS",
        dataset="fakenewsnet",
        df=df,
        output_path=out,
        groq_client=client,
        few_shot_examples=None,
    )
    assert client.calls == 1
    assert summary["parse_failures"] == 1

    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert line["error"] == "PARSE_FAIL"
    assert line["predicted_label"] == ""
