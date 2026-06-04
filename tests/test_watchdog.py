"""Unit tests for scripts/watchdog_benchmark.py — no real Groq calls."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import watchdog_benchmark as wb  # noqa: E402

# --- collect_api_keys --------------------------------------------------------


def test_key_pool_collection_primary_only():
    assert wb.collect_api_keys({"GROQ_API_KEY": "key1"}) == ["key1"]


def test_key_pool_collection_three_keys():
    env = {
        "GROQ_API_KEY": "key1",
        "GROQ_API_KEY_2": "key2",
        "GROQ_API_KEY_3": "key3",
    }
    assert wb.collect_api_keys(env) == ["key1", "key2", "key3"]


def test_key_pool_skips_blank_and_whitespace():
    env = {
        "GROQ_API_KEY": "key1",
        "GROQ_API_KEY_2": "",
        "GROQ_API_KEY_3": "   ",
        "GROQ_API_KEY_4": "key4",
    }
    assert wb.collect_api_keys(env) == ["key1", "key4"]


def test_key_pool_dedupes_duplicates():
    env = {"GROQ_API_KEY": "key1", "GROQ_API_KEY_2": "key1", "GROQ_API_KEY_3": "key2"}
    assert wb.collect_api_keys(env) == ["key1", "key2"]


def test_key_pool_empty_when_no_env():
    assert wb.collect_api_keys({}) == []


# --- compute_pending_work ----------------------------------------------------


def _seed_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if n == 0:
        return
    path.write_text("\n".join("{}" for _ in range(n)) + "\n", encoding="utf-8")


def test_compute_pending_work(tmp_path: Path, monkeypatch):
    llms_dir = tmp_path / "results" / "llms"
    llms_dir.mkdir(parents=True)

    combos = [
        wb.LogicalCombo("foo", "foo", "", "ZS"),
        wb.LogicalCombo("bar", "bar", "", "ZS"),
        wb.LogicalCombo("baz", "baz", "", "ZS"),
    ]
    # foo: fnn complete (210), er missing (file does not exist)
    _seed_jsonl(llms_dir / "foo__ZS__fakenewsnet.jsonl", 210)
    # bar: fnn partial (50), er complete (204)
    _seed_jsonl(llms_dir / "bar__ZS__fakenewsnet.jsonl", 50)
    _seed_jsonl(llms_dir / "bar__ZS__employee_reviews.jsonl", 204)
    # baz: both empty files (count as zero lines once stripped)
    _seed_jsonl(llms_dir / "baz__ZS__fakenewsnet.jsonl", 0)
    _seed_jsonl(llms_dir / "baz__ZS__employee_reviews.jsonl", 0)

    def fake_path(logical, prompt, dataset):
        safe = logical.replace("/", "-")
        return llms_dir / f"{safe}__{prompt}__{dataset}.jsonl"

    monkeypatch.setattr(wb, "jsonl_path", fake_path)

    pending = wb.compute_pending_work(combos)
    got = sorted((c.logical_name, d, cur, exp) for (c, d, cur, exp) in pending)

    assert got == [
        ("bar", "fakenewsnet", 50, 210),
        ("baz", "employee_reviews", 0, 204),
        ("baz", "fakenewsnet", 0, 210),
        ("foo", "employee_reviews", 0, 204),
    ]


def test_compute_pending_work_all_complete_returns_empty(tmp_path: Path, monkeypatch):
    llms_dir = tmp_path / "llms"
    llms_dir.mkdir(parents=True)
    combos = [wb.LogicalCombo("x", "x", "", "ZS")]
    _seed_jsonl(llms_dir / "x__ZS__fakenewsnet.jsonl", 210)
    _seed_jsonl(llms_dir / "x__ZS__employee_reviews.jsonl", 204)

    def fake_path(logical, prompt, dataset):
        return llms_dir / f"{logical}__{prompt}__{dataset}.jsonl"

    monkeypatch.setattr(wb, "jsonl_path", fake_path)
    assert wb.compute_pending_work(combos) == []


# --- backoff_seconds ---------------------------------------------------------


def test_backoff_sequence_default():
    assert wb.backoff_seconds(0) == 300
    assert wb.backoff_seconds(1) == 600
    assert wb.backoff_seconds(2) == 900
    assert wb.backoff_seconds(3) == 1800


def test_backoff_caps_at_last_value():
    assert wb.backoff_seconds(4) == 1800
    assert wb.backoff_seconds(10) == 1800
    assert wb.backoff_seconds(100) == 1800


def test_backoff_negative_clamps_to_first():
    assert wb.backoff_seconds(-1) == 300
    assert wb.backoff_seconds(-99) == 300


def test_backoff_custom_sequence():
    seq = (60, 120, 240)
    assert wb.backoff_seconds(0, seq) == 60
    assert wb.backoff_seconds(1, seq) == 120
    assert wb.backoff_seconds(2, seq) == 240
    assert wb.backoff_seconds(7, seq) == 240


# --- format_progress_report --------------------------------------------------


def test_progress_format_contains_expected_layout():
    combos = [
        wb.LogicalCombo("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "", "ZS_CoT"),
    ]
    per_slug = {"llama-3.3-70b-versatile": (135, 414)}
    per_combo = [
        (combos[0], "fakenewsnet", 135, 210),
        (combos[0], "employee_reviews", 0, 204),
    ]

    report = wb.format_progress_report(
        start_time=0.0,
        now=60.0,
        per_slug_counts=per_slug,
        per_combo_counts=per_combo,
        slug_state={"llama-3.3-70b-versatile": "active"},
        rate_per_min=10.0,
        eta_str="30m",
        total_done=135,
        total_target=414,
    )

    assert "Phase 3 Watchdog" in report
    assert "elapsed: 1m" in report
    assert "135/414" in report
    assert "32.6%" in report  # 135/414
    assert "10.0 calls/min" in report
    assert "ETA: ~30m" in report
    assert "active" in report
    assert "Incomplete combos:" in report
    assert "llama-3.3-70b-versatile / ZS_CoT / fakenewsnet: 135/210" in report
    assert "llama-3.3-70b-versatile / ZS_CoT / employee_reviews: 0/204" in report
    assert "============================================================" in report


def test_progress_format_done_slug_uses_check_mark():
    combos = [wb.LogicalCombo("foo", "foo", "", "ZS")]
    report = wb.format_progress_report(
        start_time=0.0,
        now=0.0,
        per_slug_counts={"foo": (414, 414)},
        per_combo_counts=[
            (combos[0], "fakenewsnet", 210, 210),
            (combos[0], "employee_reviews", 204, 204),
        ],
        slug_state={"foo": "done"},
        rate_per_min=0.0,
        eta_str="0m",
        total_done=414,
        total_target=414,
    )
    assert "✓ done" in report
    assert "Incomplete combos:" not in report


def test_progress_format_sleeping_label_humanized():
    combos = [wb.LogicalCombo("foo", "foo", "", "ZS")]
    report = wb.format_progress_report(
        start_time=0.0,
        now=0.0,
        per_slug_counts={"foo": (0, 414)},
        per_combo_counts=[
            (combos[0], "fakenewsnet", 0, 210),
            (combos[0], "employee_reviews", 0, 204),
        ],
        slug_state={"foo": "sleeping_10min"},
        rate_per_min=0.0,
        eta_str="calculating...",
        total_done=0,
        total_target=414,
    )
    assert "sleeping 10min" in report
    assert "ETA: calculating..." in report


def test_progress_format_minutes_formatter():
    assert wb._format_minutes(0) == "0m"
    assert wb._format_minutes(5) == "5m"
    assert wb._format_minutes(83) == "1h 23m"
    assert wb._format_minutes(120) == "2h 0m"


# --- graceful shutdown -------------------------------------------------------


def test_shutdown_handler_sets_event():
    event = threading.Event()
    handler = wb._make_shutdown_handler(event)
    assert not event.is_set()
    handler(2, None)  # signum=SIGINT, frame=None
    assert event.is_set()
    # Calling again is idempotent and doesn't raise.
    handler(2, None)
    assert event.is_set()


def test_worker_exits_promptly_when_shutdown_preset(tmp_path: Path, monkeypatch):
    """If shutdown is set before the worker starts, it must mark itself done
    and never instantiate a Groq client."""
    llms_dir = tmp_path / "llms"
    llms_dir.mkdir(parents=True)

    combos = [wb.LogicalCombo("model-x", "model-x", "", "ZS")]

    def fake_path(logical, prompt, dataset):
        return llms_dir / f"{logical}__{prompt}__{dataset}.jsonl"

    monkeypatch.setattr(wb, "jsonl_path", fake_path)

    shutdown = threading.Event()
    shutdown.set()  # pre-set: worker should exit on first check

    state: dict[str, str] = {}
    state_lock = threading.Lock()

    factory_called = {"n": 0}

    def fake_factory(**kwargs):
        factory_called["n"] += 1
        raise AssertionError("factory should not be called when shutdown is preset")

    wb._slug_worker(
        physical_slug="model-x",
        combos=combos,
        dataset_dfs={
            "fakenewsnet": pd.DataFrame({"id": [], "text": [], "label": []}),
            "employee_reviews": pd.DataFrame({"id": [], "text": [], "label": []}),
        },
        few_shot_examples={"fakenewsnet": [], "employee_reviews": []},
        key_pool=["fake-key"],
        slug_state=state,
        state_lock=state_lock,
        shutdown_event=shutdown,
        backoff_sequence=(60,),
        labeler_factory=fake_factory,
        log=logging.getLogger("test"),
    )

    assert state["model-x"] == "done"
    assert factory_called["n"] == 0


def test_worker_exits_done_when_no_pending(tmp_path: Path, monkeypatch):
    """A fully-complete slug exits cleanly without calling the factory."""
    llms_dir = tmp_path / "llms"
    llms_dir.mkdir(parents=True)
    combos = [wb.LogicalCombo("y", "y", "", "ZS")]
    _seed_jsonl(llms_dir / "y__ZS__fakenewsnet.jsonl", 210)
    _seed_jsonl(llms_dir / "y__ZS__employee_reviews.jsonl", 204)

    def fake_path(logical, prompt, dataset):
        return llms_dir / f"{logical}__{prompt}__{dataset}.jsonl"

    monkeypatch.setattr(wb, "jsonl_path", fake_path)

    state: dict[str, str] = {}

    def boom_factory(**kwargs):
        raise AssertionError("no factory call expected when nothing pending")

    wb._slug_worker(
        physical_slug="y",
        combos=combos,
        dataset_dfs={
            "fakenewsnet": pd.DataFrame(),
            "employee_reviews": pd.DataFrame(),
        },
        few_shot_examples={"fakenewsnet": [], "employee_reviews": []},
        key_pool=["k"],
        slug_state=state,
        state_lock=threading.Lock(),
        shutdown_event=threading.Event(),
        backoff_sequence=(60,),
        labeler_factory=boom_factory,
        log=logging.getLogger("test"),
    )

    assert state["y"] == "done"
