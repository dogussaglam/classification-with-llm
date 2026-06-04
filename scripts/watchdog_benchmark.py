"""Self-managing watchdog that drives Phase 3 to completion unattended.

Per physical slug (4 workers) it computes pending work from on-disk JSONL line
counts, rotates through a pool of GROQ_API_KEY[_N] env vars on RPD caps, and
backs off (5/10/15/30 min, capped) when every key is exhausted. A reporter
thread prints a progress board every --check-interval-min minutes. Ctrl+C
flips a shared shutdown event so in-flight API calls finish before exit.

Reuses ``run_benchmark`` (idempotent JSONL append) — does not modify it.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import load_env, project_root, read_parquet  # noqa: E402
from llm_textcls.llms.few_shot import (  # noqa: E402
    build_er_few_shot_examples,
    build_fnn_few_shot_examples,
)
from llm_textcls.llms.groq_client import GroqLabeler, RPDCapHitError  # noqa: E402
from llm_textcls.llms.prompts import FewShotExample  # noqa: E402
from llm_textcls.llms.runner import jsonl_path, run_benchmark  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402


@dataclass(frozen=True)
class LogicalCombo:
    """One (logical_name, model_slug, reasoning_effort, prompt_code) tuple."""

    logical_name: str
    model_slug: str
    reasoning_effort: str
    prompt_code: str


# Mirror of run_all_benchmarks.LOGICAL_COMBOS — duplicated rather than imported
# to avoid fragile sibling-script imports. Keep in sync if combos ever change.
LOGICAL_COMBOS: list[LogicalCombo] = [
    LogicalCombo("llama-3.1-8b-instant", "llama-3.1-8b-instant", "", "ZS"),
    LogicalCombo("llama-3.1-8b-instant", "llama-3.1-8b-instant", "", "ZS_CoT"),
    LogicalCombo("llama-3.1-8b-instant", "llama-3.1-8b-instant", "", "FS_CoT_RP_NA"),
    LogicalCombo("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "", "ZS"),
    LogicalCombo("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "", "ZS_CoT"),
    LogicalCombo("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "", "FS_CoT_RP_NA"),
    LogicalCombo("qwen3-32b-reasoning-none", "qwen/qwen3-32b", "none", "ZS"),
    LogicalCombo("qwen3-32b-reasoning-default", "qwen/qwen3-32b", "default", "ZS"),
    LogicalCombo("openai-gpt-oss-120b", "openai/gpt-oss-120b", "medium", "ZS"),
    LogicalCombo("openai-gpt-oss-120b", "openai/gpt-oss-120b", "medium", "ZS_CoT"),
    LogicalCombo("openai-gpt-oss-120b", "openai/gpt-oss-120b", "medium", "FS_CoT_RP_NA"),
]

DATASETS: tuple[str, ...] = ("fakenewsnet", "employee_reviews")
EXPECTED_PER_DATASET: dict[str, int] = {"fakenewsnet": 210, "employee_reviews": 204}
DATASET_PATHS: dict[str, Path] = {
    "fakenewsnet": project_root() / "data" / "fakenewsnet" / "sample.parquet",
    "employee_reviews": project_root() / "data" / "employee_reviews" / "labeled.parquet",
}

ENV_KEY_NAMES: tuple[str, ...] = (
    "GROQ_API_KEY",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_3",
    "GROQ_API_KEY_4",
    "GROQ_API_KEY_5",
)

BACKOFF_SEQUENCE_SEC: tuple[int, ...] = (300, 600, 900, 1800)
SHUTDOWN_POLL_SEC = 0.5


# ---------------------------------------------------------------------------
# Pure helpers (no threading, no I/O on globals) — kept testable.
# ---------------------------------------------------------------------------


def collect_api_keys(env: dict[str, str] | None = None) -> list[str]:
    """Read GROQ_API_KEY{,_2..5} from env and return a deduped pool.

    Args:
        env: Optional mapping to read from (defaults to ``os.environ``).

    Returns:
        Ordered list of non-empty, deduplicated keys.
    """
    src = env if env is not None else os.environ
    out: list[str] = []
    for name in ENV_KEY_NAMES:
        raw = src.get(name)
        if raw is None:
            continue
        val = raw.strip()
        if not val:
            continue
        if val in out:
            continue
        out.append(val)
    return out


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def compute_pending_work(
    combos: list[LogicalCombo],
    datasets: tuple[str, ...] = DATASETS,
    expected_per_dataset: dict[str, int] | None = None,
) -> list[tuple[LogicalCombo, str, int, int]]:
    """Inspect on-disk JSONLs to find (combo, dataset) pairs still under target.

    Args:
        combos: Logical combos to consider.
        datasets: Datasets to check per combo.
        expected_per_dataset: Target row counts; defaults to
            :data:`EXPECTED_PER_DATASET`.

    Returns:
        List of ``(combo, dataset, current_count, expected_count)`` tuples
        whose ``current_count < expected_count``.
    """
    expected = expected_per_dataset or EXPECTED_PER_DATASET
    out: list[tuple[LogicalCombo, str, int, int]] = []
    for combo in combos:
        for dataset in datasets:
            target = expected[dataset]
            path = jsonl_path(combo.logical_name, combo.prompt_code, dataset)
            current = _count_jsonl_lines(path)
            if current < target:
                out.append((combo, dataset, current, target))
    return out


def backoff_seconds(
    attempt_idx: int,
    sequence: tuple[int, ...] = BACKOFF_SEQUENCE_SEC,
) -> int:
    """Return the backoff for ``attempt_idx`` (0-indexed), capped at sequence end.

    Args:
        attempt_idx: How many full key-pool exhaustions have already happened.
        sequence: Ascending backoff schedule in seconds.

    Returns:
        Sleep duration in seconds.
    """
    if attempt_idx < 0:
        attempt_idx = 0
    return sequence[min(attempt_idx, len(sequence) - 1)]


def _make_progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    pct = max(0.0, min(1.0, current / total))
    filled = int(round(pct * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _format_minutes(total_min: float) -> str:
    total_min = max(0.0, total_min)
    h = int(total_min // 60)
    m = int(round(total_min - h * 60))
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def _slug_status_label(raw: str) -> str:
    if raw == "done":
        return "✓ done"
    if raw == "active":
        return "active"
    if raw == "rotating":
        return "rotating"
    if raw == "crashed":
        return "crashed"
    if raw.startswith("sleeping_"):
        return raw.replace("_", " ")
    return raw or "?"


def format_progress_report(
    start_time: float,
    now: float,
    per_slug_counts: dict[str, tuple[int, int]],
    per_combo_counts: list[tuple[LogicalCombo, str, int, int]],
    slug_state: dict[str, str],
    rate_per_min: float,
    eta_str: str,
    total_done: int,
    total_target: int,
) -> str:
    """Render the progress board as a single multi-line string.

    Args:
        start_time: ``time.monotonic()`` of watchdog start.
        now: Current ``time.monotonic()``.
        per_slug_counts: ``{slug: (done, target)}``.
        per_combo_counts: All ``(combo, dataset, current, expected)`` rows.
        slug_state: ``{slug: state}`` where state is "active", "rotating",
            "done", "crashed", or ``sleeping_<N>min``.
        rate_per_min: Rolling 5-min throughput.
        eta_str: Pre-formatted ETA, e.g. ``"1h 40m"`` or ``"calculating..."``.
        total_done: Sum across all slugs.
        total_target: Sum across all slugs.

    Returns:
        Multi-line report ready for ``print``.
    """
    clock = datetime.now().strftime("%H:%M:%S")
    elapsed_min = (now - start_time) / 60.0
    pct = (100.0 * total_done / total_target) if total_target else 0.0
    bar = _make_progress_bar(total_done, total_target)

    lines: list[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"Phase 3 Watchdog — {clock} (elapsed: {_format_minutes(elapsed_min)})")
    lines.append(sep)
    lines.append(f"Overall: {bar} {pct:.1f}% ({total_done}/{total_target})")
    if rate_per_min > 0:
        lines.append(f"Recent throughput: {rate_per_min:.1f} calls/min | ETA: ~{eta_str}")
    else:
        lines.append(f"Recent throughput: 0.0 calls/min | ETA: {eta_str}")
    lines.append("")
    lines.append("Per physical slug:")

    pad = max((len(s) for s in per_slug_counts), default=0)
    for slug in sorted(per_slug_counts):
        done, target = per_slug_counts[slug]
        b = _make_progress_bar(done, target)
        sp = (100.0 * done / target) if target else 0.0
        status = _slug_status_label(slug_state.get(slug, ""))
        lines.append(f"  {slug:<{pad}}  {b} {sp:5.1f}% ({done}/{target}) {status}")

    incomplete = [(c, d, cur, exp) for (c, d, cur, exp) in per_combo_counts if cur < exp]
    if incomplete:
        lines.append("")
        lines.append("Incomplete combos:")
        for combo, dataset, cur, exp in incomplete:
            p = (100.0 * cur / exp) if exp else 0.0
            lines.append(
                f"  {combo.logical_name} / {combo.prompt_code} / {dataset}: {cur}/{exp} ({p:.1f}%)"
            )
    lines.append(sep)
    return "\n".join(lines)


def _make_shutdown_handler(event: threading.Event) -> Callable:
    """Return a SIGINT handler that flips ``event``."""

    def _handler(signum, frame):  # noqa: ARG001 — signal API
        if not event.is_set():
            sys.stderr.write(
                "\n[watchdog] Ctrl+C received; finishing in-flight call and exiting...\n"
            )
            event.set()

    return _handler


def _sleep_with_shutdown(total_sec: float, shutdown_event: threading.Event) -> None:
    end = time.monotonic() + total_sec
    while True:
        if shutdown_event.is_set():
            return
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(SHUTDOWN_POLL_SEC, remaining))


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _load_test_df(dataset: str) -> pd.DataFrame:
    df = read_parquet(DATASET_PATHS[dataset])
    if "held_out_fs" in df.columns:
        df = df[~df["held_out_fs"]].reset_index(drop=True)
    return df


def _group_combos_by_slug(combos: list[LogicalCombo]) -> dict[str, list[LogicalCombo]]:
    groups: dict[str, list[LogicalCombo]] = {}
    for c in combos:
        groups.setdefault(c.model_slug, []).append(c)
    return groups


def _default_labeler_factory(**kwargs) -> GroqLabeler:
    return GroqLabeler(**kwargs)


def _slug_worker(
    physical_slug: str,
    combos: list[LogicalCombo],
    dataset_dfs: dict[str, pd.DataFrame],
    few_shot_examples: dict[str, list[FewShotExample]],
    key_pool: list[str],
    slug_state: dict[str, str],
    state_lock: threading.Lock,
    shutdown_event: threading.Event,
    backoff_sequence: tuple[int, ...],
    labeler_factory: Callable[..., object] | None,
    log,
) -> None:
    """Drive all combos under one physical slug, rotating keys on RPD.

    Each iteration recomputes pending from disk (so another worker / process /
    rerun can have advanced state in the meantime). On every key in the pool
    failing to make any progress, sleeps with the next backoff value.
    """
    factory = labeler_factory or _default_labeler_factory

    def set_state(s: str) -> None:
        with state_lock:
            slug_state[physical_slug] = s

    set_state("active")
    backoff_attempt = 0
    keys_no_progress: set[int] = set()

    try:
        while True:
            if shutdown_event.is_set():
                set_state("done")
                return

            pending_full = compute_pending_work(combos)
            if not pending_full:
                set_state("done")
                return
            pending = [(c, d) for (c, d, _cur, _exp) in pending_full]

            if not key_pool:
                log.error("[%s] no API keys in pool; cannot proceed", physical_slug)
                set_state("crashed")
                return

            if len(keys_no_progress) >= len(key_pool):
                sleep_sec = backoff_seconds(backoff_attempt, backoff_sequence)
                backoff_attempt += 1
                set_state(f"sleeping_{max(1, sleep_sec // 60)}min")
                log.warning(
                    "[%s] all %d key(s) capped; sleeping %ds",
                    physical_slug,
                    len(key_pool),
                    sleep_sec,
                )
                _sleep_with_shutdown(sleep_sec, shutdown_event)
                keys_no_progress = set()
                continue

            key_idx = next(i for i in range(len(key_pool)) if i not in keys_no_progress)
            key = key_pool[key_idx]
            set_state("active")

            progress_this_round = 0
            for combo, dataset in pending:
                if shutdown_event.is_set():
                    set_state("done")
                    return

                client = factory(
                    model_slug=combo.model_slug,
                    api_key=key,
                    reasoning_effort=combo.reasoning_effort or None,
                    hide_reasoning=True,
                )
                fs_arg = (
                    few_shot_examples.get(dataset) if combo.prompt_code == "FS_CoT_RP_NA" else None
                )
                out_path = jsonl_path(combo.logical_name, combo.prompt_code, dataset)
                prev_count = _count_jsonl_lines(out_path)

                try:
                    summary = run_benchmark(
                        logical_name=combo.logical_name,
                        model_slug=combo.model_slug,
                        reasoning_effort=combo.reasoning_effort,
                        prompt_code=combo.prompt_code,
                        dataset=dataset,
                        df=dataset_dfs[dataset],
                        output_path=out_path,
                        groq_client=client,
                        few_shot_examples=fs_arg,
                    )
                    progress_this_round += int(summary.get("completed", 0))
                    log.warning(
                        "[%s] key#%d %s/%s/%s -> completed=%d skipped=%d",
                        physical_slug,
                        key_idx,
                        combo.logical_name,
                        combo.prompt_code,
                        dataset,
                        summary.get("completed", 0),
                        summary.get("skipped", 0),
                    )
                except RPDCapHitError:
                    new_count = _count_jsonl_lines(out_path)
                    progress_this_round += max(0, new_count - prev_count)
                    log.warning(
                        "[%s] RPD on key#%d for %s/%s/%s; rotating",
                        physical_slug,
                        key_idx,
                        combo.logical_name,
                        combo.prompt_code,
                        dataset,
                    )
                    set_state("rotating")
                    break

            if progress_this_round > 0:
                keys_no_progress.clear()
                backoff_attempt = 0
            else:
                keys_no_progress.add(key_idx)
    except Exception as exc:
        log.error(
            "[%s] worker crashed: %s\n%s",
            physical_slug,
            exc,
            traceback.format_exc(),
        )
        set_state("crashed")


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _snapshot_counts(
    combos: list[LogicalCombo],
) -> tuple[dict[str, tuple[int, int]], list[tuple[LogicalCombo, str, int, int]]]:
    physical_slugs = {c.model_slug for c in combos}
    per_slug: dict[str, list[int]] = {s: [0, 0] for s in physical_slugs}
    per_combo: list[tuple[LogicalCombo, str, int, int]] = []
    for combo in combos:
        for dataset in DATASETS:
            path = jsonl_path(combo.logical_name, combo.prompt_code, dataset)
            current = _count_jsonl_lines(path)
            expected = EXPECTED_PER_DATASET[dataset]
            clipped = min(current, expected)
            per_slug[combo.model_slug][0] += clipped
            per_slug[combo.model_slug][1] += expected
            per_combo.append((combo, dataset, clipped, expected))
    return {s: (v[0], v[1]) for s, v in per_slug.items()}, per_combo


def _progress_reporter(
    combos: list[LogicalCombo],
    slug_state: dict[str, str],
    state_lock: threading.Lock,
    shutdown_event: threading.Event,
    interval_min: int,
    log,
) -> None:
    """Print a progress board immediately, then every ``interval_min`` minutes."""
    start = time.monotonic()
    history: list[tuple[float, int]] = []
    interval_sec = max(1, interval_min) * 60

    def emit() -> None:
        now = time.monotonic()
        per_slug, per_combo = _snapshot_counts(combos)
        total_done = sum(v[0] for v in per_slug.values())
        total_target = sum(v[1] for v in per_slug.values())

        history.append((now, total_done))
        cutoff = now - 300
        history[:] = [(t, c) for (t, c) in history if t >= cutoff]

        rate_per_min = 0.0
        if len(history) >= 2:
            old_t, old_c = history[0]
            dt_min = (now - old_t) / 60.0
            if dt_min > 0:
                rate_per_min = max(0.0, (total_done - old_c) / dt_min)

        if rate_per_min > 0 and total_done < total_target:
            eta_str = _format_minutes((total_target - total_done) / rate_per_min)
        elif total_done >= total_target:
            eta_str = "0m"
        else:
            eta_str = "calculating..."

        with state_lock:
            state_snapshot = dict(slug_state)

        report = format_progress_report(
            start_time=start,
            now=now,
            per_slug_counts=per_slug,
            per_combo_counts=per_combo,
            slug_state=state_snapshot,
            rate_per_min=rate_per_min,
            eta_str=eta_str,
            total_done=total_done,
            total_target=total_target,
        )
        try:
            print(report, flush=True)
        except UnicodeEncodeError:
            print(report.encode("ascii", "replace").decode("ascii"), flush=True)
        log.info("progress total=%d/%d rate=%.1f", total_done, total_target, rate_per_min)

    emit()
    while not shutdown_event.is_set():
        deadline = time.monotonic() + interval_sec
        while time.monotonic() < deadline:
            if shutdown_event.is_set():
                break
            time.sleep(min(SHUTDOWN_POLL_SEC, deadline - time.monotonic()))
        emit()


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def _print_dry_run_plan(key_pool: list[str], combos: list[LogicalCombo]) -> None:
    print("Dry-run plan (no API calls will be made):")
    print(f"  API keys loaded: {len(key_pool)}")
    total = sum(EXPECTED_PER_DATASET[d] for _ in combos for d in DATASETS)
    print(f"  Combos: {len(combos)} logical x {len(DATASETS)} datasets")
    print(f"  Target rows (across all combos): {total}")
    pending = compute_pending_work(combos)
    print(f"  Pending (combo, dataset) pairs: {len(pending)}")
    for combo, dataset, cur, exp in pending:
        pct = (100.0 * cur / exp) if exp else 0.0
        print(
            f"    - {combo.logical_name} / {combo.prompt_code} / {dataset}: "
            f"{cur}/{exp} ({pct:.1f}%)"
        )


def _run_aggregator(log) -> int:
    agg = project_root() / "scripts" / "aggregate_llm_results.py"
    if not agg.exists():
        log.error("aggregator script not found at %s", agg)
        return 1
    log.warning("Running aggregator: %s", agg)
    result = subprocess.run([sys.executable, str(agg)], check=False)
    return int(result.returncode or 0)


def _final_summary(
    slug_state: dict[str, str],
    combos: list[LogicalCombo],
) -> tuple[int, int, int, list[str]]:
    """Return (completed_jsonl_count, total_rows, crashed_count, crashed_slugs)."""
    per_slug, per_combo = _snapshot_counts(combos)
    completed_files = sum(1 for (_c, _d, cur, exp) in per_combo if cur >= exp)
    total_rows = sum(cur for (_c, _d, cur, _exp) in per_combo)
    crashed = [s for s, st in slug_state.items() if st == "crashed"]
    return completed_files, total_rows, len(crashed), crashed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-managing watchdog for Phase 3 LLM benchmarks."
    )
    parser.add_argument("--check-interval-min", type=int, default=5)
    parser.add_argument("--max-sleep-min", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paid-tier-mode",
        action="store_true",
        help="Accepted for parity with run_all_benchmarks.py; key rotation in "
        "the watchdog already supersedes the original RPD-survival semantics.",
    )
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    load_env()
    log = get_logger("watchdog_benchmark", phase="phase3")

    key_pool = collect_api_keys()
    print(f"Loaded {len(key_pool)} API keys")
    log.warning("Loaded %d API keys from env", len(key_pool))

    if args.dry_run:
        _print_dry_run_plan(key_pool, LOGICAL_COMBOS)
        return 0

    if not key_pool:
        log.error("No GROQ_API_KEY[_2..5] set; cannot proceed.")
        return 2

    # Build backoff sequence respecting --max-sleep-min cap.
    cap_sec = max(60, args.max_sleep_min * 60)
    backoff_seq = tuple(min(s, cap_sec) for s in BACKOFF_SEQUENCE_SEC)

    fnn_df = _load_test_df("fakenewsnet")
    er_df = _load_test_df("employee_reviews")
    dataset_dfs = {"fakenewsnet": fnn_df, "employee_reviews": er_df}
    log.warning("Loaded %d FNN, %d ER test rows", len(fnn_df), len(er_df))

    fs_map: dict[str, list[FewShotExample]] = {
        "fakenewsnet": build_fnn_few_shot_examples(),
        "employee_reviews": build_er_few_shot_examples(),
    }
    log.warning(
        "Built %d FNN FS examples, %d ER FS examples",
        len(fs_map["fakenewsnet"]),
        len(fs_map["employee_reviews"]),
    )

    groups = _group_combos_by_slug(LOGICAL_COMBOS)
    log.warning("Launching %d workers: %s", len(groups), sorted(groups.keys()))

    shutdown_event = threading.Event()
    state_lock = threading.Lock()
    slug_state: dict[str, str] = {slug: "active" for slug in groups}

    # Install SIGINT handler so Ctrl+C lets workers wind down cleanly.
    prev_handler = signal.signal(signal.SIGINT, _make_shutdown_handler(shutdown_event))

    workers: list[threading.Thread] = []
    for slug, combos in groups.items():
        t = threading.Thread(
            target=_slug_worker,
            args=(
                slug,
                combos,
                dataset_dfs,
                fs_map,
                key_pool,
                slug_state,
                state_lock,
                shutdown_event,
                backoff_seq,
                None,  # default labeler factory (real Groq client)
                log,
            ),
            daemon=False,
            name=f"watchdog-{slug.replace('/', '-')}",
        )
        workers.append(t)

    reporter = threading.Thread(
        target=_progress_reporter,
        args=(
            LOGICAL_COMBOS,
            slug_state,
            state_lock,
            shutdown_event,
            args.check_interval_min,
            log,
        ),
        daemon=True,
        name="watchdog-reporter",
    )

    try:
        reporter.start()
        for t in workers:
            t.start()
        for t in workers:
            t.join()
    finally:
        shutdown_event.set()
        reporter.join(timeout=10)
        # Restore previous SIGINT handler.
        signal.signal(signal.SIGINT, prev_handler or signal.SIG_DFL)

    completed_files, total_rows, n_crashed, crashed_slugs = _final_summary(
        slug_state, LOGICAL_COMBOS
    )
    print("=" * 60)
    print("Watchdog complete.")
    print(f"  Completed JSONL files: {completed_files}/{len(LOGICAL_COMBOS) * len(DATASETS)}")
    print(f"  Total rows on disk:    {total_rows}")
    if crashed_slugs:
        print(f"  Crashed workers:       {n_crashed} ({', '.join(sorted(crashed_slugs))})")
    print("=" * 60)

    agg_rc = _run_aggregator(log)
    if agg_rc != 0:
        log.warning("aggregator exited %d", agg_rc)

    return 1 if n_crashed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
