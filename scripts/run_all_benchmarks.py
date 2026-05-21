"""Orchestrate all 11 logical combos × 2 datasets in 4 per-model threads.

Idempotent: a session-2 rerun after Groq's UTC RPD reset picks up where
session-1 left off, since each combo's JSONL is the source of truth for
"what's done".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread

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
    logical_name: str
    model_slug: str
    reasoning_effort: str  # "" if not applicable
    prompt_code: str


LOGICAL_COMBOS: list[LogicalCombo] = [
    # Llama 3.1 8B — paper-era anchor, 3 prompts
    LogicalCombo("llama-3.1-8b-instant", "llama-3.1-8b-instant", "", "ZS"),
    LogicalCombo("llama-3.1-8b-instant", "llama-3.1-8b-instant", "", "ZS_CoT"),
    LogicalCombo("llama-3.1-8b-instant", "llama-3.1-8b-instant", "", "FS_CoT_RP_NA"),
    # Llama 3.3 70B — non-reasoning flagship, 3 prompts
    LogicalCombo("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "", "ZS"),
    LogicalCombo("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "", "ZS_CoT"),
    LogicalCombo("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "", "FS_CoT_RP_NA"),
    # Qwen3 32B — reasoning ablation, ZS ONLY for both variants
    LogicalCombo("qwen3-32b-reasoning-none", "qwen/qwen3-32b", "none", "ZS"),
    LogicalCombo("qwen3-32b-reasoning-default", "qwen/qwen3-32b", "default", "ZS"),
    # GPT-OSS 120B — Axis B headliner, 3 prompts
    LogicalCombo("openai-gpt-oss-120b", "openai/gpt-oss-120b", "medium", "ZS"),
    LogicalCombo("openai-gpt-oss-120b", "openai/gpt-oss-120b", "medium", "ZS_CoT"),
    LogicalCombo("openai-gpt-oss-120b", "openai/gpt-oss-120b", "medium", "FS_CoT_RP_NA"),
]

DATASETS = ("fakenewsnet", "employee_reviews")
DATASET_PATHS = {
    "fakenewsnet": project_root() / "data" / "fakenewsnet" / "sample.parquet",
    "employee_reviews": project_root() / "data" / "employee_reviews" / "labeled.parquet",
}


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


def _model_worker(
    physical_slug: str,
    combos: list[LogicalCombo],
    fnn_df: pd.DataFrame,
    er_df: pd.DataFrame,
    fnn_fs: list[FewShotExample],
    er_fs: list[FewShotExample],
    paid_tier_mode: bool,
    status_q: Queue,
    log,
) -> None:
    try:
        for combo in combos:
            client = GroqLabeler(
                model_slug=combo.model_slug,
                reasoning_effort=combo.reasoning_effort or None,
                hide_reasoning=True,
            )
            for dataset, df, fs in [
                ("fakenewsnet", fnn_df, fnn_fs),
                ("employee_reviews", er_df, er_fs),
            ]:
                out_path = jsonl_path(combo.logical_name, combo.prompt_code, dataset)
                fs_arg = fs if combo.prompt_code == "FS_CoT_RP_NA" else None
                try:
                    summary = run_benchmark(
                        logical_name=combo.logical_name,
                        model_slug=combo.model_slug,
                        reasoning_effort=combo.reasoning_effort,
                        prompt_code=combo.prompt_code,
                        dataset=dataset,
                        df=df,
                        output_path=out_path,
                        groq_client=client,
                        few_shot_examples=fs_arg,
                    )
                    status_q.put((combo, dataset, summary, None))
                    log.warning(
                        "Done %s/%s/%s: %s",
                        combo.logical_name,
                        combo.prompt_code,
                        dataset,
                        summary,
                    )
                except RPDCapHitError as e:
                    status_q.put((combo, dataset, None, e))
                    log.warning(
                        "RPD cap hit on %s (%s/%s/%s); pausing remaining combos for this model.",
                        physical_slug,
                        combo.logical_name,
                        combo.prompt_code,
                        dataset,
                    )
                    if not paid_tier_mode:
                        return
    except Exception as e:  # noqa: BLE001 — surface worker crashes via status_q
        log.exception("Worker for %s crashed: %s", physical_slug, e)
        status_q.put((None, None, None, e))


def _drain(q: Queue) -> list[tuple]:
    out: list[tuple] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _print_session_summary(statuses: list[tuple]) -> None:
    completed = sum(1 for s in statuses if s[2] is not None)
    rpd = sum(1 for s in statuses if isinstance(s[3], RPDCapHitError))
    crashed = sum(1 for s in statuses if s[3] is not None and not isinstance(s[3], RPDCapHitError))
    total_calls = sum(s[2]["completed"] for s in statuses if s[2] is not None)
    print()
    print("=" * 60)
    print(f"Session complete: {completed} combos finished, {rpd} RPD-capped, {crashed} crashed.")
    print(f"Calls completed this session: {total_calls}")
    if rpd:
        print("\nRPD-capped combos (rerun after UTC daily reset):")
        for s in statuses:
            if isinstance(s[3], RPDCapHitError):
                print(f"  - {s[0].logical_name} / {s[0].prompt_code} / {s[1]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Phase 3 LLM benchmarks.")
    parser.add_argument(
        "--paid-tier-mode",
        action="store_true",
        help="On 429: treat Retry-After>60s as transient (sleep & retry) instead of raising.",
    )
    args = parser.parse_args()

    load_env()
    log = get_logger("run_all_benchmarks", phase="phase3")

    fnn_df = _load_test_df("fakenewsnet")
    er_df = _load_test_df("employee_reviews")
    log.warning("Loaded %d FNN test rows, %d ER test rows.", len(fnn_df), len(er_df))

    fnn_fs = build_fnn_few_shot_examples()
    er_fs = build_er_few_shot_examples()
    log.warning("Built %d FNN FS examples, %d ER FS examples.", len(fnn_fs), len(er_fs))

    groups = _group_combos_by_slug(LOGICAL_COMBOS)
    log.warning(
        "Launching %d threads (one per physical slug): %s",
        len(groups),
        sorted(groups.keys()),
    )

    status_q: Queue = Queue()
    threads = [
        Thread(
            target=_model_worker,
            args=(slug, combos, fnn_df, er_df, fnn_fs, er_fs, args.paid_tier_mode, status_q, log),
            daemon=False,
            name=f"worker-{slug.replace('/', '-')}",
        )
        for slug, combos in groups.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = _drain(status_q)
    _print_session_summary(statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
