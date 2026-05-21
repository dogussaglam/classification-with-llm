"""CLI for one (logical_name, model_slug, reasoning_effort, prompt_code, dataset) combo.

Idempotent: appends to the canonical JSONL output path; skips records already done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.io import load_env, project_root, read_parquet  # noqa: E402
from llm_textcls.llms.few_shot import (  # noqa: E402
    build_er_few_shot_examples,
    build_fnn_few_shot_examples,
)
from llm_textcls.llms.groq_client import GroqLabeler, RPDCapHitError  # noqa: E402
from llm_textcls.llms.runner import jsonl_path, run_benchmark  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402

DATASET_PATHS = {
    "fakenewsnet": ("data", "fakenewsnet", "sample.parquet"),
    "employee_reviews": ("data", "employee_reviews", "labeled.parquet"),
}


def _load_test_df(dataset: str) -> pd.DataFrame:
    path = project_root().joinpath(*DATASET_PATHS[dataset])
    df = read_parquet(path)
    if "held_out_fs" in df.columns:
        df = df[~df["held_out_fs"]].reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one LLM benchmark combo.")
    parser.add_argument("--logical-name", required=True, help="Combo logical name.")
    parser.add_argument("--model-slug", required=True, help="Physical Groq slug.")
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help='"" / "none" / "default" / "medium" etc.',
    )
    parser.add_argument(
        "--prompt-code",
        required=True,
        choices=["ZS", "ZS_CoT", "FS_CoT_RP_NA"],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["fakenewsnet", "employee_reviews"],
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override; default is 1024 for reasoning models, 300 otherwise.",
    )
    parser.add_argument(
        "--no-hide-reasoning",
        action="store_true",
        help="Disable reasoning_format=hidden (debug only).",
    )
    args = parser.parse_args()

    load_env()
    log = get_logger("run_llm_benchmark", phase="phase3")

    df = _load_test_df(args.dataset)
    log.warning("Dataset %s: %d test rows (held_out filtered).", args.dataset, len(df))

    fs_examples = None
    if args.prompt_code == "FS_CoT_RP_NA":
        fs_examples = (
            build_fnn_few_shot_examples()
            if args.dataset == "fakenewsnet"
            else build_er_few_shot_examples()
        )

    reasoning_effort_param = args.reasoning_effort or None
    client = GroqLabeler(
        model_slug=args.model_slug,
        reasoning_effort=reasoning_effort_param,
        hide_reasoning=not args.no_hide_reasoning,
        max_tokens=args.max_tokens,
    )

    out_path = jsonl_path(args.logical_name, args.prompt_code, args.dataset)
    log.warning("Output: %s", out_path)

    try:
        summary = run_benchmark(
            logical_name=args.logical_name,
            model_slug=args.model_slug,
            reasoning_effort=args.reasoning_effort,
            prompt_code=args.prompt_code,
            dataset=args.dataset,
            df=df,
            output_path=out_path,
            groq_client=client,
            few_shot_examples=fs_examples,
        )
    except RPDCapHitError as e:
        log.warning(
            "RPD cap hit for %s on %s/%s/%s: %s",
            e.model,
            args.logical_name,
            args.prompt_code,
            args.dataset,
            e,
        )
        print(f"RPD_CAP {args.logical_name} {args.prompt_code} {args.dataset}")
        return 3

    log.warning("Done %s/%s/%s: %s", args.logical_name, args.prompt_code, args.dataset, summary)
    print(f"OK {args.logical_name} {args.prompt_code} {args.dataset} {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
