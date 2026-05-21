"""Idempotent JSONL-append benchmark loop and result schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from llm_textcls.io import project_root
from llm_textcls.llms.groq_client import CompletionResult, GroqLabeler, RPDCapHitError
from llm_textcls.llms.parsers import parse_label
from llm_textcls.llms.prompts import PROMPT_RENDERERS, FewShotExample


@dataclass(frozen=True)
class LLMCallResult:
    """One LLM call result, written as a single JSONL line."""

    record_id: str
    dataset: str
    model: str
    model_slug: str
    reasoning_effort: str
    prompt_code: str
    raw_output: str
    predicted_label: str
    true_label: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    error: str
    timestamp_utc: str


def jsonl_path(logical_name: str, prompt_code: str, dataset: str) -> Path:
    """Compute the canonical JSONL output path for a combo.

    Slashes in ``logical_name`` are replaced with ``-`` for filesystem safety.

    Args:
        logical_name: Combo logical name (e.g. ``"qwen3-32b-reasoning-none"``).
        prompt_code: ``"ZS"`` / ``"ZS_CoT"`` / ``"FS_CoT_RP_NA"``.
        dataset: ``"fakenewsnet"`` / ``"employee_reviews"``.

    Returns:
        Absolute path under ``results/llms/``.
    """
    safe = logical_name.replace("/", "-")
    return project_root() / "results" / "llms" / f"{safe}__{prompt_code}__{dataset}.jsonl"


def _existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = obj.get("record_id")
        if rid:
            done.add(str(rid))
    return done


def _render_prompt(
    dataset: str,
    prompt_code: str,
    text: str,
    few_shot_examples: list[FewShotExample] | None,
) -> str:
    renderer = PROMPT_RENDERERS[(dataset, prompt_code)]
    if prompt_code == "FS_CoT_RP_NA":
        if few_shot_examples is None:
            raise RuntimeError(f"FS_CoT_RP_NA requires few_shot_examples for {dataset}; got None.")
        return renderer(text=text, examples=few_shot_examples)
    return renderer(text=text)


def run_benchmark(
    logical_name: str,
    model_slug: str,
    reasoning_effort: str,
    prompt_code: str,
    dataset: str,
    df: pd.DataFrame,
    output_path: Path,
    groq_client: GroqLabeler,
    few_shot_examples: list[FewShotExample] | None = None,
) -> dict:
    """Run one (logical_name, prompt_code, dataset) combination idempotently.

    Reads ``output_path`` to collect ``record_id``s already processed and
    skips them. Appends one JSONL line per call, flushing after each line so
    a mid-loop crash leaves a valid prefix file.

    Args:
        logical_name: Combo logical name written into each row's ``model``.
        model_slug: Physical Groq slug (echoed into each row's ``model_slug``).
        reasoning_effort: ``""`` if not applicable; otherwise ``"none"`` /
            ``"default"`` / ``"medium"`` etc.
        prompt_code: One of ``"ZS"``, ``"ZS_CoT"``, ``"FS_CoT_RP_NA"``.
        dataset: ``"fakenewsnet"`` or ``"employee_reviews"``.
        df: Test rows (must already have ``held_out_fs=False`` filter applied).
        output_path: JSONL file to append to.
        groq_client: A :class:`GroqLabeler` already configured for this combo.
        few_shot_examples: Required when ``prompt_code == "FS_CoT_RP_NA"``.

    Returns:
        Summary dict with ``total``, ``completed``, ``parse_failures``,
        ``api_failures``, ``rpd_capped``, ``skipped``.

    Raises:
        RPDCapHitError: bubbles up so the orchestrator can pause this model.
        RuntimeError: if the test DF still contains ``held_out_fs=True`` rows.
    """
    if "held_out_fs" in df.columns and bool(df["held_out_fs"].any()):
        raise RuntimeError(
            "run_benchmark received test rows with held_out_fs=True; filter before calling."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_record_ids(output_path)
    summary: dict = {
        "total": len(df),
        "completed": 0,
        "parse_failures": 0,
        "api_failures": 0,
        "rpd_capped": False,
        "skipped": 0,
    }

    with output_path.open("a", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            rid = str(row.id)
            if rid in done:
                summary["skipped"] += 1
                continue
            prompt = _render_prompt(dataset, prompt_code, str(row.text), few_shot_examples)
            err = ""
            try:
                cr = groq_client.complete(prompt)
            except RPDCapHitError:
                summary["rpd_capped"] = True
                raise
            except Exception as e:  # noqa: BLE001 — record and continue
                cr = CompletionResult(
                    text="",
                    reasoning_text="",
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    model=model_slug,
                    error=f"API_ERROR:{type(e).__name__}",
                )
                summary["api_failures"] += 1
                err = cr.error

            pred_label, _reasoning = parse_label(cr.text, dataset) if cr.text else ("", "")
            if not err and cr.text and not pred_label:
                err = "PARSE_FAIL"
                summary["parse_failures"] += 1
            elif cr.error and not err:
                err = cr.error

            result = LLMCallResult(
                record_id=rid,
                dataset=dataset,
                model=logical_name,
                model_slug=model_slug,
                reasoning_effort=reasoning_effort or "",
                prompt_code=prompt_code,
                raw_output=cr.text,
                predicted_label=pred_label,
                true_label=str(row.label),
                latency_ms=cr.latency_ms,
                input_tokens=cr.input_tokens,
                output_tokens=cr.output_tokens,
                reasoning_tokens=cr.reasoning_tokens,
                error=err,
                timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            fh.flush()
            summary["completed"] += 1

    return summary
