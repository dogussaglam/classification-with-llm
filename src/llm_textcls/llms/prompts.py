"""Six prompt templates: 2 datasets x 3 strategies (ZS, ZS_CoT, FS_CoT_RP_NA)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    """One few-shot example: text + ground-truth label + hand-written reasoning."""

    text: str
    label: str
    reasoning: str


# ---------------------------------------------------------------------------
# FakeNewsNet templates
# ---------------------------------------------------------------------------

_FNN_ZS = """Classify the following news article as either "fake" or "real".

Article:
{text}

Output strict JSON with exactly this schema and nothing else:
{{"label": "fake" | "real", "reasoning": "<one short sentence>"}}

Output JSON only. No prose, no markdown fences.
"""

_FNN_ZS_COT = """Classify the following news article as either "fake" or "real".

Article:
{text}

Before producing the JSON, think step by step about the cues that
support your decision — sensationalism, verifiability of named entities
and dates, and source credibility.

Output strict JSON with exactly this schema and nothing else:
{{"label": "fake" | "real", "reasoning": "<one short sentence summarising the decisive cue>"}}

Output JSON only. No prose, no markdown fences.
"""

_FNN_FS_HEADER = """You are Claire, an experienced fact-checking journalist who has reviewed
thousands of news articles. You decide each case by examining sensationalism,
verifiability of named entities and dates, and source credibility.

Given the examples below, classify the new article as "fake" or "real".

"""

_FNN_FS_FOOTER = """Now classify this article:
{text}

Think step by step about the same cues, then output strict JSON with
exactly this schema and nothing else:
{{"label": "fake" | "real", "reasoning": "<one short sentence summarising the decisive cue>"}}

Output JSON only. No prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Employee Reviews templates
# ---------------------------------------------------------------------------

_ER_RUBRIC = """Classify the following employee review based on whether it mentions
the work-location arrangement. Choose exactly one label:
- "remote": review explicitly mentions remote work, work-from-home,
  telecommuting, or other off-site arrangements.
- "not_remote": review explicitly mentions on-site work, the office,
  in-person work, or commuting.
- "not_mentioned": review does not address work location either way.

Review:
{text}
"""

_ER_ZS = (
    _ER_RUBRIC
    + """
Output strict JSON with exactly this schema and nothing else:
{{"label": "remote" | "not_remote" | "not_mentioned", "reasoning": "<one short sentence>"}}

Output JSON only. No prose, no markdown fences.
"""
)

_ER_ZS_COT = (
    _ER_RUBRIC
    + """
Before producing the JSON, think step by step about the cues that
support your decision — direct mentions of remote work, commute,
office, or any other workplace-arrangement signal.

Output strict JSON with exactly this schema and nothing else:
{{"label": "remote" | "not_remote" | "not_mentioned", "reasoning": "<one short sentence summarising the decisive cue>"}}

Output JSON only. No prose, no markdown fences.
"""
)

_ER_FS_HEADER = """You are Mark, an HR analyst who has reviewed thousands of employee
reviews to extract workplace-arrangement signals. You decide each case
by looking for direct mentions of remote work, on-site work, or the
absence of either.

Given the examples below, classify the new review. Choose exactly one
label: "remote", "not_remote", or "not_mentioned".

"""

_ER_FS_FOOTER = """Now classify this review:
{text}

Think step by step about the same cues, then output strict JSON with
exactly this schema and nothing else:
{{"label": "remote" | "not_remote" | "not_mentioned", "reasoning": "<one short sentence summarising the decisive cue>"}}

Output JSON only. No prose, no markdown fences.
"""


def _format_example_block(idx: int, item_label: str, ex: FewShotExample) -> str:
    return (
        f"Example {idx}:\n{item_label}: {ex.text}\nReasoning: {ex.reasoning}\nLabel: {ex.label}\n\n"
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_fnn_zs(text: str) -> str:
    """Render the FakeNewsNet zero-shot prompt.

    Args:
        text: Article body.

    Returns:
        Fully populated prompt string.
    """
    return _FNN_ZS.format(text=text)


def render_fnn_zs_cot(text: str) -> str:
    """Render the FakeNewsNet zero-shot CoT prompt.

    Args:
        text: Article body.

    Returns:
        Fully populated prompt string.
    """
    return _FNN_ZS_COT.format(text=text)


def render_fnn_fs_cot_rp_na(text: str, examples: list[FewShotExample]) -> str:
    """Render the FakeNewsNet FS+CoT+RolePlay+NamedAssistant prompt.

    Args:
        text: Article body to classify.
        examples: List of exactly 2 FewShotExample, ordered ``[fake, real]``.

    Returns:
        Fully populated prompt string.

    Raises:
        ValueError: if ``len(examples) != 2``.
    """
    if len(examples) != 2:
        raise ValueError(f"FNN FS prompt needs exactly 2 examples, got {len(examples)}")
    body = _FNN_FS_HEADER
    for i, ex in enumerate(examples, start=1):
        body += _format_example_block(i, "Article", ex)
    body += _FNN_FS_FOOTER.format(text=text)
    return body


def render_er_zs(text: str) -> str:
    """Render the Employee Reviews zero-shot prompt.

    Args:
        text: Review text.

    Returns:
        Fully populated prompt string.
    """
    return _ER_ZS.format(text=text)


def render_er_zs_cot(text: str) -> str:
    """Render the Employee Reviews zero-shot CoT prompt.

    Args:
        text: Review text.

    Returns:
        Fully populated prompt string.
    """
    return _ER_ZS_COT.format(text=text)


def render_er_fs_cot_rp_na(text: str, examples: list[FewShotExample]) -> str:
    """Render the Employee Reviews FS+CoT+RolePlay+NamedAssistant prompt.

    Args:
        text: Review text to classify.
        examples: List of exactly 3 FewShotExample, ordered
            ``[remote, not_remote, not_mentioned]``.

    Returns:
        Fully populated prompt string.

    Raises:
        ValueError: if ``len(examples) != 3``.
    """
    if len(examples) != 3:
        raise ValueError(f"ER FS prompt needs exactly 3 examples, got {len(examples)}")
    body = _ER_FS_HEADER
    for i, ex in enumerate(examples, start=1):
        body += _format_example_block(i, "Review", ex)
    body += _ER_FS_FOOTER.format(text=text)
    return body


PROMPT_RENDERERS: dict[tuple[str, str], Callable] = {
    ("fakenewsnet", "ZS"): render_fnn_zs,
    ("fakenewsnet", "ZS_CoT"): render_fnn_zs_cot,
    ("fakenewsnet", "FS_CoT_RP_NA"): render_fnn_fs_cot_rp_na,
    ("employee_reviews", "ZS"): render_er_zs,
    ("employee_reviews", "ZS_CoT"): render_er_zs_cot,
    ("employee_reviews", "FS_CoT_RP_NA"): render_er_fs_cot_rp_na,
}
