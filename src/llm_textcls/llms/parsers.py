"""Strict + lenient JSON label parser. Handles <think> tags and prose preambles."""

from __future__ import annotations

import json
import re

VALID_LABELS: dict[str, tuple[str, ...]] = {
    "fakenewsnet": ("fake", "real"),
    "employee_reviews": ("remote", "not_remote", "not_mentioned"),
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text)


def _strip_markdown_fences(text: str) -> str:
    """Remove outer ```json ... ``` or ``` ... ``` fences."""
    return _FENCE_RE.sub("", text)


def _last_balanced_json_object(text: str) -> str | None:
    """Return the LAST balanced ``{...}`` substring, or None if absent.

    Scans the text, tracking brace depth and respecting strings + escapes,
    yielding every top-level object boundary. The lenient parser prefers the
    final object because reasoning models typically emit prose first and the
    final-answer JSON last.
    """
    starts: list[int] = []
    depth = 0
    in_str = False
    escape = False
    candidates: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                starts.append(i)
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and starts:
                    candidates.append((starts.pop(), i + 1))
    if not candidates:
        return None
    s, e = candidates[-1]
    return text[s:e]


def parse_label(raw_output: str, dataset: str) -> tuple[str, str]:
    """Extract (label, reasoning) from a model response.

    Strategy (defensive, per design §5.4): strip ``<think>`` blocks; strip
    markdown fences; ``json.loads``; on failure scan for the last balanced
    ``{...}`` substring and parse that; validate ``label`` is in the dataset's
    valid set.

    Args:
        raw_output: Raw assistant text (``message.content``).
        dataset: ``"fakenewsnet"`` or ``"employee_reviews"``.

    Returns:
        ``(label, reasoning)`` on success; ``("", "")`` on any failure.

    Raises:
        KeyError: if ``dataset`` is not in :data:`VALID_LABELS`.
    """
    valid = VALID_LABELS[dataset]
    if not raw_output:
        return ("", "")

    stripped = _strip_think_blocks(raw_output).strip()
    fenced = _strip_markdown_fences(stripped).strip()

    for candidate in (fenced, stripped, raw_output):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            label = str(obj.get("label", "")).strip()
            if label in valid:
                reasoning = str(obj.get("reasoning", "")).strip()
                return (label, reasoning)
            return ("", "")

    lenient = _last_balanced_json_object(raw_output)
    if lenient is not None:
        try:
            obj = json.loads(lenient)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            label = str(obj.get("label", "")).strip()
            if label in valid:
                reasoning = str(obj.get("reasoning", "")).strip()
                return (label, reasoning)
    return ("", "")
