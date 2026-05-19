"""Shared tokenizer for length filtering (cl100k_base, cached)."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the cl100k_base token count for `text`.

    Args:
        text: Input string. Non-strings and empty strings return 0.

    Returns:
        Token count.
    """
    if not isinstance(text, str) or not text:
        return 0
    return len(_enc().encode(text, disallowed_special=()))
