"""Tests for tokenization.count_tokens."""

from __future__ import annotations

from llm_textcls.data.tokenization import count_tokens


def test_empty_and_non_string():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0  # type: ignore[arg-type]
    assert count_tokens(123) == 0  # type: ignore[arg-type]


def test_deterministic_known_text():
    a = count_tokens("hello world")
    b = count_tokens("hello world")
    assert a == b
    assert a > 0


def test_longer_text_more_tokens():
    short = count_tokens("hi")
    long = count_tokens("hi " * 100)
    assert long > short
