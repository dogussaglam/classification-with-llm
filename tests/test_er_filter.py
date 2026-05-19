"""Tests for Employee Reviews keyword filter + text builder."""

from __future__ import annotations

import pandas as pd

from llm_textcls.data.employee_reviews import build_text, filter_by_keywords


def test_filter_remote_match():
    df = pd.DataFrame({"text": ["I work from home daily and love it."]})
    out = filter_by_keywords(df)
    assert out.iloc[0]["keyword_class"] == "remote"
    assert out.iloc[0]["keyword_match_count"] >= 1


def test_filter_tie_breaker_favors_not_remote():
    # Hits one keyword from each group → tie → not_remote wins.
    df = pd.DataFrame({"text": ["I work remote some days and in the office others."]})
    out = filter_by_keywords(df)
    assert out.iloc[0]["keyword_class"] == "not_remote"


def test_filter_zero_hits_not_mentioned():
    df = pd.DataFrame({"text": ["Great managers, good pay, supportive team."]})
    out = filter_by_keywords(df)
    assert out.iloc[0]["keyword_class"] == "not_mentioned"
    assert out.iloc[0]["keyword_match_count"] == 0


def test_filter_not_remote_match():
    df = pd.DataFrame({"text": ["The office is downtown, parking is hard, expected in 5 days."]})
    out = filter_by_keywords(df)
    assert out.iloc[0]["keyword_class"] == "not_remote"


def test_build_text_handles_nulls():
    row = pd.Series({"headline": "Good place", "pros": None, "cons": "Long hours"})
    assert build_text(row) == "Good place Long hours"


def test_build_text_all_null_returns_empty():
    row = pd.Series({"headline": None, "pros": None, "cons": None})
    assert build_text(row) == ""


def test_build_text_strips_whitespace():
    row = pd.Series({"headline": "  hi  ", "pros": "  pros body  ", "cons": ""})
    assert build_text(row) == "hi pros body"
