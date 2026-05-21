"""Tests for the strict + lenient JSON label parser."""

from __future__ import annotations

from llm_textcls.llms.parsers import parse_label


def test_clean_fnn_json():
    raw = '{"label": "fake", "reasoning": "sensational headline"}'
    assert parse_label(raw, "fakenewsnet") == ("fake", "sensational headline")


def test_clean_er_json():
    raw = '{"label": "remote", "reasoning": "wfh mentioned"}'
    assert parse_label(raw, "employee_reviews") == ("remote", "wfh mentioned")


def test_json_in_markdown_json_fence():
    raw = '```json\n{"label": "real", "reasoning": "verifiable"}\n```'
    assert parse_label(raw, "fakenewsnet") == ("real", "verifiable")


def test_json_in_plain_fence():
    raw = '```\n{"label": "not_remote", "reasoning": "office"}\n```'
    assert parse_label(raw, "employee_reviews") == ("not_remote", "office")


def test_json_with_prose_preamble():
    raw = (
        "Let me think about this. The article cites specific dates and named "
        "sources, so it looks legitimate. Final answer:\n"
        '{"label": "real", "reasoning": "named sources"}'
    )
    label, reasoning = parse_label(raw, "fakenewsnet")
    assert label == "real"
    assert reasoning == "named sources"


def test_json_inside_think_tags_followed_by_final_json():
    raw = (
        "<think>\nThe review explicitly mentions work from home.\n"
        '{"label": "should_not_use", "reasoning": "draft"}\n</think>\n'
        '{"label": "remote", "reasoning": "wfh signal"}'
    )
    assert parse_label(raw, "employee_reviews") == ("remote", "wfh signal")


def test_two_json_blocks_takes_last():
    raw = (
        '{"label": "fake", "reasoning": "draft"}\n'
        "After reconsidering, the named sources are verifiable.\n"
        '{"label": "real", "reasoning": "final answer"}'
    )
    label, reasoning = parse_label(raw, "fakenewsnet")
    assert label == "real"
    assert reasoning == "final answer"


def test_uppercased_label_fails():
    raw = '{"label": "REMOTE", "reasoning": "case wrong"}'
    assert parse_label(raw, "employee_reviews") == ("", "")


def test_misspelled_label_fails():
    raw = '{"label": "remote_work", "reasoning": "wrong slug"}'
    assert parse_label(raw, "employee_reviews") == ("", "")


def test_missing_closing_brace_fails():
    raw = '{"label": "fake", "reasoning": "no close"'
    assert parse_label(raw, "fakenewsnet") == ("", "")


def test_empty_output_fails():
    assert parse_label("", "fakenewsnet") == ("", "")
    assert parse_label("", "employee_reviews") == ("", "")


def test_label_without_reasoning_still_works():
    raw = '{"label": "fake"}'
    assert parse_label(raw, "fakenewsnet") == ("fake", "")
