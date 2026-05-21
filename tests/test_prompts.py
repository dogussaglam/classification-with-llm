"""Tests for prompt template renderers."""

from __future__ import annotations

import pytest

from llm_textcls.llms.prompts import (
    FewShotExample,
    render_er_fs_cot_rp_na,
    render_er_zs,
    render_er_zs_cot,
    render_fnn_fs_cot_rp_na,
    render_fnn_zs,
    render_fnn_zs_cot,
)


def _fnn_examples() -> list[FewShotExample]:
    return [
        FewShotExample(text="Fake article body 1.", label="fake", reasoning="r-fake-1"),
        FewShotExample(text="Real article body 1.", label="real", reasoning="r-real-1"),
    ]


def _er_examples() -> list[FewShotExample]:
    return [
        FewShotExample(text="WFH review.", label="remote", reasoning="r-remote"),
        FewShotExample(text="On site review.", label="not_remote", reasoning="r-onsite"),
        FewShotExample(text="Generic review.", label="not_mentioned", reasoning="r-na"),
    ]


def test_fnn_zs_inserts_text():
    out = render_fnn_zs("ARTICLE_BODY_TOKEN")
    assert "ARTICLE_BODY_TOKEN" in out
    assert "{text}" not in out
    assert out.strip()


def test_fnn_zs_cot_inserts_text_and_cot_hint():
    out = render_fnn_zs_cot("ARTICLE_BODY_TOKEN")
    assert "ARTICLE_BODY_TOKEN" in out
    assert "step by step" in out.lower()
    assert "{text}" not in out


def test_fnn_fs_inserts_all_examples_and_role_name():
    exs = _fnn_examples()
    out = render_fnn_fs_cot_rp_na("NEW_ARTICLE", exs)
    assert "Claire" in out
    assert "NEW_ARTICLE" in out
    for ex in exs:
        assert ex.text in out
        assert ex.reasoning in out
        assert ex.label in out
    assert out.index("Example 1") < out.index("Example 2") < out.index("Now classify")


def test_fnn_fs_rejects_wrong_example_count():
    with pytest.raises(ValueError):
        render_fnn_fs_cot_rp_na("X", _fnn_examples()[:1])


def test_er_zs_lists_three_labels():
    out = render_er_zs("REVIEW_TEXT")
    assert "REVIEW_TEXT" in out
    for lbl in ("remote", "not_remote", "not_mentioned"):
        assert lbl in out


def test_er_zs_cot_has_cot_hint():
    out = render_er_zs_cot("REVIEW_TEXT")
    assert "REVIEW_TEXT" in out
    assert "step by step" in out.lower()


def test_er_fs_inserts_all_examples_and_role_name():
    exs = _er_examples()
    out = render_er_fs_cot_rp_na("NEW_REVIEW", exs)
    assert "Mark" in out
    assert "NEW_REVIEW" in out
    for ex in exs:
        assert ex.text in out
        assert ex.reasoning in out
        assert ex.label in out
    # Examples appear in order
    indices = [out.index(f"Example {i}") for i in range(1, 4)]
    assert indices == sorted(indices)


def test_er_fs_rejects_wrong_example_count():
    with pytest.raises(ValueError):
        render_er_fs_cot_rp_na("X", _er_examples()[:2])
