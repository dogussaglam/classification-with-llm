"""Few-shot example builders for FNN (2 examples) and ER (3 examples)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_textcls.io import project_root, read_parquet
from llm_textcls.llms.prompts import FewShotExample

FNN_LABEL_ORDER: tuple[str, ...] = ("fake", "real")
ER_LABEL_ORDER: tuple[str, ...] = ("remote", "not_remote", "not_mentioned")


def _default_fnn_sample_path() -> Path:
    return project_root() / "data" / "fakenewsnet" / "sample.parquet"


def _default_fnn_reasonings_path() -> Path:
    return project_root() / "data" / "fakenewsnet" / "fs_reasonings.json"


def _default_er_labeled_path() -> Path:
    return project_root() / "data" / "employee_reviews" / "held_out_fs_labeled.parquet"


def _default_er_reasonings_path() -> Path:
    return project_root() / "data" / "employee_reviews" / "fs_reasonings.json"


def _load_reasonings(path: Path) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(
            f"Missing few-shot reasonings file: {path}. Architect-supplied "
            "JSON must be present before benchmark runs (design §6.4)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_fnn_few_shot_examples(
    sample_path: Path | None = None,
    reasonings_path: Path | None = None,
) -> list[FewShotExample]:
    """Build the 2 FNN few-shot examples, ordered ``[fake, real]``.

    Reads ``sample.parquet`` and filters ``held_out_fs=True``, picking the
    first row per label in parquet row order. Reasoning text is read from
    the JSON file keyed by row id.

    Args:
        sample_path: Override for ``data/fakenewsnet/sample.parquet``.
        reasonings_path: Override for ``data/fakenewsnet/fs_reasonings.json``.

    Returns:
        List of exactly 2 :class:`FewShotExample`.

    Raises:
        RuntimeError: if either file is missing or a label class has no
            held-out example, or if the reasoning JSON is missing the row id.
    """
    sp = sample_path or _default_fnn_sample_path()
    rp = reasonings_path or _default_fnn_reasonings_path()
    df = read_parquet(sp)
    held = df[df["held_out_fs"]].reset_index(drop=True)
    reasonings = _load_reasonings(rp)

    out: list[FewShotExample] = []
    for label in FNN_LABEL_ORDER:
        rows = held[held["label"] == label]
        if rows.empty:
            raise RuntimeError(
                f"FNN held-out has no '{label}' row. Phase 1 OD-2 guarantees "
                "2 per class; check data/fakenewsnet/sample.parquet."
            )
        first = rows.iloc[0]
        rid = str(first["id"])
        if rid not in reasonings:
            raise RuntimeError(
                f"Missing reasoning for FNN row id '{rid}' in {rp}. "
                "Add an entry or update the architect-supplied JSON."
            )
        out.append(
            FewShotExample(
                text=str(first["text"]),
                label=label,
                reasoning=reasonings[rid],
            )
        )
    return out


def build_er_few_shot_examples(
    labeled_held_out_path: Path | None = None,
    reasonings_path: Path | None = None,
) -> list[FewShotExample]:
    """Build the 3 ER few-shot examples, ordered ``[remote, not_remote, not_mentioned]``.

    Reads ``held_out_fs_labeled.parquet`` (produced by
    ``scripts/label_held_out_fs.py --merge``), picks the first row per label
    in parquet row order, and pairs with reasoning text from JSON.

    Args:
        labeled_held_out_path: Override for
            ``data/employee_reviews/held_out_fs_labeled.parquet``.
        reasonings_path: Override for
            ``data/employee_reviews/fs_reasonings.json``.

    Returns:
        List of exactly 3 :class:`FewShotExample`.

    Raises:
        RuntimeError: if either file is missing, any class has no row, or
            the reasoning JSON is missing the chosen row id.
    """
    lp = labeled_held_out_path or _default_er_labeled_path()
    rp = reasonings_path or _default_er_reasonings_path()
    if not lp.exists():
        raise RuntimeError(
            f"Missing {lp}. Run `python scripts/label_held_out_fs.py` and then "
            "`python scripts/label_held_out_fs.py --merge` first."
        )
    df = read_parquet(lp)
    reasonings = _load_reasonings(rp)

    out: list[FewShotExample] = []
    for label in ER_LABEL_ORDER:
        rows = df[df["label"] == label]
        if rows.empty:
            raise RuntimeError(
                f"ER held-out labeled set has no '{label}' row. Edit "
                "data/employee_reviews/held_out_labeled.csv and re-run merge."
            )
        first = rows.iloc[0]
        rid = str(first["id"])
        if rid not in reasonings:
            raise RuntimeError(
                f"Missing reasoning for ER row id '{rid}' in {rp}. "
                "Add an entry to the JSON before running FS prompts."
            )
        out.append(
            FewShotExample(
                text=str(first["text"]),
                label=label,
                reasoning=reasonings[rid],
            )
        )
    return out
