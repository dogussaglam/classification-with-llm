"""Employee Reviews (Glassdoor) acquisition + keyword pre-classification."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

from llm_textcls.data.tokenization import count_tokens

DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "remote": [
        "remote",
        "work from home",
        "wfh",
        "telecommute",
        "telecommuting",
        "telework",
        "remotely",
        "from home",
        "distributed team",
        "fully remote",
    ],
    "not_remote": [
        "office",
        "on-site",
        "on site",
        "onsite",
        "in person",
        "in-person",
        "commute",
        "commuting",
        "headquarters",
        "on premises",
        "on-premises",
        "in the office",
        "return to office",
        "rto",
    ],
}


def build_text(row: pd.Series) -> str:
    """Concatenate the non-empty headline/pros/cons text fields.

    Args:
        row: Pandas row with at least headline, pros, cons columns.

    Returns:
        Single space-joined string of non-empty parts.
    """
    parts = [row.get("headline"), row.get("pros"), row.get("cons")]
    parts = [p.strip() for p in parts if p and isinstance(p, str) and p.strip()]
    return " ".join(parts)


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read glassdoor_reviews.csv with only the 3 text columns, build text + id.

    Args:
        raw_dir: Directory containing ``glassdoor_reviews.csv``.

    Returns:
        DataFrame with columns id, text. Rows where both pros and cons are
        empty are dropped, as are rows with len(text) < 20.

    Raises:
        FileNotFoundError: if the CSV is missing.
    """
    csv_path = raw_dir / "glassdoor_reviews.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download davidgauthier/glassdoor-job-reviews from Kaggle."
        )

    df = pd.read_csv(csv_path, usecols=["headline", "pros", "cons"])
    pros_ok = df["pros"].fillna("").astype(str).str.strip().ne("")
    cons_ok = df["cons"].fillna("").astype(str).str.strip().ne("")
    df = df[pros_ok | cons_ok].reset_index(drop=True)

    df["text"] = df.apply(build_text, axis=1)
    df = df[df["text"].str.len() >= 20].reset_index(drop=True)
    df["id"] = df["text"].map(lambda s: "er_" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:12])
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df[["id", "text"]]


def _compile_patterns(keyword_groups: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    compiled: dict[str, list[re.Pattern]] = {}
    for group, keywords in keyword_groups.items():
        compiled[group] = [
            re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords
        ]
    return compiled


def filter_by_keywords(
    df: pd.DataFrame,
    keyword_groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Assign a provisional `keyword_class` to each row from regex hit counts.

    Tie-breaker: `not_remote` wins. Zero hits → `not_mentioned`. Provisional
    only — used for sampling diversity, NOT as a final label.

    Args:
        df: DataFrame with a `text` column.
        keyword_groups: Mapping of class name → keyword list. Defaults to DEFAULT_KEYWORDS.

    Returns:
        Copy with new `keyword_class` and `keyword_match_count` columns.
    """
    groups = keyword_groups if keyword_groups is not None else DEFAULT_KEYWORDS
    patterns = _compile_patterns(groups)
    group_names = list(patterns.keys())

    classes: list[str] = []
    counts: list[int] = []
    for text in df["text"].fillna("").astype(str).tolist():
        hits = {g: sum(1 for p in patterns[g] if p.search(text)) for g in group_names}
        total = sum(hits.values())
        if total == 0:
            classes.append("not_mentioned")
            counts.append(0)
            continue
        max_hits = max(hits.values())
        winners = [g for g in group_names if hits[g] == max_hits]
        if len(winners) == 1:
            classes.append(winners[0])
        elif "not_remote" in winners:
            classes.append("not_remote")
        else:
            classes.append(winners[0])
        counts.append(total)

    out = df.copy()
    out["keyword_class"] = classes
    out["keyword_match_count"] = pd.Series(counts, dtype="int64").values
    return out


def sample_candidates(
    df: pd.DataFrame,
    n_per_class: int = 70,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic shuffle, then take n_per_class of each keyword_class.

    Args:
        df: DataFrame with a `keyword_class` column.
        n_per_class: Target rows per class.
        seed: RNG seed.

    Returns:
        DataFrame of (approximately) n_per_class * 3 rows.

    Raises:
        ValueError: if a class has fewer than n_per_class rows.
    """
    parts: list[pd.DataFrame] = []
    for cls, group in df.groupby("keyword_class", sort=True):
        if len(group) < n_per_class:
            raise ValueError(f"keyword_class '{cls}' has {len(group)} rows, need {n_per_class}")
        shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        parts.append(shuffled.head(n_per_class))
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def filter_by_length(df: pd.DataFrame, max_tokens: int = 4096) -> pd.DataFrame:
    """Add token_count and keep rows with token_count <= max_tokens.

    Args:
        df: DataFrame with a `text` column.
        max_tokens: Inclusive upper bound.

    Returns:
        Filtered DataFrame with added int64 `token_count` column.
    """
    out = df.copy()
    out["token_count"] = out["text"].map(count_tokens).astype("int64")
    return out[out["token_count"] <= max_tokens].reset_index(drop=True)


def mark_held_out_fs(df: pd.DataFrame, n_per_class: int = 2) -> pd.DataFrame:
    """Reserve the last n_per_class rows of each keyword_class for Phase 3 FS.

    Args:
        df: DataFrame already produced by `sample_candidates`.
        n_per_class: FS rows per class.

    Returns:
        Copy with a `held_out_fs` bool column.
    """
    out = df.copy()
    out["held_out_fs"] = False
    for _cls, group in out.groupby("keyword_class", sort=True):
        idx = group.index.tolist()[-n_per_class:]
        out.loc[idx, "held_out_fs"] = True
    return out


def existing_ids_in_batches(to_label_dir: Path, logger: logging.Logger | None = None) -> set[str]:
    """Return ids already present in any batch CSV (labeled or unlabeled) under to_label_dir.

    Args:
        to_label_dir: Directory holding batch_*.csv files.
        logger: Optional logger.

    Returns:
        Set of id strings.
    """
    log = logger or logging.getLogger(__name__)
    ids: set[str] = set()
    if not to_label_dir.exists():
        return ids
    for csv_path in sorted(to_label_dir.glob("batch_*.csv")):
        try:
            sub = pd.read_csv(csv_path, usecols=["id"])
            ids.update(sub["id"].astype(str).tolist())
        except Exception as exc:
            log.warning("Could not read ids from %s: %s", csv_path, exc)
    return ids
