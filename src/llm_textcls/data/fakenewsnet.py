"""FakeNewsNet acquisition: HF primary route, CSV+scrape fallback."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from llm_textcls.data.tokenization import count_tokens

_HF_TEXT_CANDIDATES = ("text", "content", "article", "body", "news")
_HF_TITLE_CANDIDATES = ("title", "headline")
_HF_URL_CANDIDATES = ("url", "link", "news_url")
_HF_LABEL_CANDIDATES = ("label", "labels", "class", "category", "type")

_FAKE_STRINGS = {"fake", "f", "false", "0", "mf", "hf"}
_REAL_STRINGS = {"real", "true", "r", "t", "1", "mr", "hr"}


def _normalize_label_value(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in _FAKE_STRINGS:
        return "fake"
    if s in _REAL_STRINGS:
        return "real"
    return None


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_from_huggingface(
    hf_dataset_id: str = "Jinyan1/PolitiFact",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load a FakeNewsNet-style dataset from HuggingFace and normalize columns.

    Prints sample rows + unique label values before mapping. If the label
    encoding cannot be resolved to {"fake","real"}, returns an empty DataFrame
    so the caller can fall back to the CSV route (per OD-3).

    Args:
        hf_dataset_id: HuggingFace dataset identifier.
        logger: Optional logger for route + label decisions.

    Returns:
        DataFrame with columns id, label, title, text, url. Empty on ambiguity.
    """
    log = logger or logging.getLogger(__name__)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        log.warning("`datasets` not importable: %s", exc)
        return _empty_fnn_frame()

    try:
        ds = load_dataset(hf_dataset_id)
    except Exception as exc:
        log.warning("HF load_dataset(%s) failed: %s", hf_dataset_id, exc)
        return _empty_fnn_frame()

    frames: list[pd.DataFrame] = []
    split_names_seen: list[str] = []
    for split_name in ds:
        split_df = ds[split_name].to_pandas()
        # If the dataset encodes the label in the split name (e.g.,
        # Jinyan1/PolitiFact uses MF/HF/MR/HR), inject it as a `_split_label` col.
        split_label = _normalize_label_value(split_name)
        split_df["_split_label"] = split_label  # may be None
        frames.append(split_df)
        split_names_seen.append(split_name)
        log.debug(
            "HF split %s columns=%s rows=%d split_label=%s",
            split_name,
            list(split_df.columns),
            len(split_df),
            split_label,
        )
    if not frames:
        log.warning("HF dataset %s has no splits", hf_dataset_id)
        return _empty_fnn_frame()

    raw = pd.concat(frames, ignore_index=True)
    cols = [c for c in raw.columns if c != "_split_label"]
    text_col = _pick_column(cols, _HF_TEXT_CANDIDATES)
    title_col = _pick_column(cols, _HF_TITLE_CANDIDATES)
    url_col = _pick_column(cols, _HF_URL_CANDIDATES)
    label_col = _pick_column(cols, _HF_LABEL_CANDIDATES)

    split_labels_resolved = raw["_split_label"].notna().all() and raw["_split_label"].nunique() >= 2
    if label_col is None and split_labels_resolved:
        log.warning("HF label inferred from split names: %s", sorted(set(split_names_seen)))
        raw["_norm_label"] = raw["_split_label"]
    elif label_col is not None:
        unique_labels = sorted({str(v) for v in raw[label_col].dropna().unique()})
        log.warning("HF label column '%s' unique values: %s", label_col, unique_labels)
        raw["_norm_label"] = raw[label_col].map(_normalize_label_value)
    else:
        log.warning(
            "HF dataset missing label column AND split names unresolved. cols=%s splits=%s",
            cols,
            split_names_seen,
        )
        return _empty_fnn_frame()

    if text_col is None:
        log.warning("HF dataset missing text column. cols=%s", cols)
        return _empty_fnn_frame()

    unresolved = raw["_norm_label"].isna().sum()
    total = len(raw)
    if total == 0 or unresolved / total > 0.5:
        log.warning(
            "HF label encoding ambiguous (%d/%d unresolved). Falling back to CSV route.",
            unresolved,
            total,
        )
        return _empty_fnn_frame()

    if unresolved:
        log.warning(
            "HF label normalization: %d/%d rows dropped (unresolved labels)", unresolved, total
        )
    raw = raw[raw["_norm_label"].notna()].copy()

    out = pd.DataFrame(
        {
            "id": [f"fnn_hf_{i:06d}" for i in range(len(raw))],
            "label": raw["_norm_label"].astype(str).values,
            "title": _safe_str_col(raw, title_col),
            "text": _safe_str_col(raw, text_col),
            "url": _safe_str_col(raw, url_col),
        }
    )
    log.warning(
        "HF route OK: %d rows; label counts=%s",
        len(out),
        out["label"].value_counts().to_dict(),
    )
    return out


def _safe_str_col(df: pd.DataFrame, col: str | None) -> list[str]:
    if col is None or col not in df.columns:
        return ["" for _ in range(len(df))]
    return df[col].fillna("").astype(str).tolist()


def _empty_fnn_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "label", "title", "text", "url"])


def load_from_csv(
    raw_dir: Path,
    logger: logging.Logger | None = None,
    request_delay_s: float = 1.0,
    timeout_s: float = 10.0,
) -> pd.DataFrame:
    """Fallback loader: read politifact_{fake,real}.csv and scrape article text.

    Caches each fetched article body at ``raw_dir/../fnn_articles/<sha256>.txt``.
    One retry per URL; permanent failures are skipped and logged.

    Args:
        raw_dir: Directory containing politifact_fake.csv and politifact_real.csv.
        logger: Optional logger.
        request_delay_s: Sleep between scrape requests.
        timeout_s: HTTP timeout.

    Returns:
        DataFrame with columns id, label, title, text, url.
    """
    log = logger or logging.getLogger(__name__)
    fake_path = raw_dir / "politifact_fake.csv"
    real_path = raw_dir / "politifact_real.csv"
    if not fake_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            f"CSV fallback requires {fake_path} and {real_path}. "
            "Download from KaiDMML/FakeNewsNet GitHub politifact CSVs."
        )

    cache_dir = raw_dir.parent / "fnn_articles"
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for label, path in (("fake", fake_path), ("real", real_path)):
        df = pd.read_csv(path)
        df["label"] = label
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    if "id" not in df.columns:
        df["id"] = [f"fnn_csv_{i:06d}" for i in range(len(df))]
    if "title" not in df.columns:
        df["title"] = ""
    url_col = "news_url" if "news_url" in df.columns else ("url" if "url" in df.columns else None)
    if url_col is None:
        raise ValueError("politifact CSVs must have a url/news_url column")

    texts: list[str] = []
    failures = 0
    for url in df[url_col].fillna("").astype(str).tolist():
        if not url:
            texts.append("")
            failures += 1
            continue
        body = _fetch_article(url, cache_dir, logger=log, timeout_s=timeout_s)
        if body is None:
            failures += 1
            texts.append("")
        else:
            texts.append(body)
        time.sleep(request_delay_s)

    df["text"] = texts
    df["url"] = df[url_col].fillna("").astype(str)
    total = len(df)
    if total and failures / total > 0.15:
        log.warning("CSV scrape loss %d/%d (>15%%); proceed with caution", failures, total)
    df = df[df["text"].str.len() > 0].copy()
    return df[["id", "label", "title", "text", "url"]]


def _fetch_article(
    url: str,
    cache_dir: Path,
    logger: logging.Logger,
    timeout_s: float = 10.0,
) -> str | None:
    """Fetch + extract body text for a URL with one retry; cache to disk."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8") or None

    from trafilatura import extract  # imported lazily

    for attempt in (1, 2):
        try:
            resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code >= 400:
                logger.debug("scrape %s -> HTTP %d (attempt %d)", url, resp.status_code, attempt)
                continue
            body = extract(resp.text) or ""
            cache_path.write_text(body, encoding="utf-8")
            return body or None
        except Exception as exc:
            logger.debug("scrape %s exc=%s (attempt %d)", url, exc, attempt)
    return None


def filter_by_length(df: pd.DataFrame, max_tokens: int = 4096) -> pd.DataFrame:
    """Add a `token_count` column and keep only rows with token_count <= max_tokens.

    Args:
        df: DataFrame with a `text` column.
        max_tokens: Inclusive upper bound on cl100k_base token count.

    Returns:
        Filtered DataFrame (copy) with new `token_count` column.
    """
    out = df.copy()
    out["token_count"] = out["text"].map(count_tokens).astype("int64")
    return out[out["token_count"] <= max_tokens].reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    n_per_class: int = 107,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic per-label shuffle, take first n_per_class of each label.

    Args:
        df: DataFrame with a `label` column.
        n_per_class: Rows per label.
        seed: RNG seed for shuffle.

    Returns:
        DataFrame of length n_per_class * num_labels.

    Raises:
        ValueError: if any label has fewer than n_per_class rows.
    """
    parts: list[pd.DataFrame] = []
    for label, group in df.groupby("label", sort=True):
        if len(group) < n_per_class:
            raise ValueError(f"Label '{label}' has {len(group)} rows, need {n_per_class}")
        shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        parts.append(shuffled.head(n_per_class))
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def mark_held_out_fs(df: pd.DataFrame, n_per_class: int = 2) -> pd.DataFrame:
    """Reserve the last `n_per_class` rows of each label for Few-Shot use.

    The seeded shuffle in `stratified_sample` makes "last N" deterministic.

    Args:
        df: DataFrame already produced by `stratified_sample`.
        n_per_class: Number of FS examples per label.

    Returns:
        DataFrame copy with a `held_out_fs` bool column.
    """
    out = df.copy()
    out["held_out_fs"] = False
    for _label, group in out.groupby("label", sort=True):
        idx = group.index.tolist()[-n_per_class:]
        out.loc[idx, "held_out_fs"] = True
    return out
