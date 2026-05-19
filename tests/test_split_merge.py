"""End-to-end test: synthesize unlabeled set, split, label, merge — round-trip preserves rows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from llm_textcls.io import ERUnlabeledRow, write_parquet

ROOT = Path(__file__).resolve().parents[1]


def _make_unlabeled(n: int = 60) -> pd.DataFrame:
    classes = ["remote", "not_remote", "not_mentioned"]
    rows = []
    for i in range(n):
        cls = classes[i % 3]
        rows.append(
            {
                "id": f"er_test_{i:04d}",
                "text": f"sample review {i} for class {cls}",
                "keyword_class": cls,
                "keyword_match_count": 1,
                "token_count": 10 + i,
                "held_out_fs": False,
            }
        )
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["keyword_class"] = df["keyword_class"].astype(str)
    df["keyword_match_count"] = df["keyword_match_count"].astype("int64")
    df["token_count"] = df["token_count"].astype("int64")
    df["held_out_fs"] = df["held_out_fs"].astype(bool)
    return df


def test_split_then_label_then_merge_roundtrip(tmp_path):
    unlabeled_path = tmp_path / "unlabeled.parquet"
    to_label_dir = tmp_path / "to_label"
    out_path = tmp_path / "labeled.parquet"

    df = _make_unlabeled(n=210)  # 70 per class, > GATE_MIN_TOTAL
    write_parquet(df, unlabeled_path, ERUnlabeledRow)

    env_paths = {
        "PYTHONPATH": str(ROOT / "src"),
    }
    import os

    env = {**os.environ, **env_paths}

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_for_labeling.py"),
            "--in",
            str(unlabeled_path),
            "--out-dir",
            str(to_label_dir),
            "--n-batches",
            "4",
        ],
        check=True,
        env=env,
    )

    csv_paths = sorted(to_label_dir.glob("batch_*.csv"))
    assert len(csv_paths) == 4
    total_rows_in_batches = 0
    for p in csv_paths:
        sub = pd.read_csv(p)
        assert set(sub.columns) == {"id", "text", "label"}
        # Auto-label: copy keyword_class from unlabeled for this id.
        kw = df.set_index("id")["keyword_class"].to_dict()
        sub["label"] = sub["id"].map(kw)
        labeled_path = to_label_dir / p.name.replace(".csv", "_labeled.csv")
        sub.to_csv(labeled_path, index=False)
        total_rows_in_batches += len(sub)

    assert total_rows_in_batches == 210

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "merge_labeled.py"),
            "--in-dir",
            str(to_label_dir),
            "--unlabeled",
            str(unlabeled_path),
            "--out",
            str(out_path),
        ],
        check=True,
        env=env,
    )

    merged = pd.read_parquet(out_path)
    assert len(merged) == 210
    assert set(merged["id"]) == set(df["id"])
    assert set(merged["label"].unique()) == {"remote", "not_remote", "not_mentioned"}
    # Label should match the auto-assigned keyword_class.
    joined = merged.merge(df[["id", "keyword_class"]], on="id")
    assert (joined["label"] == joined["keyword_class"]).all()


def test_supplemental_batch_excludes_held_out_fs_ids(tmp_path):
    """B1 regression: generate_supplemental_batch must not re-sample held-out FS rows.

    Otherwise a Few-Shot example could end up in labeled.parquet, leaking the FS
    set into the Phase 3 test set.
    """
    import hashlib
    import os

    raw_dir = tmp_path / "raw" / "glassdoor"
    raw_dir.mkdir(parents=True)
    to_label_dir = tmp_path / "to_label"
    to_label_dir.mkdir()

    # 5 reviews that the keyword filter will classify as 'remote'.
    csv_rows = []
    for i in range(5):
        csv_rows.append(
            {
                "headline": f"Headline {i} for remote review",
                "pros": f"Working from home full time was great for review {i}",
                "cons": f"Almost nothing bad to say about this remote job {i}",
            }
        )
    pd.DataFrame(csv_rows).to_csv(raw_dir / "glassdoor_reviews.csv", index=False)

    # Reproduce load_raw's id derivation for row 0 -> that's our held-out FS id.
    r = csv_rows[0]
    text = f"{r['headline']} {r['pros']} {r['cons']}"
    held_out_id = "er_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    held_out_df = pd.DataFrame(
        {
            "id": pd.array([held_out_id], dtype="object"),
            "text": pd.array([text], dtype="object"),
            "keyword_class": pd.array(["remote"], dtype="object"),
            "keyword_match_count": pd.array([1], dtype="int64"),
            "token_count": pd.array([10], dtype="int64"),
            "held_out_fs": pd.array([True], dtype="bool"),
        }
    )
    write_parquet(held_out_df, to_label_dir.parent / "held_out_fs.parquet", ERUnlabeledRow)

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    # Request n == total: without the fix, sample(n=5) returns ALL rows (held-out
    # guaranteed present). With the fix, only 4 are available so the script raises.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_supplemental_batch.py"),
            "--class",
            "remote",
            "--n",
            "5",
            "--raw-dir",
            str(raw_dir),
            "--to-label-dir",
            str(to_label_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    out_csv = to_label_dir / "batch_supplemental_remote.csv"
    if out_csv.exists():
        # Script succeeded — must not contain the held-out id.
        written = pd.read_csv(out_csv)
        assert held_out_id not in set(written["id"].astype(str)), (
            f"Held-out FS id {held_out_id} leaked into supplemental batch — "
            "Phase 3 FS↔test leakage risk."
        )
    else:
        # Script refused because the held-out row was excluded from the pool.
        assert result.returncode != 0, (
            f"Script wrote no CSV but exited 0 — unexpected: stderr={result.stderr[:500]}"
        )
        assert "available" in result.stderr.lower(), (
            f"Script failed for the wrong reason: stderr={result.stderr[:500]}"
        )
