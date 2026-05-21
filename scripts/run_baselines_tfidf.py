"""Run TF-IDF baselines (NB, SVM, RF, XGBoost) on both Phase 1 datasets.

Writes `results/baselines/raw_tfidf.parquet` with 4 models × 2 datasets × 5 folds = 40 rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.baselines.tfidf_models import cross_validate_baseline  # noqa: E402
from llm_textcls.evaluation.metrics import BaselineResult, results_to_dataframe  # noqa: E402
from llm_textcls.io import project_root, read_parquet, write_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402

MODELS = ("nb", "svm", "rf", "xgboost")
DATASETS = (
    ("fakenewsnet", "data/fakenewsnet/sample.parquet"),
    ("employee_reviews", "data/employee_reviews/labeled.parquet"),
)


def _load_eval_df(parquet_path: Path) -> pd.DataFrame:
    df = read_parquet(parquet_path)
    if "held_out_fs" in df.columns:
        df = df[~df["held_out_fs"]].reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TF-IDF baselines (NB, SVM, RF, XGBoost)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root() / "results" / "baselines" / "raw_tfidf.parquet",
    )
    args = parser.parse_args()

    log = get_logger("run_baselines_tfidf", phase="phase2")
    all_results: list[BaselineResult] = []

    for dataset_name, rel_path in DATASETS:
        df = _load_eval_df(project_root() / rel_path)
        log.warning(
            "Dataset %s: %d rows, label counts=%s",
            dataset_name,
            len(df),
            df["label"].value_counts().to_dict(),
        )
        for model_name in MODELS:
            log.warning("Running %s on %s (%d-fold CV)", model_name, dataset_name, args.n_splits)
            fold_results = cross_validate_baseline(
                model_name=model_name,
                df=df,
                dataset_name=dataset_name,
                n_splits=args.n_splits,
                seed=args.seed,
            )
            all_results.extend(fold_results)
            f1s = [r.f1_weighted for r in fold_results]
            log.warning(
                "%s/%s: f1_weighted folds=%s mean=%.4f",
                model_name,
                dataset_name,
                [round(x, 4) for x in f1s],
                sum(f1s) / len(f1s),
            )

    out_df = results_to_dataframe(all_results)
    write_parquet(out_df, args.out, BaselineResult)
    log.warning("Wrote %s (%d rows)", args.out, len(out_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
