"""Re-run RoBERTa on Employee Reviews ONLY with debug hyperparameters.

The initial Phase 2 RoBERTa run failed to learn on ER: train loss stayed near
log(3) across all folds, F1=0.41. Root cause hypothesis: 163 train samples
× batch 8 × 3 epochs = ~60 optimizer steps is too few for the random-init
classifier head to escape its initialization, especially with class weights
adding noise on a 3-class imbalanced task.

This script keeps the existing FNN rows in raw_roberta.parquet untouched and
re-runs ONLY the ER folds with: epochs=10, lr=5e-5, use_class_weights=False.

Per ODs:
- OD-3 disables class weights here as part of the debug — documented as a
  deviation in reports/baselines.md after this script + aggregate run.
- Other ODs (fp16, max_len=512, 5-fold, seed=42) unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.evaluation.metrics import BaselineResult, results_to_dataframe  # noqa: E402
from llm_textcls.io import project_root, read_parquet, write_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402


def _check_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch not installed. Install CUDA-enabled torch:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. RoBERTa fine-tuning requires a CUDA GPU.")


def _load_eval_df(parquet_path: Path) -> pd.DataFrame:
    df = read_parquet(parquet_path)
    if "held_out_fs" in df.columns:
        df = df[~df["held_out_fs"]].reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run RoBERTa on ER only (debug params)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--use-class-weights", action="store_true", default=False)
    parser.add_argument(
        "--raw-roberta",
        type=Path,
        default=project_root() / "results" / "baselines" / "raw_roberta.parquet",
    )
    args = parser.parse_args()

    _check_cuda()
    log = get_logger("rerun_roberta_er_debug", phase="phase2")
    log.warning(
        "Debug rerun params: epochs=%d, lr=%g, batch=%d, class_weights=%s",
        args.epochs,
        args.lr,
        args.batch_size,
        args.use_class_weights,
    )

    # Preserve existing FNN rows.
    existing = read_parquet(args.raw_roberta)
    fnn_rows = existing[existing["dataset"] == "fakenewsnet"].copy()
    log.warning("Preserving %d existing FNN RoBERTa rows", len(fnn_rows))

    # Re-run ER with debug hyperparameters.
    from llm_textcls.baselines.roberta import cross_validate_roberta

    er_df = _load_eval_df(project_root() / "data" / "employee_reviews" / "labeled.parquet")
    log.warning("RoBERTa ER (debug): %d rows, %d-fold", len(er_df), args.n_splits)

    er_results = cross_validate_roberta(
        df=er_df,
        dataset_name="employee_reviews",
        n_splits=args.n_splits,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_class_weights=args.use_class_weights,
        logger=log,
    )
    er_df_new = results_to_dataframe(er_results)
    log.warning(
        "RoBERTa ER (debug) f1_weighted folds=%s mean=%.4f",
        [round(r.f1_weighted, 4) for r in er_results],
        sum(r.f1_weighted for r in er_results) / len(er_results),
    )

    # Concat FNN (preserved) + ER (rerun) and overwrite raw_roberta.parquet.
    combined = pd.concat([fnn_rows, er_df_new], ignore_index=True)
    write_parquet(combined, args.raw_roberta, BaselineResult)
    log.warning(
        "Wrote %s (%d rows: %d FNN preserved + %d ER rerun)",
        args.raw_roberta,
        len(combined),
        len(fnn_rows),
        len(er_df_new),
    )
    log.warning("Next: run scripts/aggregate_baselines.py to refresh summary + report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
