"""Run RoBERTa-base 5-fold CV on both Phase 1 datasets (CUDA required).

Per OD-1: if the FIRST FNN fold exceeds 4 minutes wall-clock, ER is downgraded
to 3-fold for the rest of the run. Writes `results/baselines/raw_roberta.parquet`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_textcls.evaluation.metrics import BaselineResult, results_to_dataframe  # noqa: E402
from llm_textcls.io import project_root, read_parquet, write_parquet  # noqa: E402
from llm_textcls.logging_setup import get_logger  # noqa: E402

FNN_PATH = "data/fakenewsnet/sample.parquet"
ER_PATH = "data/employee_reviews/labeled.parquet"

SLOW_FIRST_FOLD_SECONDS = 4 * 60.0


def _check_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch not installed. Install CUDA-enabled torch:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. RoBERTa fine-tuning requires a CUDA GPU.\n"
            "Install CUDA-enabled torch:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            f"torch version installed: {torch.__version__}"
        )


def _load_eval_df(parquet_path: Path) -> pd.DataFrame:
    df = read_parquet(parquet_path)
    if "held_out_fs" in df.columns:
        df = df[~df["held_out_fs"]].reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RoBERTa baseline on FNN + ER")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n-splits-fnn", type=int, default=5)
    parser.add_argument("--n-splits-er", type=int, default=5)
    parser.add_argument(
        "--slow-first-fold-fallback",
        type=float,
        default=SLOW_FIRST_FOLD_SECONDS,
        help="If first FNN fold takes longer than this many seconds, drop ER to 3 folds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root() / "results" / "baselines" / "raw_roberta.parquet",
    )
    args = parser.parse_args()

    _check_cuda()

    log = get_logger("run_baselines_roberta", phase="phase2")
    import torch  # safe after _check_cuda

    log.warning(
        "CUDA available: %s, device=%s, torch=%s",
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0),
        torch.__version__,
    )

    # Import roberta module only after CUDA check (transformers + torch are heavy).
    from llm_textcls.baselines.roberta import cross_validate_roberta

    all_results: list[BaselineResult] = []

    # FNN — first fold timed.
    fnn_df = _load_eval_df(project_root() / FNN_PATH)
    log.warning("RoBERTa FNN: %d rows", len(fnn_df))
    t0 = time.perf_counter()
    fnn_results = cross_validate_roberta(
        df=fnn_df,
        dataset_name="fakenewsnet",
        n_splits=args.n_splits_fnn,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        logger=log,
    )
    fnn_wall = time.perf_counter() - t0
    log.warning("RoBERTa FNN done: %d folds in %.1fs", len(fnn_results), fnn_wall)
    all_results.extend(fnn_results)

    # OD-1 fallback: if FIRST FNN fold was slow, drop ER to 3 folds.
    first_fnn_fold_time = fnn_results[0].fit_time_sec + fnn_results[0].inference_time_sec
    er_splits = args.n_splits_er
    if first_fnn_fold_time > args.slow_first_fold_fallback:
        log.warning(
            "First FNN fold took %.1fs (>%.1fs threshold). Dropping ER to 3-fold per OD-1.",
            first_fnn_fold_time,
            args.slow_first_fold_fallback,
        )
        er_splits = 3
    else:
        log.warning(
            "First FNN fold %.1fs <= %.1fs threshold. Keeping ER at %d-fold.",
            first_fnn_fold_time,
            args.slow_first_fold_fallback,
            er_splits,
        )

    # ER.
    er_df = _load_eval_df(project_root() / ER_PATH)
    log.warning("RoBERTa ER: %d rows, %d-fold", len(er_df), er_splits)
    er_results = cross_validate_roberta(
        df=er_df,
        dataset_name="employee_reviews",
        n_splits=er_splits,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        logger=log,
    )
    all_results.extend(er_results)

    out_df = results_to_dataframe(all_results)
    write_parquet(out_df, args.out, BaselineResult)
    log.warning(
        "Wrote %s (%d rows: %d FNN + %d ER)",
        args.out,
        len(out_df),
        len(fnn_results),
        len(er_results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
