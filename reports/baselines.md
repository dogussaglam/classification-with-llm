# Phase 2 — Baseline results

Generated: 2026-05-19T19:23:05.705589+00:00

## Gate

**STATUS: FAIL — outside ±10 on [('nb', 'employee_reviews')]**

- Hard tolerance: ±5 F1 (paper target)
- Soft warning:   ±10 F1

### Breaches

- nb       / employee_reviews  : actual=45.90, target=61.3, |delta|=15.40
- roberta  / fakenewsnet       : actual=85.64, target=93.0, |delta|=7.36
- roberta  / employee_reviews  : actual=74.99, target=83.8, |delta|=8.81

## Per-model summary

| Model | Dataset | F1 mean | F1 std | Acc mean | Folds | Fit s (mean) | Inf s (mean) |
|---|---|---|---|---|---|---|---|
| nb | employee_reviews | 0.4590 | 0.0553 | 0.5591 | 5 | 0.09 | 0.008 |
| rf | employee_reviews | 0.6238 | 0.0713 | 0.6866 | 5 | 0.64 | 0.091 |
| roberta | employee_reviews | 0.7499 | 0.0636 | 0.7452 | 5 | 63.89 | 0.453 |
| svm | employee_reviews | 0.6575 | 0.0813 | 0.6671 | 5 | 0.08 | 0.009 |
| xgboost | employee_reviews | 0.7141 | 0.0778 | 0.7252 | 5 | 2.32 | 0.010 |
| nb | fakenewsnet | 0.8997 | 0.0412 | 0.9000 | 5 | 0.17 | 0.022 |
| rf | fakenewsnet | 0.7898 | 0.0408 | 0.7952 | 5 | 0.57 | 0.087 |
| roberta | fakenewsnet | 0.8564 | 0.0427 | 0.8571 | 5 | 20.30 | 0.462 |
| svm | fakenewsnet | 0.8804 | 0.0262 | 0.8810 | 5 | 0.18 | 0.023 |
| xgboost | fakenewsnet | 0.7736 | 0.0501 | 0.7762 | 5 | 3.27 | 0.080 |

## Caveats

- Our Employee Reviews dataset is 204 rows vs. the paper's ~1000. Higher F1 variance is expected, especially for RoBERTa where the minority class (`not_remote`, ~50 rows) gets ~10 test samples per fold.
- RoBERTa input is truncated to 512 tokens (paper's choice). FNN p95 token count is 1692, so the long tail loses content.
- RF and XGBoost have no paper reference; their numbers are reported as-is.
- **NB-ER hard FAIL (Δ=15.4) is a small-N effect, not a pipeline bug.** NB-FNN hits the paper target exactly (0.900 / 0.900), demonstrating the TF-IDF + MultinomialNB pipeline is correct. With 204 ER rows (mean 69 tokens) vs. the paper's ~1000, MultinomialNB's high-bias / low-capacity nature is what fails — SVM on the same pipeline lands inside ±5 of target (Δ=3.0).
- **RoBERTa-ER used debug hyperparameters** (epochs=10, lr=5e-5, use_class_weights=False) via `scripts/rerun_roberta_er_debug.py`. The original paper-aligned params (epochs=3, lr=2e-5, class_weights=True) produced a degenerate result (F1=0.41, train loss stuck at log(3)≈1.10) on our small-N ER split; the classifier head could not escape its random initialisation in ~60 optimiser steps. The deviation is documented in `docs/decisions/phase2_baselines.md` (OD-3 post-run note). RoBERTa-FNN keeps the original params (5-fold, 3 epochs, lr=2e-5).

## Run metadata

- raw.parquet row count: 50 (expected: one of [46, 50])
- Folds per (model, dataset):
  - nb / employee_reviews: 5
  - rf / employee_reviews: 5
  - roberta / employee_reviews: 5
  - svm / employee_reviews: 5
  - xgboost / employee_reviews: 5
  - nb / fakenewsnet: 5
  - rf / fakenewsnet: 5
  - roberta / fakenewsnet: 5
  - svm / fakenewsnet: 5
  - xgboost / fakenewsnet: 5
