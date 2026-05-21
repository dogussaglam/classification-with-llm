"""RoBERTa-base fine-tune + StratifiedKFold CV with optional class weighting.

Imports torch + transformers at module load — only import this module when a
CUDA-enabled torch is available.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from llm_textcls.evaluation.metrics import BaselineResult, weighted_f1

ROBERTA_MODEL_ID = "roberta-base"
ROBERTA_MAX_LEN = 512


class _TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights to cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **_kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _tokenize(
    texts: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    max_len: int = ROBERTA_MAX_LEN,
) -> _TextClassificationDataset:
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors=None,
    )
    return _TextClassificationDataset(encodings, labels)


def cross_validate_roberta(
    df: pd.DataFrame,
    dataset_name: str,
    text_col: str = "text",
    label_col: str = "label",
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
    use_class_weights: bool = True,
    fp16: bool = True,
    logger: logging.Logger | None = None,
) -> list[BaselineResult]:
    """Fine-tune roberta-base per fold and return one BaselineResult per fold.

    Args:
        df: DataFrame with text_col + label_col (string labels).
        dataset_name: Tag stored on each BaselineResult.
        text_col: Name of text column.
        label_col: Name of label column.
        n_splits: CV folds.
        seed: Random seed for fold splitter + Trainer.
        epochs: Fine-tune epochs per fold.
        lr: Learning rate.
        batch_size: Per-device train / eval batch size.
        use_class_weights: If True, compute per-fold balanced class weights.
        fp16: Mixed precision training (Ampere+ GPUs).
        logger: Optional logger.

    Returns:
        List of `n_splits` BaselineResult objects.

    Raises:
        RuntimeError: if CUDA is not available.
    """
    log = logger or logging.getLogger(__name__)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cross_validate_roberta requires CUDA. Install a CUDA-enabled torch: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    texts = df[text_col].astype(str).to_numpy()
    label_strs = df[label_col].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(label_strs).tolist()))
    label2id = {lbl: i for i, lbl in enumerate(classes)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    y_int = np.array([label2id[lbl] for lbl in label_strs])

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_ID)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: list[BaselineResult] = []
    for fold_i, (tr, te) in enumerate(skf.split(texts, y_int)):
        log.warning(
            "RoBERTa %s fold %d/%d: n_train=%d n_test=%d",
            dataset_name,
            fold_i + 1,
            n_splits,
            len(tr),
            len(te),
        )
        train_ds = _tokenize(texts[tr], y_int[tr], tokenizer)
        eval_ds = _tokenize(texts[te], y_int[te], tokenizer)

        weights_tensor: torch.Tensor | None = None
        if use_class_weights:
            w = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_int[tr])
            weights_tensor = torch.tensor(w, dtype=torch.float32)

        model = AutoModelForSequenceClassification.from_pretrained(
            ROBERTA_MODEL_ID,
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=lr,
                fp16=fp16,
                eval_strategy="no",
                save_strategy="no",
                logging_strategy="no",
                report_to=[],
                seed=seed,
                dataloader_pin_memory=True,
            )
            trainer = WeightedTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                class_weights=weights_tensor if use_class_weights else None,
            )
            t0 = time.perf_counter()
            trainer.train()
            fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            preds = trainer.predict(eval_ds)
            inf_time = time.perf_counter() - t0

        y_hat_int = preds.predictions.argmax(-1)
        y_hat = np.array([id2label[i] for i in y_hat_int])
        y_true = label_strs[te]

        report = classification_report(y_true, y_hat, output_dict=True, zero_division=0)
        results.append(
            BaselineResult(
                model_name="roberta",
                dataset=dataset_name,
                fold=fold_i,
                f1_weighted=weighted_f1(y_true, y_hat),
                accuracy=float(accuracy_score(y_true, y_hat)),
                fit_time_sec=float(fit_time),
                inference_time_sec=float(inf_time),
                n_train=int(len(tr)),
                n_test=int(len(te)),
                classification_report=json.dumps(report),
            )
        )

        del model, trainer
        torch.cuda.empty_cache()

    return results
