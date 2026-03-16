"""Metrics shared by PhoBERT training and final evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def predicted_class_ids(predictions: Any) -> np.ndarray:
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2-D logits, received shape {logits.shape}")
    return logits.argmax(axis=-1)


def basic_classification_metrics(
    label_ids: Sequence[int],
    predicted_ids: Sequence[int],
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(label_ids, predicted_ids)),
        "macro_f1": float(
            f1_score(
                label_ids,
                predicted_ids,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                label_ids,
                predicted_ids,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def detailed_classification_metrics(
    label_ids: Sequence[int],
    predicted_ids: Sequence[int],
    target_names: Sequence[str],
) -> dict[str, Any]:
    labels = list(range(len(target_names)))
    return {
        **basic_classification_metrics(label_ids, predicted_ids, labels=labels),
        "classification_report": classification_report(
            label_ids,
            predicted_ids,
            labels=labels,
            target_names=list(target_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            label_ids, predicted_ids, labels=labels
        ).tolist(),
    }


def trainer_compute_metrics(num_labels: int) -> Callable[[Any], dict[str, float]]:
    def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
        predicted_ids = predicted_class_ids(evaluation_prediction.predictions)
        return basic_classification_metrics(
            evaluation_prediction.label_ids,
            predicted_ids,
            labels=list(range(num_labels)),
        )

    return compute_metrics
