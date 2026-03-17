"""Load a selected model backend once and expose batch-safe prediction."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import joblib
import numpy as np

from ml.data import ROOT, clean_text
from ml.inference.types import FeedbackPrediction, TaskPrediction
from ml.phobert_common import TARGETS


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores -= scores.max(axis=-1, keepdims=True)
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


class _BaselineBackend:
    name = "linear_svm"

    def __init__(self, artifacts_dir: Path) -> None:
        self.models = {
            task: joblib.load(artifacts_dir / f"{task}_model.joblib")
            for task in TARGETS
        }

    def predict_many(self, texts: list[str]) -> list[FeedbackPrediction]:
        task_predictions: dict[str, list[TaskPrediction]] = {}
        for task, model in self.models.items():
            scores = np.asarray(model.decision_function(texts))
            probabilities = _softmax(scores)
            classes = [str(label) for label in model.classes_]
            indices = probabilities.argmax(axis=-1)
            task_predictions[task] = [
                TaskPrediction(
                    label=classes[index],
                    confidence=float(probabilities[row, index]),
                    probabilities={
                        label: float(probabilities[row, column])
                        for column, label in enumerate(classes)
                    },
                    confidence_is_calibrated=False,
                )
                for row, index in enumerate(indices)
            ]
        return [
            FeedbackPrediction(
                text=text,
                sentiment=task_predictions["sentiment"][index],
                topic=task_predictions["topic"][index],
                model_backend=self.name,
            )
            for index, text in enumerate(texts)
        ]


class _PhoBERTBackend:
    name = "phobert"

    def __init__(
        self,
        artifacts_dir: Path,
        vncorenlp_dir: Path,
        max_length: int = 128,
        batch_size: int = 32,
    ) -> None:
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("PhoBERT max_length and batch_size must be positive")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from ml.prepare_phobert_data import build_segmenter, segment_text

        self.torch = torch
        self.segment_text = segment_text
        self.segmenter = build_segmenter(vncorenlp_dir.expanduser().resolve())
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizers: dict[str, Any] = {}
        self.models: dict[str, Any] = {}
        for task in TARGETS:
            path = artifacts_dir / "phobert" / task / "best_model"
            self.tokenizers[task] = AutoTokenizer.from_pretrained(
                path, local_files_only=True
            )
            self.models[task] = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True
            ).to(self.device).eval()

    def predict_many(self, texts: list[str]) -> list[FeedbackPrediction]:
        segmented = [self.segment_text(self.segmenter, text) for text in texts]
        task_predictions: dict[str, list[TaskPrediction]] = {}
        for task, labels in TARGETS.items():
            chunks = []
            for offset in range(0, len(segmented), self.batch_size):
                encoded = self.tokenizers[task](
                    segmented[offset : offset + self.batch_size],
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    name: value.to(self.device) for name, value in encoded.items()
                }
                with self.torch.inference_mode():
                    chunks.append(
                        self.models[task](**encoded)
                        .logits.softmax(dim=-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            probabilities = np.concatenate(chunks, axis=0)
            indices = probabilities.argmax(axis=-1)
            task_predictions[task] = [
                TaskPrediction(
                    label=labels[index],
                    confidence=float(probabilities[row, index]),
                    probabilities={
                        label: float(probabilities[row, column])
                        for column, label in enumerate(labels)
                    },
                    confidence_is_calibrated=False,
                )
                for row, index in enumerate(indices)
            ]
        return [
            FeedbackPrediction(
                text=text,
                sentiment=task_predictions["sentiment"][index],
                topic=task_predictions["topic"][index],
                model_backend=self.name,
            )
            for index, text in enumerate(texts)
        ]


class FeedbackPredictor:
    """Thread-safe facade for baseline or PhoBERT feedback inference."""

    def __init__(
        self,
        backend: str | None = None,
        artifacts_dir: Path | None = None,
        vncorenlp_dir: Path | None = None,
    ) -> None:
        backend = (backend or os.getenv("MODEL_BACKEND", "baseline")).lower()
        artifacts_dir = artifacts_dir or ROOT / "artifacts"
        if backend == "baseline":
            self._backend = _BaselineBackend(artifacts_dir)
        elif backend == "phobert":
            model_dir = vncorenlp_dir or Path(
                os.getenv("VNCORENLP_DIR", str(ROOT / "artifacts" / "vncorenlp"))
            )
            self._backend = _PhoBERTBackend(
                artifacts_dir,
                model_dir,
                max_length=int(os.getenv("PHOBERT_MAX_LENGTH", "128")),
                batch_size=int(os.getenv("PHOBERT_BATCH_SIZE", "32")),
            )
        else:
            raise ValueError("MODEL_BACKEND must be 'baseline' or 'phobert'")
        self._lock = Lock()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def predict(self, text: str) -> FeedbackPrediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: Iterable[str]) -> list[FeedbackPrediction]:
        cleaned = [clean_text(text) for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Feedback text must not be empty")
        with self._lock:
            return self._backend.predict_many(cleaned)
