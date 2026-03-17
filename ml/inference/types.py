"""Dependency-free inference result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    confidence_is_calibrated: bool = False


@dataclass(frozen=True)
class FeedbackPrediction:
    text: str
    sentiment: TaskPrediction
    topic: TaskPrediction
    model_backend: str

    def as_dict(self) -> dict:
        return asdict(self)
