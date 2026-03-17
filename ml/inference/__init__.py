"""Stable, lazy inference interface used by the API and local applications."""

from threading import Lock

from ml.inference.predictor import FeedbackPredictor


_default_predictor: FeedbackPredictor | None = None
_default_lock = Lock()


def get_default_predictor() -> FeedbackPredictor:
    global _default_predictor
    if _default_predictor is None:
        with _default_lock:
            if _default_predictor is None:
                _default_predictor = FeedbackPredictor()
    return _default_predictor


def predict_feedback(text: str) -> dict:
    return get_default_predictor().predict(text).as_dict()


def predict_sentiment(text: str) -> dict:
    return predict_feedback(text)["sentiment"]


def predict_topic(text: str) -> dict:
    return predict_feedback(text)["topic"]


__all__ = [
    "FeedbackPredictor",
    "get_default_predictor",
    "predict_feedback",
    "predict_sentiment",
    "predict_topic",
]
