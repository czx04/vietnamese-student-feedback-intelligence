from pathlib import Path

import joblib
import numpy as np
import pytest

import ml.inference as inference
from ml.inference import FeedbackPredictor


class FakeClassifier:
    classes_ = np.array(["negative", "neutral", "positive"])

    def decision_function(self, texts):
        return np.tile([0.1, 0.2, 1.4], (len(texts), 1))


class FakeTopicClassifier:
    classes_ = np.array(["facility", "lecturer", "others", "program"])

    def decision_function(self, texts):
        return np.tile([0.1, 1.5, 0.2, 0.0], (len(texts), 1))


def test_baseline_predictor_has_stable_contract(tmp_path: Path):
    joblib.dump(FakeClassifier(), tmp_path / "sentiment_model.joblib")
    joblib.dump(FakeTopicClassifier(), tmp_path / "topic_model.joblib")
    predictor = FeedbackPredictor(backend="baseline", artifacts_dir=tmp_path)

    result = predictor.predict("  giảng viên dạy tốt  ").as_dict()

    assert result["text"] == "giảng viên dạy tốt"
    assert result["sentiment"]["label"] == "positive"
    assert result["topic"]["label"] == "lecturer"
    assert result["model_backend"] == "linear_svm"
    assert sum(result["sentiment"]["probabilities"].values()) == pytest.approx(1.0)


def test_predictor_rejects_empty_text(tmp_path: Path):
    joblib.dump(FakeClassifier(), tmp_path / "sentiment_model.joblib")
    joblib.dump(FakeTopicClassifier(), tmp_path / "topic_model.joblib")
    predictor = FeedbackPredictor(backend="baseline", artifacts_dir=tmp_path)
    with pytest.raises(ValueError, match="must not be empty"):
        predictor.predict("  ")


def test_convenience_task_functions_use_stable_dictionary_contract(
    tmp_path: Path, monkeypatch
):
    joblib.dump(FakeClassifier(), tmp_path / "sentiment_model.joblib")
    joblib.dump(FakeTopicClassifier(), tmp_path / "topic_model.joblib")
    predictor = FeedbackPredictor(backend="baseline", artifacts_dir=tmp_path)
    monkeypatch.setattr(inference, "_default_predictor", predictor)

    assert inference.predict_sentiment("dạy tốt")["label"] == "positive"
    assert inference.predict_topic("dạy tốt")["label"] == "lecturer"
