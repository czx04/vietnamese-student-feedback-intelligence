import numpy as np

from ml.metrics import (
    basic_classification_metrics,
    detailed_classification_metrics,
    predicted_class_ids,
    trainer_compute_metrics,
)


def test_predicted_class_ids_accepts_tuple_logits():
    logits = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    assert predicted_class_ids((logits,)).tolist() == [1, 0]


def test_metrics_include_missing_classes_in_macro_f1():
    metrics = basic_classification_metrics([0, 0], [0, 0], labels=[0, 1, 2])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1 / 3


def test_detailed_metrics_have_fixed_confusion_matrix_shape():
    metrics = detailed_classification_metrics(
        [0, 1, 2], [0, 2, 2], ["negative", "neutral", "positive"]
    )
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
    assert set(metrics["classification_report"]) >= {
        "negative",
        "neutral",
        "positive",
    }


def test_trainer_metric_adapter():
    prediction = type(
        "Prediction",
        (),
        {
            "predictions": np.array([[4.0, 1.0], [1.0, 4.0]]),
            "label_ids": np.array([0, 1]),
        },
    )()
    metrics = trainer_compute_metrics(2)(prediction)
    assert metrics == {"accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}
