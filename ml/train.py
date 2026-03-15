import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.data import ROOT, load_dataset, prepare_training_frame, validate_labels


REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"
TARGETS = {
    "sentiment": ["negative", "neutral", "positive"],
    "topic": ["lecturer", "program", "facility", "others"],
}


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )


def candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=42),
        "complement_nb": ComplementNB(alpha=0.5),
    }


def evaluate(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_confusion_matrix(task: str, labels: list[str], matrix: list[list[int]]) -> None:
    figure, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title=f"{task.title()} validation confusion matrix", xlabel="Predicted", ylabel="True")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"confusion_matrix_{task}.png", dpi=160)
    plt.close(figure)


def collect_errors(
    frame: pd.DataFrame, task: str, predictions: np.ndarray, limit: int = 50
) -> pd.DataFrame:
    errors = frame.loc[frame[task].to_numpy() != predictions, ["text", task]].copy()
    errors = errors.rename(columns={task: "true_label"})
    errors["predicted_label"] = predictions[frame[task].to_numpy() != predictions]
    errors["text_length"] = errors["text"].str.len()
    errors = errors.sort_values(["true_label", "predicted_label", "text_length"])
    if len(errors) <= limit:
        return errors.reset_index(drop=True)

    counts = errors.groupby(["true_label", "predicted_label"]).size()
    per_group = max(2, limit // len(counts))
    sample = errors.groupby(["true_label", "predicted_label"], group_keys=False).head(per_group)
    if len(sample) < limit:
        remaining = errors.drop(sample.index).head(limit - len(sample))
        sample = pd.concat([sample, remaining])
    return sample.head(limit).reset_index(drop=True)


def train_task(
    task: str, train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[list[dict], dict]:
    labels = TARGETS[task]
    results = []
    fitted = {}

    for name, classifier in candidates().items():
        pipeline = Pipeline([("tfidf", vectorizer()), ("classifier", classifier)])
        pipeline.fit(train["text"], train[task])
        predictions = pipeline.predict(valid["text"])
        metrics = evaluate(valid[task], predictions, labels)
        results.append({"task": task, "model": name, **metrics})
        fitted[name] = (pipeline, predictions)

    best = max(results, key=lambda item: item["macro_f1"])
    best_pipeline, best_predictions = fitted[best["model"]]
    joblib.dump(best_pipeline, ARTIFACTS_DIR / f"{task}_model.joblib")
    collect_errors(valid, task, best_predictions).to_csv(
        REPORTS_DIR / f"error_cases_{task}.csv", index=False
    )
    save_confusion_matrix(task, labels, best["confusion_matrix"])
    return results, best


def run_training() -> list[dict]:
    frames = load_dataset()
    validate_labels(frames)
    frames["train"], preprocessing = prepare_training_frame(frames["train"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_models = {}
    for task in TARGETS:
        results, best = train_task(task, frames["train"], frames["valid"])
        all_results.extend(results)
        best_models[task] = best["model"]

    serializable = {
        "preprocessing": preprocessing,
        "validation_results": all_results,
        "selected_models": best_models,
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison = pd.DataFrame(all_results)[
        ["task", "model", "accuracy", "macro_f1", "weighted_f1"]
    ]
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    return all_results


if __name__ == "__main__":
    results = run_training()
    print(
        pd.DataFrame(results)[
            ["task", "model", "accuracy", "macro_f1", "weighted_f1"]
        ].to_string(index=False)
    )
