"""Step 10: compare the locked Linear SVM and PhoBERT models on test data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ml.data import ROOT, load_split
from ml.phobert_common import TARGETS, to_jsonable, write_json


ARTIFACTS_DIR = ROOT / "artifacts"
PHOBERT_REPORTS = ROOT / "reports" / "phobert"
COMPARISON_DIR = ROOT / "reports" / "comparison"


def file_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (
        1024 * 1024
    )


def classification_metrics(
    true_labels: pd.Series | np.ndarray,
    predicted_labels: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, labels=labels, average="macro")
        ),
        "weighted_f1": float(
            f1_score(true_labels, predicted_labels, labels=labels, average="weighted")
        ),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=labels
        ).tolist(),
    }


def benchmark_baseline(model: Any, texts: list[str], repeats: int = 3) -> float:
    model.predict(texts[: min(8, len(texts))])
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict(texts)
        durations.append(time.perf_counter() - started)
    return 1000 * min(durations) / len(texts)


def benchmark_phobert(
    task: str,
    texts: list[str],
    model_dir: Path,
    max_length: int,
    batch_size: int,
) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).eval()
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        model(**{name: values[:2] for name, values in encoded.items()})
        started = time.perf_counter()
        for offset in range(0, len(texts), batch_size):
            model(
                **{
                    name: values[offset : offset + batch_size]
                    for name, values in encoded.items()
                }
            )
        duration = time.perf_counter() - started
    del model
    return 1000 * duration / len(texts)


def save_metric_chart(rows: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, task in zip(axes, TARGETS):
        subset = frame.loc[frame["task"].eq(task)]
        x = np.arange(len(subset))
        width = 0.25
        for index, metric in enumerate(("accuracy", "macro_f1", "weighted_f1")):
            axis.bar(x + (index - 1) * width, subset[metric], width, label=metric)
        axis.set_xticks(x, subset["model"])
        axis.set_ylim(0, 1)
        axis.set_title(task.title())
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "test_metric_comparison.png", dpi=160)
    plt.close(figure)


def compare_models(
    output_dir: Path = COMPARISON_DIR,
    benchmark_samples: int = 256,
    phobert_batch_size: int = 16,
) -> list[dict[str, Any]]:
    test = load_split("test")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for task, labels in TARGETS.items():
        baseline_path = ARTIFACTS_DIR / f"{task}_model.joblib"
        phobert_dir = ARTIFACTS_DIR / "phobert" / task / "best_model"
        phobert_report_dir = PHOBERT_REPORTS / task
        for required in (baseline_path, phobert_dir, phobert_report_dir):
            if not required.exists():
                raise FileNotFoundError(f"Missing required step 10 input: {required}")

        baseline = joblib.load(baseline_path)
        baseline_predictions = baseline.predict(test["text"])
        baseline_metrics = classification_metrics(test[task], baseline_predictions, labels)
        baseline_ms = benchmark_baseline(
            baseline,
            test["text"].head(benchmark_samples).tolist(),
        )
        write_json(output_dir / f"linear_svm_{task}_test_metrics.json", baseline_metrics)

        phobert_metrics = json.loads(
            (phobert_report_dir / "test_metrics.json").read_text(encoding="utf-8")
        )
        phobert_predictions = pd.read_csv(
            phobert_report_dir / "test_predictions.csv"
        )
        if len(phobert_predictions) != len(test):
            raise ValueError(f"PhoBERT {task} predictions do not match the test split")
        if not phobert_predictions["true_label"].astype(str).equals(
            test[task].astype(str)
        ):
            raise ValueError(f"PhoBERT {task} report rows are not aligned to the test split")
        run_config = json.loads(
            (phobert_report_dir / "run_config.json").read_text(encoding="utf-8")
        )
        sample_texts = phobert_predictions["phobert_text"].head(benchmark_samples).tolist()
        phobert_ms = benchmark_phobert(
            task,
            sample_texts,
            phobert_dir,
            max_length=int(run_config["max_length"]),
            batch_size=phobert_batch_size,
        )

        rows.extend(
            [
                {
                    "model": "linear_svm",
                    "task": task,
                    "split": "test",
                    "accuracy": baseline_metrics["accuracy"],
                    "macro_f1": baseline_metrics["macro_f1"],
                    "weighted_f1": baseline_metrics["weighted_f1"],
                    "inference_ms_per_sample_cpu": baseline_ms,
                    "size_mb": file_size_mb(baseline_path),
                },
                {
                    "model": "phobert_base",
                    "task": task,
                    "split": "test",
                    "accuracy": float(phobert_metrics["accuracy"]),
                    "macro_f1": float(phobert_metrics["macro_f1"]),
                    "weighted_f1": float(phobert_metrics["weighted_f1"]),
                    "inference_ms_per_sample_cpu": phobert_ms,
                    "size_mb": file_size_mb(phobert_dir),
                },
            ]
        )

        comparison = pd.DataFrame(
            {
                "text": test["text"],
                "true_label": test[task],
                "linear_svm_prediction": baseline_predictions,
                "phobert_prediction": phobert_predictions["predicted_label"],
            }
        )
        comparison["linear_svm_correct"] = comparison["true_label"].eq(
            comparison["linear_svm_prediction"]
        )
        comparison["phobert_correct"] = comparison["true_label"].eq(
            comparison["phobert_prediction"]
        )
        comparison["outcome"] = np.select(
            [
                comparison["phobert_correct"] & ~comparison["linear_svm_correct"],
                comparison["linear_svm_correct"] & ~comparison["phobert_correct"],
                comparison["phobert_correct"] & comparison["linear_svm_correct"],
            ],
            ["phobert_only_correct", "linear_svm_only_correct", "both_correct"],
            default="both_wrong",
        )
        comparison.to_csv(output_dir / f"{task}_error_comparison.csv", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "model_comparison.csv", index=False)
    gains = {}
    for task in TARGETS:
        subset = result.loc[result["task"].eq(task)].set_index("model")
        gains[task] = {
            "phobert_macro_f1_gain": float(
                subset.loc["phobert_base", "macro_f1"]
                - subset.loc["linear_svm", "macro_f1"]
            ),
            "quality_winner": str(subset["macro_f1"].idxmax()),
            "speed_winner": str(subset["inference_ms_per_sample_cpu"].idxmin()),
            "size_winner": str(subset["size_mb"].idxmin()),
        }
    write_json(output_dir / "summary.json", {"rows": rows, "conclusions": gains})
    save_metric_chart(rows, output_dir)
    return to_jsonable(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument("--benchmark-samples", type=int, default=256)
    parser.add_argument("--phobert-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_models(
        output_dir=args.output_dir,
        benchmark_samples=args.benchmark_samples,
        phobert_batch_size=args.phobert_batch_size,
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
