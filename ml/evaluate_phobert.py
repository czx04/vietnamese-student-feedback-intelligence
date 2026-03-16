"""Evaluate a locked PhoBERT model on the held-out test split once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.metrics import detailed_classification_metrics, predicted_class_ids
from ml.phobert_common import (
    PHOBERT_REPORTS_DIR,
    PHOBERT_ROOT,
    SEGMENTED_DIR,
    TARGETS,
    EncodedTextDataset,
    load_segmented_split,
    select_samples,
    target_labels,
    validate_max_length,
    validate_target_values,
    write_json,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _load_max_length(report_root: Path, task: str, fallback: int) -> int:
    path = report_root / task / "run_config.json"
    if not path.exists():
        return fallback
    config = json.loads(path.read_text(encoding="utf-8"))
    return int(config.get("max_length", fallback))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "PhoBERT dependencies are missing. Run "
            "`python -m pip install -r requirements-phobert.txt`."
        ) from exc

    labels = target_labels(args.task)
    model_path = args.model_path or args.output_root / args.task / "best_model"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")
    test_frame = load_segmented_split("test", args.segmented_dir)
    validate_target_values(test_frame, args.task)
    test_frame = select_samples(test_frame, args.max_test_samples)
    label_column = f"{args.task}_id"
    max_length = args.max_length or _load_max_length(
        args.report_root, args.task, fallback=128
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    validate_max_length(tokenizer, max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    test_dataset = EncodedTextDataset(
        test_frame["phobert_text"],
        test_frame[label_column],
        tokenizer,
        max_length,
    )
    fp16 = args.precision == "fp16" or (
        args.precision == "auto" and torch.cuda.is_available()
    )
    if args.precision == "fp16" and not torch.cuda.is_available():
        raise RuntimeError("FP16 evaluation requires an NVIDIA CUDA GPU")

    task_dir = args.output_root / args.task
    reports_dir = args.report_root / args.task
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(task_dir / "evaluation"),
            per_device_eval_batch_size=args.eval_batch_size,
            fp16=fp16,
            report_to="none",
            seed=args.seed,
            dataloader_num_workers=args.dataloader_num_workers,
        ),
        processing_class=tokenizer,
    )
    prediction_output = trainer.predict(test_dataset, metric_key_prefix="test")
    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    predicted_ids = predicted_class_ids(logits)
    probabilities = _softmax(logits)
    metrics = {
        **prediction_output.metrics,
        **detailed_classification_metrics(
            prediction_output.label_ids, predicted_ids, labels
        ),
        "split": "test",
        "model_path": str(model_path.resolve()),
        "rows": len(test_frame),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "test_metrics.json", metrics)
    pd.DataFrame(
        metrics["confusion_matrix"], index=labels, columns=labels
    ).rename_axis("true_label").to_csv(reports_dir / "confusion_matrix_test.csv")
    predictions = pd.DataFrame(
        {
            "text": test_frame["text"],
            "phobert_text": test_frame["phobert_text"],
            "true_id": prediction_output.label_ids,
            "true_label": [labels[index] for index in prediction_output.label_ids],
            "predicted_id": predicted_ids,
            "predicted_label": [labels[index] for index in predicted_ids],
            "confidence": probabilities.max(axis=-1),
            "correct": prediction_output.label_ids == predicted_ids,
        }
    )
    for index, label in enumerate(labels):
        predictions[f"probability_{label}"] = probabilities[:, index]
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the locked PhoBERT model on the test split."
    )
    parser.add_argument("--task", choices=sorted(TARGETS), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--segmented-dir", type=Path, default=SEGMENTED_DIR)
    parser.add_argument("--output-root", type=Path, default=PHOBERT_ROOT)
    parser.add_argument("--report-root", type=Path, default=PHOBERT_REPORTS_DIR)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument(
        "--precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    metrics = evaluate(parse_args())
    print(
        f"test accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
