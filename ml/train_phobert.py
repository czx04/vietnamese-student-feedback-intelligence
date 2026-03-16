"""Fine-tune PhoBERT using train/valid only and save the best checkpoint."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ml.metrics import (
    detailed_classification_metrics,
    predicted_class_ids,
    trainer_compute_metrics,
)
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


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _environment_text() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else result.stderr


def _use_fp16(precision: str, torch_module: Any) -> bool:
    cuda_available = bool(torch_module.cuda.is_available())
    if precision == "fp16" and not cuda_available:
        raise RuntimeError("FP16 training requires an NVIDIA CUDA GPU")
    return precision == "fp16" or (precision == "auto" and cuda_available)


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PhoBERT dependencies are missing. Run "
            "`python -m pip install -r requirements-phobert.txt`."
        ) from exc

    labels = target_labels(args.task)
    train_frame = load_segmented_split("train", args.segmented_dir)
    valid_frame = load_segmented_split("valid", args.segmented_dir)
    validate_target_values(train_frame, args.task)
    validate_target_values(valid_frame, args.task)
    train_frame = select_samples(train_frame, args.max_train_samples)
    valid_frame = select_samples(valid_frame, args.max_valid_samples)
    label_column = f"{args.task}_id"

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    validate_max_length(tokenizer, args.max_length)
    train_dataset = EncodedTextDataset(
        train_frame["phobert_text"],
        train_frame[label_column],
        tokenizer,
        args.max_length,
    )
    valid_dataset = EncodedTextDataset(
        valid_frame["phobert_text"],
        valid_frame[label_column],
        tokenizer,
        args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
    )

    task_dir = args.output_root / args.task
    checkpoints_dir = task_dir / "checkpoints"
    reports_dir = args.report_root / args.task
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
    if (
        existing_checkpoints
        and args.resume_from_checkpoint is None
        and not args.overwrite_output_dir
    ):
        raise FileExistsError(
            f"{checkpoints_dir} already contains checkpoints. Use a new "
            "--output-root, pass --resume-from-checkpoint latest, or explicitly "
            "pass --overwrite-output-dir."
        )
    fp16 = _use_fp16(args.precision, torch)
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        overwrite_output_dir=args.overwrite_output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=fp16,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        compute_metrics=trainer_compute_metrics(len(labels)),
    )

    resume: bool | str | None = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    train_result = trainer.train(resume_from_checkpoint=resume)

    # load_best_model_at_end ensures this is the best validation checkpoint.
    best_model_dir = task_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(best_model_dir)
    prediction_output = trainer.predict(valid_dataset, metric_key_prefix="validation")
    predicted_ids = predicted_class_ids(prediction_output.predictions)
    validation_metrics = {
        **prediction_output.metrics,
        **detailed_classification_metrics(
            prediction_output.label_ids, predicted_ids, labels
        ),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "split": "valid",
    }

    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "model_name": args.model_name,
        "labels": labels,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "precision": "fp16" if fp16 else "fp32",
        "train_rows": len(train_frame),
        "valid_rows": len(valid_frame),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
    }
    write_json(reports_dir / "run_config.json", run_config)
    write_json(reports_dir / "train_metrics.json", train_result.metrics)
    write_json(reports_dir / "validation_metrics.json", validation_metrics)
    write_json(reports_dir / "log_history.json", trainer.state.log_history)
    (reports_dir / "environment.txt").write_text(
        _environment_text(), encoding="utf-8"
    )
    return validation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT on UIT-VSFC.")
    parser.add_argument("--task", choices=sorted(TARGETS), required=True)
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--segmented-dir", type=Path, default=SEGMENTED_DIR)
    parser.add_argument("--output-root", type=Path, default=PHOBERT_ROOT)
    parser.add_argument("--report-root", type=Path, default=PHOBERT_REPORTS_DIR)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument(
        "--precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint path, or 'latest' to resume the newest checkpoint.",
    )
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    metrics = train(parse_args())
    print(
        f"validation accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
