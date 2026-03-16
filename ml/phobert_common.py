"""Shared constants and helpers for the PhoBERT training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ml.data import ROOT


PHOBERT_ROOT = ROOT / "artifacts" / "phobert"
SEGMENTED_DIR = PHOBERT_ROOT / "segmented"
PHOBERT_REPORTS_DIR = ROOT / "reports" / "phobert"
TARGETS = {
    "sentiment": ["negative", "neutral", "positive"],
    "topic": ["lecturer", "program", "facility", "others"],
}
REQUIRED_SEGMENTED_COLUMNS = {
    "text",
    "phobert_text",
    "topic",
    "topic_id",
    "sentiment",
    "sentiment_id",
    "split",
}


class EncodedTextDataset:
    """Minimal map-style dataset accepted by Hugging Face Trainer."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {name: values[index] for name, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def target_labels(task: str) -> list[str]:
    try:
        return TARGETS[task]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task}. Expected one of {sorted(TARGETS)}") from exc


def validate_target_values(frame: pd.DataFrame, task: str) -> None:
    labels = target_labels(task)
    expected = {label: index for index, label in enumerate(labels)}
    pairs = frame[[task, f"{task}_id"]].drop_duplicates()
    for label, label_id in pairs.itertuples(index=False, name=None):
        if label not in expected or int(label_id) != expected[label]:
            raise ValueError(
                f"Unexpected {task} mapping in segmented data: {label!r} -> "
                f"{label_id!r}; expected {expected}"
            )


def load_segmented_split(
    split: str,
    segmented_dir: Path = SEGMENTED_DIR,
) -> pd.DataFrame:
    path = segmented_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing segmented data: {path}. Run "
            "`python -m ml.prepare_phobert_data --download-model` first."
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_SEGMENTED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} is empty")
    if frame["phobert_text"].isna().any() or frame["phobert_text"].str.strip().eq("").any():
        raise ValueError(f"{path} contains empty segmented text")
    if set(frame["split"].unique()) != {split}:
        raise ValueError(f"{path} contains rows from another split")
    return frame


def select_samples(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None:
        return frame.reset_index(drop=True)
    if maximum <= 0:
        raise ValueError("Maximum sample count must be greater than zero")
    return frame.head(maximum).reset_index(drop=True)


def validate_max_length(tokenizer: Any, max_length: int) -> None:
    if max_length <= 0:
        raise ValueError("max_length must be greater than zero")
    model_max = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if 0 < model_max < 100_000 and max_length > model_max:
        raise ValueError(
            f"max_length={max_length} exceeds tokenizer limit {model_max}"
        )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
