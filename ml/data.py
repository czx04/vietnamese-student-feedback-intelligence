import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS = ("train", "valid", "test")
EXPECTED_COLUMNS = {
    "Sentence",
    "Topic",
    "Sentiment",
    "Encoded_topic",
    "Encoded_sentiment",
}


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def load_split(split: str, clean: bool = True) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    path = RAW_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_data.sh first.")

    with path.open(encoding="utf-8") as file:
        frame = pd.DataFrame(json.load(file))

    missing = EXPECTED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    frame = frame.rename(
        columns={
            "Sentence": "text",
            "Topic": "topic",
            "Sentiment": "sentiment",
            "Encoded_topic": "topic_id",
            "Encoded_sentiment": "sentiment_id",
        }
    )
    frame["split"] = split

    if clean:
        frame["text"] = frame["text"].map(clean_text)
        frame = frame.loc[frame["text"].ne("")].reset_index(drop=True)

    return frame


def load_dataset(clean: bool = True) -> dict[str, pd.DataFrame]:
    return {split: load_split(split, clean=clean) for split in SPLITS}


def prepare_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    prepared = frame.copy()
    prepared["_text_key"] = prepared["text"].str.casefold()
    grouped = prepared.groupby("_text_key")
    conflicting_rows = grouped["sentiment"].transform("nunique").gt(1)
    conflicting_rows |= grouped["topic"].transform("nunique").gt(1)
    prepared = prepared.loc[~conflicting_rows]
    duplicate_rows = int(prepared["_text_key"].duplicated().sum())
    prepared = prepared.drop_duplicates("_text_key", keep="first").drop(columns="_text_key")
    return prepared.reset_index(drop=True), {
        "conflicting_rows_removed": int(conflicting_rows.sum()),
        "duplicate_rows_removed": duplicate_rows,
    }


def validate_labels(frames: dict[str, pd.DataFrame]) -> None:
    expected = {
        "sentiment": {"negative": 0, "neutral": 1, "positive": 2},
        "topic": {"lecturer": 0, "program": 1, "facility": 2, "others": 3},
    }
    for split, frame in frames.items():
        for target, mapping in expected.items():
            pairs = frame[[target, f"{target}_id"]].drop_duplicates()
            actual = dict(zip(pairs[target], pairs[f"{target}_id"]))
            if actual != mapping:
                raise ValueError(f"Unexpected {target} labels in {split}: {actual}")


def export_processed() -> dict[str, Path]:
    frames = load_dataset()
    validate_labels(frames)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, frame in frames.items():
        if split == "train":
            frame, _ = prepare_training_frame(frame)
        path = PROCESSED_DIR / f"{split}.csv"
        frame.to_csv(path, index=False)
        paths[split] = path
    return paths


if __name__ == "__main__":
    for name, path in export_processed().items():
        print(f"{name}: {path}")
