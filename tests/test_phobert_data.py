from pathlib import Path

import pandas as pd

import pytest

from ml.phobert_common import (
    EncodedTextDataset,
    load_segmented_split,
    validate_target_values,
)
from ml.prepare_phobert_data import segment_text


class FakeSegmenter:
    def word_segment(self, text: str) -> list[str]:
        return text.replace("giảng viên", "giảng_viên").split()


class FakeTokenizer:
    model_max_length = 256

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "truncation": True,
            "padding": "max_length",
            "max_length": 4,
        }
        return {
            "input_ids": [[1, 2, 0, 0] for _ in texts],
            "attention_mask": [[1, 1, 0, 0] for _ in texts],
        }


def test_segment_text_joins_vncorenlp_tokens():
    assert segment_text(FakeSegmenter(), "giảng viên tốt") == "giảng_viên tốt"


def test_encoded_dataset_exposes_transformer_fields():
    dataset = EncodedTextDataset(["một câu"], [2], FakeTokenizer(), max_length=4)
    assert len(dataset) == 1
    assert dataset[0] == {
        "input_ids": [1, 2, 0, 0],
        "attention_mask": [1, 1, 0, 0],
        "labels": 2,
    }


def test_load_segmented_split_validates_cache(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "text": ["giảng viên tốt"],
            "phobert_text": ["giảng_viên tốt"],
            "topic": ["lecturer"],
            "topic_id": [0],
            "sentiment": ["positive"],
            "sentiment_id": [2],
            "split": ["train"],
        }
    )
    frame.to_csv(tmp_path / "train.csv", index=False)
    loaded = load_segmented_split("train", tmp_path)
    assert loaded["phobert_text"].tolist() == ["giảng_viên tốt"]


def test_validate_target_values_rejects_changed_label_mapping():
    frame = pd.DataFrame({"sentiment": ["positive"], "sentiment_id": [0]})
    with pytest.raises(ValueError, match="Unexpected sentiment mapping"):
        validate_target_values(frame, "sentiment")
