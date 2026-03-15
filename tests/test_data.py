import pytest

import pandas as pd

from ml.data import clean_text, load_dataset, prepare_training_frame, validate_labels


def test_clean_text_normalizes_whitespace():
    assert clean_text("  giảng viên\n dạy tốt  ") == "giảng viên dạy tốt"


def test_load_dataset_has_official_splits():
    frames = load_dataset()
    assert set(frames) == {"train", "valid", "test"}
    assert all(len(frame) > 0 for frame in frames.values())


def test_label_mapping_matches_dataset():
    validate_labels(load_dataset())


def test_unknown_split_is_rejected():
    from ml.data import load_split

    with pytest.raises(ValueError):
        load_split("development")


def test_conflicting_duplicate_is_removed():
    frame = pd.DataFrame(
        {
            "text": ["câu giống nhau", "câu giống nhau", "câu khác"],
            "sentiment": ["positive", "negative", "neutral"],
            "topic": ["lecturer", "lecturer", "program"],
        }
    )
    prepared, summary = prepare_training_frame(frame)
    assert prepared["text"].tolist() == ["câu khác"]
    assert summary["conflicting_rows_removed"] == 2
