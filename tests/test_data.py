from pathlib import Path

import pandas as pd
import pytest

from sentiment_analysis.data import (
    DataValidationError,
    load_and_clean,
    split_data,
    stratified_sample,
    validate_prepared_data,
    validate_raw_data,
)


def test_validate_raw_columns() -> None:
    with pytest.raises(DataValidationError, match="Colunas"):
        validate_raw_data(pd.DataFrame({"review_text": ["x"]}))


def test_load_clean_maps_ratings_and_removes_duplicates(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "review_title": ["Ruim", "Ok", "Ótimo", "Ótimo"],
            "review_text": ["quebrou", "recebi", "adorei", "adorei"],
            "overall_rating": [1, 3, 5, 5],
        }
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    clean = load_and_clean(path)
    assert clean["label"].tolist() == ["negative", "neutral", "positive"]
    assert len(clean) == 3


def test_stratified_sample_and_split_are_consistent(labeled_frame: pd.DataFrame) -> None:
    sample = stratified_sample(labeled_frame, per_class=10)
    assert sample["label"].value_counts().to_dict() == {
        "negative": 10,
        "neutral": 10,
        "positive": 10,
    }
    splits = split_data(sample)
    assert sum(map(len, splits.values())) == len(sample)
    text_sets = [set(frame["text"]) for frame in splits.values()]
    assert not (text_sets[0] & text_sets[1])
    assert not (text_sets[0] & text_sets[2])
    assert not (text_sets[1] & text_sets[2])
    for frame in splits.values():
        validate_prepared_data(frame)


def test_invalid_label_is_rejected() -> None:
    invalid = pd.DataFrame({"text": ["x"], "label": ["mixed"]})
    with pytest.raises(DataValidationError, match="Rotulos invalidos"):
        validate_prepared_data(invalid)
