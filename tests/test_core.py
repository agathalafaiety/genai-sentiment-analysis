from __future__ import annotations

import json
from pathlib import Path

import joblib
import nbformat
import pandas as pd
import pytest

from sentiment_analysis.data import (
    DataValidationError,
    combine_review_text,
    normalize_text,
    split_data,
    stratified_sample,
    validate_prepared_data,
)
from sentiment_analysis.training import (
    ClassicalPredictor,
    build_pipeline,
    classification_metrics,
    train_classical_models,
)


@pytest.fixture
def labeled_frame() -> pd.DataFrame:
    phrases = {
        "negative": ["produto horrível", "compra péssima", "não gostei", "veio quebrado"],
        "neutral": ["produto entregue", "chegou ontem", "duas peças", "pedido registrado"],
        "positive": ["produto excelente", "compra ótima", "adorei muito", "perfeito"],
    }
    ratings = {"negative": 1, "neutral": 3, "positive": 5}
    return pd.DataFrame(
        [
            {"text": f"{text} exemplo {repeat}", "label": label, "rating": ratings[label]}
            for label, texts in phrases.items()
            for repeat in range(5)
            for text in texts
        ]
    )


def test_preprocessing_preserves_useful_signals() -> None:
    assert normalize_text(" NÃO gostei! https://example.com ") == "não gostei! url"
    assert combine_review_text("Ótimo", None) == "ótimo"
    assert combine_review_text(float("nan"), "Chegou bem") == "chegou bem"


def test_data_sample_split_and_validation(labeled_frame: pd.DataFrame) -> None:
    sample = stratified_sample(labeled_frame, per_class=10)
    assert sample["label"].value_counts().to_dict() == {
        "negative": 10,
        "neutral": 10,
        "positive": 10,
    }
    splits = split_data(sample)
    assert sum(map(len, splits.values())) == 30
    text_sets = [set(frame["text"]) for frame in splits.values()]
    assert not any(text_sets[left] & text_sets[right] for left, right in ((0, 1), (0, 2), (1, 2)))
    with pytest.raises(DataValidationError):
        validate_prepared_data(pd.DataFrame({"text": ["x"], "label": ["mixed"]}))


def test_metrics_contract() -> None:
    metrics = classification_metrics(
        ["negative", "neutral", "positive", "positive"],
        ["negative", "positive", "positive", "positive"],
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["errors"] == 1
    assert len(metrics["confusion_matrix"]) == 3


def test_pipeline_and_inference(labeled_frame: pd.DataFrame, tmp_path: Path) -> None:
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(labeled_frame["text"], labeled_frame["label"])
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    predictor = ClassicalPredictor(path)
    cases = {
        "produto excelente ótimo": "positive",
        "pedido entregue ontem": "neutral",
        "compra horrível péssima": "negative",
    }
    for text, expected in cases.items():
        prediction = predictor.predict(text)
        assert prediction.sentiment == expected
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_training_saves_selected_pipeline(labeled_frame: pd.DataFrame, tmp_path: Path) -> None:
    splits = {
        "train": labeled_frame.groupby("label", group_keys=False).head(12).reset_index(drop=True),
        "validation": labeled_frame.groupby("label", group_keys=False)
        .tail(4)
        .reset_index(drop=True),
        "test": pd.DataFrame(
            [
                {"text": "excelente adorei", "label": "positive", "rating": 5},
                {"text": "pedido entregue", "label": "neutral", "rating": 3},
                {"text": "horrível quebrado", "label": "negative", "rating": 1},
            ]
        ),
    }
    metadata = train_classical_models(
        splits,
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
    )
    assert metadata["best_model"] in {"logistic_regression", "complement_naive_bayes"}
    assert (tmp_path / "models" / "best_classical_quick.joblib").exists()


def test_only_two_valid_notebooks_remain() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {"analysis.ipynb", "demo_colab.ipynb"}
    found = {path.name for path in (root / "notebooks").glob("*.ipynb")}
    assert found == expected
    for name in expected:
        notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        nbformat.validate(nbformat.from_dict(notebook))
