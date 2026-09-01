from pathlib import Path

import joblib
import pandas as pd
import pytest

from sentiment_analysis.inference import ClassicalPredictor
from sentiment_analysis.training import build_pipeline, train_classical_models


def test_pipeline_predicts_known_classes(labeled_frame: pd.DataFrame) -> None:
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(labeled_frame["text"], labeled_frame["label"])
    predictions = pipeline.predict(
        ["produto excelente e ótimo", "pedido entregue ontem", "produto péssimo e quebrado"]
    )
    assert predictions.tolist() == ["positive", "neutral", "negative"]


def test_train_saves_best_pipeline(
    labeled_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train = labeled_frame.groupby("label", group_keys=False).head(12).reset_index(drop=True)
    validation = labeled_frame.groupby("label", group_keys=False).tail(4).reset_index(drop=True)
    test = pd.DataFrame(
        [
            {"text": "excelente adorei", "label": "positive", "rating": 5},
            {"text": "pedido chegou", "label": "neutral", "rating": 3},
            {"text": "horrível quebrado", "label": "negative", "rating": 1},
        ]
    )
    metadata = train_classical_models(
        {"train": train, "validation": validation, "test": test},
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
    )
    model_path = tmp_path / "models" / "best_classical_quick.joblib"
    assert model_path.exists()
    assert metadata["best_model"] in {"logistic_regression", "complement_naive_bayes"}
    assert joblib.load(model_path).predict(["adorei excelente"])[0] == "positive"


def test_inference_format_and_examples(labeled_frame: pd.DataFrame, tmp_path: Path) -> None:
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
        assert set(prediction.probabilities) == {"negative", "neutral", "positive"}
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
