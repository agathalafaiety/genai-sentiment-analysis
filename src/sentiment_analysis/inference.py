"""Inferencia consistente para CLI, testes e Streamlit."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field

from sentiment_analysis.config import LABELS, MODELS_DIR


class Prediction(BaseModel):
    sentiment: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    latency_ms: float = Field(ge=0)


class ClassicalPredictor:
    """Carrega uma unica vez o pipeline e expoe probabilidades normalizadas."""

    def __init__(self, model_path: Path | None = None, *, mode: str = "quick") -> None:
        self.model_path = Path(model_path or MODELS_DIR / f"best_classical_{mode}.joblib")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado em {self.model_path}. "
                f"Execute: python -m sentiment_analysis.training --mode {mode} --prepare"
            )
        self.pipeline = joblib.load(self.model_path)

    def predict(self, text: str) -> Prediction:
        if not str(text).strip():
            raise ValueError("Informe um texto nao vazio")
        started = time.perf_counter()
        predicted = str(self.pipeline.predict([text])[0])
        probabilities = self._probabilities(text)
        elapsed = (time.perf_counter() - started) * 1_000
        return Prediction(
            sentiment=predicted,
            confidence=probabilities[predicted],
            probabilities=probabilities,
            latency_ms=elapsed,
        )

    def _probabilities(self, text: str) -> dict[str, float]:
        if hasattr(self.pipeline, "predict_proba"):
            values = self.pipeline.predict_proba([text])[0]
            classes = [str(value) for value in self.pipeline.classes_]
            mapping = {label: float(value) for label, value in zip(classes, values, strict=True)}
        else:
            scores = np.asarray(self.pipeline.decision_function([text])).reshape(-1)
            exponentials = np.exp(scores - scores.max())
            values = exponentials / exponentials.sum()
            classes = [str(value) for value in self.pipeline.classes_]
            mapping = {label: float(value) for label, value in zip(classes, values, strict=True)}
        return {label: mapping.get(label, 0.0) for label in LABELS}


def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    return prediction.model_dump()
