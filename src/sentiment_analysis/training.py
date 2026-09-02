"""Treinamento reproduzivel dos baselines e modelos classicos."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from pydantic import BaseModel, Field
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from sentiment_analysis.config import (
    FIGURES_DIR,
    LABELS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from sentiment_analysis.data import file_sha256, load_splits, normalize_text, prepare_dataset


@dataclass(frozen=True)
class Candidate:
    name: str
    parameter: str
    values: tuple[float, ...]


CANDIDATES = (
    Candidate("logistic_regression", "classifier__C", (0.5, 1.0, 2.0)),
    Candidate("complement_naive_bayes", "classifier__alpha", (0.25, 0.5, 1.0)),
)


class Prediction(BaseModel):
    sentiment: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    latency_ms: float = Field(ge=0)


class ClassicalPredictor:
    """Carrega o pipeline salvo e retorna classe, probabilidades e latencia."""

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
        return Prediction(
            sentiment=predicted,
            confidence=probabilities[predicted],
            probabilities=probabilities,
            latency_ms=(time.perf_counter() - started) * 1_000,
        )

    def _probabilities(self, text: str) -> dict[str, float]:
        if hasattr(self.pipeline, "predict_proba"):
            values = self.pipeline.predict_proba([text])[0]
        else:
            scores = np.asarray(self.pipeline.decision_function([text])).reshape(-1)
            exponentials = np.exp(scores - scores.max())
            values = exponentials / exponentials.sum()
        classes = [str(value) for value in self.pipeline.classes_]
        mapping = {label: float(value) for label, value in zip(classes, values, strict=True)}
        return {label: mapping.get(label, 0.0) for label in LABELS}


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """Calcula metricas globais, por classe, erros e matriz de confusao."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho nao vazio")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    errors = int(
        sum(expected != predicted for expected, predicted in zip(y_true, y_pred, strict=True))
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "errors": errors,
        "error_rate": errors / len(y_true),
        "n_samples": len(y_true),
    }


def measure_prediction_latency(
    predict: Callable[[Sequence[str]], Sequence[str]],
    texts: Sequence[str],
    *,
    repeats: int = 3,
) -> tuple[list[str], float]:
    if not texts:
        raise ValueError("texts nao pode ser vazio")
    predictions: list[str] = []
    elapsed_values = []
    for _ in range(repeats):
        started = time.perf_counter()
        predictions = list(predict(texts))
        elapsed_values.append(time.perf_counter() - started)
    return predictions, float(np.mean(elapsed_values) * 1_000 / len(texts))


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    destination: Path,
    *,
    title: str,
    labels: Sequence[str] = LABELS,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axis,
    )
    axis.set(title=title, xlabel="Predito", ylabel="Real")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def build_pipeline(model_name: str, *, parameter_value: float = 1.0) -> Pipeline:
    """Constroi pipeline TF-IDF + classificador; o vetor e ajustado so no treino."""
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=50_000,
        sublinear_tf=True,
        strip_accents=None,
    )
    if model_name == "logistic_regression":
        classifier: ClassifierMixin = LogisticRegression(
            C=parameter_value,
            class_weight="balanced",
            max_iter=1_000,
            random_state=RANDOM_SEED,
        )
    elif model_name == "complement_naive_bayes":
        classifier = ComplementNB(alpha=parameter_value, fit_prior=False)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def _select_candidate(
    candidate: Candidate,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[float, float]:
    best_value = candidate.values[0]
    best_score = -1.0
    for value in candidate.values:
        pipeline = build_pipeline(candidate.name, parameter_value=value)
        pipeline.fit(train["text"], train["label"])
        prediction = pipeline.predict(validation["text"])
        score = f1_score(validation["label"], prediction, average="macro", zero_division=0)
        if score > best_score:
            best_value, best_score = value, float(score)
    return best_value, best_score


def _evaluate_model(model: Any, test: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    predictions, latency = measure_prediction_latency(
        model.predict,
        test["text"].tolist(),
    )
    metrics = classification_metrics(test["label"].tolist(), predictions)
    metrics["latency_ms_per_sample"] = latency
    return metrics, predictions


def train_classical_models(
    splits: dict[str, pd.DataFrame],
    *,
    mode: str = "quick",
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Any]:
    """Seleciona no validation, reajusta em train+validation e avalia uma vez no test."""
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    development = pd.concat([train, validation], ignore_index=True)

    results: dict[str, Any] = {}
    predictions_by_model: dict[str, list[str]] = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train[["text"]], train["label"])
    dummy_metrics, dummy_predictions = _evaluate_model(
        _DummyTextAdapter(dummy),
        test,
    )
    dummy_metrics["selection"] = {"strategy": "most_frequent"}
    results["majority_baseline"] = dummy_metrics
    predictions_by_model["majority_baseline"] = dummy_predictions

    selected: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for candidate in CANDIDATES:
        value, validation_macro_f1 = _select_candidate(candidate, train, validation)
        model = build_pipeline(candidate.name, parameter_value=value)
        model.fit(development["text"], development["label"])
        metrics, predictions = _evaluate_model(model, test)
        metrics["selection"] = {
            "parameter": candidate.parameter,
            "value": value,
            "validation_macro_f1": validation_macro_f1,
        }
        results[candidate.name] = metrics
        predictions_by_model[candidate.name] = predictions
        selected[candidate.name] = {"value": value, "validation_macro_f1": validation_macro_f1}
        fitted[candidate.name] = model

    best_name = max(selected, key=lambda name: selected[name]["validation_macro_f1"])
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"best_classical_{mode}.joblib"
    joblib.dump(fitted[best_name], model_path, compress=3)

    metadata = {
        "mode": mode,
        "random_seed": RANDOM_SEED,
        "labels": list(LABELS),
        "best_model": best_name,
        "selection_metric": "validation_macro_f1",
        "best_model_test_metrics": {
            key: results[best_name][key]
            for key in ("accuracy", "macro_f1", "weighted_f1", "latency_ms_per_sample")
        },
        "artifact": {"filename": model_path.name, "sha256": file_sha256(model_path)},
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
        "results": results,
    }

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{mode}_classical_metrics.json"
    metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = []
    for name, metrics in results.items():
        comparison.append(
            {
                "model": name,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
                "latency_ms_per_sample": metrics["latency_ms_per_sample"],
                "n_samples": metrics["n_samples"],
            }
        )
        save_confusion_matrix(
            metrics["confusion_matrix"],
            Path(figures_dir) / f"{mode}_{name}_confusion_matrix.png",
            title=f"Matriz de confusao — {name.replace('_', ' ')}",
        )
    pd.DataFrame(comparison).to_csv(metrics_dir / f"{mode}_comparison.csv", index=False)

    prediction_frame = test[["text", "label", "rating"]].copy()
    for name, predictions in predictions_by_model.items():
        prediction_frame[f"prediction_{name}"] = predictions
    prediction_frame.to_csv(metrics_dir / f"{mode}_test_predictions.csv", index=False)

    model_metadata_path = models_dir / f"best_classical_{mode}.metadata.json"
    model_metadata_path.write_text(
        json.dumps({key: value for key, value in metadata.items() if key != "results"}, indent=2),
        encoding="utf-8",
    )
    return metadata


class _DummyTextAdapter:
    """Adapta DummyClassifier treinado com matriz 2D ao contrato de texto."""

    def __init__(self, model: DummyClassifier) -> None:
        self.model = model

    def predict(self, texts: list[str]) -> Any:
        return self.model.predict([[text] for text in texts])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--prepare", action="store_true", help="Prepara os dados antes de treinar")
    args = parser.parse_args()
    if args.prepare:
        prepare_dataset(args.mode)
    splits = load_splits(args.mode)
    metadata = train_classical_models(splits, mode=args.mode)
    print(
        json.dumps({"best_model": metadata["best_model"], "results": metadata["results"]}, indent=2)
    )


if __name__ == "__main__":
    main()
