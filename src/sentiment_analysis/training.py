"""Treinamento reproduzivel dos baselines e modelos classicos."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from sentiment_analysis.config import (
    FIGURES_DIR,
    LABELS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from sentiment_analysis.data import load_splits, prepare_dataset
from sentiment_analysis.evaluation import (
    classification_metrics,
    measure_prediction_latency,
    save_confusion_matrix,
)
from sentiment_analysis.preprocessing import normalize_text


@dataclass(frozen=True)
class Candidate:
    name: str
    parameter: str
    values: tuple[float, ...]


CANDIDATES = (
    Candidate("logistic_regression", "classifier__C", (0.5, 1.0, 2.0)),
    Candidate("complement_naive_bayes", "classifier__alpha", (0.25, 0.5, 1.0)),
)


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
