"""Metricas e visualizacoes comparaveis entre abordagens."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sentiment_analysis.config import LABELS


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """Retorna metricas globais, por classe, erros e matriz de confusao."""
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
    """Mede latencia media por item usando chamadas em lote repetidas."""
    if not texts:
        raise ValueError("texts nao pode ser vazio")
    predictions: list[str] = []
    elapsed_values = []
    for _ in range(repeats):
        started = time.perf_counter()
        predictions = list(predict(texts))
        elapsed_values.append(time.perf_counter() - started)
    return predictions, float(np.mean(elapsed_values) * 1000 / len(texts))


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
