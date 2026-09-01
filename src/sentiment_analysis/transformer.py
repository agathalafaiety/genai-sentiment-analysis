"""Avaliacao opcional de um Transformer multilíngue no mesmo teste."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_analysis.config import (
    FIGURES_DIR,
    METRICS_DIR,
    TRANSFORMER_MODEL_ID,
    TRANSFORMER_REVISION,
)
from sentiment_analysis.data import load_splits
from sentiment_analysis.training import classification_metrics, save_confusion_matrix


class TransformerPredictor:
    def __init__(self, *, device: int = -1) -> None:
        try:
            import torch
            import transformers
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Instale o extra opcional: pip install -e .[transformer]") from exc
        self.torch_version = torch.__version__
        self.transformers_version = transformers.__version__
        self.classifier = pipeline(
            "text-classification",
            model=TRANSFORMER_MODEL_ID,
            revision=TRANSFORMER_REVISION,
            device=device,
            top_k=None,
        )

    def predict_many(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
    ) -> tuple[list[str], list[dict[str, float]], float]:
        started = time.perf_counter()
        output = self.classifier(texts, batch_size=batch_size, truncation=True, max_length=512)
        elapsed_ms = (time.perf_counter() - started) * 1_000 / len(texts)
        probabilities = [
            {str(item["label"]).lower(): float(item["score"]) for item in scores}
            for scores in output
        ]
        predictions = [max(scores, key=scores.get) for scores in probabilities]
        return predictions, probabilities, elapsed_ms


def evaluate_transformer(
    test: pd.DataFrame,
    *,
    limit: int | None = None,
    device: int = -1,
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    mode: str = "quick",
) -> dict[str, Any]:
    if limit and limit < len(test):
        per_class = max(1, limit // test["label"].nunique())
        test = (
            test.groupby("label", group_keys=False)
            .sample(n=per_class, random_state=42)
            .reset_index(drop=True)
        )
    predictor = TransformerPredictor(device=device)
    predictions, probabilities, latency = predictor.predict_many(test["text"].tolist())
    metrics = classification_metrics(test["label"].tolist(), predictions)
    metrics.update(
        {
            "latency_ms_per_sample": latency,
            "model_id": TRANSFORMER_MODEL_ID,
            "revision": TRANSFORMER_REVISION,
            "runtime": {
                "python_platform": platform.platform(),
                "torch": predictor.torch_version,
                "transformers": predictor.transformers_version,
                "device_argument": device,
            },
        }
    )
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{mode}_transformer_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output = test[["text", "label"]].copy()
    output["prediction_transformer"] = predictions
    output["probabilities_transformer"] = [json.dumps(value) for value in probabilities]
    output.to_csv(metrics_dir / f"{mode}_transformer_predictions.csv", index=False)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        Path(figures_dir) / f"{mode}_transformer_confusion_matrix.png",
        title="Matriz de confusao — Transformer",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", type=int, default=-1, help="-1 para CPU; 0 para primeira GPU")
    args = parser.parse_args()
    metrics = evaluate_transformer(
        load_splits(args.mode)["test"],
        limit=args.limit,
        device=args.device,
        mode=args.mode,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
