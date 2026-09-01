"""Consolida resultados observados; abordagens nao executadas ficam como N/A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_analysis.config import METRICS_DIR, REPORTS_DIR


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build(mode: str = "quick") -> pd.DataFrame:
    classical = _read_json(METRICS_DIR / f"{mode}_classical_metrics.json")
    if classical is None:
        raise FileNotFoundError("Execute o treinamento classico antes de consolidar resultados")
    rows = []
    names = {
        "majority_baseline": "Baseline majoritário",
        "logistic_regression": "Regressão Logística",
        "complement_naive_bayes": "Complement Naive Bayes",
    }
    for key, display_name in names.items():
        metrics = classical["results"][key]
        rows.append(
            {
                "Modelo": display_name,
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "Latência média (ms/item)": metrics["latency_ms_per_sample"],
                "Amostras": metrics["n_samples"],
                "Observações": "Teste quick balanceado; métrica observada em CPU.",
            }
        )

    transformer = _read_json(METRICS_DIR / f"{mode}_transformer_metrics.json")
    rows.append(
        {
            "Modelo": "Transformer",
            "Macro F1": transformer.get("macro_f1") if transformer else None,
            "Weighted F1": transformer.get("weighted_f1") if transformer else None,
            "Latência média (ms/item)": transformer.get("latency_ms_per_sample")
            if transformer
            else None,
            "Amostras": transformer.get("n_samples") if transformer else None,
            "Observações": (
                "Zero-shot, revisão fixada; domínio de treino diferente do B2W."
                if transformer
                else "N/A — avaliação opcional ainda não executada."
            ),
        }
    )

    for strategy, display in (("zero_shot", "GenAI zero-shot"), ("few_shot", "GenAI few-shot")):
        candidates = sorted(METRICS_DIR.glob(f"{mode}_genai_*_{strategy}.json"))
        metrics = _read_json(candidates[-1]) if candidates else None
        rows.append(
            {
                "Modelo": display,
                "Macro F1": metrics.get("macro_f1") if metrics else None,
                "Weighted F1": metrics.get("weighted_f1") if metrics else None,
                "Latência média (ms/item)": metrics.get("latency_ms_per_sample")
                if metrics
                else None,
                "Amostras": metrics.get("n_samples") if metrics else None,
                "Observações": (
                    f"Cobertura válida: {metrics['coverage']:.1%}."
                    if metrics
                    else "N/A — exige provedor opcional."
                ),
            }
        )

    frame = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    print(build(args.mode).to_string(index=False))


if __name__ == "__main__":
    main()
