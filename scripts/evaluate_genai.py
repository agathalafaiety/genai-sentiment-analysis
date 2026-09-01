"""Avalia GenAI zero/few-shot apenas quando um provedor e escolhido explicitamente."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sentiment_analysis.config import METRICS_DIR, RANDOM_SEED
from sentiment_analysis.data import load_splits
from sentiment_analysis.evaluation import classification_metrics
from sentiment_analysis.genai import GenAIClassifier, LocalTransformersProvider, OpenAIProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "local"), required=True)
    parser.add_argument("--strategy", choices=("zero_shot", "few_shot"), required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--limit", type=int, default=90)
    args = parser.parse_args()

    provider = OpenAIProvider() if args.provider == "openai" else LocalTransformersProvider()
    classifier = GenAIClassifier(provider, strategy=args.strategy)
    test = load_splits(args.mode)["test"]
    per_class = max(1, args.limit // test["label"].nunique())
    sample = (
        test.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    rows = []
    for row in sample.itertuples(index=False):
        attempt = classifier.safe_classify(row.text)
        rows.append(
            {
                "text": row.text,
                "label": row.label,
                "prediction": attempt.result.sentiment if attempt.result else None,
                "confidence": attempt.result.confidence if attempt.result else None,
                "explanation": attempt.result.explanation if attempt.result else None,
                "valid": attempt.valid,
                "latency_ms": attempt.latency_ms,
                "error": attempt.error,
            }
        )

    output = pd.DataFrame(rows)
    valid = output[output["valid"]]
    metrics = {
        "provider": args.provider,
        "strategy": args.strategy,
        "coverage": float(output["valid"].mean()),
        "valid_responses": int(output["valid"].sum()),
        "n_samples": len(output),
        "latency_ms_per_sample": float(output["latency_ms"].mean()),
    }
    if not valid.empty:
        metrics.update(
            classification_metrics(valid["label"].tolist(), valid["prediction"].tolist())
        )
    else:
        metrics.update({"macro_f1": None, "weighted_f1": None, "accuracy": None})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}_genai_{args.provider}_{args.strategy}"
    Path(METRICS_DIR / f"{stem}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.to_csv(METRICS_DIR / f"{stem}_predictions.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
