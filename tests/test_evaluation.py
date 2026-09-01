import pytest

from sentiment_analysis.evaluation import classification_metrics, measure_prediction_latency


def test_metrics_values() -> None:
    metrics = classification_metrics(
        ["negative", "neutral", "positive", "positive"],
        ["negative", "positive", "positive", "positive"],
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["errors"] == 1
    assert metrics["n_samples"] == 4
    assert len(metrics["confusion_matrix"]) == 3


def test_latency_contract() -> None:
    predictions, latency = measure_prediction_latency(
        lambda texts: ["neutral"] * len(texts), ["a", "b"]
    )
    assert predictions == ["neutral", "neutral"]
    assert latency >= 0


def test_empty_metrics_rejected() -> None:
    with pytest.raises(ValueError):
        classification_metrics([], [])
