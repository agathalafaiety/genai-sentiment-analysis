import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from sentiment_analysis.genai import (
    GenAIClassifier,
    GenAIError,
    InvalidGenAIResponse,
    OpenAIProvider,
    balanced_evaluation_sample,
    evaluate_genai,
    parse_genai_response,
)


class MockProvider:
    provider_name = "mock"
    model_id = "deterministic-mock"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, *, instructions: str, prompt: str) -> str:
        self.calls += 1
        assert "classificador" in instructions
        assert "<texto>" in prompt
        return self.response


def test_valid_response_with_code_fence() -> None:
    result = parse_genai_response(
        '```json\n{"sentiment":"neutral","confidence":0.72,"explanation":"Texto factual sem opinião."}\n```'
    )
    assert result.sentiment == "neutral"


@pytest.mark.parametrize(
    "raw",
    [
        "sem json",
        '{"sentiment":"mixed","confidence":0.5,"explanation":"Inválido."}',
        '{"sentiment":"positive","confidence":2,"explanation":"Inválido."}',
        '{"sentiment":"positive","confidence":0.8,"explanation":""}',
        '{"sentiment":"positive","confidence":0.8,"explanation":"Elogio.","extra":true}',
    ],
)
def test_invalid_responses(raw: str) -> None:
    with pytest.raises(InvalidGenAIResponse):
        parse_genai_response(raw)


def test_mock_classifier_zero_and_few_shot() -> None:
    response = '{"sentiment":"positive","confidence":0.9,"explanation":"Elogio direto."}'
    for strategy in ("zero_shot", "few_shot"):
        provider = MockProvider(response)
        result = GenAIClassifier(provider, strategy=strategy).classify("Produto excelente")
        assert result.sentiment == "positive"
        assert provider.calls == 1


def test_safe_classifier_captures_provider_failure() -> None:
    attempt = GenAIClassifier(
        MockProvider("resposta inválida"), strategy="zero_shot"
    ).safe_classify("texto")
    assert not attempt.valid
    assert attempt.result is None
    assert "InvalidGenAIResponse" in attempt.error


def test_openai_provider_stays_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(GenAIError, match="não configurada"):
        OpenAIProvider()


def test_openai_provider_requests_strict_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return types.SimpleNamespace(
                output_text=(
                    '{"sentiment":"positive","confidence":0.9,"explanation":"Elogio direto."}'
                )
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client_options"] = kwargs
            self.responses = FakeResponses()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")

    raw = OpenAIProvider(model="test-model").complete(
        instructions="classifique",
        prompt="texto",
    )
    assert "positive" in raw
    assert captured["model"] == "test-model"
    assert captured["store"] is False
    assert captured["max_output_tokens"] == 250
    schema_format = captured["text"]["format"]
    assert schema_format["type"] == "json_schema"
    assert schema_format["strict"] is True
    assert schema_format["schema"]["additionalProperties"] is False


def test_balanced_evaluation_sample() -> None:
    frame = pd.DataFrame(
        [
            {"text": f"{label}-{index}", "label": label}
            for label in ("negative", "neutral", "positive")
            for index in range(10)
        ]
    )
    sample = balanced_evaluation_sample(frame, 9)
    assert sample["label"].value_counts().to_dict() == {
        "negative": 3,
        "neutral": 3,
        "positive": 3,
    }


def test_evaluate_genai_persists_auditable_results(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"text": f"texto {label}", "label": label}
            for label in ("negative", "neutral", "positive")
        ]
    )
    provider = MockProvider(
        '{"sentiment":"positive","confidence":0.8,"explanation":"Resposta válida."}'
    )
    metrics = evaluate_genai(
        frame,
        provider,
        strategy="zero_shot",
        limit=3,
        metrics_dir=tmp_path,
    )
    assert metrics["n_samples"] == 3
    assert metrics["validity_rate"] == 1
    assert (tmp_path / "quick_genai_mock_zero_shot_metrics.json").exists()
    assert (tmp_path / "quick_genai_mock_zero_shot_predictions.csv").exists()
