import sys
import types

import pytest

from sentiment_analysis.genai import (
    GenAIClassifier,
    GenAIError,
    InvalidGenAIResponse,
    OpenAIProvider,
    parse_genai_response,
)


class MockProvider:
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
    with pytest.raises(GenAIError, match="nao configurada"):
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
    schema_format = captured["text"]["format"]
    assert schema_format["type"] == "json_schema"
    assert schema_format["strict"] is True
    assert schema_format["schema"]["additionalProperties"] is False
