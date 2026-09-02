"""Classificação e avaliação GenAI opcionais com saída estruturada."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from sentiment_analysis.config import (
    LOCAL_GENAI_MODEL_ID,
    LOCAL_GENAI_REVISION,
    METRICS_DIR,
    PROMPTS_DIR,
    RANDOM_SEED,
)
from sentiment_analysis.data import load_splits
from sentiment_analysis.training import classification_metrics

Strategy = Literal["zero_shot", "few_shot"]


class GenAIError(RuntimeError):
    """Erro controlado de provedor, timeout ou formato."""


class InvalidGenAIResponse(GenAIError):
    """Resposta não satisfez o contrato estruturado."""


class GenAIResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=3, max_length=500)

    @field_validator("explanation")
    @classmethod
    def explanation_must_be_clear(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("explanation não pode ser vazia")
        return value


class TextProvider(Protocol):
    provider_name: str
    model_id: str

    def complete(self, *, instructions: str, prompt: str) -> str: ...


@dataclass(frozen=True)
class GenAIAttempt:
    result: GenAIResult | None
    valid: bool
    latency_ms: float
    error: str | None = None


def parse_genai_response(raw_response: str) -> GenAIResult:
    """Extrai o primeiro objeto JSON e valida classe, confiança e explicação."""
    raw = str(raw_response).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise InvalidGenAIResponse("A resposta não contém um objeto JSON")
    try:
        payload = json.loads(raw[start : end + 1])
        return GenAIResult.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise InvalidGenAIResponse(f"Resposta GenAI inválida: {exc}") from exc


def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> str:
    path = Path(directory) / "prompts.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt versionado não encontrado: {path}")
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if name not in prompts:
        raise KeyError(f"Prompt desconhecido: {name}")
    return str(prompts[name]).strip()


class GenAIClassifier:
    def __init__(self, provider: TextProvider, *, strategy: Strategy) -> None:
        self.provider = provider
        self.strategy = strategy

    def classify(self, text: str) -> GenAIResult:
        if not str(text).strip():
            raise ValueError("Informe um texto não vazio")
        instructions = load_prompt("system_v1")
        template = load_prompt(f"{self.strategy}_v1")
        raw = self.provider.complete(instructions=instructions, prompt=template.format(text=text))
        return parse_genai_response(raw)

    def safe_classify(self, text: str) -> GenAIAttempt:
        started = time.perf_counter()
        try:
            result = self.classify(text)
            return GenAIAttempt(result, True, (time.perf_counter() - started) * 1_000)
        except Exception as exc:  # SDKs de provedores expõem exceções diferentes
            return GenAIAttempt(
                None,
                False,
                (time.perf_counter() - started) * 1_000,
                f"{type(exc).__name__}: {exc}",
            )


class OpenAIProvider:
    """Adaptador para Responses API com Structured Outputs e telemetria de tokens."""

    provider_name = "openai"
    _INPUT_USD_PER_MILLION = 0.25
    _OUTPUT_USD_PER_MILLION = 2.00

    def __init__(self, *, model: str | None = None, timeout: float = 30.0) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise GenAIError("OPENAI_API_KEY não configurada; o provedor permanece desativado")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenAIError("Instale o extra opcional: pip install -e .[genai]") from exc
        self.model_id = model or os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
        self.client = OpenAI(timeout=timeout, max_retries=2)
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, *, instructions: str, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_id,
            instructions=instructions,
            input=prompt,
            max_output_tokens=250,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "sentiment_classification",
                    "strict": True,
                    "schema": GenAIResult.model_json_schema(),
                }
            },
            store=False,
        )
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        if not response.output_text:
            raise GenAIError("O provedor retornou uma resposta vazia")
        return response.output_text

    def usage(self) -> dict[str, int | float | None]:
        cost = None
        if self.model_id == "gpt-5-mini-2025-08-07":
            cost = (
                self.input_tokens * self._INPUT_USD_PER_MILLION
                + self.output_tokens * self._OUTPUT_USD_PER_MILLION
            ) / 1_000_000
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": cost,
        }


class LocalTransformersProvider:
    """Provedor local pequeno; carregamento e download são deliberadamente lazy."""

    provider_name = "local"

    def __init__(
        self,
        model_id: str = LOCAL_GENAI_MODEL_ID,
        revision: str | None = LOCAL_GENAI_REVISION,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise GenAIError("Instale o extra opcional: pip install -e .[transformer]") from exc
        self.model_id = model_id
        model_options: dict[str, Any] = {
            "model": model_id,
            "device_map": "auto",
            "dtype": "auto",
        }
        if revision:
            model_options["revision"] = revision
        self.generator = pipeline("text-generation", **model_options)
        self.generation_config = copy.deepcopy(self.generator.model.generation_config)
        self.generation_config.max_length = None
        self.generation_config.max_new_tokens = 96
        self.generation_config.do_sample = False
        self.generation_config.temperature = None
        self.generation_config.top_p = None
        self.generation_config.top_k = None

    def complete(self, *, instructions: str, prompt: str) -> str:
        local_schema = (
            "Retorne somente um JSON válido neste formato exato: "
            '{"sentiment":"positive|neutral|negative","confidence":0.0,'
            '"explanation":"justificativa curta"}.'
        )
        messages = [
            {"role": "system", "content": f"{instructions}\n{local_schema}"},
            {"role": "user", "content": prompt},
        ]
        output = self.generator(
            messages,
            generation_config=self.generation_config,
            clean_up_tokenization_spaces=False,
        )
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1]["content"])
        return str(generated)


def balanced_evaluation_sample(test: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Seleciona a mesma quantidade por classe para uma avaliação de custo controlado."""
    if limit < 3:
        raise ValueError("limit deve ser pelo menos 3")
    labels = sorted(test["label"].unique())
    per_class, remainder = divmod(limit, len(labels))
    samples = []
    for index, label in enumerate(labels):
        amount = per_class + int(index < remainder)
        subset = test.loc[test["label"] == label]
        samples.append(subset.sample(n=min(amount, len(subset)), random_state=RANDOM_SEED))
    return pd.concat(samples, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)


def evaluate_genai(
    test: pd.DataFrame,
    provider: TextProvider,
    *,
    strategy: Strategy,
    limit: int = 30,
    mode: str = "quick",
    metrics_dir: Path = METRICS_DIR,
) -> dict[str, Any]:
    """Avalia um provedor real e persiste métricas e previsões auditáveis."""
    sample = balanced_evaluation_sample(test, limit)
    classifier = GenAIClassifier(provider, strategy=strategy)
    usage_before = provider.usage() if hasattr(provider, "usage") else None
    records: list[dict[str, Any]] = []
    predictions: list[str] = []

    for row in sample.itertuples(index=False):
        attempt = classifier.safe_classify(str(row.text))
        prediction = attempt.result.sentiment if attempt.result else "invalid"
        predictions.append(prediction)
        records.append(
            {
                "text": row.text,
                "label": row.label,
                "prediction": prediction,
                "confidence": attempt.result.confidence if attempt.result else None,
                "explanation": attempt.result.explanation if attempt.result else None,
                "valid": attempt.valid,
                "latency_ms": attempt.latency_ms,
                "error": attempt.error,
            }
        )

    metrics = classification_metrics(sample["label"].tolist(), predictions)
    valid_count = sum(record["valid"] for record in records)
    metrics.update(
        {
            "provider": provider.provider_name,
            "model_id": provider.model_id,
            "strategy": strategy,
            "valid_responses": valid_count,
            "validity_rate": valid_count / len(records),
            "latency_ms_per_sample": sum(record["latency_ms"] for record in records) / len(records),
            "runtime": {"platform": platform.platform(), "sample_seed": RANDOM_SEED},
        }
    )
    if usage_before is not None:
        usage_after = provider.usage()
        metrics["usage"] = {
            key: (None if usage_after[key] is None else usage_after[key] - (usage_before[key] or 0))
            for key in usage_after
        }

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{mode}_genai_{provider.provider_name}_{strategy}"
    (metrics_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(metrics_dir / f"{prefix}_predictions.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "local"), required=True)
    parser.add_argument("--strategy", choices=("zero_shot", "few_shot", "both"), default="both")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", help="Sobrescreve o modelo padrão do provedor")
    args = parser.parse_args()

    if args.provider == "openai":
        provider: TextProvider = OpenAIProvider(model=args.model)
    else:
        provider = LocalTransformersProvider(
            model_id=args.model or LOCAL_GENAI_MODEL_ID,
            revision=None if args.model else LOCAL_GENAI_REVISION,
        )
    strategies: tuple[Strategy, ...] = (
        ("zero_shot", "few_shot") if args.strategy == "both" else (args.strategy,)
    )
    test = load_splits(args.mode)["test"]
    for strategy in strategies:
        metrics = evaluate_genai(
            test,
            provider,
            strategy=strategy,
            limit=args.limit,
            mode=args.mode,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
