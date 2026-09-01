"""Classificacao GenAI opcional com prompts versionados e validacao estrita."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from sentiment_analysis.config import LOCAL_GENAI_MODEL_ID, LOCAL_GENAI_REVISION, PROMPTS_DIR


class GenAIError(RuntimeError):
    """Erro controlado de provedor, timeout ou formato."""


class InvalidGenAIResponse(GenAIError):
    """Resposta nao satisfez o contrato estruturado."""


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
            raise ValueError("explanation nao pode ser vazia")
        return value


class TextProvider(Protocol):
    def complete(self, *, instructions: str, prompt: str) -> str: ...


@dataclass(frozen=True)
class GenAIAttempt:
    result: GenAIResult | None
    valid: bool
    latency_ms: float
    error: str | None = None


def parse_genai_response(raw_response: str) -> GenAIResult:
    """Extrai o primeiro objeto JSON e valida classes, confianca e explicacao."""
    raw = str(raw_response).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise InvalidGenAIResponse("A resposta nao contem um objeto JSON")
    try:
        payload = json.loads(raw[start : end + 1])
        return GenAIResult.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise InvalidGenAIResponse(f"Resposta GenAI invalida: {exc}") from exc


def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> str:
    path = Path(directory) / "prompts.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt versionado nao encontrado: {path}")
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if name not in prompts:
        raise KeyError(f"Prompt desconhecido: {name}")
    return str(prompts[name]).strip()


class GenAIClassifier:
    def __init__(
        self, provider: TextProvider, *, strategy: Literal["zero_shot", "few_shot"]
    ) -> None:
        self.provider = provider
        self.strategy = strategy

    def classify(self, text: str) -> GenAIResult:
        if not str(text).strip():
            raise ValueError("Informe um texto nao vazio")
        instructions = load_prompt("system_v1")
        template = load_prompt(f"{self.strategy}_v1")
        raw = self.provider.complete(instructions=instructions, prompt=template.format(text=text))
        return parse_genai_response(raw)

    def safe_classify(self, text: str) -> GenAIAttempt:
        started = time.perf_counter()
        try:
            result = self.classify(text)
            return GenAIAttempt(result, True, (time.perf_counter() - started) * 1_000)
        except Exception as exc:  # provider SDKs expose different timeout/error classes
            return GenAIAttempt(
                None,
                False,
                (time.perf_counter() - started) * 1_000,
                f"{type(exc).__name__}: {exc}",
            )


class OpenAIProvider:
    """Adaptador opcional para Responses API com Structured Outputs."""

    def __init__(self, *, model: str | None = None, timeout: float = 30.0) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise GenAIError("OPENAI_API_KEY nao configurada; o provedor permanece desativado")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenAIError("Instale o extra opcional: pip install -e .[genai]") from exc
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
        self.client = OpenAI(timeout=timeout, max_retries=2)

    def complete(self, *, instructions: str, prompt: str) -> str:
        schema = GenAIResult.model_json_schema()
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "sentiment_classification",
                    "strict": True,
                    "schema": schema,
                }
            },
            store=False,
        )
        if not response.output_text:
            raise GenAIError("O provedor retornou uma resposta vazia")
        return response.output_text


class LocalTransformersProvider:
    """Provedor aberto para Colab; carregamento e download sao deliberadamente lazy."""

    def __init__(
        self,
        model_id: str = LOCAL_GENAI_MODEL_ID,
        revision: str = LOCAL_GENAI_REVISION,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise GenAIError("Instale o extra opcional: pip install -e .[transformer]") from exc
        self.generator = pipeline(
            "text-generation",
            model=model_id,
            revision=revision,
            device_map="auto",
            model_kwargs={"torch_dtype": "auto"},
        )

    def complete(self, *, instructions: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        output = self.generator(messages, max_new_tokens=180, do_sample=False)
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1]["content"])
        return str(generated)
