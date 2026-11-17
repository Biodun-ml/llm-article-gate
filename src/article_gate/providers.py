import os
from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> Completion: ...


class OllamaProvider:
    def __init__(self, model: str, host: str | None = None, timeout: float | None = None):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        # CPU-only laptops can take minutes per article on a 7B model, mostly on the first call
        # while the model loads.
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT", "600"))

    def complete(self, system: str, prompt: str) -> Completion:
        response = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return Completion(
            body["message"]["content"], body.get("prompt_eval_count"), body.get("eval_count")
        )


class AnthropicProvider:
    def __init__(self, model: str, max_tokens: int = 2048):
        import anthropic

        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> Completion:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(text, message.usage.input_tokens, message.usage.output_tokens)


class OpenAICompatibleProvider:
    # OPENAI_BASE_URL points this at any /v1/chat/completions server (vLLM, LM Studio, Groq).

    def __init__(self, model: str):
        import openai

        self.model = model
        self.name = f"openai:{model}"
        self.client = openai.OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))

    def complete(self, system: str, prompt: str) -> Completion:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        usage = response.usage
        return Completion(
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
        )


def make_provider(spec: str) -> Provider:
    from article_gate.fake import FakeProvider, heuristic_responder
    from article_gate.tracing import maybe_trace

    kind, _, model = spec.partition(":")
    if kind == "fake":
        provider: Provider = FakeProvider(heuristic_responder, name="fake")
    elif not model:
        raise ValueError(f"model spec {spec!r} needs a model, e.g. ollama:qwen2.5:7b")
    elif kind == "ollama":
        provider = OllamaProvider(model)
    elif kind == "anthropic":
        provider = AnthropicProvider(model)
    elif kind == "openai":
        provider = OpenAICompatibleProvider(model)
    else:
        raise ValueError(f"unknown provider {kind!r}; use ollama, anthropic, openai or fake")
    return maybe_trace(provider)
