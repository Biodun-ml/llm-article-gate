import httpx
import pytest

from article_gate.fake import FakeProvider
from article_gate.providers import OllamaProvider, make_provider
from article_gate.tracing import LangfuseTracedProvider, maybe_trace


def test_ollama_reads_content_and_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = {}

    def fake_post(url: str, json: dict, timeout: float) -> httpx.Response:
        sent.update(url=url, body=json)
        body = {"message": {"content": "{}"}, "prompt_eval_count": 120, "eval_count": 8}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    completion = OllamaProvider("qwen2.5:7b", host="http://ollama:11434").complete("sys", "hi")

    assert (completion.text, completion.input_tokens, completion.output_tokens) == ("{}", 120, 8)
    assert sent["url"] == "http://ollama:11434/api/chat"
    assert sent["body"]["format"] == "json"


def test_make_provider_rejects_bad_specs() -> None:
    with pytest.raises(ValueError, match="needs a model"):
        make_provider("ollama")
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider("gemini:pro")


def test_no_tracing_without_langfuse_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    provider = FakeProvider(["{}"])
    assert maybe_trace(provider) is provider


class RecordingGeneration:
    def __init__(self) -> None:
        self.ended_with: dict = {}

    def end(self, **kwargs) -> None:
        self.ended_with = kwargs


class RecordingClient:
    def __init__(self) -> None:
        self.generations: list[RecordingGeneration] = []

    def generation(self, **kwargs) -> RecordingGeneration:
        self.generations.append(RecordingGeneration())
        return self.generations[-1]


def test_traced_provider_records_output() -> None:
    client = RecordingClient()
    traced = LangfuseTracedProvider(FakeProvider(['{"ok": 1}']), client)

    assert traced.complete("sys", "prompt").text == '{"ok": 1}'
    assert client.generations[0].ended_with["output"] == '{"ok": 1}'
