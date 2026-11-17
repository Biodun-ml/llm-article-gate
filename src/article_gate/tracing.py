import atexit
import os

from article_gate.providers import Completion, Provider


class LangfuseTracedProvider:
    def __init__(self, inner: Provider, client):
        self.inner = inner
        self.name = inner.name
        self.client = client

    def complete(self, system: str, prompt: str) -> Completion:
        generation = self.client.generation(
            name="article-gate",
            model=self.name,
            input=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        try:
            result = self.inner.complete(system, prompt)
        except Exception as exc:
            generation.end(level="ERROR", status_message=str(exc))
            raise
        usage = {"input": result.input_tokens, "output": result.output_tokens}
        generation.end(
            output=result.text,
            usage_details={k: v for k, v in usage.items() if v is not None} or None,
        )
        return result


def maybe_trace(provider: Provider) -> Provider:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return provider
    try:
        from langfuse import Langfuse
    except ImportError:
        return provider
    client = Langfuse()
    # The SDK batches events in a background thread; without a flush short CLI runs lose them.
    atexit.register(client.flush)
    return LangfuseTracedProvider(provider, client)
