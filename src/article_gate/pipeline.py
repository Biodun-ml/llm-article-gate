import re
import time
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from article_gate.fetch import Article
from article_gate.grounding import ground
from article_gate.providers import Provider
from article_gate.schemas import Classification, Extraction
from article_gate.url_gate import check_url, is_url

# Long-form guides rarely add new places after this point, and small local models
# degrade badly on long contexts.
MAX_ARTICLE_CHARS = 12_000

ModelT = TypeVar("ModelT", bound=BaseModel)


class InvalidModelOutput(Exception):
    pass


@dataclass
class CallStats:
    calls: int = 0
    invalid_replies: int = 0
    failures: int = 0
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def add_tokens(self, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens


@dataclass
class ArticleResult:
    source: str
    gate_rule: str | None = None
    classification: Classification | None = None
    extraction: Extraction | None = None
    dropped: list[str] = field(default_factory=list)
    stats: CallStats = field(default_factory=CallStats)
    error: str | None = None

    @property
    def is_bookable(self) -> bool:
        return bool(self.classification and self.classification.is_bookable)


def load_prompt(name: str) -> str:
    return files("article_gate").joinpath("prompts", f"{name}.txt").read_text(encoding="utf-8")


def parse_json_reply(text: str, model: type[ModelT]) -> ModelT:
    # Models sometimes wrap JSON in ```json fences or add a sentence before it.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return model.model_validate_json(match.group(0) if match else text)


def complete_json(
    provider: Provider, system: str, prompt: str, model: type[ModelT], stats: CallStats
) -> ModelT:
    attempt_prompt = prompt
    for attempt in range(2):
        started = time.perf_counter()
        completion = provider.complete(system, attempt_prompt)
        stats.latency_s += time.perf_counter() - started
        stats.calls += 1
        stats.add_tokens(completion.input_tokens, completion.output_tokens)
        try:
            return parse_json_reply(completion.text, model)
        except ValidationError as exc:
            stats.invalid_replies += 1
            if attempt == 1:
                stats.failures += 1
                raise InvalidModelOutput(str(exc)) from exc
            attempt_prompt = (
                f"{prompt}\n\nYour previous reply was:\n{completion.text}\n\n"
                f"It failed validation:\n{exc}\n\nReply again with only the corrected JSON object."
            )
    raise AssertionError("unreachable")


def classify(provider: Provider, article: Article, stats: CallStats) -> Classification:
    prompt = f"TITLE: {article.title}\n\nARTICLE:\n{article.text[:MAX_ARTICLE_CHARS]}"
    return complete_json(provider, load_prompt("classify"), prompt, Classification, stats)


def extract(provider: Provider, article: Article, stats: CallStats) -> Extraction:
    prompt = f"TITLE: {article.title}\n\nARTICLE:\n{article.text[:MAX_ARTICLE_CHARS]}"
    return complete_json(provider, load_prompt("extract"), prompt, Extraction, stats)


def gate_url(source: str, url: str | None = None) -> str | None:
    target = url or source
    if not is_url(target):
        return None
    decision = check_url(target)
    return None if decision.passed else decision.rule


def process_article(provider: Provider, article: Article) -> ArticleResult:
    result = ArticleResult(source=article.source)
    try:
        result.classification = classify(provider, article, result.stats)
        if result.classification.is_bookable:
            raw = extract(provider, article, result.stats)
            result.extraction, result.dropped = ground(raw, article.text)
    except InvalidModelOutput as exc:
        result.error = f"invalid JSON after retry: {exc.args[0].splitlines()[0]}"
    return result
