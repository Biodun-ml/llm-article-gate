import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from article_gate.fetch import read_file
from article_gate.grounding import normalize
from article_gate.pipeline import ArticleResult, gate_url, process_article
from article_gate.providers import Provider


@dataclass
class LabelledArticle:
    id: str
    url: str
    path: Path
    is_bookable: bool
    places: list[str]


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class ModelReport:
    model: str
    gate: PRF = field(default_factory=PRF)
    places: PRF = field(default_factory=PRF)
    articles: int = 0
    url_gated: int = 0
    llm_calls: int = 0
    invalid_replies: int = 0
    failures: int = 0
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def invalid_rate(self) -> float:
        return self.invalid_replies / self.llm_calls if self.llm_calls else 0.0

    @property
    def avg_latency_s(self) -> float:
        scored = self.articles - self.url_gated
        return self.latency_s / scored if scored else 0.0


def load_labels(path: Path) -> list[LabelledArticle]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            LabelledArticle(
                id=row["id"],
                url=row["url"],
                path=path.parent / row["file"],
                is_bookable=row["is_bookable"],
                places=row.get("places", []),
            )
        )
    return rows


def match_key(name: str) -> str:
    key = re.sub(r"[^\w\s]", " ", normalize(name))
    key = re.sub(r"\s+", " ", key).strip()
    return key.removeprefix("the ")


def score_places(predicted: list[str], gold: list[str]) -> PRF:
    pred_keys = {match_key(p) for p in predicted}
    gold_keys = {match_key(g) for g in gold}
    return PRF(
        tp=len(pred_keys & gold_keys),
        fp=len(pred_keys - gold_keys),
        fn=len(gold_keys - pred_keys),
    )


def update_report(report: ModelReport, label: LabelledArticle, result: ArticleResult) -> None:
    report.articles += 1
    if result.gate_rule:
        report.url_gated += 1

    predicted = result.is_bookable
    if predicted and label.is_bookable:
        report.gate.tp += 1
    elif predicted:
        report.gate.fp += 1
    elif label.is_bookable:
        report.gate.fn += 1

    predicted_places = [p.name for p in result.extraction.places] if result.extraction else []
    place_score = score_places(predicted_places, label.places)
    report.places.tp += place_score.tp
    report.places.fp += place_score.fp
    report.places.fn += place_score.fn

    stats = result.stats
    report.llm_calls += stats.calls
    report.invalid_replies += stats.invalid_replies
    report.failures += stats.failures
    report.latency_s += stats.latency_s
    if stats.input_tokens is not None:
        report.input_tokens = (report.input_tokens or 0) + stats.input_tokens
    if stats.output_tokens is not None:
        report.output_tokens = (report.output_tokens or 0) + stats.output_tokens


def evaluate(
    provider: Provider,
    labels: list[LabelledArticle],
    results: list[tuple[LabelledArticle, ArticleResult]] | None = None,
) -> ModelReport:
    report = ModelReport(model=provider.name)
    for position, label in enumerate(labels, start=1):
        rule = gate_url(label.url)
        if rule:
            result = ArticleResult(source=label.url, gate_rule=rule)
        else:
            article = read_file(label.path)
            article.source = label.url
            result = process_article(provider, article)
        status = (
            result.error
            or result.gate_rule
            or ("bookable" if result.is_bookable else "not bookable")
        )
        print(
            f"  [{position}/{len(labels)}] {label.id}: {status} ({result.stats.latency_s:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        update_report(report, label, result)
        if results is not None:
            results.append((label, result))
    return report


def format_table(reports: list[ModelReport]) -> str:
    headers = [
        "model",
        "gate P",
        "gate R",
        "gate F1",
        "place P",
        "place R",
        "place F1",
        "invalid JSON",
        "failed",
        "LLM calls",
        "avg s/article",
        "tokens in/out",
    ]
    rows = [headers]
    for r in reports:
        tokens = "n/a" if r.input_tokens is None else f"{r.input_tokens}/{r.output_tokens}"
        rows.append(
            [
                r.model,
                f"{r.gate.precision:.2f}",
                f"{r.gate.recall:.2f}",
                f"{r.gate.f1:.2f}",
                f"{r.places.precision:.2f}",
                f"{r.places.recall:.2f}",
                f"{r.places.f1:.2f}",
                f"{r.invalid_rate:.0%}",
                str(r.failures),
                str(r.llm_calls),
                f"{r.avg_latency_s:.2f}",
                tokens,
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
