import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

from article_gate.evaluate import evaluate, format_table, load_labels
from article_gate.fetch import DEFAULT_CACHE_DIR, load_article
from article_gate.pipeline import ArticleResult, gate_url, process_article
from article_gate.providers import make_provider

DEFAULT_MODEL = "ollama:qwen2.5:7b"


def run(sources: list[str], model: str, cache_dir: Path, as_json: bool) -> int:
    provider = make_provider(model)
    results: list[ArticleResult] = []
    for source in sources:
        rule = gate_url(source)
        if rule:
            result = ArticleResult(source=source, gate_rule=rule)
        else:
            try:
                article = load_article(source, cache_dir)
            except (httpx.HTTPError, OSError) as exc:
                result = ArticleResult(source=source, error=f"fetch failed: {exc}")
            else:
                result = process_article(provider, article)
        results.append(result)
        if not as_json:
            print(format_result(result))

    if as_json:
        print(json.dumps([result_to_dict(r) for r in results], indent=2, ensure_ascii=False))
    else:
        print(format_summary(results))
    return 0


def format_result(result: ArticleResult) -> str:
    if result.gate_rule:
        return f"SKIP  {result.source}  (url gate: {result.gate_rule})\n"
    if result.error:
        return f"ERROR {result.source}  ({result.error})\n"

    c = result.classification
    assert c is not None
    verdict = "BOOK" if c.is_bookable else "NO  "
    lines = [f"{verdict}  {result.source}  [{c.article_type}, {c.confidence:.2f}] {c.reason}"]
    if result.extraction:
        for place in result.extraction.places:
            country = f", {place.country}" if place.country else ""
            lines.append(f"      place     {place.name} ({place.type}{country})")
        for activity in result.extraction.activities:
            where = f" @ {activity.place}" if activity.place else ""
            lines.append(f"      activity  {activity.name}{where}")
    if result.dropped:
        lines.append(f"      dropped (no evidence in text): {', '.join(result.dropped)}")
    return "\n".join(lines) + "\n"


def format_summary(results: list[ArticleResult]) -> str:
    gated = sum(1 for r in results if r.gate_rule)
    bookable = sum(1 for r in results if r.is_bookable)
    calls = sum(r.stats.calls for r in results)
    return (
        f"{len(results)} inputs: {gated} rejected by URL gate (saved at least {gated} LLM calls), "
        f"{bookable} bookable, {calls} LLM calls made"
    )


def result_to_dict(result: ArticleResult) -> dict:
    return {
        "source": result.source,
        "gate_rule": result.gate_rule,
        "classification": result.classification.model_dump() if result.classification else None,
        "extraction": result.extraction.model_dump() if result.extraction else None,
        "dropped": result.dropped,
        "stats": asdict(result.stats),
        "error": result.error,
    }


def run_eval(labels_path: Path, models: list[str]) -> int:
    labels = load_labels(labels_path)
    reports = []
    for spec in models:
        print(f"evaluating {spec} on {len(labels)} articles...", file=sys.stderr)
        reports.append(evaluate(make_provider(spec), labels))
    gated = reports[0].url_gated
    print(f"URL gate rejected {gated}/{len(labels)} articles before any model call.\n")
    print(format_table(reports))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="article-gate")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="classify and extract from URLs or local files")
    run_cmd.add_argument("sources", nargs="+", help="article URLs or .txt/.md/.html files")
    run_cmd.add_argument("--model", default=DEFAULT_MODEL, help="provider:model, or 'fake'")
    run_cmd.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    run_cmd.add_argument("--json", action="store_true", help="print results as JSON")

    eval_cmd = sub.add_parser("eval", help="score models against a labelled JSONL file")
    eval_cmd.add_argument("labels", type=Path)
    eval_cmd.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return run(args.sources, args.model, args.cache_dir, args.json)
        return run_eval(args.labels, args.models)
    except httpx.ConnectError as exc:
        print(
            f"could not reach the model server: {exc}. Is `ollama serve` running?", file=sys.stderr
        )
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
