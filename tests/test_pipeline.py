import json

import pytest

from article_gate.fake import FakeProvider
from article_gate.fetch import Article
from article_gate.pipeline import (
    CallStats,
    InvalidModelOutput,
    complete_json,
    parse_json_reply,
    process_article,
)
from article_gate.schemas import Classification

VALID_CLASSIFICATION = json.dumps(
    {"is_bookable": True, "article_type": "itinerary", "confidence": 0.9, "reason": "Day plan."}
)
ARTICLE = Article(
    source="test.txt",
    title="Two days in Porto",
    text="Start at the Ribeira waterfront. Cross the Dom Luís I Bridge to the port cellars.",
)


def test_parses_json_inside_code_fence() -> None:
    reply = f"Here you go:\n```json\n{VALID_CLASSIFICATION}\n```"
    assert parse_json_reply(reply, Classification).is_bookable


def test_retries_once_and_feeds_back_the_error() -> None:
    provider = FakeProvider(['{"is_bookable": "maybe"}', VALID_CLASSIFICATION])
    stats = CallStats()

    result = complete_json(provider, "system", "PROMPT", Classification, stats)

    assert result.article_type == "itinerary"
    assert stats.calls == 2
    assert stats.invalid_replies == 1
    retry_prompt = provider.calls[1][1]
    assert retry_prompt.startswith("PROMPT")
    assert '{"is_bookable": "maybe"}' in retry_prompt
    assert "is_bookable" in retry_prompt.split("failed validation", 1)[1]


def test_gives_up_after_second_invalid_reply() -> None:
    provider = FakeProvider(["not json", "still not json"])
    stats = CallStats()

    with pytest.raises(InvalidModelOutput):
        complete_json(provider, "system", "prompt", Classification, stats)
    assert stats.calls == 2
    assert stats.failures == 1


def test_process_article_reports_failure_instead_of_raising() -> None:
    result = process_article(FakeProvider(["{}", "{}"]), ARTICLE)
    assert result.error and result.error.startswith("invalid JSON")
    assert result.classification is None


def test_not_bookable_skips_extraction() -> None:
    reply = json.dumps(
        {"is_bookable": False, "article_type": "news", "confidence": 0.8, "reason": "News."}
    )
    provider = FakeProvider([reply])

    result = process_article(provider, ARTICLE)

    assert result.extraction is None
    assert len(provider.calls) == 1


def test_bookable_article_is_extracted_and_grounded() -> None:
    extraction = json.dumps(
        {
            "places": [
                {
                    "name": "Ribeira",
                    "type": "neighbourhood",
                    "country": None,
                    "evidence": "Start at the Ribeira waterfront.",
                },
                {
                    "name": "Livraria Lello",
                    "type": "venue",
                    "country": "Portugal",
                    "evidence": "Visit the famous Livraria Lello bookshop.",
                },
            ],
            "activities": [],
        }
    )
    provider = FakeProvider([VALID_CLASSIFICATION, extraction])

    result = process_article(provider, ARTICLE)

    assert result.extraction is not None
    assert [p.name for p in result.extraction.places] == ["Ribeira"]
    assert result.dropped == ["place:Livraria Lello"]
