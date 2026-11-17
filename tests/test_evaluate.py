from pathlib import Path

import pytest

from article_gate.evaluate import PRF, evaluate, format_table, load_labels, score_places
from article_gate.fake import FakeProvider

LABELS = Path(__file__).resolve().parents[1] / "data" / "labels.jsonl"


def test_prf_math() -> None:
    prf = PRF(tp=6, fp=2, fn=4)
    assert prf.precision == pytest.approx(0.75)
    assert prf.recall == pytest.approx(0.6)
    assert prf.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_prf_with_no_predictions_is_zero_not_error() -> None:
    assert PRF(fn=3).precision == 0.0
    assert PRF().f1 == 0.0


def test_place_matching_ignores_case_accents_and_leading_article() -> None:
    score = score_places(
        predicted=["jeronimos monastery", "The Walled City", "Hoi An", "Cascais"],
        gold=["Jerónimos Monastery", "Walled City", "Hội An", "Sintra"],
    )
    assert (score.tp, score.fp, score.fn) == (3, 1, 1)


def test_duplicate_predictions_count_once() -> None:
    score = score_places(predicted=["Lisbon", "lisbon", "LISBON"], gold=["Lisbon"])
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)


def test_eval_on_bundled_labels_with_scripted_provider() -> None:
    labels = load_labels(LABELS)
    always_no = '{"is_bookable": false, "article_type": "other", "confidence": 0.5, "reason": "x"}'
    provider = FakeProvider(lambda _system, _prompt: always_no, name="always-no")

    report = evaluate(provider, labels)

    bookable = sum(label.is_bookable for label in labels)
    assert report.articles == len(labels)
    assert report.url_gated == 2
    assert report.llm_calls == len(labels) - 2
    assert (report.gate.tp, report.gate.fp, report.gate.fn) == (0, 0, bookable)
    assert report.places.tp == 0
    assert "always-no" in format_table([report])
