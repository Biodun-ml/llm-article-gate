from article_gate.grounding import ground, normalize
from article_gate.schemas import Activity, Extraction, Place

TEXT = (
    "Day 3 is a day trip to Sintra, about forty minutes by train.\n"
    "The   Jerónimos Monastery opens at 9:30 and the queue is shortest in the first hour. "
    "Book a table at a fado house in Alfama; it’s worth it."
)


def place(name: str, evidence: str) -> Place:
    return Place(name=name, type="landmark", evidence=evidence)


def test_normalize_ignores_case_accents_quotes_and_whitespace() -> None:
    assert normalize("  Jerónimos\n  MONASTERY ") == "jeronimos monastery"
    assert normalize("it’s") == normalize("it's")


def test_keeps_items_with_evidence_in_text() -> None:
    extraction = Extraction(
        places=[place("Jeronimos Monastery", "the jeronimos monastery opens at 9:30")],
        activities=[Activity(name="fado night", evidence="fado house in Alfama; it's worth it")],
    )
    grounded, dropped = ground(extraction, TEXT)
    assert len(grounded.places) == 1
    assert len(grounded.activities) == 1
    assert dropped == []


def test_drops_paraphrased_evidence() -> None:
    extraction = Extraction(places=[place("Sintra", "Sintra is a short train ride from Lisbon.")])
    grounded, dropped = ground(extraction, TEXT)
    assert grounded.places == []
    assert dropped == ["place:Sintra"]


def test_drops_place_whose_name_is_not_in_its_evidence() -> None:
    extraction = Extraction(places=[place("Cascais", "Day 3 is a day trip to Sintra")])
    grounded, dropped = ground(extraction, TEXT)
    assert grounded.places == []
    assert dropped == ["place:Cascais"]


def test_drops_activity_with_invented_evidence() -> None:
    extraction = Extraction(
        activities=[Activity(name="surfing", evidence="Surf lessons run every morning.")]
    )
    grounded, dropped = ground(extraction, TEXT)
    assert grounded.activities == []
    assert dropped == ["activity:surfing"]
