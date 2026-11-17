import json
import re
from collections.abc import Callable, Iterable

from article_gate.providers import Completion

Responder = Callable[[str, str], str]


class FakeProvider:
    def __init__(self, responder: Responder | Iterable[str], name: str = "fake"):
        self.name = name
        self.calls: list[tuple[str, str]] = []
        if callable(responder):
            self._respond = responder
        else:
            scripted = iter(responder)
            self._respond = lambda _system, _prompt: next(scripted)

    def complete(self, system: str, prompt: str) -> Completion:
        self.calls.append((system, prompt))
        text = self._respond(system, prompt)
        return Completion(text)


# A keyword baseline so `--models fake` exercises the whole pipeline offline.
# It is a sanity check for the harness, not a model to compare against.
TRAVEL_CUES = re.compile(
    r"\b(itinerary|day \d|where to stay|getting there|ferry|hike|trail|beach|museum|"
    r"old town|book (a|ahead)|guesthouse|viewpoint|tour)\b",
    re.IGNORECASE,
)
PLACE_AFTER_PREPOSITION = re.compile(
    r"\b(?:in|to|at|visit|from|around)\s+((?:[A-Z][\w'’-]+)(?:\s+(?:[A-Z][\w'’-]+|de|da|do))*)"
)
ACTIVITY_WORDS = re.compile(
    r"\b(hik(?:e|ing)|kayak(?:ing)?|snorkel(?:ling)?|walking tour|food tour|boat trip|"
    r"cooking class|wine tasting|surf(?:ing)?)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
NOT_PLACES = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Day|The|I|We)$"
)


def heuristic_responder(system: str, prompt: str) -> str:
    text = prompt.split("ARTICLE:", 1)[-1]
    if '"is_bookable"' in system:
        return _classify(text)
    return _extract(text)


def _classify(text: str) -> str:
    cues = len(TRAVEL_CUES.findall(text))
    is_bookable = cues >= 3
    return json.dumps(
        {
            "is_bookable": is_bookable,
            "article_type": "destination_guide" if is_bookable else "other",
            "confidence": min(1.0, cues / 6),
            "reason": f"{cues} travel keywords",
        }
    )


def _extract(text: str) -> str:
    places: dict[str, dict] = {}
    activities: dict[str, dict] = {}
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        for match in PLACE_AFTER_PREPOSITION.finditer(sentence):
            name = match.group(1)
            if not NOT_PLACES.match(name):
                places.setdefault(name, {"name": name, "type": "other", "evidence": sentence})
        for match in ACTIVITY_WORDS.finditer(sentence):
            name = match.group(1).lower()
            activities.setdefault(name, {"name": name, "evidence": sentence})
    return json.dumps({"places": list(places.values()), "activities": list(activities.values())})
