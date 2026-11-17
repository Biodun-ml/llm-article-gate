import re
import unicodedata

from article_gate.schemas import Extraction

QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(QUOTE_MAP))
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents).strip().casefold()


def ground(extraction: Extraction, article_text: str) -> tuple[Extraction, list[str]]:
    haystack = normalize(article_text)
    dropped: list[str] = []

    places = []
    for place in extraction.places:
        evidence = normalize(place.evidence)
        if evidence in haystack and normalize(place.name) in evidence:
            places.append(place)
        else:
            dropped.append(f"place:{place.name}")

    # Activity names are often rephrased ("sunset kayak tour"), so only the evidence must match.
    activities = []
    for activity in extraction.activities:
        if normalize(activity.evidence) in haystack:
            activities.append(activity)
        else:
            dropped.append(f"activity:{activity.name}")

    return Extraction(places=places, activities=activities), dropped
