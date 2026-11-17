import re
from dataclasses import dataclass
from urllib.parse import urlparse

# Ordered so the first matching rule is the one reported; specific rules before generic ones.
RULES: list[tuple[str, re.Pattern[str]]] = [
    ("file_type", re.compile(r"\.(pdf|jpe?g|png|gif|webp|svg|mp4|mp3|zip|xml|json|css|js)$")),
    ("feed", re.compile(r"/(feed|rss|atom)/?$")),
    ("taxonomy", re.compile(r"/(tag|tags|category|categories|topics?|archive|archives)(/|$)")),
    ("author", re.compile(r"/(author|authors|writer|contributors?)(/|$)")),
    ("pagination", re.compile(r"/page/\d+/?$")),
    (
        "site_page",
        re.compile(
            r"^/(about(-us)?|contact(-us)?|privacy(-policy)?|terms|cookies?|disclaimer|"
            r"advertise|work-with-me|shop|cart|checkout|login|signup|search)(/|$)"
        ),
    ),
    ("date_index", re.compile(r"^/\d{4}(/\d{2}){0,2}/?$")),
]

PAGINATION_QUERY = re.compile(r"(^|&)(page|paged|p)=\d+")


@dataclass(frozen=True)
class GateDecision:
    url: str
    passed: bool
    rule: str | None = None


def check_url(url: str) -> GateDecision:
    parsed = urlparse(url)
    path = parsed.path.lower() or "/"

    if path == "/":
        return GateDecision(url, False, "homepage")
    if PAGINATION_QUERY.search(parsed.query):
        return GateDecision(url, False, "pagination")
    for name, pattern in RULES:
        if pattern.search(path):
            return GateDecision(url, False, name)
    return GateDecision(url, True)


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))
