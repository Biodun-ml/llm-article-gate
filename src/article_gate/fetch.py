import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "llm-article-gate/0.1"
DEFAULT_CACHE_DIR = Path(".cache/articles")
BOILERPLATE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]


@dataclass
class Article:
    source: str
    title: str
    text: str


def load_article(source: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Article:
    if source.startswith(("http://", "https://")):
        return fetch_url(source, cache_dir)
    return read_file(Path(source))


def read_file(path: Path) -> Article:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".htm"}:
        title, text = extract_main_text(raw)
        return Article(str(path), title, text)
    first_line, _, rest = raw.partition("\n")
    title = first_line.lstrip("# ").strip()
    return Article(str(path), title, rest.strip() if rest.strip() else raw.strip())


def fetch_url(url: str, cache_dir: Path) -> Article:
    cache_file = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"
    if cache_file.exists():
        return Article(**json.loads(cache_file.read_text(encoding="utf-8")))

    response = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=20.0
    )
    response.raise_for_status()
    title, text = extract_main_text(response.text)
    article = Article(url, title, text)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(asdict(article), ensure_ascii=False), encoding="utf-8")
    return article


def extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    try:
        import trafilatura
    except ImportError:
        trafilatura = None
    if trafilatura is not None:
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text:
            return title, text

    return title, _extract_with_soup(soup)


def _extract_with_soup(soup: BeautifulSoup) -> str:
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    # Short paragraphs are mostly captions, share buttons and newsletter prompts.
    paragraphs = [p.get_text(" ", strip=True) for p in root.find_all(["p", "h2", "h3", "li"])]
    kept = [p for p in paragraphs if len(p) >= 40 or p.endswith((".", "?", "!"))]
    return re.sub(r"[ \t]+", " ", "\n\n".join(kept)).strip()
