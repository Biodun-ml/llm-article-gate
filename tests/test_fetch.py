from pathlib import Path

import httpx
import pytest
from bs4 import BeautifulSoup

from article_gate.fetch import _extract_with_soup, fetch_url

HTML = """
<html><head><title>Two days in Porto</title></head><body>
<nav><p>Home | Destinations | About this blog and its many many writers.</p></nav>
<article>
  <h2>Day 1</h2>
  <p>Start at the Ribeira waterfront and cross the Dom Luis I Bridge to the port cellars.</p>
  <p>Share</p>
</article>
<footer><p>Subscribe to our newsletter for weekly travel inspiration in your inbox.</p></footer>
</body></html>
"""


def test_soup_fallback_keeps_article_body_only() -> None:
    text = _extract_with_soup(BeautifulSoup(HTML, "html.parser"))
    assert "Ribeira waterfront" in text
    assert "newsletter" not in text
    assert "Destinations" not in text
    assert "Share" not in text


def test_fetch_uses_disk_cache_on_second_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_get(url: str, **_kwargs) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, text=HTML, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    url = "https://blog.example.com/porto-two-days/"

    first = fetch_url(url, tmp_path)
    second = fetch_url(url, tmp_path)

    assert calls == [url]
    assert second == first
    assert first.title == "Two days in Porto"
