import pytest

from article_gate.url_gate import check_url


@pytest.mark.parametrize(
    ("url", "rule"),
    [
        ("https://blog.example.com/", "homepage"),
        ("https://blog.example.com/tag/portugal/", "taxonomy"),
        ("https://blog.example.com/category/europe/spain", "taxonomy"),
        ("https://blog.example.com/author/ada/", "author"),
        ("https://blog.example.com/about/", "site_page"),
        ("https://blog.example.com/privacy-policy", "site_page"),
        ("https://blog.example.com/europe/page/3/", "pagination"),
        ("https://blog.example.com/europe/?paged=2", "pagination"),
        ("https://blog.example.com/wp-content/uploads/map.PDF", "file_type"),
        ("https://blog.example.com/images/lisbon.jpg", "file_type"),
        ("https://blog.example.com/feed/", "feed"),
        ("https://blog.example.com/2024/05/", "date_index"),
    ],
)
def test_rejects_non_articles(url: str, rule: str) -> None:
    decision = check_url(url)
    assert not decision.passed
    assert decision.rule == rule


@pytest.mark.parametrize(
    "url",
    [
        "https://blog.example.com/portugal/lisbon-three-days-itinerary/",
        "https://blog.example.com/2024/05/oaxaca-city-guide/",
        "https://blog.example.com/about-the-amalfi-coast-in-winter/",
        "https://blog.example.com/tagine-cooking-class-marrakech/",
        "https://blog.example.com/europe/?utm_source=newsletter",
    ],
)
def test_passes_article_urls(url: str) -> None:
    assert check_url(url).passed
