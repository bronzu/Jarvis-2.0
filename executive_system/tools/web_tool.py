from html import unescape
from urllib.parse import quote_plus

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:  # pragma: no cover - fallback per ambienti senza dipendenze extra
    BeautifulSoup = None


class WebBrowserTool:
    """Tool di navigazione web per scraping di pagine e ricerca testuale."""

    def __init__(self, timeout=10, headers=None):
        self.timeout = timeout
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (compatible; JarvisBot/2.0; +https://example.com/bot)"
            )
        }

    def fetch(self, url):
        if not url or not isinstance(url, str):
            raise ValueError("URL non valida")

        response = requests.get(url, timeout=self.timeout, headers=self.headers)
        response.raise_for_status()
        return response.text

    def _parse_html_text(self, html):
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return " ".join(soup.get_text(separator=" ", strip=True).split())

        import re
        cleaned = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = unescape(cleaned)
        return " ".join(re.sub(r"\s+", " ", cleaned).split())

    def extract_text(self, url):
        html = self.fetch(url)
        return self._parse_html_text(html)

    def search(self, query, limit=5):
        if not query or not isinstance(query, str):
            raise ValueError("query non valida")

        encoded = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded}"
        html = self.fetch(url)

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            results = []
            for entry in soup.select(".result")[:limit]:
                title = entry.select_one(".result__title")
                link = entry.select_one("a.result__a")
                snippet = entry.select_one(".result__snippet")
                if not title and not link:
                    continue
                results.append({
                    "title": title.get_text(" ", strip=True) if title else "",
                    "url": link.get("href") if link else "",
                    "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                })
            return results

        import re
        matches = re.findall(r"<a rel=\"nofollow\" class=\"result-link\" href=\"([^\"]+)\".*?>(.*?)</a>", html, flags=re.DOTALL)
        results = []
        for href, raw_title in matches[:limit]:
            title = re.sub(r"<[^>]+>", " ", raw_title)
            results.append({
                "title": " ".join(title.split()),
                "url": href,
                "snippet": "",
            })
        return results

    def search_and_extract(self, query, limit=3):
        results = self.search(query, limit=limit)
        for item in results:
            url = item.get("url")
            if url:
                item["content"] = self.extract_text(url)
        return results
