import re
from urllib.request import Request, urlopen


class InternetTool:
    def __init__(self, timeout=10, headers=None, network_policy=None):
        self.timeout = timeout
        self.headers = headers or {
            "User-Agent": "Jarvis/2.0 InternetTool"
        }
        self.network_policy = network_policy

    def _check_policy(self, url: str):
        if self.network_policy is None:
            return {"allowed": True}
        return self.network_policy.check_url(url)

    def fetch(self, url):
        if not url or not isinstance(url, str):
            raise ValueError("URL non valida")
        policy = self._check_policy(url)
        if not policy.get("allowed", False):
            raise PermissionError(f"Accesso rete non autorizzato: {policy.get('reason')}")
        req = Request(url, headers=self.headers)
        with urlopen(req, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def extract_text(self, html):
        if not isinstance(html, str):
            raise TypeError("HTML deve essere una stringa")
        cleaned = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"&[a-zA-Z]+;", " ", cleaned)
        cleaned = re.sub(r"&\d+;", " ", cleaned)
        cleaned = " ".join(re.sub(r"\s+", " ", cleaned).split())
        return cleaned

    def search(self, query):
        if not query or not isinstance(query, str):
            raise ValueError("query non valida")
        html = self.fetch(query)
        return self.extract_text(html)
