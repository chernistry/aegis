from __future__ import annotations

"""Web search retriever module.

Fetches web search results (DuckDuckGo) at query time and converts them into
RAG-compatible document dicts. No external state is preserved; results are
fetched ad-hoc and fed directly to the reranking stage.

Dependencies: `duckduckgo-search` which is lightweight and requires no API key.
"""

from typing import List, Dict, Any
import os
from contextlib import suppress
import httpx

from duckduckgo_search import DDGS
from ..web.page_fetcher import AsyncWebPageFetcher

# Optional DDG client
with suppress(ImportError):
    from duckduckgo_search import DDGS  # type: ignore


class WebSearchRetriever:  # pylint: disable=too-few-public-methods
    """Retrieve search snippets from DuckDuckGo.

    This class performs a web search and returns the page *title + snippet* as
    the document text. Each document dict is structured identically to the
    `HybridRetriever` output so that it can be fused and reused downstream.
    """

    def __init__(self, source: str | None = "duckduckgo", max_results: int = 10, fetch_full_pages: bool = False):
        self.source = source or "duckduckgo"
        self.max_results = max_results
        self.fetch_full_pages = fetch_full_pages
        self._page_fetcher: AsyncWebPageFetcher | None = AsyncWebPageFetcher() if fetch_full_pages else None
        self.api_key = os.getenv("BRAVE_API_KEY")
        self._use_brave = self.api_key is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:  # noqa: D401
        """Return search snippets for *query* limited to *top_k* results.

        If a `BRAVE_API_KEY` environment variable is set and the `brave-search`
        library is installed, Brave Search API is used. Otherwise, DuckDuckGo
        is used as a fallback (no key required).
        """
        limit = top_k or self.max_results
        if self._use_brave:
            return self._retrieve_brave(query, limit)
        return self._retrieve_duckduckgo(query, limit)

    async def retrieve_async(self, query: str, top_k: int | None = None):
        """Async version that optionally fetches full page content."""
        docs = self.retrieve(query, top_k=top_k)
        if self._page_fetcher is None:
            for d in docs:
                d.setdefault("source", "web")
            return docs
        urls = [d.get("url") for d in docs if d.get("url")]
        if not urls:
            return docs
        full_pages = await self._page_fetcher.fetch_batch(urls)
        url_to_text = {item["url"]: item["text"] for item in full_pages}
        for d in docs:
            if d.get("url") in url_to_text:
                d["text"] = url_to_text[d["url"]]
        return docs

    # ------------------------------------------------------------------
    # Brave implementation
    # ------------------------------------------------------------------
    def _retrieve_brave(self, query: str, limit: int) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
            params = {"q": query, "count": limit, "result_filter": "web"}
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
            web_results = data.get("web", {}).get("results", [])
            for rank, item in enumerate(web_results[:limit]):
                snippet = (
                    (item.get("title", "") + "\n" + item.get("description", "")).strip()
                )
                if not snippet:
                    continue
                score = 1.0 / (rank + 1)
                docs.append({
                    "id": item.get("url", f"web:{rank}"),
                    "text": snippet,
                    "score": score,
                    "source": "web",
                    "url": item.get("url", ""),
                })
        except Exception:  # pylint: disable=broad-except
            return self._retrieve_duckduckgo(query, limit)
        return docs

    # ------------------------------------------------------------------
    # DuckDuckGo implementation
    # ------------------------------------------------------------------
    def _retrieve_duckduckgo(self, query: str, limit: int) -> List[Dict[str, Any]]:
        if 'DDGS' not in globals():
            return []
        docs: List[Dict[str, Any]] = []
        with DDGS() as ddgs:  # type: ignore
            for rank, result in enumerate(ddgs.text(query, max_results=limit)):
                snippet = (result.get("title", "") + "\n" + result.get("body", "")).strip()
                if not snippet:
                    continue
                score = 1.0 / (rank + 1)
                docs.append({
                    "id": result.get("href", f"web:{rank}"),
                    "text": snippet,
                    "score": score,
                    "source": "web",
                    "url": result.get("href", ""),
                })

        # After building initial docs, optionally fetch full pages -- handled in async path only
        return docs[:limit] 