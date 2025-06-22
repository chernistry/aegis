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

from duckduckgo_search import DDGS

# Optional Brave Search client
with suppress(ImportError):
    from brave import Brave  # type: ignore


class WebSearchRetriever:  # pylint: disable=too-few-public-methods
    """Retrieve search snippets from DuckDuckGo.

    This class performs a web search and returns the page *title + snippet* as
    the document text. Each document dict is structured identically to the
    `HybridRetriever` output so that it can be fused and reused downstream.
    """

    def __init__(self, source: str | None = None, max_results: int = 10):
        self.max_results = max_results
        self.api_key = os.getenv("BRAVE_API_KEY")
        self._use_brave = self.api_key is not None and 'Brave' in globals()
        self.source = source or ("brave" if self._use_brave else "duckduckgo")
        if self._use_brave:
            self._client = Brave()  # Brave reads API key from env var

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

    # ------------------------------------------------------------------
    # Brave implementation
    # ------------------------------------------------------------------
    def _retrieve_brave(self, query: str, limit: int) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            data = self._client.search(q=query, count=limit, raw=True)
            web_results = (
                data.get("web", {}).get("results", []) if isinstance(data, dict) else []
            )
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
            # Fallback to DDG in case of any Brave failure
            return self._retrieve_duckduckgo(query, limit)
        return docs

    # ------------------------------------------------------------------
    # DuckDuckGo implementation
    # ------------------------------------------------------------------
    def _retrieve_duckduckgo(self, query: str, limit: int) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
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
        return docs[:limit] 