from __future__ import annotations

"""Web search retriever module.

Fetches web search results (DuckDuckGo) at query time and converts them into
RAG-compatible document dicts. No external state is preserved; results are
fetched ad-hoc and fed directly to the reranking stage.

Dependencies: `duckduckgo-search` which is lightweight and requires no API key.
"""

from typing import List, Dict, Any

from duckduckgo_search import DDGS


class WebSearchRetriever:  # pylint: disable=too-few-public-methods
    """Retrieve search snippets from DuckDuckGo.

    This class performs a web search and returns the page *title + snippet* as
    the document text. Each document dict is structured identically to the
    `HybridRetriever` output so that it can be fused and reused downstream.
    """

    def __init__(self, source: str | None = "duckduckgo", max_results: int = 10):
        self.source = source or "duckduckgo"
        self.max_results = max_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:  # noqa: D401
        """Return web search snippets for *query* limited to *top_k* results."""
        limit = top_k or self.max_results
        docs: List[Dict[str, Any]] = []
        # Using DDGS in a context manager keeps underlying session tidy
        with DDGS() as ddgs:
            # "text" search yields title, href, body
            for rank, result in enumerate(ddgs.text(query, max_results=limit)):
                snippet = (result.get("title", "") + "\n" + result.get("body", "")).strip()
                if not snippet:
                    continue
                score = 1.0 / (rank + 1)  # simple rank-based prior
                docs.append({
                    "id": result.get("href", f"web:{rank}"),
                    "text": snippet,
                    "score": score,
                    "source": "web",
                    "url": result.get("href", ""),
                })
        return docs[:limit] 