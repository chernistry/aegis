from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


class TextChunker:
    """Utility class for splitting documents into semantic chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        # SentenceSplitter supports token-based chunking; fallback to char if tokenizer missing
        self._parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, docs: List[Document]):
        """Convert raw Documents to a list of Nodes (chunks)."""
        return self._parser.get_nodes_from_documents(docs) 