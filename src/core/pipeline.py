from typing import List, Optional
import os
from pathlib import Path
import json

import httpx
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .chunking import TextChunker
from .embeddings import EmbeddingModel
from .retrievers import HybridRetriever
from .rerankers import CrossEncoderReranker
from .retrievers.web_search import WebSearchRetriever


class AegisRAGPipeline:
    """End-to-end retrieval-augmented generation pipeline.

    This class is a thin orchestration layer delegating to chunking, embedding,
    retrieval, reranking and generation components.
    """

    def __init__(self, collection_name: str = "aegis_docs", data_dir: Optional[Path] = None):
        self.collection_name = collection_name
        self.data_dir = data_dir or Path("data/raw")

        # Initialize embedding model
        self.embed_model = EmbeddingModel()

        # Setup Qdrant vector store
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Load or build index if collection already has vectors
        if self.qdrant_client.collection_exists(collection_name=self.collection_name) and \
           (self.qdrant_client.get_collection(self.collection_name).points_count or 0) > 0:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model=self.embed_model._model)
        else:
            # Build index from local docs if any
            self.index = self._build_index_from_directory()

        self.chat_engine = self.index.as_chat_engine(streaming=True)

        # Instantiate hybrid retriever & reranker
        self.hybrid_retriever = HybridRetriever(
            client=self.qdrant_client,
            embed_model=self.embed_model,
            collection_name=self.collection_name,
        )
        self.web_search_enabled = os.getenv("ENABLE_WEB_SEARCH", "0") == "1"
        if self.web_search_enabled:
            self.web_retriever = WebSearchRetriever()
        else:
            self.web_retriever = None
        self.reranker = CrossEncoderReranker()

        # Configuration for generation backend (Ollama by default)
        self._ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")

    def _build_index_from_directory(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found. Ingestion required.")
        docs = SimpleDirectoryReader(str(self.data_dir)).load_data()
        if len(docs) == 0:
            raise ValueError(f"No documents found in {self.data_dir}")
        chunker = TextChunker()
        nodes = chunker.split(docs)

        return VectorStoreIndex(nodes, storage_context=self.storage_context, embed_model=self.embed_model._model)

    async def _generate_stream_async(self, prompt: str):
        """Call Ollama generation endpoint asynchronously and stream response."""
        url = f"{self._ollama_url}/api/generate"
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": True,  # Enable streaming
        }
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, timeout=60) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        # Ollama streams JSON objects, decode and extract token
                        try:
                            data = json.loads(chunk)
                            if data.get("done") is False:
                                token = data.get("response", "")
                                yield token
                        except json.JSONDecodeError:
                            # Handle potential decoding errors for incomplete chunks
                            continue

    async def _generate_async(self, prompt: str) -> str:
        """Call Ollama generation endpoint asynchronously."""
        url = f"{self._ollama_url}/api/generate"
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    async def query_stream(self, question: str, top_k: int = 5):
        """Execute full RAG flow and stream the generated answer token by token."""
        # 1. Retrieve candidates using hybrid search
        candidates = self.hybrid_retriever.retrieve(question, top_k=top_k * 3)
        # Optionally augment with web search results
        if self.web_retriever is not None:
            candidates.extend(self.web_retriever.retrieve(question, top_k=top_k))

        # 2. Rerank with cross-encoder
        top_docs = self.reranker.rerank(question, candidates, top_k=top_k)

        # 3. Build prompt with context
        context = "\n\n".join([d["text"] for d in top_docs])
        prompt = (
            "You are Aegis, an expert assistant. Answer the question using ONLY the provided context. "
            "If the context does not contain enough information, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        # 4. Generate answer via LLM stream
        async for token in self._generate_stream_async(prompt):
            yield token

    async def query(self, question: str, top_k: int = 5) -> str:
        """Execute full RAG flow and return generated answer using hybrid retrieval + rerank."""
        # 1. Retrieve candidates using hybrid search (retrieve more for reranking)
        candidates = self.hybrid_retriever.retrieve(question, top_k=top_k * 3)
        if self.web_retriever is not None:
            candidates.extend(self.web_retriever.retrieve(question, top_k=top_k))

        # 2. Rerank with cross-encoder
        top_docs = self.reranker.rerank(question, candidates, top_k=top_k)

        # 3. Build prompt with context
        context = "\n\n".join([d["text"] for d in top_docs])
        prompt = (
            "You are Aegis, an expert assistant. Answer the question using ONLY the provided context. "
            "If the context does not contain enough information, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        # 4. Generate answer via LLM
        answer = await self._generate_async(prompt)
        return answer.strip() 