import os
from typing import List

from llama_index.embeddings.jinaai import JinaEmbedding


class EmbeddingModel:
    """Wrapper around Jina AI embeddings with simple interface."""

    def __init__(self, model_name: str = "jina-embeddings-v3-base"):
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise EnvironmentError("JINA_API_KEY env variable not set.")
        self._model = JinaEmbedding(model_name=model_name, api_key=api_key)

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        return await self._model.aget_text_embedding_batch(texts)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.get_text_embedding_batch(texts)

    @property
    def dimension(self) -> int:
        return self._model.embedding_dim 