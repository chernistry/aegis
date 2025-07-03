import argparse
from pathlib import Path
import asyncio
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.core.chunking import TextChunker
from src.core.embeddings import EmbeddingModel


async def main(data_dir: Path, collection_name: str):
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")
    docs = SimpleDirectoryReader(str(data_dir)).load_data()
    if len(docs) == 0:
        raise ValueError("No documents found to ingest")

    chunker = TextChunker()
    nodes = chunker.split(docs)

    embed_model = EmbeddingModel()

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build and persist index
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model._model)
    print(f"Ingested {len(nodes)} chunks into collection '{collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant store")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"), help="Directory with source docs")
    parser.add_argument("--collection", type=str, default="aegis_docs", help="Qdrant collection name")
    args = parser.parse_args()

    asyncio.run(main(args.data_dir, args.collection)) 