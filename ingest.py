#!/usr/bin/env python3
"""
Ingestion script for Aegis RAG system using Jina AI API.
This script reads files from data/raw, segments them using Jina's Segmenter API,
creates embeddings using Jina's Embeddings API, and ingests them into Qdrant.
"""

import os
import sys
import requests
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Get your Jina AI API key for free: https://jina.ai/?sui=apikey
load_dotenv()

# Constants
JINA_API_KEY = os.getenv("JINA_API_KEY")
if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY environment variable is not set. Get your key from https://jina.ai/?sui=apikey")

COLLECTION_NAME = "aegis_docs_v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = "jina-embeddings-v3-base-en" # Using v3 model
VECTOR_SIZE = 1024 # jina-embeddings-v3 models have 1024 dimensions

JINA_API_HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

def get_chunks(text: str) -> List[str]:
    """Segment text into chunks using Jina's Segmenter API."""
    response = requests.post(
        "https://segment.jina.ai/",
        headers=JINA_API_HEADERS,
        json={
            "content": text,
            "return_chunks": True,
        }
    )
    response.raise_for_status()
    return response.json().get("chunks", [])

def get_embeddings(chunks: List[str]) -> List[List[float]]:
    """Get embeddings for text chunks using Jina's Embeddings API."""
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers=JINA_API_HEADERS,
        json={
            "input": chunks,
            "model": EMBEDDING_MODEL,
        }
    )
    response.raise_for_status()
    embeddings = [item['embedding'] for item in response.json().get("data", [])]
    return embeddings

def main():
    """Main function to run the ingestion process."""
    client = QdrantClient(url=QDRANT_URL)

    # Recreate collection with new vector size
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"Could not create collection: {e}")
        # Check if it already exists with correct config
        try:
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            if collection_info.vectors_config.params.size != VECTOR_SIZE:
                 print(f"Collection '{COLLECTION_NAME}' exists but with wrong vector size. Please delete it manually.")
                 sys.exit(1)
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        except Exception:
            sys.exit(1)


    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist. Please create it and add documents.")
        sys.exit(1)
        
    documents = list(data_dir.glob("**/*.md"))
    if not documents:
        print(f"No markdown documents found in {data_dir}. Nothing to ingest.")
        return

    point_id = 0
    for doc_path in documents:
        print(f"Processing {doc_path}...")
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = get_chunks(content)
        if not chunks:
            print(f"  No chunks generated for {doc_path}. Skipping.")
            continue
        print(f"  Split into {len(chunks)} chunks.")

        embeddings = get_embeddings(chunks)
        print(f"  Generated {len(embeddings)} embeddings.")

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": str(doc_path.name),
                        "chunk_id": i,
                    }
                )
            )
            point_id += 1
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
        print(f"  Ingested {len(points)} points from {doc_path}.")

    print("\nIngestion complete!")

if __name__ == "__main__":
    main() 