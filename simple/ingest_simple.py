#!/usr/bin/env python3
"""
Simple ingestion script for Aegis RAG system.
This script reads files from data/raw and ingests them into Qdrant.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import re
import hashlib
import numpy as np
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# Load environment variables from .env file
load_dotenv()

# Constants
COLLECTION_NAME = "aegis_docs"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
VECTOR_SIZE = 256  # Simple hash-based embedding size

def read_markdown_file(file_path: Path) -> str:
    """Read content from a markdown file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size, store current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep the overlap from the end of the previous chunk
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else ""
        
        # Add paragraph to current chunk
        current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create a collection in Qdrant if it doesn't exist."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection {collection_name}")
    else:
        print(f"Collection {collection_name} already exists")

def simple_embedding(text: str, size: int = VECTOR_SIZE) -> List[float]:
    """Create a simple embedding vector from text using hash methods.
    This is not an ML-based embedding, just a simple hashing approach for demo purposes.
    """
    # Generate a hash of the text
    text_hash = hashlib.sha256(text.encode()).digest()
    
    # Use the hash to seed a random number generator
    np.random.seed(int.from_bytes(text_hash[:4], byteorder='big'))
    
    # Create a random vector
    vector = np.random.normal(0, 1, size)
    
    # Normalize to unit length for cosine similarity
    vector = vector / np.linalg.norm(vector)
    
    return vector.tolist()

def main():
    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)
    
    # Create collection
    create_qdrant_collection(client, COLLECTION_NAME, VECTOR_SIZE)
    
    # Process files
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        sys.exit(1)
    
    # Start ID counter
    point_id = 1
    
    for file_path in data_dir.glob("**/*.md"):
        print(f"Processing {file_path}...")
        
        # Read content
        content = read_markdown_file(file_path)
        
        # Split into chunks
        chunks = split_text_into_chunks(content)
        print(f"Split into {len(chunks)} chunks")
        
        # Generate embeddings using simple hashing method
        embeddings = [simple_embedding(chunk) for chunk in chunks]
        print(f"Generated {len(embeddings)} embeddings")
        
        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=point_id,  # Use integer IDs
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": str(file_path),
                        "chunk_id": i,
                    }
                )
            )
            point_id += 1  # Increment ID for next point
        
        # Upsert to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        
        print(f"Ingested {len(points)} chunks from {file_path}")
    
    print("Ingestion complete!")

if __name__ == "__main__":
    main() 