#!/usr/bin/env python3
"""
Simple search script for Aegis RAG system.
This script searches for documents in Qdrant based on a query.
"""

import sys
import hashlib
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# Constants
COLLECTION_NAME = "aegis_docs"
VECTOR_SIZE = 256  # Should match the size used in ingest_simple.py
TOP_K = 3  # Number of results to return

def simple_embedding(text: str, size: int = VECTOR_SIZE) -> list:
    """Create a simple embedding vector from text using hash methods.
    This is not an ML-based embedding, just a simple hashing approach for demo purposes.
    Must match the approach used in ingest_simple.py.
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

def search_documents(query: str, top_k: int = TOP_K):
    """Search for documents in Qdrant based on the query."""
    # Create embedding for the query
    query_vector = simple_embedding(query)
    
    # Connect to Qdrant
    client = QdrantClient("localhost", port=6333)
    
    # Search for similar documents
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    
    print(f"\nSearch results for query: '{query}'")
    print("-" * 50)
    
    for i, result in enumerate(search_results):
        print(f"\nResult {i+1} (Score: {result.score:.4f}):")
        print(f"Source: {result.payload.get('source', 'Unknown')}")
        print("-" * 30)
        print(result.payload.get("text", "No text available"))
        print("-" * 50)

def main():
    """Main function to execute the script."""
    if len(sys.argv) < 2:
        print("Please provide a search query.")
        print("Usage: python search_docs.py 'your search query'")
        sys.exit(1)
    
    # Get query from command line arguments
    query = " ".join(sys.argv[1:])
    
    # Search for documents
    search_documents(query)

if __name__ == "__main__":
    main() 