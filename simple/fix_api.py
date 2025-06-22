#!/usr/bin/env python3
"""
Basic FastAPI application for Aegis RAG system.
"""

import os
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional
import uvicorn

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Initialize FastAPI app
app = FastAPI(title="Aegis RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
COLLECTION_NAME = "aegis_docs"
VECTOR_SIZE = 256
TOP_K = 3

# Models
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []

# Helper functions
def simple_embedding(text: str, size: int = VECTOR_SIZE) -> List[float]:
    """Create a simple embedding vector from text using hash methods."""
    # Generate a hash of the text
    text_hash = hashlib.sha256(text.encode()).digest()
    
    # Use the hash to seed a random number generator
    np.random.seed(int.from_bytes(text_hash[:4], byteorder='big'))
    
    # Create a random vector
    vector = np.random.normal(0, 1, size)
    
    # Normalize to unit length for cosine similarity
    vector = vector / np.linalg.norm(vector)
    
    return vector.tolist()

def search_documents(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Search for documents in Qdrant based on the query."""
    try:
        # Create embedding for the query
        query_vector = simple_embedding(query)
        
        # Connect to Qdrant
        client = QdrantClient("qdrant", port=6333)
        
        # Search for similar documents
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
        )
        
        # Format results
        results = []
        for i, result in enumerate(search_results):
            results.append({
                "id": result.id,
                "score": float(result.score),
                "text": result.payload.get("text", "No text available"),
                "source": result.payload.get("source", "Unknown"),
            })
        
        return results
    except Exception as e:
        print(f"Error in search_documents: {str(e)}")
        return []

# Routes
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to get answer based on context."""
    try:
        # Search for relevant documents
        search_results = search_documents(request.question)
        
        # Simple answer
        answer = f"На ваш вопрос '{request.question}' я нашел следующую информацию:\n\n"
        if search_results:
            answer += search_results[0]["text"]
        else:
            answer += "К сожалению, я не нашел релевантной информации."
        
        return ChatResponse(answer=answer, sources=search_results)
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return ChatResponse(
            answer=f"Произошла ошибка: {str(e)}",
            sources=[]
        )

async def stream_generator():
    """Generate a simple stream response."""
    tokens = ["Это", " временный", " потоковый", " ответ"]
    for token in tokens:
        yield f"data: {token}\n\n"
        # In a real implementation, you would await between tokens

@app.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest):
    """Stream response endpoint."""
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run("fix_api:app", host="0.0.0.0", port=8910, reload=True) 