#!/usr/bin/env python3
"""
Simple FastAPI application for Aegis RAG system.
"""

import os
import re
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
    question: str = Field(..., description="User query")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Chat history")

class SearchResult(BaseModel):
    id: int
    score: float
    text: str
    source: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]

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
        
        # Connect to Qdrant - use Docker service name when in container
        qdrant_host = "qdrant"
        qdrant_port = 6333
        
        # Debug info
        print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        
        client = QdrantClient(qdrant_host, port=qdrant_port)
        
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
        print(f"Error searching documents: {str(e)}")
        return []

def generate_answer(query: str, context: List[Dict[str, Any]]) -> str:
    """Generate an answer based on the query and context."""
    if not context:
        return f"Извините, я не нашел информации по вашему вопросу '{query}'."
        
    # For simplicity, we'll just concatenate the contexts
    context_text = "\n".join([item["text"] for item in context])
    
    # Simple answer generation logic - in a real application, you'd use an LLM here
    answer = f"Отвечая на ваш вопрос '{query}', я нашел следующую информацию:\n\n"
    
    # Extract relevant sentences from context
    query_words = re.findall(r'\w+', query.lower())
    sentences = re.split(r'(?<=[.!?])\s+', context_text)
    
    relevant_sentences = []
    for sentence in sentences:
        if any(word.lower() in sentence.lower() for word in query_words):
            relevant_sentences.append(sentence)
    
    if relevant_sentences:
        answer += " ".join(relevant_sentences[:3])
    else:
        # If no relevant sentences, just use the first few sentences
        answer += " ".join(sentences[:3] if sentences else ["Извините, не нашел релевантной информации."])
    
    return answer

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
        
        # Generate an answer
        answer = generate_answer(request.question, search_results)
        
        # Format response
        response = ChatResponse(
            answer=answer,
            sources=[SearchResult(**result) for result in search_results]
        )
        
        return response
    except Exception as e:
        # Log error
        print(f"Error in chat endpoint: {str(e)}")
        
        # Return error response
        return ChatResponse(
            answer=f"Произошла ошибка при обработке вашего запроса: {str(e)}",
            sources=[]
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8910, reload=True) 