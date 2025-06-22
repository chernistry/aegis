#!/usr/bin/env python3
"""
Main FastAPI application for the Aegis RAG system.
"""

import os
import requests
import ollama
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

# Get your Jina AI API key for free: https://jina.ai/?sui=apikey
from dotenv import load_dotenv
load_dotenv()

# --- Constants & Global Settings ---
JINA_API_KEY = os.getenv("JINA_API_KEY")
if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY environment variable is not set. Get your key from https://jina.ai/?sui=apikey")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
COLLECTION_NAME = "aegis_docs_v2"
EMBEDDING_MODEL = "jina-embeddings-v3"
RERANKER_MODEL = "jina-reranker-v2-base-multilingual"
TOP_K_RETRIEVAL = 10  # Retrieve more documents initially for the reranker
TOP_K_RERANK = 3    # The final number of documents to use for the answer

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Aegis RAG API",
    description="A RAG system using Jina AI and Ollama",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Clients ---
qdrant_client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_URL)
JINA_API_HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Chat history")

class Source(BaseModel):
    text: str
    source: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

# --- Core RAG Functions ---
def get_query_embedding(query: str) -> List[float]:
    """Get embedding for a query using Jina's Embeddings API."""
    try:
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=JINA_API_HEADERS,
            json={"input": [query], "model": EMBEDDING_MODEL}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Jina Embeddings API request failed: {e}")

def rerank_documents(query: str, documents: List[Dict]) -> List[Dict]:
    """Rerank documents using Jina's Reranker API."""
    if not documents:
        return []
    
    texts_to_rerank = [doc['payload']['text'] for doc in documents]

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers=JINA_API_HEADERS,
            json={
                "model": RERANKER_MODEL,
                "query": query,
                "documents": texts_to_rerank,
                "top_n": TOP_K_RERANK
            }
        )
        response.raise_for_status()
        reranked_results = response.json()["results"]
        
        # Map reranked results back to original documents
        final_docs = []
        for res in reranked_results:
            original_doc = documents[res['index']]
            final_docs.append({
                'text': original_doc['payload']['text'],
                'source': original_doc['payload']['source'],
                'score': res['relevance_score']
            })
        return final_docs
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Jina Reranker API request failed: {e}")

def generate_llm_answer(query: str, context: List[Dict]) -> str:
    """Generate an answer using Ollama LLM."""
    context_str = "\n\n---\n\n".join([doc['text'] for doc in context])
    prompt = f"""
    Based on the following context, please provide a comprehensive answer to the user's question.
    If the context does not contain the answer, state that you could not find the information.

    Context:
    {context_str}

    User's Question: {query}
    """
    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")

# --- API Endpoints ---
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_handler(req: ChatRequest):
    """Handles the main chat logic: retrieval, reranking, and generation."""
    # 1. Get query embedding
    query_embedding = get_query_embedding(req.question)

    # 2. Retrieve documents from Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=TOP_K_RETRIEVAL,
        with_payload=True,
    )
    documents = [result.model_dump() for result in search_results]
    
    # 3. Rerank documents for relevance
    reranked_docs = rerank_documents(req.question, documents)

    # 4. Generate answer using LLM
    answer = generate_llm_answer(req.question, reranked_docs)

    return ChatResponse(answer=answer, sources=reranked_docs)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8910, reload=True) 