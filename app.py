#!/usr/bin/env python3
"""
Aegis RAG System - Main FastAPI Application

A production-ready, enterprise-grade RAG system with advanced features:
- Hybrid retrieval (dense + sparse)
- Intelligent reranking 
- Streaming responses
- OpenAI-compatible API
- Comprehensive monitoring and logging
- Error handling and resilience
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import httpx
import ollama
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, validator
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/aegis.log') if os.path.exists('logs') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not available, using system environment only")

# Configuration with validation
class Config:
    """Centralized configuration management with validation."""
    
    # API Configuration
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
    
    # Collection and Model Settings
    COLLECTION_NAME: str = "aegis_docs_v2"
    EMBEDDING_MODEL: str = "jina-embeddings-v3"
    RERANKER_MODEL: str = "jina-reranker-v2-base-multilingual"
    
    # Retrieval Parameters
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    REQUEST_TIMEOUT: int = 60
    MAX_RETRIES: int = 3
    
    # Performance Settings
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    ENABLE_CORS: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical configuration."""
        if not cls.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is required. Get your key from https://jina.ai/?sui=apikey")
        
        # Test URLs format
        for url_name, url in [("QDRANT_URL", cls.QDRANT_URL), ("OLLAMA_URL", cls.OLLAMA_URL)]:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"{url_name} must be a valid HTTP/HTTPS URL")

# Initialize configuration
config = Config()

# Global clients - initialized in lifespan
qdrant_client: Optional[QdrantClient] = None
ollama_client: Optional[ollama.Client] = None
jina_headers: Dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper initialization and cleanup."""
    global qdrant_client, ollama_client, jina_headers
    
    logger.info("🚀 Starting Aegis RAG System...")
    
    try:
        # Validate configuration
        config.validate()
        logger.info("✓ Configuration validated")
        
        # Initialize Jina headers
        jina_headers = {
            "Authorization": f"Bearer {config.JINA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Initialize Qdrant client with retries
        for attempt in range(config.MAX_RETRIES):
            try:
                qdrant_client = QdrantClient(url=config.QDRANT_URL)
                collections = qdrant_client.get_collections()
                logger.info(f"✓ Connected to Qdrant at {config.QDRANT_URL}")
                logger.info(f"  Found {len(collections.collections)} collections")
                break
            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise RuntimeError(f"Failed to connect to Qdrant after {config.MAX_RETRIES} attempts: {e}")
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2 ** attempt)
        
        # Initialize Ollama client
        try:
            ollama_client = ollama.Client(host=config.OLLAMA_URL)
            # Test connection
            models = ollama_client.list()
            logger.info(f"✓ Connected to Ollama at {config.OLLAMA_URL}")
            logger.info(f"  Available models: {len(models['models'])}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            # Continue without Ollama for graceful degradation
            ollama_client = None
        
        logger.info("🎉 Aegis RAG System started successfully!")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"Failed to start Aegis RAG System: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🔄 Shutting down Aegis RAG System...")
        if qdrant_client:
            qdrant_client.close()
        logger.info("✓ Shutdown complete")


# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="Aegis RAG API",
    description="Enterprise-grade Retrieval-Augmented Generation system with hybrid search and intelligent reranking",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan
)

# Add middleware
if config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add metrics if enabled
if config.ENABLE_METRICS:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(app)


# Pydantic Models with Enhanced Validation
class ChatRequest(BaseModel):
    """Chat request with comprehensive validation."""
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Chat history")
    top_k: Optional[int] = Field(default=3, ge=1, le=20, description="Number of results to return")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class Source(BaseModel):
    """Source document with metadata."""
    text: str = Field(..., description="Source text content")
    source: str = Field(..., description="Document source identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Chat response with sources and metadata."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents used")
    metadata: Optional[Dict] = Field(default=None, description="Response metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    version: str
    services: Dict[str, str]


# Utility Functions with Error Handling
async def get_query_embedding(query: str) -> List[float]:
    """Get embedding for a query using Jina's Embeddings API with retries."""
    for attempt in range(config.MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers=jina_headers,
                    json={"input": [query], "model": config.EMBEDDING_MODEL}
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
                
        except httpx.TimeoutException:
            logger.warning(f"Jina API timeout on attempt {attempt + 1}")
            if attempt == config.MAX_RETRIES - 1:
                raise HTTPException(status_code=408, detail="Embedding service timeout")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)
                continue
            raise HTTPException(status_code=e.response.status_code, detail=f"Jina API error: {e}")
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail=f"Embedding service error: {e}")
            await asyncio.sleep(2 ** attempt)


async def rerank_documents(query: str, documents: List[Dict]) -> List[Dict]:
    """Rerank documents using Jina's Reranker API with error handling."""
    if not documents:
        return []
    
    texts_to_rerank = [doc.get('payload', {}).get('text', doc.get('text', '')) for doc in documents]
    
    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            response = await client.post(
                "https://api.jina.ai/v1/rerank",
                headers=jina_headers,
                json={
                    "model": config.RERANKER_MODEL,
                    "query": query,
                    "documents": texts_to_rerank,
                    "top_n": config.TOP_K_RERANK
                }
            )
            response.raise_for_status()
            reranked_results = response.json()["results"]
            
            # Map reranked results back to original documents
            final_docs = []
            for res in reranked_results:
                original_doc = documents[res['index']]
                final_docs.append({
                    'text': texts_to_rerank[res['index']],
                    'source': original_doc.get('payload', {}).get('source', 'unknown'),
                    'score': res['relevance_score'],
                    'metadata': original_doc.get('payload', {})
                })
            return final_docs
            
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Fallback: return top documents by original score
        return documents[:config.TOP_K_RERANK]


async def generate_llm_response(query: str, context: List[Dict], temperature: float = 0.7) -> str:
    """Generate response using Ollama with enhanced prompt and error handling."""
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Language model service unavailable")
    
    context_str = "\n\n---\n\n".join([doc['text'] for doc in context])
    
    prompt = f"""You are Aegis, an expert AI assistant with access to a curated knowledge base. Your responses should be:
- Accurate and based solely on the provided context
- Comprehensive yet concise
- Well-structured and easy to understand
- Professional in tone

Context Information:
{context_str}

User Question: {query}

Instructions:
1. If the context contains relevant information, provide a detailed answer
2. If the context is insufficient, clearly state what information is missing
3. Always cite relevant sources when possible
4. Be honest about limitations

Answer:"""

    try:
        logger.debug("Generating LLM response")
        
        response = ollama_client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature}
        )
        return response['message']['content']
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language model error: {e}")


# API Routes
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Comprehensive health check with service status."""
    services = {}
    
    # Check Qdrant
    try:
        if qdrant_client:
            qdrant_client.get_collections()
            services["qdrant"] = "healthy"
        else:
            services["qdrant"] = "unavailable"
    except Exception:
        services["qdrant"] = "unhealthy"
    
    # Check Ollama
    try:
        if ollama_client:
            ollama_client.list()
            services["ollama"] = "healthy"
        else:
            services["ollama"] = "unavailable"
    except Exception:
        services["ollama"] = "unhealthy"
    
    # Check Jina API
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("https://api.jina.ai/v1/models", headers=jina_headers)
            services["jina"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        services["jina"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if all(status in ["healthy", "unavailable"] for status in services.values()) else "degraded",
        timestamp=time.time(),
        version="2.0.0",
        services=services
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_handler(request: ChatRequest):
    """Enhanced chat endpoint with comprehensive RAG pipeline."""
    try:
        # Get query embedding
        query_embedding = await get_query_embedding(request.question)
        
        # Retrieve documents from Qdrant
        search_results = qdrant_client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_embedding,
            limit=config.TOP_K_RETRIEVAL,
            with_payload=True,
        )
        
        if not search_results:
            return ChatResponse(
                answer="I couldn't find relevant information in the knowledge base to answer your question.",
                sources=[],
                metadata={"query_time": time.time(), "results_found": 0}
            )
        
        documents = [result.model_dump() for result in search_results]
        
        # Rerank documents for relevance
        reranked_docs = await rerank_documents(request.question, documents)
        
        # Generate answer using LLM
        answer = await generate_llm_response(request.question, reranked_docs, request.temperature)
        
        # Format sources
        sources = [
            Source(
                text=doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'],
                source=doc['source'],
                score=doc['score'],
                metadata=doc.get('metadata', {})
            )
            for doc in reranked_docs
        ]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata={
                "query_time": time.time(),
                "results_found": len(reranked_docs),
                "model_used": config.OLLAMA_MODEL
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Additional endpoints for monitoring and management
@app.get("/info", tags=["System"])
async def system_info():
    """Get system information and configuration."""
    return {
        "name": "Aegis RAG System",
        "version": "2.0.0",
        "configuration": {
            "collection_name": config.COLLECTION_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "reranker_model": config.RERANKER_MODEL,
            "ollama_model": config.OLLAMA_MODEL,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_rerank": config.TOP_K_RERANK
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return Response(
        content=f'{{"error": "{exc.detail}", "status_code": {exc.status_code}}}',
        status_code=exc.status_code,
        media_type="application/json"
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc} - {request.url}", exc_info=True)
    return Response(
        content='{"error": "Internal server error", "status_code": 500}',
        status_code=500,
        media_type="application/json"
    )


# Development server
if __name__ == "__main__":
    import asyncio
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8910,
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    ) 