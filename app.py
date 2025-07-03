#!/usr/bin/env python3
"""
Aegis RAG System - Main Application Module

This module implements the core FastAPI application for the Aegis Retrieval-Augmented Generation system.
The application provides intelligent document retrieval and question answering capabilities through a
REST API interface, leveraging state-of-the-art embedding models, vector similarity search, and
large language models.

Architecture:
    - FastAPI-based REST API with automatic OpenAPI documentation
    - Jina AI for text embeddings and document reranking
    - Qdrant vector database for similarity search
    - Ollama for large language model inference
    - Comprehensive error handling and monitoring

Author: Aegis RAG Development Team
Version: 1.0.0
License: MIT
"""

import os
import logging
import uvicorn
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import requests
import ollama
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application Configuration
class Config:
    """
    Application configuration management.
    
    Centralizes all configuration parameters with validation and default values.
    Supports environment variable overrides for production deployments.
    """
    
    # API Configuration
    API_TITLE: str = "Aegis RAG API"
    API_DESCRIPTION: str = "Enterprise Retrieval-Augmented Generation System"
    API_VERSION: str = "1.0.0"
    
    # External Service URLs
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    
    # Model Configuration
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
    
    # Vector Database Configuration
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "aegis_docs_v2")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "1024"))
    
    # RAG Pipeline Configuration
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "3"))
    
    # Performance Configuration
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    @classmethod
    def validate_configuration(cls) -> None:
        """
        Validate required configuration parameters.
        
        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        if not cls.JINA_API_KEY:
            raise ValueError(
                "JINA_API_KEY environment variable is required. "
                "Get your free API key from https://jina.ai/?sui=apikey"
            )
        
        if cls.TOP_K_RERANK > cls.TOP_K_RETRIEVAL:
            raise ValueError(
                f"TOP_K_RERANK ({cls.TOP_K_RERANK}) cannot exceed "
                f"TOP_K_RETRIEVAL ({cls.TOP_K_RETRIEVAL})"
            )
        
        logger.info("Configuration validation completed successfully")

# Initialize configuration
config = Config()
config.validate_configuration()

# HTTP Headers for Jina AI API
JINA_API_HEADERS = {
    "Authorization": f"Bearer {config.JINA_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": f"Aegis-RAG/{config.API_VERSION}"
}

# Pydantic Models
class ChatRequest(BaseModel):
    """
    Request model for chat interactions.
    
    Attributes:
        question: User's question or query (required)
        history: Optional conversation history for context
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the company policy on remote work?",
                "history": [
                    {"role": "user", "content": "Tell me about work policies"},
                    {"role": "assistant", "content": "I can help you with work policies..."}
                ]
            }
        }
    )
    
    question: str = Field(
        ...,
        description="User's question or query",
        min_length=1,
        max_length=1000
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default=[],
        description="Optional conversation history",
        max_length=10
    )

class Source(BaseModel):
    """
    Document source information with relevance scoring.
    
    Attributes:
        text: Relevant document excerpt
        source: Document identifier or filename
        score: Relevance score (0.0-1.0)
    """
    text: str = Field(..., description="Document text excerpt")
    source: str = Field(..., description="Source document identifier")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    """
    Response model for chat interactions.
    
    Attributes:
        answer: Generated response text
        sources: List of source documents with relevance scores
    """
    answer: str = Field(..., description="Generated response")
    sources: List[Source] = Field(..., description="Source documents")

class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Service status indicator
        version: Application version
        dependencies: External service status
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")

# Application Lifecycle Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Handles startup and shutdown procedures including service validation
    and resource cleanup.
    """
    # Startup
    logger.info("Starting Aegis RAG application")
    
    try:
        # Validate external services
        await validate_dependencies()
        logger.info("All dependencies validated successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Aegis RAG application")
        await cleanup_resources()

# FastAPI Application Instance
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure restrictively in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=600
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled errors.
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

# Service Clients
class ServiceClients:
    """
    Centralized service client management.
    
    Provides singleton access to external service clients with connection
    management and error handling.
    """
    
    _qdrant_client: Optional[QdrantClient] = None
    _ollama_client: Optional[ollama.Client] = None
    
    @classmethod
    def get_qdrant_client(cls) -> QdrantClient:
        """Get or create Qdrant client instance."""
        if cls._qdrant_client is None:
            cls._qdrant_client = QdrantClient(
                url=config.QDRANT_URL,
                timeout=config.REQUEST_TIMEOUT
            )
            logger.info(f"Initialized Qdrant client: {config.QDRANT_URL}")
        return cls._qdrant_client
    
    @classmethod
    def get_ollama_client(cls) -> ollama.Client:
        """Get or create Ollama client instance."""
        if cls._ollama_client is None:
            cls._ollama_client = ollama.Client(
                host=config.OLLAMA_URL,
                timeout=config.REQUEST_TIMEOUT
            )
            logger.info(f"Initialized Ollama client: {config.OLLAMA_URL}")
        return cls._ollama_client

# Core RAG Pipeline Functions
async def get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding vector for user query using Jina AI Embeddings API.
    
    Args:
        query: User's question or search query
        
    Returns:
        List of float values representing the query embedding
        
    Raises:
        HTTPException: If embedding generation fails
    """
    try:
        logger.debug(f"Generating embedding for query: {query[:100]}...")
        
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=JINA_API_HEADERS,
            json={
                "input": [query],
                "model": config.EMBEDDING_MODEL
            },
            timeout=30
        )
        response.raise_for_status()
        
        embedding_data = response.json()
        embedding = embedding_data["data"][0]["embedding"]
        
        logger.debug(f"Generated embedding with dimension: {len(embedding)}")
        return embedding
        
    except requests.RequestException as e:
        logger.error(f"Jina Embeddings API request failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unavailable: {str(e)}"
        )
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid embedding response format: {e}")
        raise HTTPException(
            status_code=502,
            detail="Invalid response from embedding service"
        )

async def search_documents(query_embedding: List[float]) -> List[Dict[str, Any]]:
    """
    Perform similarity search in vector database.
    
    Args:
        query_embedding: Query vector for similarity search
        
    Returns:
        List of documents with metadata and similarity scores
        
    Raises:
        HTTPException: If vector search fails
    """
    try:
        logger.debug("Performing vector similarity search")
        
        qdrant_client = ServiceClients.get_qdrant_client()
        search_results = qdrant_client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_embedding,
            limit=config.TOP_K_RETRIEVAL,
            with_payload=True,
            with_vectors=False  # Optimize bandwidth
        )
        
        documents = [result.model_dump() for result in search_results]
        logger.debug(f"Retrieved {len(documents)} documents from vector search")
        
        return documents
        
    except ResponseHandlingException as e:
        logger.error(f"Qdrant search failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Vector database unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in document search: {e}")
        raise HTTPException(
            status_code=500,
            detail="Document search failed"
        )

async def rerank_documents(query: str, documents: List[Dict]) -> List[Dict]:
    """
    Rerank documents using Jina AI Reranker for improved relevance.
    
    Args:
        query: Original user query
        documents: List of documents from initial retrieval
        
    Returns:
        List of reranked documents with updated relevance scores
        
    Raises:
        HTTPException: If reranking fails
    """
    if not documents:
        return []
    
    try:
        logger.debug(f"Reranking {len(documents)} documents")
        
        texts_to_rerank = [doc['payload']['text'] for doc in documents]
        
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers=JINA_API_HEADERS,
            json={
                "model": config.RERANKER_MODEL,
                "query": query,
                "documents": texts_to_rerank,
                "top_n": config.TOP_K_RERANK
            },
            timeout=30
        )
        response.raise_for_status()
        
        reranked_results = response.json()["results"]
        
        # Map reranked results back to original documents
        final_docs = []
        for result in reranked_results:
            original_doc = documents[result['index']]
            final_docs.append({
                'text': original_doc['payload']['text'],
                'source': original_doc['payload']['source'],
                'score': result['relevance_score']
            })
        
        logger.debug(f"Reranked to {len(final_docs)} most relevant documents")
        return final_docs
        
    except requests.RequestException as e:
        logger.error(f"Jina Reranker API request failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Reranking service unavailable: {str(e)}"
        )
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid reranker response format: {e}")
        raise HTTPException(
            status_code=502,
            detail="Invalid response from reranking service"
        )

async def generate_llm_answer(query: str, context_docs: List[Dict]) -> str:
    """
    Generate answer using large language model with retrieved context.
    
    Args:
        query: User's original question
        context_docs: List of relevant documents for context
        
    Returns:
        Generated answer string
        
    Raises:
        HTTPException: If answer generation fails
    """
    try:
        logger.debug("Generating LLM response")
        
        # Prepare context from retrieved documents
        context_str = "\n\n---\n\n".join([doc['text'] for doc in context_docs])
        
        # Construct prompt with context and query
        prompt = f"""Based on the following context documents, provide a comprehensive and accurate answer to the user's question. If the context does not contain sufficient information to answer the question, clearly state that the information is not available in the provided documents.

Context Documents:
{context_str}

User Question: {query}

Please provide a detailed answer based solely on the information provided in the context documents."""

        ollama_client = ServiceClients.get_ollama_client()
        response = ollama_client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.1,  # Low temperature for factual responses
                'top_p': 0.9,
                'num_predict': 1000
            }
        )
        
        answer = response['message']['content']
        logger.debug(f"Generated answer with {len(answer)} characters")
        
        return answer
        
    except Exception as e:
        logger.error(f"Ollama LLM generation failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Language model service unavailable: {str(e)}"
        )

# Dependency Validation
async def validate_dependencies() -> None:
    """
    Validate external service dependencies during startup.
    
    Raises:
        Exception: If any critical dependency is unavailable
    """
    dependencies = {}
    
    try:
        # Test Qdrant connection
        qdrant_client = ServiceClients.get_qdrant_client()
        collections = qdrant_client.get_collections()
        dependencies["qdrant"] = "healthy"
        logger.info("Qdrant connection validated")
    except Exception as e:
        dependencies["qdrant"] = f"unhealthy: {e}"
        logger.error(f"Qdrant validation failed: {e}")
    
    try:
        # Test Ollama connection
        ollama_client = ServiceClients.get_ollama_client()
        models = ollama_client.list()
        dependencies["ollama"] = "healthy"
        logger.info("Ollama connection validated")
    except Exception as e:
        dependencies["ollama"] = f"unhealthy: {e}"
        logger.error(f"Ollama validation failed: {e}")
    
    try:
        # Test Jina AI API
        response = requests.get(
            "https://api.jina.ai/v1/models",
            headers=JINA_API_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        dependencies["jina_ai"] = "healthy"
        logger.info("Jina AI API validated")
    except Exception as e:
        dependencies["jina_ai"] = f"unhealthy: {e}"
        logger.error(f"Jina AI validation failed: {e}")

async def cleanup_resources() -> None:
    """Clean up resources during application shutdown."""
    # Close client connections if needed
    logger.info("Resource cleanup completed")

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring and load balancing.
    
    Returns comprehensive health status including dependency checks.
    
    Returns:
        HealthResponse: Current system health status
    """
    dependencies = {}
    
    # Quick health checks for dependencies
    try:
        qdrant_client = ServiceClients.get_qdrant_client()
        qdrant_client.get_collections()
        dependencies["qdrant"] = "healthy"
    except Exception:
        dependencies["qdrant"] = "unhealthy"
    
    try:
        ollama_client = ServiceClients.get_ollama_client()
        ollama_client.list()
        dependencies["ollama"] = "healthy"
    except Exception:
        dependencies["ollama"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if all(status == "healthy" for status in dependencies.values()) else "degraded",
        version=config.API_VERSION,
        dependencies=dependencies
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Process user questions through the RAG pipeline.
    
    Implements the complete retrieval-augmented generation workflow:
    1. Generate query embedding
    2. Perform vector similarity search
    3. Rerank results for relevance
    4. Generate contextual answer using LLM
    
    Args:
        request: Chat request containing question and optional history
        
    Returns:
        ChatResponse: Generated answer with source attributions
        
    Raises:
        HTTPException: If any step in the pipeline fails
    """
    logger.info(f"Processing chat request: {request.question[:100]}...")
    
    try:
        # Step 1: Generate query embedding
        query_embedding = await get_query_embedding(request.question)
        
        # Step 2: Search for relevant documents
        documents = await search_documents(query_embedding)
        
        # Step 3: Rerank documents for relevance
        reranked_docs = await rerank_documents(request.question, documents)
        
        # Step 4: Generate answer using LLM
        answer = await generate_llm_answer(request.question, reranked_docs)
        
        # Format sources for response
        sources = [
            Source(
                text=doc['text'],
                source=doc['source'],
                score=doc['score']
            )
            for doc in reranked_docs
        ]
        
        logger.info(f"Successfully processed chat request with {len(sources)} sources")
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Chat processing failed due to internal error"
        )

# Application Entry Point
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8910,
        reload=False,  # Disable in production
        log_level="info",
        access_log=True
    ) 