#!/usr/bin/env python3
"""
Advanced embedding service with caching, error handling, and optimization.

This module provides a production-ready interface to Jina AI's embedding models
with features like automatic retries, caching, batch processing, and monitoring.
"""

import asyncio
import hashlib
import logging
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Union

import httpx
from llama_index.embeddings.jinaai import JinaEmbedding

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class EmbeddingCache:
    """Simple in-memory cache for embeddings with TTL."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _get_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache if available and not expired."""
        key = self._get_key(text, model_name)
        entry = self.cache.get(key)
        
        if entry is None:
            return None
            
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            del self.cache[key]
            return None
            
        return entry['embedding']
    
    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache with timestamp."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        key = self._get_key(text, model_name)
        self.cache[key] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


class EmbeddingModel:
    """
    Advanced embedding model wrapper with enterprise features.
    
    Features:
    - Automatic retries with exponential backoff
    - In-memory caching with TTL
    - Batch processing optimization
    - Comprehensive error handling
    - Performance monitoring
    - Async and sync interfaces
    """
    
    def __init__(
        self, 
        model_name: str = "jina-embeddings-v3",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        batch_size: int = 100
    ):
        """
        Initialize embedding model with configuration.
        
        Args:
            model_name: Jina embedding model name
            api_key: Jina AI API key (defaults to JINA_API_KEY env var)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            cache_enabled: Whether to enable embedding caching
            cache_size: Maximum number of cached embeddings
            cache_ttl: Cache time-to-live in seconds
            batch_size: Maximum batch size for processing
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size
        
        # Get API key
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise EmbeddingError("JINA_API_KEY environment variable not set")
        
        # Initialize LlamaIndex embedding model
        try:
            self._model = JinaEmbedding(model_name=model_name, api_key=self.api_key)
            logger.info(f"✓ Initialized Jina embedding model: {model_name}")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embedding model: {e}")
        
        # Initialize cache
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = EmbeddingCache(max_size=cache_size, ttl_seconds=cache_ttl)
            logger.info(f"✓ Embedding cache enabled (size: {cache_size}, TTL: {cache_ttl}s)")
        else:
            self.cache = None
            logger.info("Cache disabled for embeddings")
        
        # Performance tracking
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'total_time': 0.0
        }
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        try:
            return self._model.embedding_dim
        except Exception:
            # Default for jina-embeddings-v3
            return 1024
    
    def _update_stats(self, hit: bool = False, error: bool = False, duration: float = 0.0) -> None:
        """Update performance statistics."""
        self.stats['requests'] += 1
        if hit:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        if error:
            self.stats['errors'] += 1
        self.stats['total_time'] += duration
    
    def _check_cache(self, text: str) -> Optional[List[float]]:
        """Check cache for existing embedding."""
        if not self.cache_enabled or not self.cache:
            return None
        return self.cache.get(text, self.model_name)
    
    def _store_cache(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        if self.cache_enabled and self.cache:
            self.cache.set(text, self.model_name, embedding)
    
    async def embed_async_single(self, text: str) -> List[float]:
        """
        Get embedding for a single text asynchronously.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding fails after retries
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cached = self._check_cache(text)
            if cached is not None:
                self._update_stats(hit=True, duration=time.time() - start_time)
                return cached
            
            # Get embedding with retries
            for attempt in range(self.max_retries):
                try:
                    embedding = await self._model.aget_text_embedding(text)
                    
                    # Store in cache
                    self._store_cache(text, embedding)
                    
                    duration = time.time() - start_time
                    self._update_stats(hit=False, duration=duration)
                    
                    logger.debug(f"Generated embedding for text (length: {len(text)}, time: {duration:.2f}s)")
                    return embedding
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise EmbeddingError(f"Failed to get embedding after {self.max_retries} attempts: {e}")
                    
                    wait_time = 2 ** attempt
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
        
        except Exception as e:
            self._update_stats(error=True, duration=time.time() - start_time)
            raise EmbeddingError(f"Embedding failed: {e}")
    
    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts asynchronously with batching.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if len(texts) == 1:
            return [await self.embed_async_single(texts[0])]
        
        # Process in batches to avoid overwhelming the API
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Check cache for each text in batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                cached = self._check_cache(text)
                if cached is not None:
                    batch_embeddings.append(cached)
                    self._update_stats(hit=True)
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Get embeddings for uncached texts
            if uncached_texts:
                try:
                    start_time = time.time()
                    new_embeddings = await self._model.aget_text_embedding_batch(uncached_texts)
                    duration = time.time() - start_time
                    
                    # Store in cache and fill placeholders
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        self._store_cache(text, embedding)
                        batch_idx = uncached_indices[idx]
                        batch_embeddings[batch_idx] = embedding
                        self._update_stats(hit=False, duration=duration / len(uncached_texts))
                    
                    logger.debug(f"Generated {len(new_embeddings)} embeddings in batch (time: {duration:.2f}s)")
                    
                except Exception as e:
                    # Fallback to individual requests
                    logger.warning(f"Batch embedding failed, falling back to individual requests: {e}")
                    for idx, text in zip(uncached_indices, uncached_texts):
                        try:
                            embedding = await self.embed_async_single(text)
                            batch_embeddings[idx] = embedding
                        except Exception as individual_error:
                            logger.error(f"Failed to embed individual text: {individual_error}")
                            # Use zero vector as fallback
                            batch_embeddings[idx] = [0.0] * self.dimension
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Synchronous embedding interface.
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(texts, str):
            # Single text
            try:
                return self._model.get_text_embedding(texts)
            except Exception as e:
                raise EmbeddingError(f"Failed to get embedding: {e}")
        else:
            # Multiple texts
            try:
                return self._model.get_text_embedding_batch(texts)
            except Exception as e:
                raise EmbeddingError(f"Failed to get batch embeddings: {e}")
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        total_requests = self.stats['requests']
        if total_requests == 0:
            return self.stats.copy()
        
        stats = self.stats.copy()
        stats.update({
            'cache_hit_rate': self.stats['cache_hits'] / total_requests,
            'error_rate': self.stats['errors'] / total_requests,
            'avg_time_per_request': self.stats['total_time'] / total_requests
        })
        
        if self.cache_enabled and self.cache:
            stats['cache_stats'] = self.cache.stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(model={self.model_name}, cache={self.cache_enabled}, dim={self.dimension})" 