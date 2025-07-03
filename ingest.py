#!/usr/bin/env python3
"""
Aegis RAG System - Document Ingestion Module

This module implements the document ingestion pipeline for the Aegis RAG system.
It processes documents from various formats (PDF, Markdown, TXT), segments them using
Jina AI's segmentation service, generates embeddings using Jina AI's embedding models,
and stores them in the Qdrant vector database for efficient similarity search.

Features:
    - Multi-format document support (PDF, Markdown, TXT)
    - Intelligent text segmentation via Jina AI Segmenter
    - High-quality embeddings via Jina AI Embeddings v3
    - Batch processing with progress tracking
    - Robust error handling and retry mechanisms
    - Comprehensive logging and monitoring

Architecture:
    1. Document Discovery: Scan directory for supported file types
    2. Content Extraction: Extract text while preserving structure
    3. Text Segmentation: Split content into optimal chunks
    4. Embedding Generation: Create vector representations
    5. Metadata Enrichment: Add document metadata and indexing information
    6. Vector Storage: Batch insert into Qdrant collection

Author: Aegis RAG Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time, sleep

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from pypdf import PdfReader
from tqdm import tqdm

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingestion.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IngestionConfig:
    """
    Configuration class for document ingestion pipeline.
    
    Centralizes all configuration parameters with validation and type safety.
    Supports environment variable overrides for different deployment environments.
    """
    
    # External Service Configuration
    jina_api_key: str = os.getenv("JINA_API_KEY", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # Collection Configuration
    collection_name: str = os.getenv("COLLECTION_NAME", "aegis_docs_v2")
    vector_size: int = int(os.getenv("VECTOR_SIZE", "1024"))
    
    # Model Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
    
    # Processing Configuration
    data_directory: str = os.getenv("DOCUMENT_STORAGE_PATH", "/data/raw")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Performance Configuration
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))
    
    # Content Processing Configuration
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "8192"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "256"))
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        if not self.jina_api_key:
            raise ValueError(
                "JINA_API_KEY environment variable is required. "
                "Get your free API key from https://jina.ai/?sui=apikey"
            )
        
        if not Path(self.data_directory).exists():
            logger.warning(f"Data directory {self.data_directory} does not exist")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        
        logger.info("Configuration validation completed successfully")

@dataclass
class DocumentMetadata:
    """
    Document metadata container.
    
    Stores comprehensive metadata about processed documents for tracking
    and debugging purposes.
    """
    filename: str
    file_path: str
    file_size: int
    file_type: str
    chunk_count: int
    processing_time: float
    content_hash: str
    created_at: str

class JinaAPIClient:
    """
    Jina AI API client with retry logic and error handling.
    
    Provides robust interface to Jina AI services with automatic retry
    mechanisms and comprehensive error handling.
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.jina_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Aegis-RAG-Ingest/1.0.0"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _make_request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Dict[str, Any],
        operation_name: str
    ) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            json_data: Request JSON payload
            operation_name: Operation name for logging
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"{operation_name} failed after {self.config.max_retries} attempts: {e}"
                    )
                    raise
                
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"{operation_name} attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                sleep(wait_time)
    
    def segment_text(self, content: str) -> List[str]:
        """
        Segment text into chunks using Jina's Segmenter API.
        
        Args:
            content: Raw text content to segment
            
        Returns:
            List of text chunks
            
        Raises:
            requests.RequestException: If segmentation fails
        """
        try:
            logger.debug(f"Segmenting text of {len(content)} characters")
            
            response = self._make_request_with_retry(
                method="POST",
                url="https://segment.jina.ai/",
                json_data={
                    "content": content,
                    "return_chunks": True,
                    "max_chunk_length": self.config.max_chunk_size,
                    "overlap_length": self.config.chunk_overlap
                },
                operation_name="Text segmentation"
            )
            
            chunks = response.json().get("chunks", [])
            logger.debug(f"Generated {len(chunks)} text chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}")
            # Fallback to simple chunking
            return self._fallback_segmentation(content)
    
    def _fallback_segmentation(self, content: str) -> List[str]:
        """
        Fallback text segmentation when API is unavailable.
        
        Args:
            content: Text content to segment
            
        Returns:
            List of text chunks
        """
        logger.info("Using fallback segmentation")
        
        chunks = []
        chunk_size = self.config.max_chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks using Jina's Embeddings API.
        
        Args:
            texts: List of text chunks to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            requests.RequestException: If embedding generation fails
        """
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            response = self._make_request_with_retry(
                method="POST",
                url="https://api.jina.ai/v1/embeddings",
                json_data={
                    "input": texts,
                    "model": self.config.embedding_model
                },
                operation_name="Embedding generation"
            )
            
            embedding_data = response.json()
            embeddings = [item['embedding'] for item in embedding_data.get("data", [])]
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

class DocumentProcessor:
    """
    Document processing pipeline for content extraction and preparation.
    
    Handles multiple document formats with robust error handling and
    content validation.
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.txt', '.markdown'}
    
    def __init__(self, config: IngestionConfig):
        self.config = config
    
    def discover_documents(self) -> List[Path]:
        """
        Discover supported documents in the data directory.
        
        Returns:
            List of document file paths
        """
        data_path = Path(self.config.data_directory)
        
        if not data_path.exists():
            logger.error(f"Data directory {data_path} does not exist")
            return []
        
        documents = []
        for ext in self.SUPPORTED_EXTENSIONS:
            documents.extend(data_path.glob(f"**/*{ext}"))
        
        logger.info(f"Discovered {len(documents)} documents in {data_path}")
        return sorted(documents)
    
    def extract_content(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """
        Extract text content from document file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Tuple of (content, metadata)
            
        Raises:
            Exception: If content extraction fails
        """
        start_time = time()
        
        try:
            if file_path.suffix.lower() == '.pdf':
                content = self._extract_pdf_content(file_path)
            else:
                content = self._extract_text_content(file_path)
            
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                file_type=file_path.suffix.lower(),
                chunk_count=0,  # Will be updated later
                processing_time=time() - start_time,
                content_hash=content_hash,
                created_at=str(time())
            )
            
            logger.debug(f"Extracted {len(content)} characters from {file_path.name}")
            return content, metadata
            
        except Exception as e:
            logger.error(f"Content extraction failed for {file_path}: {e}")
            raise
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(file_path)
            content = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num + 1} from {file_path}: {e}")
                    continue
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            raise
    
    def _extract_text_content(self, file_path: Path) -> str:
        """
        Extract text content from plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content as string
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                    return content
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Could not decode {file_path} with any supported encoding")
            
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            raise

class VectorStore:
    """
    Vector database operations for document storage and management.
    
    Provides interface to Qdrant vector database with collection management,
    batch operations, and error handling.
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant_url,
            timeout=config.request_timeout
        )
        logger.info(f"Initialized Qdrant client: {config.qdrant_url}")
    
    def setup_collection(self) -> None:
        """
        Create or recreate the vector collection with appropriate configuration.
        
        Raises:
            ResponseHandlingException: If collection operations fail
        """
        try:
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(self.config.collection_name)
                existing_vector_size = collection_info.config.params.vectors.size
                
                if existing_vector_size != self.config.vector_size:
                    logger.warning(
                        f"Collection exists with vector size {existing_vector_size}, "
                        f"but configuration requires {self.config.vector_size}. Recreating..."
                    )
                    self.client.delete_collection(self.config.collection_name)
                    self._create_collection()
                else:
                    logger.info(f"Collection '{self.config.collection_name}' already exists")
                    
            except ResponseHandlingException:
                # Collection doesn't exist, create it
                self._create_collection()
                
        except Exception as e:
            logger.error(f"Collection setup failed: {e}")
            raise
    
    def _create_collection(self) -> None:
        """Create new vector collection with optimal configuration."""
        try:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=0
                ),
                hnsw_config=models.HnswConfig(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                )
            )
            logger.info(f"Created collection '{self.config.collection_name}' successfully")
            
        except Exception as e:
            logger.error(f"Collection creation failed: {e}")
            raise
    
    def store_document_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: DocumentMetadata
    ) -> int:
        """
        Store document chunks with embeddings in vector database.
        
        Args:
            chunks: List of text chunks
            embeddings: Corresponding embedding vectors
            metadata: Document metadata
            
        Returns:
            Number of points successfully stored
            
        Raises:
            ResponseHandlingException: If storage operation fails
        """
        try:
            points = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = int(hashlib.md5(
                    f"{metadata.file_path}_{i}".encode()
                ).hexdigest()[:8], 16)
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": metadata.filename,
                            "file_path": metadata.file_path,
                            "chunk_id": i,
                            "chunk_count": len(chunks),
                            "file_type": metadata.file_type,
                            "file_size": metadata.file_size,
                            "content_hash": metadata.content_hash,
                            "created_at": metadata.created_at
                        }
                    )
                )
            
            # Batch insert with progress tracking
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"Stored {len(points)} points for {metadata.filename}")
            return len(points)
            
        except Exception as e:
            logger.error(f"Point storage failed for {metadata.filename}: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics for monitoring.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "disk_data_size": collection_info.disk_data_size,
                "ram_data_size": collection_info.ram_data_size,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

class IngestionPipeline:
    """
    Main ingestion pipeline orchestrator.
    
    Coordinates the entire document ingestion process from discovery
    to vector storage with comprehensive progress tracking and error handling.
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.jina_client = JinaAPIClient(config)
        self.document_processor = DocumentProcessor(config)
        self.vector_store = VectorStore(config)
        
        # Statistics tracking
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "total_vectors": 0,
            "processing_time": 0.0
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline.
        
        Returns:
            Dictionary containing ingestion statistics and results
        """
        start_time = time()
        logger.info("Starting document ingestion pipeline")
        
        try:
            # Setup vector collection
            self.vector_store.setup_collection()
            
            # Discover documents
            documents = self.document_processor.discover_documents()
            self.stats["total_documents"] = len(documents)
            
            if not documents:
                logger.warning("No documents found for ingestion")
                return self.stats
            
            # Process documents with progress tracking
            with tqdm(total=len(documents), desc="Processing documents") as pbar:
                for doc_path in documents:
                    try:
                        self._process_single_document(doc_path)
                        self.stats["processed_documents"] += 1
                    except Exception as e:
                        logger.error(f"Failed to process {doc_path}: {e}")
                        self.stats["failed_documents"] += 1
                    finally:
                        pbar.update(1)
            
            # Final statistics
            self.stats["processing_time"] = time() - start_time
            collection_stats = self.vector_store.get_collection_stats()
            
            logger.info(f"Ingestion completed in {self.stats['processing_time']:.2f} seconds")
            logger.info(f"Processed: {self.stats['processed_documents']}/{self.stats['total_documents']} documents")
            logger.info(f"Generated: {self.stats['total_chunks']} chunks, {self.stats['total_vectors']} vectors")
            
            return {**self.stats, "collection_stats": collection_stats}
            
        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            raise
    
    def _process_single_document(self, doc_path: Path) -> None:
        """
        Process a single document through the complete pipeline.
        
        Args:
            doc_path: Path to document file
        """
        logger.debug(f"Processing document: {doc_path}")
        
        # Extract content
        content, metadata = self.document_processor.extract_content(doc_path)
        
        if not content.strip():
            logger.warning(f"No content extracted from {doc_path}")
            return
        
        # Segment content
        chunks = self.jina_client.segment_text(content)
        
        if not chunks:
            logger.warning(f"No chunks generated for {doc_path}")
            return
        
        metadata.chunk_count = len(chunks)
        self.stats["total_chunks"] += len(chunks)
        
        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]
            batch_embeddings = self.jina_client.generate_embeddings(batch_chunks)
            embeddings.extend(batch_embeddings)
        
        self.stats["total_vectors"] += len(embeddings)
        
        # Store in vector database
        stored_count = self.vector_store.store_document_chunks(chunks, embeddings, metadata)
        logger.info(f"Successfully processed {doc_path.name}: {stored_count} vectors stored")

def main() -> None:
    """
    Main entry point for the ingestion script.
    
    Sets up configuration, initializes the pipeline, and executes ingestion
    with comprehensive error handling and reporting.
    """
    try:
        # Load and validate configuration
        config = IngestionConfig()
        config.validate()
        
        logger.info("Starting Aegis RAG document ingestion")
        logger.info(f"Configuration: {config}")
        
        # Initialize and run pipeline
        pipeline = IngestionPipeline(config)
        results = pipeline.run()
        
        # Report final results
        logger.info("=== INGESTION SUMMARY ===")
        logger.info(f"Total documents: {results['total_documents']}")
        logger.info(f"Successfully processed: {results['processed_documents']}")
        logger.info(f"Failed documents: {results['failed_documents']}")
        logger.info(f"Total chunks generated: {results['total_chunks']}")
        logger.info(f"Total vectors stored: {results['total_vectors']}")
        logger.info(f"Processing time: {results['processing_time']:.2f} seconds")
        
        if results.get('collection_stats'):
            stats = results['collection_stats']
            logger.info(f"Collection vectors: {stats.get('vectors_count', 'unknown')}")
            logger.info(f"Collection size: {stats.get('disk_data_size', 'unknown')} bytes")
        
        # Exit with appropriate code
        if results['failed_documents'] > 0:
            logger.warning(f"{results['failed_documents']} documents failed processing")
            sys.exit(1)
        else:
            logger.info("All documents processed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Ingestion failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 