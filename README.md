# Aegis RAG System

## Overview

Aegis is a production-grade Retrieval-Augmented Generation (RAG) system designed for enterprise document processing and intelligent question answering. The system leverages state-of-the-art embedding models, semantic reranking, and large language models to deliver accurate, contextually relevant responses from organizational knowledge bases.

## Architecture

### Core Components

- **API Layer**: FastAPI-based REST API with OpenAPI specification
- **Vector Database**: Qdrant for high-performance similarity search
- **Embedding Service**: Jina AI embeddings (v3) for semantic understanding
- **Reranking Engine**: Jina AI reranker for result optimization
- **Language Model**: Ollama-hosted LLMs for response generation
- **Document Processing**: Multi-format ingestion pipeline

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Open WebUI    │    │   API Gateway   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
          ┌─────────────────────────────────────────────┐
          │              Aegis API                      │
          │        (FastAPI + CORS)                     │
          └─────────────────┬───────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
┌───▼────┐         ┌────────▼────────┐     ┌────────▼────────┐
│ Qdrant │         │   Jina AI APIs  │     │   Ollama LLM    │
│Vector  │         │ (Embed/Rerank)  │     │   (DeepSeek)    │
│Database│         └─────────────────┘     └─────────────────┘
└────────┘
```

## Features

### Production Capabilities
- **Scalable Architecture**: Containerized microservices with horizontal scaling support
- **High Availability**: Service redundancy and health monitoring
- **Security**: API key management and CORS configuration
- **Monitoring**: Health checks and error handling
- **Multi-format Support**: PDF, Markdown document processing

### Technical Features
- **Advanced Retrieval**: Semantic search with cosine similarity
- **Result Optimization**: Two-stage retrieval with reranking
- **Context Management**: Intelligent document chunking and segmentation
- **Flexible LLM Integration**: Pluggable language model architecture
- **Real-time Processing**: Streaming responses and concurrent request handling

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Jina AI API key ([Get free key](https://jina.ai/?sui=apikey))
- 8GB+ RAM for optimal performance

### Environment Setup

1. Clone the repository and configure environment:
```bash
cp .env.example .env
# Edit .env with your Jina AI API key
```

2. Launch the system:
```bash
docker-compose up -d
```

3. Verify deployment:
```bash
curl http://localhost:8910/health
```

### Document Ingestion

Place documents in `data/raw/` directory:
```bash
mkdir -p data/raw
cp your-documents.pdf data/raw/
```

Run ingestion:
```bash
docker-compose up ingestor
```

### API Usage

Query the system:
```bash
curl -X POST http://localhost:8910/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the company policy on remote work?",
    "history": []
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `JINA_API_KEY` | Jina AI API authentication key | - | Yes |
| `QDRANT_URL` | Qdrant database connection URL | `http://qdrant:6333` | No |
| `OLLAMA_URL` | Ollama service endpoint | `http://ollama:11434` | No |
| `OLLAMA_MODEL` | Language model identifier | `deepseek-r1:7b` | No |

### Service Configuration

#### Retrieval Parameters
- **Initial Retrieval**: 10 documents
- **Post-Rerank Results**: 3 documents
- **Vector Dimensions**: 1024 (Jina v3)
- **Distance Metric**: Cosine similarity

#### Model Configuration
- **Embedding Model**: `jina-embeddings-v3`
- **Reranking Model**: `jina-reranker-v2-base-multilingual`
- **LLM Model**: `deepseek-r1:7b`

## API Reference

### Endpoints

#### Health Check
```
GET /health
```
Returns system status and service availability.

#### Chat Interface
```
POST /chat
```

**Request Body:**
```json
{
  "question": "string",
  "history": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ]
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "text": "string",
      "source": "string",
      "score": "number"
    }
  ]
}
```

## Deployment

### Production Considerations

#### Scaling
- Configure Qdrant cluster for high availability
- Implement API gateway for load balancing
- Use container orchestration (Kubernetes) for auto-scaling

#### Security
- Implement API authentication and rate limiting
- Configure TLS termination at load balancer
- Use secrets management for API keys
- Network segmentation and firewall rules

#### Monitoring
- Implement structured logging with correlation IDs
- Configure metrics collection (Prometheus/Grafana)
- Set up alerting for service degradation
- Monitor embedding API rate limits

#### Performance Optimization
- Implement connection pooling for database clients
- Configure appropriate resource limits and requests
- Use caching for frequently accessed embeddings
- Optimize vector database indices

### Infrastructure Requirements

#### Minimum Specifications
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **Network**: 100Mbps bandwidth

#### Recommended Production
- **CPU**: 8+ cores
- **Memory**: 16GB+ RAM
- **Storage**: 200GB+ NVMe SSD
- **Network**: 1Gbps+ bandwidth

## Development

### Local Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start services locally:
```bash
docker-compose up qdrant ollama
```

3. Run API in development mode:
```bash
python app.py
```

### Testing

Execute test suite:
```bash
pytest tests/
```

### Code Quality

The codebase follows enterprise development standards:
- Type hints throughout the codebase
- Comprehensive error handling
- Structured configuration management
- Separation of concerns architecture
- Production-ready logging

## Troubleshooting

### Common Issues

#### Jina AI API Errors
- Verify API key validity and quota limits
- Check network connectivity to Jina AI services
- Review rate limiting and retry logic

#### Vector Database Issues
- Ensure Qdrant service is healthy
- Verify collection configuration and indices
- Check available disk space for vector storage

#### LLM Generation Failures
- Confirm Ollama service status and model availability
- Monitor resource utilization during inference
- Validate model compatibility and version

### Support

For technical support and enterprise inquiries, please refer to the following resources:
- Review system logs for detailed error information
- Check service health endpoints for component status
- Validate configuration parameters and dependencies

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome. Please follow the established code style and include comprehensive tests for new features.