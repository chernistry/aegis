# Aegis RAG System

A state-of-the-art Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant vector database, Ollama LLM, and Jina AI for embeddings and reranking.

## 🚀 Features

- **Advanced RAG Pipeline**: Hybrid retrieval with dense vector search and keyword matching
- **Multiple Interfaces**: OpenAI-compatible API, streaming responses, and web UI integration
- **Production Ready**: Docker containerization, monitoring, and comprehensive error handling
- **Modular Architecture**: Clean separation of concerns with pluggable components
- **High Performance**: Async operations, streaming responses, and efficient reranking

## 📋 System Requirements

- Docker and Docker Compose
- 8GB+ RAM (for Ollama LLM)
- 10GB+ free disk space
- JINA_API_KEY (free at https://jina.ai/?sui=apikey)

## 🛠 Quick Start

### 1. Environment Setup

Create a `.env` file with your Jina AI API key:

```bash
# Get your free API key from https://jina.ai/?sui=apikey
JINA_API_KEY=your_jina_api_key_here

# Optional configurations
OLLAMA_MODEL=deepseek-r1:7b
QDRANT_URL=http://qdrant:6333
OLLAMA_URL=http://ollama:11434
ENABLE_WEB_SEARCH=0
```

### 2. Document Preparation

Place your documents (PDF or Markdown) in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Copy your .md and .pdf files to data/raw/
```

### 3. System Launch

```bash
# Make the run script executable
chmod +x .run.sh

# Start the entire system
./run.sh start

# Wait for services to initialize (1-2 minutes)
# The system will be available at:
# - Web UI: http://localhost:8920
# - API: http://localhost:8910
```

### 4. Document Ingestion

```bash
# Ingest your documents into the vector database
./run.sh ingest

# Check ingestion logs
./run.sh logs ingestor
```

## 🎯 System Architecture

### Core Components

1. **API Layer** (`src/api/main.py`): FastAPI application with OpenAI-compatible endpoints
2. **RAG Pipeline** (`src/core/pipeline.py`): Orchestrates retrieval, reranking, and generation
3. **Vector Database**: Qdrant for high-performance vector search
4. **LLM Engine**: Ollama running DeepSeek-R1 model
5. **Embeddings**: Jina AI's state-of-the-art embedding models
6. **Reranking**: Jina AI's multilingual reranker for precision

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Open WebUI    │────│   Aegis API     │────│   Qdrant DB     │
│   (Port 8920)   │    │   (Port 8910)   │    │   (Port 6333)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                         ┌─────────────────┐
                         │   Ollama LLM    │
                         │   (Port 11434)  │
                         └─────────────────┘
```

## 📚 API Reference

### Health Check
```bash
curl http://localhost:8910/health
```

### Chat (Non-streaming)
```bash
curl -X POST "http://localhost:8910/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 5}'
```

### Chat (Streaming)
```bash
curl -X POST "http://localhost:8910/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the main concepts", "top_k": 5}'
```

### OpenAI-Compatible Endpoint
```bash
curl -X POST "http://localhost:8910/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aegis-rag-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## 🔧 Management Commands

```bash
# System management
./run.sh start          # Start all services
./run.sh stop           # Stop all services
./run.sh restart        # Restart all services
./run.sh status         # Show service status

# Monitoring
./run.sh logs           # Show all logs
./run.sh logs api       # Show API logs
./run.sh logs ollama    # Show Ollama logs

# Data management
./run.sh ingest         # Process documents in data/raw/
./run.sh rebuild        # Rebuild containers
./run.sh clean          # Remove all data (destructive)
```

## 🧪 Testing

### Run Tests
```bash
# Install dependencies for testing
pip install -r requirements.txt

# Run test suite
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_api.py::test_health_endpoint -v
```

### Manual Testing
```bash
# Test API directly
python -c "
import requests
response = requests.get('http://localhost:8910/health')
print(response.json())
"

# Test chat functionality
python -c "
import requests
response = requests.post('http://localhost:8910/chat', 
    json={'question': 'Hello', 'top_k': 3})
print(response.json())
"
```

## 🏗 Development

### Project Structure
```
├── src/
│   ├── api/           # FastAPI application
│   ├── core/          # RAG pipeline components
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── pipeline.py
│   │   ├── rerankers/
│   │   ├── retrievers/
│   │   └── web/
│   └── scripts/       # Utility scripts
├── simple/            # Simplified implementations
├── tests/             # Test suite
├── data/
│   └── raw/          # Place documents here
├── docker-compose.yml
├── Dockerfile.api
├── requirements.txt
└── .run.sh           # Management script
```

### Adding New Documents
1. Place files in `data/raw/`
2. Run `./run.sh ingest`
3. Monitor with `./run.sh logs ingestor`

### Customizing the Model
Edit docker-compose.yml to change the LLM model:
```yaml
environment:
  - OLLAMA_MODEL=llama3.1:8b  # or any supported model
```

### Web Search Integration
Enable web search augmentation:
```bash
echo "ENABLE_WEB_SEARCH=1" >> .env
./run.sh restart
```

## 🔍 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check Docker status
   docker info
   
   # Check logs
   ./run.sh logs
   ```

2. **Ollama model not found**
   ```bash
   # Pull the model manually
   docker exec aegis-ollama-1 ollama pull deepseek-r1:7b
   ```

3. **Empty search results**
   ```bash
   # Verify documents are ingested
   ./run.sh logs ingestor
   
   # Check Qdrant collection
   curl http://localhost:6333/collections/aegis_docs_v2
   ```

4. **API errors**
   ```bash
   # Check API logs
   ./run.sh logs api
   
   # Verify environment variables
   docker exec aegis-api-1 env | grep JINA_API_KEY
   ```

### Performance Tuning

1. **Memory allocation**: Increase Docker memory limit for Ollama
2. **Concurrent requests**: Adjust uvicorn workers in Dockerfile.api
3. **Vector dimensions**: Modify embedding model in ingest.py
4. **Cache settings**: Configure response caching for frequent queries

## 🌟 Advanced Features

### Custom Rerankers
Implement custom reranking logic in `src/core/rerankers/`

### Multiple Embeddings
Add support for different embedding models in `src/core/embeddings.py`

### Custom Retrievers
Extend retrieval capabilities in `src/core/retrievers/`

### Monitoring
- Prometheus metrics at `/metrics`
- Health checks at `/health`
- Container logs via `./run.sh logs`

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs with `./run.sh logs`
3. Open an issue on GitHub

---

**Built with ❤️ for the AI community**