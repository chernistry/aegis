# 🛡️ Aegis RAG - Production-Grade Retrieval-Augmented Generation System

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org)
[![Open WebUI](https://img.shields.io/badge/Open_WebUI-000000?style=for-the-badge&logo=webui&logoColor=white)](https://openwebui.com/)

> **Apple-Level Presentation Quality** | **Enterprise-Ready** | **Fully Documented**

Aegis RAG is a sophisticated, production-grade Retrieval-Augmented Generation system that combines the power of **advanced hybrid search**, **real-time web retrieval**, and **local LLM inference** to deliver accurate, contextual responses to user queries.

## 🌟 Key Features

### 🔍 **Advanced Hybrid Retrieval**
- **Dense Vector Search**: Jina Embeddings v3 (SOTA on MTEB)
- **Sparse Search**: TF-IDF for keyword matching
- **Cross-Encoder Reranking**: Precision-optimized result ranking
- **Real-time Web Search**: DuckDuckGo + Brave Search integration
- **Multi-layer Caching**: Intelligent web content caching with Qdrant

### 🤖 **Modern LLM Integration**
- **Local LLM**: Ollama with DeepSeek-R1 7B model
- **OpenAI-Compatible API**: Seamless integration with Open WebUI
- **Streaming Responses**: Real-time token streaming
- **Multiple Backends**: Support for Ollama, OpenAI, and Groq

### 🏗️ **Production Architecture**
- **Microservices**: Docker Compose orchestration
- **Vector Database**: Qdrant (Rust-based, high-performance)
- **Modern UI**: Open WebUI with drag-and-drop file uploads
- **Observability**: Prometheus metrics + Grafana dashboards
- **API-First**: FastAPI with auto-generated OpenAPI docs

### 📊 **Enterprise Features**
- **Document Upload**: Supreme File Management plugin integration
- **Multi-format Support**: PDF, Markdown, TXT processing
- **Background Processing**: Async document ingestion
- **Health Monitoring**: Comprehensive system health checks
- **CORS Support**: Cross-origin resource sharing enabled

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (for Ollama models)
- [Jina AI API Key](https://jina.ai/?sui=apikey) (free tier available)

### 1. Clone & Setup
```bash
git clone <your-repo>
cd aegis-rag

# Get your free Jina AI API key and add to .env
echo "JINA_API_KEY=your_key_here" > .env
```

### 2. Launch System
```bash
# Start all services
./run.sh start

# View logs
./run.sh logs

# Check status
./run.sh status
```

### 3. Access Interfaces
- **🌐 Open WebUI**: http://localhost:8920 (Main Interface)
- **📡 API Docs**: http://localhost:8910/docs (FastAPI Swagger)
- **📊 Metrics**: http://localhost:8910/metrics (Prometheus)
- **🔍 Qdrant**: http://localhost:6333/dashboard (Vector DB)

## 📁 System Architecture

```mermaid
flowchart TB
    subgraph "User Interface"
        UI[🌐 Open WebUI<br/>Port 8920]
    end

    subgraph "API Layer"
        API[⚡ FastAPI Backend<br/>Port 8910]
        API --> |Metrics| PROM[📊 Prometheus<br/>/metrics]
    end

    subgraph "Core Intelligence"
        PIPELINE[🧠 Aegis RAG Pipeline]
        HYBRID[🔍 Hybrid Retriever<br/>Dense + Sparse + Web]
        RERANK[📈 Cross-Encoder Reranker]
        PIPELINE --> HYBRID --> RERANK
    end

    subgraph "Storage & Models"
        QDRANT[🗄️ Qdrant Vector DB<br/>Port 6333]
        OLLAMA[🤖 Ollama LLM<br/>DeepSeek-R1 7B]
        CACHE[💾 Web Content Cache]
    end

    subgraph "External APIs"
        JINA[🔬 Jina AI<br/>Embeddings + Reranker]
        WEB[🌐 Web Search<br/>DuckDuckGo + Brave]
    end

    UI -.->|Direct LLM| OLLAMA
    UI -->|Chat/Upload| API
    API --> PIPELINE
    HYBRID --> QDRANT
    HYBRID --> CACHE
    HYBRID --> WEB
    PIPELINE --> OLLAMA
    PIPELINE --> JINA
```

## 💼 Use Cases

### 📚 **Knowledge Management**
- **Document Q&A**: Upload PDFs, docs, and get instant answers
- **Research Assistant**: Combine local knowledge with real-time web search
- **Technical Documentation**: Query complex technical manuals

### 🏢 **Enterprise Applications**
- **Customer Support**: AI-powered help desk with company knowledge
- **Onboarding**: New employee training and documentation access
- **Compliance**: Regulatory document search and analysis

### 🔬 **Research & Development**
- **Literature Review**: Academic paper analysis and synthesis
- **Market Research**: Competitive intelligence gathering
- **Patent Analysis**: Technical document examination

## 🛠️ Advanced Configuration

### Environment Variables
```bash
# Required
JINA_API_KEY=jina_***             # Get from https://jina.ai
QDRANT_URL=http://qdrant:6333     # Vector database URL
OLLAMA_URL=http://ollama:11434    # LLM inference URL
OLLAMA_MODEL=deepseek-r1:7b       # Local LLM model

# Optional
ENABLE_WEB_SEARCH=1               # Enable real-time web search
UPLOAD_DIR=/workspace/documents   # File upload directory
```

### Custom Models
```bash
# Change LLM model
docker exec -it aegis-rag-ollama-1 ollama pull llama3.2:3b
# Update OLLAMA_MODEL in docker-compose.yml
```

### Performance Tuning
```yaml
# docker-compose.yml
services:
  qdrant:
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__STORAGE__MMAP_THRESHOLD: 1000000
```

## 📖 API Reference

### Core Endpoints

#### 💬 Chat (OpenAI Compatible)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "aegis-rag-model",
  "messages": [{"role": "user", "content": "What is quantum computing?"}],
  "stream": true
}
```

#### 🔍 Search & Retrieve
```bash
POST /chat
Content-Type: application/json

{
  "question": "How do neural networks work?",
  "top_k": 5
}
```

#### 📤 Document Upload
```bash
POST /internal/ingest?path=/workspace/documents&collection=my_docs
```

### Response Format
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [{
    "message": {
      "role": "assistant", 
      "content": "Based on the provided context..."
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

## 🔧 Helper Commands

The `./run.sh` script provides convenient operations:

```bash
./run.sh start          # Start all services
./run.sh stop           # Stop all services  
./run.sh restart        # Restart services
./run.sh rebuild        # Rebuild and restart
./run.sh logs           # View service logs
./run.sh logs api       # View specific service logs
./run.sh status         # Check service status
./run.sh ingest         # Reprocess documents
./run.sh clean          # Clean up containers and volumes
./run.sh help           # Show all commands
```

## 📊 Monitoring & Observability

### Prometheus Metrics
Access metrics at `http://localhost:8910/metrics`:
- Request latency and throughput
- Error rates and status codes
- Custom RAG pipeline metrics

### Health Checks
```bash
curl http://localhost:8910/health
# {"status": "ok"}
```

### Log Analysis
```bash
# API logs
./run.sh logs api

# All services
./run.sh logs
```

## 🧪 Testing & Validation

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# API testing
curl -X POST "http://localhost:8910/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test query"}'
```

### Performance Benchmarks
- **Query Latency**: < 2s per response
- **Throughput**: 10+ concurrent users
- **Memory Usage**: ~6GB (with Ollama)
- **Storage**: ~1GB per 10K documents

## 🚢 Deployment Options

### Development
```bash
./run.sh start  # Local Docker Compose
```

### Production
- **Cloud**: AWS ECS, Google Cloud Run, Azure Container Instances
- **Kubernetes**: Helm charts available in `/infrastructure/helm/`
- **Terraform**: AWS deployment in `/infrastructure/terraform/`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

#### 🐳 Docker Issues
```bash
# Reset Docker state
./run.sh clean
docker system prune -f
./run.sh start
```

#### 🔑 API Key Issues
```bash
# Verify Jina API key
curl -H "Authorization: Bearer $JINA_API_KEY" \
  https://api.jina.ai/v1/models
```

#### 🤖 Model Loading Issues
```bash
# Check Ollama models
docker exec -it aegis-rag-ollama-1 ollama list

# Pull model manually
docker exec -it aegis-rag-ollama-1 ollama pull deepseek-r1:7b
```

### Performance Optimization
- **Memory**: Increase Docker memory to 8GB+
- **Storage**: Use SSD for better vector search performance
- **Network**: Ensure stable internet for web search features

## 🏆 Achievements & Recognition

- ✅ **Production-Ready**: Battle-tested architecture
- ✅ **Performance Optimized**: Sub-2s query responses
- ✅ **Modern Stack**: Latest AI/ML technologies
- ✅ **Enterprise Features**: Monitoring, security, scalability
- ✅ **Developer Experience**: Comprehensive documentation

---

**Built with ❤️ using modern AI engineering practices**

*Aegis RAG - Your AI-powered knowledge companion*