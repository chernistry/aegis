# API Documentation

## Overview

The Aegis RAG API provides a RESTful interface for intelligent document retrieval and question answering. The API is built on FastAPI with automatic OpenAPI specification generation and comprehensive error handling.

## Base URL

```
Production: https://your-domain.com/
Development: http://localhost:8910/
```

## Authentication

The API currently operates without authentication for internal deployments. For production environments, implement API key authentication or OAuth 2.0 integration at the gateway level.

## Content Types

All API endpoints accept and return `application/json` unless otherwise specified.

## Rate Limiting

Default rate limits apply to prevent resource exhaustion:
- **Standard endpoints**: 100 requests per minute
- **Chat endpoints**: 20 requests per minute
- **Bulk operations**: 5 requests per minute

## Error Handling

### Standard HTTP Status Codes

| Code | Description | Usage |
|------|-------------|--------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid request parameters or body |
| 422 | Validation Error | Request body validation failed |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Dependent service unavailable |

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "request_id": "uuid-correlation-id"
}
```

## Data Models

### ChatRequest

Request model for chat interactions.

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

**Fields:**
- `question` (required): User query string, max 1000 characters
- `history` (optional): Conversation context, max 10 exchanges

### ChatResponse

Response model for chat interactions.

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

**Fields:**
- `answer`: Generated response text
- `sources`: Array of source documents with relevance scores

### Source

Document source with relevance information.

```json
{
  "text": "string",
  "source": "string",
  "score": "number"
}
```

**Fields:**
- `text`: Relevant document excerpt
- `source`: Document identifier or filename
- `score`: Relevance score (0.0-1.0)

## Endpoints

### Health Check

Monitor system health and service availability.

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "services": {
    "qdrant": "healthy",
    "ollama": "healthy",
    "jina_ai": "healthy"
  }
}
```

**Status Codes:**
- `200`: All services operational
- `503`: One or more services unavailable

### Chat Interface

Process user questions and return contextual answers.

```http
POST /chat
```

**Request Body:**
```json
{
  "question": "What are the security policies for remote access?",
  "history": [
    {
      "role": "user",
      "content": "Tell me about VPN requirements"
    },
    {
      "role": "assistant", 
      "content": "VPN access requires multi-factor authentication..."
    }
  ]
}
```

**Response:**
```json
{
  "answer": "Based on the security documentation, remote access requires the following protocols...",
  "sources": [
    {
      "text": "Remote access security policy requires VPN connection with MFA...",
      "source": "security-policy.pdf",
      "score": 0.89
    },
    {
      "text": "All remote connections must use approved VPN clients...",
      "source": "it-guidelines.md", 
      "score": 0.76
    }
  ]
}
```

**Status Codes:**
- `200`: Question processed successfully
- `400`: Invalid question format or length
- `422`: Validation error in request body
- `500`: Processing error

## Integration Examples

### Python Client

```python
import httpx
import asyncio

class AegisClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def ask_question(self, question: str, history: list = None):
        """Ask a question to the Aegis RAG system."""
        payload = {
            "question": question,
            "history": history or []
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self):
        """Check system health status."""
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()

# Usage example
async def main():
    client = AegisClient("http://localhost:8910")
    
    result = await client.ask_question(
        "What is the remote work policy?",
        history=[]
    )
    
    print(f"Answer: {result['answer']}")
    for source in result['sources']:
        print(f"Source: {source['source']} (Score: {source['score']})")

asyncio.run(main())
```

### JavaScript/TypeScript Client

```typescript
interface ChatRequest {
  question: string;
  history?: Array<{role: string; content: string}>;
}

interface ChatResponse {
  answer: string;
  sources: Array<{
    text: string;
    source: string;
    score: number;
  }>;
}

class AegisClient {
  constructor(private baseUrl: string) {}

  async askQuestion(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

// Usage
const client = new AegisClient('http://localhost:8910');
const result = await client.askQuestion({
  question: 'What are the data retention policies?',
  history: []
});
```

### cURL Examples

**Basic question:**
```bash
curl -X POST http://localhost:8910/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the incident response procedure?",
    "history": []
  }'
```

**Question with context:**
```bash
curl -X POST http://localhost:8910/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the escalation steps?",
    "history": [
      {
        "role": "user",
        "content": "Tell me about incident response"
      },
      {
        "role": "assistant", 
        "content": "Incident response follows a structured process..."
      }
    ]
  }'
```

## Performance Characteristics

### Response Times
- **Health check**: < 10ms
- **Simple queries**: < 2s
- **Complex queries**: < 5s
- **Document ingestion**: Variable based on size

### Throughput
- **Concurrent requests**: Up to 50 simultaneous connections
- **Questions per minute**: 200+ (depending on complexity)
- **Data processing**: 10MB documents per minute

## OpenAPI Specification

The complete OpenAPI specification is available at:
```
GET /docs (Swagger UI)
GET /redoc (ReDoc)
GET /openapi.json (Raw specification)
```

## Versioning

The API follows semantic versioning principles:
- **Major version**: Breaking changes to API structure
- **Minor version**: New features, backwards compatible
- **Patch version**: Bug fixes and improvements

Current version: `v1.0.0`

## Support and Maintenance

### Health Monitoring
Monitor the `/health` endpoint for service status. Implement alerting for:
- Response time degradation
- Service unavailability
- Error rate increases

### Logging
All requests include correlation IDs for tracing. Log levels:
- **INFO**: Normal operations
- **WARN**: Recoverable errors
- **ERROR**: Processing failures
- **DEBUG**: Detailed diagnostic information