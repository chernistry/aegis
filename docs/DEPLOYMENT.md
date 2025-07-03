# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Aegis RAG system in production environments. The deployment strategy emphasizes reliability, scalability, and operational excellence through infrastructure as code, automated deployment pipelines, and comprehensive monitoring.

## Deployment Architecture

### Production Environment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Load Balancer Layer                        │
│                   (AWS ALB / Azure LB)                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────────────────────┐
│                  Application Tier                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   API Pod   │  │   API Pod   │  │   API Pod   │            │
│  │     #1      │  │     #2      │  │     #3      │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────────────────────┐
│                   Services Tier                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Qdrant    │  │   Ollama    │  │  Ingestion  │            │
│  │  Cluster    │  │   Service   │  │   Service   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────────────────────┐
│                  Storage Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Vector DB   │  │ Config DB   │  │ Object      │            │
│  │ Storage     │  │ (Redis)     │  │ Storage     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Infrastructure Requirements

### Minimum Production Requirements

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| API Service | 2 cores | 4GB | 20GB | 1Gbps |
| Qdrant | 4 cores | 16GB | 200GB SSD | 1Gbps |
| Ollama | 8 cores | 32GB | 100GB | 1Gbps |
| Load Balancer | 2 cores | 4GB | 50GB | 10Gbps |
| **Total** | **16 cores** | **56GB** | **370GB** | **13Gbps** |

### Recommended Production Requirements

| Component | CPU | Memory | Storage | Network | Replicas |
|-----------|-----|--------|---------|---------|----------|
| API Service | 4 cores | 8GB | 50GB | 1Gbps | 3 |
| Qdrant Cluster | 8 cores | 32GB | 1TB NVMe | 10Gbps | 3 |
| Ollama Service | 16 cores | 64GB | 500GB | 10Gbps | 2 |
| Redis Cache | 4 cores | 16GB | 100GB | 1Gbps | 2 |
| **Total** | **64 cores** | **224GB** | **2.2TB** | **46Gbps** | **10** |

### Cloud Provider Specifications

#### AWS Deployment
- **API Service**: `t3.large` instances
- **Vector Database**: `m5.2xlarge` with GP3 storage
- **LLM Service**: `g4dn.4xlarge` (GPU acceleration)
- **Load Balancer**: Application Load Balancer (ALB)
- **Storage**: EFS for shared storage, S3 for backups

#### Azure Deployment
- **API Service**: `Standard_D2s_v3` instances
- **Vector Database**: `Standard_D8s_v3` with Premium SSD
- **LLM Service**: `Standard_NC6s_v3` (GPU acceleration)
- **Load Balancer**: Azure Load Balancer
- **Storage**: Azure Files for shared storage, Blob Storage for backups

#### Google Cloud Platform
- **API Service**: `n2-standard-2` instances
- **Vector Database**: `n2-standard-8` with SSD persistent disks
- **LLM Service**: `n1-standard-16` with T4 GPUs
- **Load Balancer**: Google Cloud Load Balancer
- **Storage**: Cloud Filestore for shared storage, Cloud Storage for backups

## Container Deployment

### Docker Production Configuration

#### Multi-Stage Dockerfile

```dockerfile
# Production Dockerfile for Aegis API
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim as production

# Security: Create non-root user
RUN useradd --create-home --shell /bin/bash aegis
USER aegis
WORKDIR /home/aegis/app

# Copy application code and dependencies
COPY --from=builder /root/.local /home/aegis/.local
COPY --chown=aegis:aegis . .

# Environment configuration
ENV PATH=/home/aegis/.local/bin:$PATH
ENV PYTHONPATH=/home/aegis/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8910/health || exit 1

# Expose application port
EXPOSE 8910

# Start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8910", "--workers", "4"]
```

#### Production Docker Compose

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.production
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - JINA_API_KEY=${JINA_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://ollama:11434
      - LOG_LEVEL=INFO
    ports:
      - "8910:8910"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8910/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    depends_on:
      qdrant:
        condition: service_healthy
      ollama:
        condition: service_healthy

  qdrant:
    image: qdrant/qdrant:v1.7.0
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
        reservations:
          cpus: '2.0'
          memory: 8G
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 32G
        reservations:
          cpus: '4.0'
          memory: 16G
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_KEEP_ALIVE=24h
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_data:
    driver: local
  ollama_models:
    driver: local

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Kubernetes Deployment

### Namespace Configuration

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: aegis-rag
  labels:
    name: aegis-rag
    environment: production
```

### ConfigMap for Application Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aegis-config
  namespace: aegis-rag
data:
  QDRANT_URL: "http://qdrant-service:6333"
  OLLAMA_URL: "http://ollama-service:11434"
  OLLAMA_MODEL: "deepseek-r1:7b"
  LOG_LEVEL: "INFO"
  COLLECTION_NAME: "aegis_docs_v2"
  TOP_K_RETRIEVAL: "10"
  TOP_K_RERANK: "3"
```

### Secret Management

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aegis-secrets
  namespace: aegis-rag
type: Opaque
data:
  JINA_API_KEY: <base64-encoded-api-key>
```

### API Service Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aegis-api
  namespace: aegis-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aegis-api
  template:
    metadata:
      labels:
        app: aegis-api
    spec:
      containers:
      - name: api
        image: aegis-rag:latest
        ports:
        - containerPort: 8910
        envFrom:
        - configMapRef:
            name: aegis-config
        - secretRef:
            name: aegis-secrets
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8910
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8910
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: aegis-api-service
  namespace: aegis-rag
spec:
  selector:
    app: aegis-api
  ports:
  - port: 8910
    targetPort: 8910
  type: ClusterIP
```

### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aegis-ingress
  namespace: aegis-rag
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.aegis.company.com
    secretName: aegis-tls
  rules:
  - host: api.aegis.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aegis-api-service
            port:
              number: 8910
```

## Database Deployment

### Qdrant Cluster Configuration

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: aegis-rag
spec:
  serviceName: qdrant-headless
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.7.0
        ports:
        - containerPort: 6333
        - containerPort: 6334
        env:
        - name: QDRANT__CLUSTER__ENABLED
          value: "true"
        - name: QDRANT__CLUSTER__P2P__PORT
          value: "6335"
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            cpu: 2000m
            memory: 8Gi
          limits:
            cpu: 4000m
            memory: 16Gi
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 500Gi
      storageClassName: fast-ssd
```

## Environment Configuration

### Production Environment Variables

```bash
# Aegis RAG Production Environment Configuration

# Core Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8910
API_WORKERS=4
API_TIMEOUT=300

# External Service URLs
QDRANT_URL=http://qdrant-cluster:6333
OLLAMA_URL=http://ollama-service:11434
REDIS_URL=redis://redis-cluster:6379

# Model Configuration
OLLAMA_MODEL=deepseek-r1:7b
EMBEDDING_MODEL=jina-embeddings-v3
RERANKER_MODEL=jina-reranker-v2-base-multilingual

# Processing Parameters
COLLECTION_NAME=aegis_docs_v2
TOP_K_RETRIEVAL=10
TOP_K_RERANK=3
VECTOR_SIZE=1024

# Security Configuration
JINA_API_KEY=${JINA_API_KEY}
CORS_ORIGINS=["https://app.aegis.company.com"]
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Performance Tuning
CONNECTION_POOL_SIZE=20
MAX_CONCURRENT_REQUESTS=50
CACHE_TTL_SECONDS=3600

# Monitoring Configuration
PROMETHEUS_METRICS_ENABLED=true
TRACING_ENABLED=true
JAEGER_ENDPOINT=http://jaeger:14268
```

## Deployment Pipeline

### CI/CD Pipeline Configuration

```yaml
# .github/workflows/deploy.yml
name: Deploy Aegis RAG

on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=app --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: |
        docker build -t aegis-rag:${{ github.sha }} .
        docker tag aegis-rag:${{ github.sha }} aegis-rag:latest
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push aegis-rag:${{ github.sha }}
        docker push aegis-rag:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/aegis-api api=aegis-rag:${{ github.sha }} -n aegis-rag
        kubectl rollout status deployment/aegis-api -n aegis-rag
```

### Blue-Green Deployment

```bash
#!/bin/bash
# Blue-Green Deployment Script

set -e

# Configuration
NAMESPACE="aegis-rag"
APP_NAME="aegis-api"
NEW_VERSION="$1"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

# Get current active deployment
CURRENT_DEPLOYMENT=$(kubectl get service ${APP_NAME}-active -n ${NAMESPACE} -o jsonpath='{.spec.selector.version}')
echo "Current active deployment: ${CURRENT_DEPLOYMENT}"

# Determine new deployment version
if [ "$CURRENT_DEPLOYMENT" = "blue" ]; then
    NEW_DEPLOYMENT="green"
else
    NEW_DEPLOYMENT="blue"
fi

echo "Deploying version ${NEW_VERSION} to ${NEW_DEPLOYMENT} environment"

# Update the inactive deployment
kubectl set image deployment/${APP_NAME}-${NEW_DEPLOYMENT} \
    api=aegis-rag:${NEW_VERSION} \
    -n ${NAMESPACE}

# Wait for rollout to complete
kubectl rollout status deployment/${APP_NAME}-${NEW_DEPLOYMENT} -n ${NAMESPACE}

# Health check
echo "Performing health checks..."
kubectl run health-check \
    --image=curlimages/curl:latest \
    --rm -i --restart=Never \
    --command -- curl -f http://${APP_NAME}-${NEW_DEPLOYMENT}:8910/health

# Switch traffic to new deployment
echo "Switching traffic to ${NEW_DEPLOYMENT} deployment"
kubectl patch service ${APP_NAME}-active \
    -n ${NAMESPACE} \
    -p '{"spec":{"selector":{"version":"'${NEW_DEPLOYMENT}'"}}}'

echo "Deployment complete. Active deployment is now: ${NEW_DEPLOYMENT}"
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus-config.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
- job_name: 'aegis-api'
  static_configs:
  - targets: ['aegis-api-service:8910']
  metrics_path: '/metrics'
  scrape_interval: 30s

- job_name: 'qdrant'
  static_configs:
  - targets: ['qdrant-service:6333']
  metrics_path: '/metrics'

- job_name: 'ollama'
  static_configs:
  - targets: ['ollama-service:11434']
  metrics_path: '/api/metrics'
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Aegis RAG System",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "Error rate"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: aegis-network-policy
  namespace: aegis-rag
spec:
  podSelector:
    matchLabels:
      app: aegis-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8910
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
  - to:
    - podSelector:
        matchLabels:
          app: ollama
    ports:
    - protocol: TCP
      port: 11434
```

### Pod Security Policy

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: aegis-psp
spec:
  privileged: false
  runAsUser:
    rule: MustRunAsNonRoot
  runAsGroup:
    rule: MustRunAs
    ranges:
    - min: 1000
      max: 65535
  fsGroup:
    rule: MustRunAs
    ranges:
    - min: 1000
      max: 65535
  volumes:
  - configMap
  - secret
  - emptyDir
  - persistentVolumeClaim
```

## Backup and Recovery

### Automated Backup Strategy

```bash
#!/bin/bash
# Qdrant Backup Script

BACKUP_DIR="/backups/qdrant"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAMESPACE="aegis-rag"

# Create backup directory
mkdir -p ${BACKUP_DIR}/${TIMESTAMP}

# Backup Qdrant collections
kubectl exec -n ${NAMESPACE} qdrant-0 -- \
    qdrant-cli collection backup aegis_docs_v2 \
    --output-dir /qdrant/storage/backups/${TIMESTAMP}

# Copy backup to external storage
kubectl cp ${NAMESPACE}/qdrant-0:/qdrant/storage/backups/${TIMESTAMP} \
    ${BACKUP_DIR}/${TIMESTAMP}

# Upload to cloud storage
aws s3 sync ${BACKUP_DIR}/${TIMESTAMP} \
    s3://aegis-backups/qdrant/${TIMESTAMP}

# Cleanup old backups (keep 30 days)
find ${BACKUP_DIR} -type d -mtime +30 -exec rm -rf {} \;
```

### Disaster Recovery Procedure

```markdown
# Disaster Recovery Runbook

## Recovery Time Objective (RTO): 4 hours
## Recovery Point Objective (RPO): 1 hour

### Critical Service Recovery Order:
1. Database services (Qdrant, Redis)
2. External service dependencies verification
3. API services deployment
4. Load balancer and ingress configuration
5. Health check and traffic validation

### Recovery Steps:
1. **Assess Impact**: Determine scope of outage
2. **Restore Infrastructure**: Deploy base infrastructure
3. **Restore Data**: Recover from latest backup
4. **Deploy Applications**: Deploy application services
5. **Validate Operation**: Perform end-to-end testing
6. **Resume Traffic**: Route production traffic
```

## Performance Tuning

### JVM and Python Optimization

```bash
# Python Application Tuning
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Uvicorn Worker Configuration
uvicorn app:app \
    --host 0.0.0.0 \
    --port 8910 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-log \
    --use-colors \
    --loop uvloop
```

### Database Performance Tuning

```yaml
# Qdrant Performance Configuration
storage:
  optimizers:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
    default_segment_number: 0
  wal:
    wal_capacity_mb: 32
    wal_segments_ahead: 0
  performance:
    max_search_threads: 0
```

This deployment guide provides comprehensive coverage of production deployment scenarios, from container-based deployments to Kubernetes orchestration, with emphasis on security, monitoring, and operational excellence.