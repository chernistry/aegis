# Operations Manual

## Overview

This operations manual provides comprehensive guidance for monitoring, maintaining, and troubleshooting the Aegis RAG system in production environments. The document follows industry best practices for operational excellence and incident management.

## System Health Monitoring

### Key Performance Indicators (KPIs)

#### Application Layer Metrics

| Metric | Target | Critical Threshold | Alert Condition |
|--------|--------|-------------------|-----------------|
| API Response Time (P95) | < 2s | > 5s | Sustained 2+ minutes |
| API Error Rate | < 1% | > 5% | Sustained 1+ minute |
| API Availability | > 99.9% | < 99% | Any downtime |
| Request Throughput | Baseline ±20% | > 50% deviation | Sustained 5+ minutes |
| Memory Usage | < 80% | > 90% | Current usage |
| CPU Usage | < 70% | > 85% | Sustained 5+ minutes |

#### Vector Database Metrics

| Metric | Target | Critical Threshold | Alert Condition |
|--------|--------|-------------------|-----------------|
| Search Latency (P95) | < 100ms | > 500ms | Sustained 2+ minutes |
| Index Memory Usage | < 80% | > 90% | Current usage |
| Disk Usage | < 70% | > 85% | Current usage |
| Connection Pool Usage | < 80% | > 95% | Current usage |
| Query Success Rate | > 99.5% | < 99% | Sustained 1+ minute |

#### Language Model Metrics

| Metric | Target | Critical Threshold | Alert Condition |
|--------|--------|-------------------|-----------------|
| Generation Latency (P95) | < 3s | > 10s | Sustained 2+ minutes |
| Model Load Time | < 30s | > 60s | Single occurrence |
| GPU Memory Usage | < 85% | > 95% | Current usage |
| GPU Utilization | 60-90% | < 30% or > 95% | Sustained 5+ minutes |
| Generation Success Rate | > 99% | < 98% | Sustained 1+ minute |

### Monitoring Infrastructure

#### Prometheus Metrics Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'aegis-production'
    environment: 'production'

scrape_configs:
  - job_name: 'aegis-api'
    static_configs:
      - targets: ['aegis-api:8910']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    metrics_path: '/api/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s

rule_files:
  - "aegis_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Critical Alert Rules

```yaml
# aegis_alerts.yml
groups:
  - name: aegis.rules
    rules:
    - alert: AegisAPIHighLatency
      expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
      for: 2m
      labels:
        severity: critical
        service: aegis-api
      annotations:
        summary: "Aegis API high latency detected"
        description: "API response time P95 is {{ $value }}s, exceeding 5s threshold"

    - alert: AegisAPIHighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
      for: 1m
      labels:
        severity: critical
        service: aegis-api
      annotations:
        summary: "Aegis API high error rate"
        description: "API error rate is {{ $value | humanizePercentage }}, exceeding 5% threshold"

    - alert: QdrantHighSearchLatency
      expr: histogram_quantile(0.95, rate(qdrant_search_duration_seconds_bucket[5m])) > 0.5
      for: 2m
      labels:
        severity: warning
        service: qdrant
      annotations:
        summary: "Qdrant high search latency"
        description: "Search latency P95 is {{ $value }}s, exceeding 500ms threshold"

    - alert: OllamaModelGenerationFailure
      expr: rate(ollama_generation_failures_total[5m]) > 0.02
      for: 1m
      labels:
        severity: critical
        service: ollama
      annotations:
        summary: "Ollama model generation failures"
        description: "Generation failure rate is {{ $value | humanizePercentage }}, exceeding 2% threshold"

    - alert: AegisServiceDown
      expr: up == 0
      for: 30s
      labels:
        severity: critical
      annotations:
        summary: "Service {{ $labels.job }} is down"
        description: "Service {{ $labels.job }} has been down for more than 30 seconds"
```

### Health Check Procedures

#### Automated Health Checks

```bash
#!/bin/bash
# health_check.sh - Comprehensive health check script

set -e

# Configuration
API_URL="http://localhost:8910"
QDRANT_URL="http://localhost:6333"
OLLAMA_URL="http://localhost:11434"
TIMEOUT=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${2:-$GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    log "$1" $RED
}

warning() {
    log "$1" $YELLOW
}

# Test API Health
test_api_health() {
    log "Testing API health..."
    if curl -sf --max-time $TIMEOUT "$API_URL/health" > /dev/null; then
        log "✓ API health check passed"
        return 0
    else
        error "✗ API health check failed"
        return 1
    fi
}

# Test API Functionality
test_api_functionality() {
    log "Testing API functionality..."
    local response=$(curl -sf --max-time $TIMEOUT \
        -H "Content-Type: application/json" \
        -d '{"question":"test query","history":[]}' \
        "$API_URL/chat" 2>/dev/null)
    
    if [[ $? -eq 0 ]] && echo "$response" | jq -e '.answer' > /dev/null 2>&1; then
        log "✓ API functionality test passed"
        return 0
    else
        error "✗ API functionality test failed"
        return 1
    fi
}

# Test Qdrant Health
test_qdrant_health() {
    log "Testing Qdrant health..."
    if curl -sf --max-time $TIMEOUT "$QDRANT_URL/health" > /dev/null; then
        log "✓ Qdrant health check passed"
        return 0
    else
        error "✗ Qdrant health check failed"
        return 1
    fi
}

# Test Ollama Health
test_ollama_health() {
    log "Testing Ollama health..."
    if curl -sf --max-time $TIMEOUT "$OLLAMA_URL/api/tags" > /dev/null; then
        log "✓ Ollama health check passed"
        return 0
    else
        error "✗ Ollama health check failed"
        return 1
    fi
}

# Test External Dependencies
test_external_dependencies() {
    log "Testing external dependencies..."
    
    # Test Jina AI API
    if curl -sf --max-time $TIMEOUT \
        -H "Authorization: Bearer $JINA_API_KEY" \
        "https://api.jina.ai/v1/models" > /dev/null; then
        log "✓ Jina AI API accessible"
    else
        warning "⚠ Jina AI API check failed"
    fi
}

# Main health check execution
main() {
    log "Starting comprehensive health check..."
    
    local failures=0
    
    test_api_health || ((failures++))
    test_api_functionality || ((failures++))
    test_qdrant_health || ((failures++))
    test_ollama_health || ((failures++))
    test_external_dependencies
    
    if [[ $failures -eq 0 ]]; then
        log "✓ All health checks passed successfully"
        exit 0
    else
        error "✗ $failures health check(s) failed"
        exit 1
    fi
}

main "$@"
```

## Incident Response Procedures

### Incident Classification

#### Severity Levels

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| P0 - Critical | Complete service outage | 15 minutes | Total system down, data loss |
| P1 - High | Major service degradation | 30 minutes | High error rates, slow responses |
| P2 - Medium | Partial service impact | 2 hours | Non-critical feature failure |
| P3 - Low | Minor issues | 24 hours | Performance degradation |

#### Incident Response Team

| Role | Responsibilities | Contact Method |
|------|------------------|----------------|
| Incident Commander | Overall incident coordination | Primary: Pager, Secondary: Phone |
| Technical Lead | Technical investigation and resolution | Slack + Pager |
| Communications Lead | Stakeholder updates and communication | Email + Slack |
| Subject Matter Expert | Domain-specific expertise | Slack + Phone |

### Runbooks

#### API Service Failure

```markdown
# Runbook: API Service Failure

## Symptoms
- API health check failures
- High error rates (5xx responses)
- Complete API unresponsiveness

## Immediate Actions
1. **Verify Impact**: Check monitoring dashboards for scope
2. **Check Dependencies**: Verify Qdrant and Ollama status
3. **Review Logs**: Check application logs for error patterns
4. **Scale Resources**: Increase API pod replicas if resource-constrained

## Investigation Steps
1. Check application logs:
   ```bash
   kubectl logs -f deployment/aegis-api -n aegis-rag --tail=100
   ```

2. Verify pod status:
   ```bash
   kubectl get pods -n aegis-rag -l app=aegis-api
   ```

3. Check resource utilization:
   ```bash
   kubectl top pods -n aegis-rag
   ```

4. Test dependencies:
   ```bash
   curl -f http://qdrant-service:6333/health
   curl -f http://ollama-service:11434/api/tags
   ```

## Resolution Actions
1. **Rolling Restart**: 
   ```bash
   kubectl rollout restart deployment/aegis-api -n aegis-rag
   ```

2. **Scale Up**:
   ```bash
   kubectl scale deployment/aegis-api --replicas=5 -n aegis-rag
   ```

3. **Emergency Rollback**:
   ```bash
   kubectl rollout undo deployment/aegis-api -n aegis-rag
   ```

## Recovery Verification
1. Monitor API health endpoint
2. Verify response times return to normal
3. Check error rates decrease below 1%
4. Validate end-to-end functionality
```

#### Vector Database Issues

```markdown
# Runbook: Vector Database Performance Degradation

## Symptoms
- Slow search response times
- High memory usage
- Connection pool exhaustion

## Immediate Actions
1. **Check Qdrant Status**: Verify cluster health
2. **Monitor Resources**: Check memory and disk usage
3. **Review Query Patterns**: Identify expensive operations
4. **Scale if Needed**: Add read replicas

## Investigation Steps
1. Check Qdrant cluster status:
   ```bash
   curl http://qdrant-service:6333/cluster
   ```

2. Monitor collection status:
   ```bash
   curl http://qdrant-service:6333/collections/aegis_docs_v2
   ```

3. Check resource usage:
   ```bash
   kubectl top pods -n aegis-rag -l app=qdrant
   ```

## Resolution Actions
1. **Optimize Queries**: Review and optimize search parameters
2. **Increase Resources**: Scale up Qdrant pods
3. **Clear Cache**: Restart Qdrant pods to clear memory
4. **Reindex**: Rebuild indices if corrupted

## Prevention
1. Implement query result caching
2. Regular index optimization
3. Monitor collection size growth
4. Set up automated scaling
```

#### LLM Service Degradation

```markdown
# Runbook: Language Model Service Issues

## Symptoms
- High generation latency
- Model loading failures
- GPU memory exhaustion

## Immediate Actions
1. **Check Ollama Status**: Verify service health
2. **Monitor GPU Usage**: Check GPU memory and utilization
3. **Review Model Status**: Verify model availability
4. **Scale Resources**: Add more Ollama instances

## Investigation Steps
1. Check Ollama service status:
   ```bash
   curl http://ollama-service:11434/api/tags
   ```

2. Monitor GPU resources:
   ```bash
   nvidia-smi
   ```

3. Check model loading:
   ```bash
   curl -X POST http://ollama-service:11434/api/show -d '{"name":"deepseek-r1:7b"}'
   ```

## Resolution Actions
1. **Restart Service**: Rolling restart of Ollama pods
2. **Clear GPU Memory**: Reset GPU state
3. **Load Balance**: Distribute requests across instances
4. **Model Reload**: Refresh model weights

## Prevention
1. Implement model pre-loading
2. Monitor GPU memory usage
3. Set up request queuing
4. Regular model health checks
```

## Maintenance Procedures

### Scheduled Maintenance

#### Monthly Maintenance Tasks

1. **Security Updates**
   - Update base Docker images
   - Apply security patches to dependencies
   - Review and update SSL certificates
   - Audit access controls and permissions

2. **Performance Optimization**
   - Analyze query performance metrics
   - Optimize database indices
   - Review and tune cache configurations
   - Clean up old logs and temporary files

3. **Backup Verification**
   - Test backup restoration procedures
   - Verify backup integrity and completeness
   - Update disaster recovery documentation
   - Test failover mechanisms

#### Weekly Maintenance Tasks

1. **System Health Review**
   - Review monitoring alerts and trends
   - Analyze performance metrics
   - Check resource utilization patterns
   - Validate all health checks

2. **Capacity Planning**
   - Monitor storage growth trends
   - Analyze traffic patterns
   - Plan resource scaling needs
   - Review cost optimization opportunities

#### Daily Maintenance Tasks

1. **Log Review**
   - Check error logs for new issues
   - Monitor warning patterns
   - Verify backup completion
   - Review security event logs

2. **Performance Monitoring**
   - Check response time trends
   - Monitor error rates
   - Verify SLA compliance
   - Review resource utilization

### Database Maintenance

#### Vector Database Optimization

```bash
#!/bin/bash
# qdrant_optimization.sh - Qdrant maintenance script

set -e

NAMESPACE="aegis-rag"
COLLECTION="aegis_docs_v2"
QDRANT_URL="http://qdrant-service:6333"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check collection health
check_collection_health() {
    log "Checking collection health..."
    curl -s "$QDRANT_URL/collections/$COLLECTION" | jq .
}

# Optimize collection indices
optimize_indices() {
    log "Optimizing collection indices..."
    curl -X POST "$QDRANT_URL/collections/$COLLECTION/index" \
        -H "Content-Type: application/json" \
        -d '{
            "operation": "optimize",
            "optimize_config": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000
            }
        }'
}

# Clean up deleted vectors
vacuum_collection() {
    log "Vacuuming deleted vectors..."
    curl -X POST "$QDRANT_URL/collections/$COLLECTION/index" \
        -H "Content-Type: application/json" \
        -d '{"operation": "vacuum"}'
}

# Update collection configuration
update_config() {
    log "Updating collection configuration..."
    curl -X PATCH "$QDRANT_URL/collections/$COLLECTION" \
        -H "Content-Type: application/json" \
        -d '{
            "optimizers_config": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 0
            }
        }'
}

# Generate collection statistics
generate_stats() {
    log "Generating collection statistics..."
    curl -s "$QDRANT_URL/collections/$COLLECTION" | \
        jq '{
            vectors_count: .result.vectors_count,
            segments_count: .result.segments_count,
            disk_data_size: .result.disk_data_size,
            ram_data_size: .result.ram_data_size
        }'
}

main() {
    log "Starting Qdrant maintenance..."
    
    check_collection_health
    optimize_indices
    vacuum_collection
    update_config
    generate_stats
    
    log "Qdrant maintenance completed"
}

main "$@"
```

### Model Management

#### Model Update Procedure

```bash
#!/bin/bash
# model_update.sh - Language model update script

set -e

NAMESPACE="aegis-rag"
OLLAMA_SERVICE="ollama-service"
NEW_MODEL="$1"
OLD_MODEL="$2"

if [[ -z "$NEW_MODEL" || -z "$OLD_MODEL" ]]; then
    echo "Usage: $0 <new_model> <old_model>"
    exit 1
fi

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Pull new model
pull_model() {
    log "Pulling model: $NEW_MODEL"
    kubectl exec -n $NAMESPACE deployment/ollama -- \
        ollama pull "$NEW_MODEL"
}

# Test model functionality
test_model() {
    local model="$1"
    log "Testing model: $model"
    
    local response=$(kubectl exec -n $NAMESPACE deployment/ollama -- \
        curl -s -X POST http://localhost:11434/api/generate \
        -d "{\"model\":\"$model\",\"prompt\":\"Hello\",\"stream\":false}")
    
    if echo "$response" | jq -e '.response' > /dev/null; then
        log "✓ Model test passed: $model"
        return 0
    else
        log "✗ Model test failed: $model"
        return 1
    fi
}

# Update configuration
update_config() {
    log "Updating model configuration..."
    kubectl patch configmap aegis-config -n $NAMESPACE \
        -p "{\"data\":{\"OLLAMA_MODEL\":\"$NEW_MODEL\"}}"
}

# Rolling restart API services
restart_api_services() {
    log "Restarting API services..."
    kubectl rollout restart deployment/aegis-api -n $NAMESPACE
    kubectl rollout status deployment/aegis-api -n $NAMESPACE
}

# Remove old model
cleanup_old_model() {
    log "Removing old model: $OLD_MODEL"
    kubectl exec -n $NAMESPACE deployment/ollama -- \
        ollama rm "$OLD_MODEL" || true
}

main() {
    log "Starting model update: $OLD_MODEL -> $NEW_MODEL"
    
    pull_model
    test_model "$NEW_MODEL"
    update_config
    restart_api_services
    
    # Verify new deployment
    sleep 30
    if test_model "$NEW_MODEL"; then
        cleanup_old_model
        log "Model update completed successfully"
    else
        log "Model update failed, rolling back..."
        update_config  # This would restore OLD_MODEL
        restart_api_services
        exit 1
    fi
}

main "$@"
```

## Performance Optimization

### Query Performance Tuning

#### Search Optimization Guidelines

1. **Vector Search Parameters**
   ```python
   # Optimal search configuration
   search_params = {
       "limit": 10,  # Initial retrieval count
       "with_payload": True,
       "with_vectors": False,  # Reduce bandwidth
       "score_threshold": 0.7,  # Filter low-relevance results
   }
   ```

2. **Embedding Cache Strategy**
   ```python
   # Implement embedding caching
   import redis
   import hashlib
   
   redis_client = redis.Redis(host='redis', port=6379, db=0)
   
   def get_cached_embedding(text):
       key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
       cached = redis_client.get(key)
       if cached:
           return json.loads(cached)
       return None
   
   def cache_embedding(text, embedding):
       key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
       redis_client.setex(key, 3600, json.dumps(embedding))
   ```

3. **Connection Pool Optimization**
   ```python
   # Configure optimal connection pools
   qdrant_client = QdrantClient(
       url=QDRANT_URL,
       timeout=60,
       prefer_grpc=True,
       grpc_options={
           'grpc.keepalive_time_ms': 30000,
           'grpc.keepalive_timeout_ms': 5000,
           'grpc.keepalive_permit_without_calls': True,
           'grpc.http2.max_pings_without_data': 0,
       }
   )
   ```

### Resource Optimization

#### Memory Management

```bash
#!/bin/bash
# memory_optimization.sh - Memory optimization script

# Configure Python garbage collection
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# JVM-style memory limits for Python
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072
export MALLOC_MMAP_MAX_=65536

# Configure uvicorn workers
uvicorn app:app \
    --host 0.0.0.0 \
    --port 8910 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --worker-connections 1000 \
    --backlog 2048 \
    --keep-alive 2 \
    --limit-max-requests 1000 \
    --limit-max-requests-jitter 50
```

## Troubleshooting Guide

### Common Issues and Solutions

#### High API Latency

**Symptoms:**
- API response times > 5 seconds
- Timeouts in client applications
- Queue buildup in load balancer

**Diagnostic Steps:**
1. Check CPU and memory usage
2. Analyze database query performance
3. Review external API latency
4. Check network connectivity

**Solutions:**
1. Scale API service horizontally
2. Optimize database queries
3. Implement request caching
4. Tune connection pools

#### Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors
- Pod restarts due to memory limits

**Diagnostic Steps:**
1. Monitor memory usage patterns
2. Analyze garbage collection logs
3. Profile application memory usage
4. Check for unclosed connections

**Solutions:**
1. Implement proper connection cleanup
2. Configure garbage collection tuning
3. Add memory limits and monitoring
4. Regular service restarts

#### Vector Search Accuracy Issues

**Symptoms:**
- Poor search result relevance
- Inconsistent answer quality
- User complaints about accuracy

**Diagnostic Steps:**
1. Analyze search score distributions
2. Review embedding quality
3. Check document segmentation
4. Validate reranking performance

**Solutions:**
1. Retune search parameters
2. Improve document preprocessing
3. Update embedding models
4. Enhance reranking algorithms

### Emergency Procedures

#### Complete System Recovery

```bash
#!/bin/bash
# emergency_recovery.sh - Emergency system recovery

set -e

NAMESPACE="aegis-rag"
BACKUP_DATE="$1"

if [[ -z "$BACKUP_DATE" ]]; then
    echo "Usage: $0 <backup_date_YYYYMMDD>"
    exit 1
fi

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Stop all services
stop_services() {
    log "Stopping all services..."
    kubectl scale deployment --all --replicas=0 -n $NAMESPACE
    kubectl wait --for=delete pods --all -n $NAMESPACE --timeout=300s
}

# Restore vector database
restore_qdrant() {
    log "Restoring Qdrant from backup: $BACKUP_DATE"
    
    # Download backup from cloud storage
    aws s3 sync s3://aegis-backups/qdrant/$BACKUP_DATE /tmp/qdrant-restore/
    
    # Restore to Qdrant
    kubectl exec -n $NAMESPACE qdrant-0 -- \
        qdrant-cli collection restore aegis_docs_v2 \
        --input-dir /tmp/qdrant-restore/
}

# Restart services
restart_services() {
    log "Restarting services..."
    kubectl scale deployment/qdrant --replicas=3 -n $NAMESPACE
    kubectl wait --for=condition=ready pods -l app=qdrant -n $NAMESPACE --timeout=300s
    
    kubectl scale deployment/ollama --replicas=2 -n $NAMESPACE
    kubectl wait --for=condition=ready pods -l app=ollama -n $NAMESPACE --timeout=300s
    
    kubectl scale deployment/aegis-api --replicas=3 -n $NAMESPACE
    kubectl wait --for=condition=ready pods -l app=aegis-api -n $NAMESPACE --timeout=300s
}

# Verify recovery
verify_recovery() {
    log "Verifying system recovery..."
    
    # Wait for services to be ready
    sleep 60
    
    # Test API functionality
    if ./health_check.sh; then
        log "✓ System recovery successful"
        return 0
    else
        log "✗ System recovery failed"
        return 1
    fi
}

main() {
    log "Starting emergency recovery procedure..."
    
    stop_services
    restore_qdrant
    restart_services
    verify_recovery
    
    log "Emergency recovery completed"
}

main "$@"
```

This operations manual provides comprehensive coverage of monitoring, maintenance, and troubleshooting procedures required for operating the Aegis RAG system at enterprise scale with principal-level operational excellence.