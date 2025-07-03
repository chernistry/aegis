#!/usr/bin/env python3
"""
Aegis RAG System Integration Test Suite
This script validates that all components are working correctly.
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8910"
OPENWEBUI_URL = "http://localhost:8920"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"

class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def log(message: str, color: str = Colors.NC):
    """Print colored log message."""
    print(f"{color}{message}{Colors.NC}")

def test_service_health(name: str, url: str, endpoint: str = "/") -> bool:
    """Test if a service is responding."""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=5)
        if response.status_code == 200:
            log(f"✅ {name} is healthy", Colors.GREEN)
            return True
        else:
            log(f"❌ {name} returned status {response.status_code}", Colors.RED)
            return False
    except requests.RequestException as e:
        log(f"❌ {name} is not accessible: {e}", Colors.RED)
        return False

def test_api_endpoints() -> bool:
    """Test main API endpoints."""
    log("\n🧪 Testing API Endpoints", Colors.BLUE)
    
    # Test health endpoint
    if not test_service_health("API Health", API_BASE_URL, "/health"):
        return False
    
    # Test OpenAI compatibility
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            log(f"✅ Models endpoint working: {len(models.get('data', []))} models", Colors.GREEN)
        else:
            log(f"❌ Models endpoint failed: {response.status_code}", Colors.RED)
            return False
    except Exception as e:
        log(f"❌ Models endpoint error: {e}", Colors.RED)
        return False
    
    return True

def test_chat_functionality() -> bool:
    """Test chat functionality."""
    log("\n💬 Testing Chat Functionality", Colors.BLUE)
    
    test_question = "What is artificial intelligence?"
    
    try:
        # Test basic chat endpoint
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": test_question, "top_k": 3},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            log(f"✅ Chat endpoint working, response length: {len(data.get('answer', ''))}", Colors.GREEN)
        else:
            log(f"❌ Chat endpoint failed: {response.status_code}", Colors.RED)
            return False
            
    except Exception as e:
        log(f"❌ Chat endpoint error: {e}", Colors.RED)
        return False
    
    return True

def test_openai_compatibility() -> bool:
    """Test OpenAI-compatible endpoint."""
    log("\n🤖 Testing OpenAI Compatibility", Colors.BLUE)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json={
                "model": "aegis-rag-model",
                "messages": [{"role": "user", "content": "Hello, test message"}],
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                log("✅ OpenAI compatibility working", Colors.GREEN)
                return True
            else:
                log("❌ Invalid OpenAI response format", Colors.RED)
                return False
        else:
            log(f"❌ OpenAI endpoint failed: {response.status_code}", Colors.RED)
            return False
            
    except Exception as e:
        log(f"❌ OpenAI endpoint error: {e}", Colors.RED)
        return False

def test_metrics_endpoint() -> bool:
    """Test Prometheus metrics."""
    log("\n📊 Testing Metrics", Colors.BLUE)
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            if "http_requests_total" in metrics:
                log("✅ Prometheus metrics available", Colors.GREEN)
                return True
            else:
                log("❌ Metrics format incorrect", Colors.RED)
                return False
        else:
            log(f"❌ Metrics endpoint failed: {response.status_code}", Colors.RED)
            return False
    except Exception as e:
        log(f"❌ Metrics endpoint error: {e}", Colors.RED)
        return False

def test_document_ingestion() -> bool:
    """Test document ingestion endpoint."""
    log("\n📄 Testing Document Ingestion", Colors.BLUE)
    
    # Create a test document
    test_dir = Path("data/raw/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "test_doc.md"
    test_content = """# Test Document
    
This is a test document for validating the Aegis RAG system.
It contains information about artificial intelligence and machine learning.

## Key Concepts
- Neural networks process information like the human brain
- Machine learning algorithms improve through experience
- RAG systems combine retrieval and generation for better responses
"""
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/internal/ingest",
            params={"path": str(test_dir), "collection": "test_collection"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            log(f"✅ Ingestion endpoint working: {data.get('message', '')}", Colors.GREEN)
            return True
        else:
            log(f"❌ Ingestion failed: {response.status_code}", Colors.RED)
            return False
            
    except Exception as e:
        log(f"❌ Ingestion error: {e}", Colors.RED)
        return False
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()

def run_performance_test() -> bool:
    """Basic performance test."""
    log("\n⚡ Running Performance Test", Colors.BLUE)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": "What is machine learning?", "top_k": 5},
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            if response_time < 10:  # Less than 10 seconds is acceptable
                log(f"✅ Performance test passed: {response_time:.2f}s response time", Colors.GREEN)
                return True
            else:
                log(f"⚠️  Slow response time: {response_time:.2f}s", Colors.YELLOW)
                return True
        else:
            log(f"❌ Performance test failed: {response.status_code}", Colors.RED)
            return False
            
    except Exception as e:
        log(f"❌ Performance test error: {e}", Colors.RED)
        return False

def main():
    """Run all tests."""
    log("🚀 Starting Aegis RAG System Tests", Colors.BLUE)
    log("=" * 50, Colors.BLUE)
    
    tests = [
        ("Core Services", lambda: all([
            test_service_health("Qdrant", QDRANT_URL, "/"),
            test_service_health("Ollama", OLLAMA_URL, "/api/tags"),
            test_service_health("Open WebUI", OPENWEBUI_URL, "/"),
        ])),
        ("API Endpoints", test_api_endpoints),
        ("Chat Functionality", test_chat_functionality),
        ("OpenAI Compatibility", test_openai_compatibility),
        ("Metrics", test_metrics_endpoint),
        ("Document Ingestion", test_document_ingestion),
        ("Performance", run_performance_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        log(f"\n🔍 Running {test_name} tests...", Colors.YELLOW)
        try:
            if test_func():
                passed += 1
                log(f"✅ {test_name} tests passed", Colors.GREEN)
            else:
                log(f"❌ {test_name} tests failed", Colors.RED)
        except Exception as e:
            log(f"❌ {test_name} tests error: {e}", Colors.RED)
    
    log("\n" + "=" * 50, Colors.BLUE)
    log(f"📊 Test Results: {passed}/{total} passed", Colors.BLUE)
    
    if passed == total:
        log("🎉 All tests passed! Aegis RAG is working perfectly.", Colors.GREEN)
        return 0
    else:
        log(f"⚠️  {total - passed} tests failed. Check the logs above.", Colors.YELLOW)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)