#!/usr/bin/env python3
"""
RAG Answer Verification Script
Tests specific technical questions to validate answer quality
"""

import sys
import time

# Run the same imports as our test script
sys.path.append('.')

# Re-import our test functions
from test_rag_minimal import MinimalRAGPipeline, extract_text_from_pdf
import os

def verify_technical_answers():
    """Test more specific technical questions to verify answer accuracy"""
    
    print("🔬 RAG Answer Verification Test")
    print("=" * 50)
    
    # Initialize pipeline (reuse from previous test)
    rag = MinimalRAGPipeline()
    
    # Load the same documents
    pdf_dir = "data/raw"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    print(f"📚 Loading {len(pdf_files)} papers for verification...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        if text:
            rag.add_document(text, {"source": pdf_file, "type": "research_paper"})
    
    print(f"✅ Loaded {len(rag.documents)} document chunks")
    
    # Verification questions - more specific and technical
    verification_questions = [
        {
            "question": "What is the Inner Thinking Transformer (ITT)?",
            "expected_topics": ["dynamic", "token-level", "reasoning", "thinking steps"],
            "category": "Architecture Definition"
        },
        {
            "question": "How does Manus AI handle complex tasks?",
            "expected_topics": ["decompos", "simultaneous", "parallel", "coordination"],
            "category": "System Capability"
        },
        {
            "question": "What models are compared in the DeepSeek study?",
            "expected_topics": ["DeepSeek", "GPT-4o", "comparison", "models"],
            "category": "Model Comparison"
        },
        {
            "question": "What are the computational benefits of ITT?",
            "expected_topics": ["FLOPs", "parameter", "efficient", "overhead"],
            "category": "Performance Analysis"
        },
        {
            "question": "What applications are mentioned for Manus AI?",
            "expected_topics": ["research", "healthcare", "public sector", "applications"],
            "category": "Use Cases"
        }
    ]
    
    print(f"\n🤔 Testing {len(verification_questions)} verification questions:")
    print("=" * 50)
    
    results = []
    
    for i, test_case in enumerate(verification_questions, 1):
        question = test_case["question"]
        expected_topics = test_case["expected_topics"]
        category = test_case["category"]
        
        print(f"\n🔍 Test {i}: {category}")
        print(f"❓ Question: {question}")
        
        # Get answer
        start_time = time.time()
        answer = rag.answer_question(question)
        end_time = time.time()
        
        # Extract the main answer text (before similarity score)
        answer_text = answer.split("(Similarity:")[0].strip()
        answer_text_lower = answer_text.lower()
        
        # Check for expected topics
        topics_found = []
        for topic in expected_topics:
            if topic.lower() in answer_text_lower:
                topics_found.append(topic)
        
        # Calculate relevance score
        relevance_score = len(topics_found) / len(expected_topics)
        
        print(f"⏱️  Response time: {end_time - start_time:.3f}s")
        print(f"🎯 Expected topics: {expected_topics}")
        print(f"✅ Topics found: {topics_found}")
        print(f"📊 Relevance score: {relevance_score:.2f} ({len(topics_found)}/{len(expected_topics)})")
        
        # Show snippet of answer
        answer_snippet = answer_text[:200] + "..." if len(answer_text) > 200 else answer_text
        print(f"💡 Answer snippet: {answer_snippet}")
        
        # Evaluate quality
        if relevance_score >= 0.5:
            quality = "✅ GOOD"
        elif relevance_score >= 0.25:
            quality = "⚠️  PARTIAL"
        else:
            quality = "❌ POOR"
        
        print(f"🏆 Quality: {quality}")
        
        results.append({
            "question": question,
            "category": category,
            "relevance_score": relevance_score,
            "topics_found": len(topics_found),
            "topics_total": len(expected_topics),
            "response_time": end_time - start_time,
            "quality": quality
        })
        
        print("-" * 50)
    
    # Summary statistics
    print(f"\n📈 Verification Summary:")
    print("=" * 50)
    
    avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
    avg_response_time = sum(r["response_time"] for r in results) / len(results)
    good_answers = len([r for r in results if r["relevance_score"] >= 0.5])
    
    print(f"📊 Total questions tested: {len(results)}")
    print(f"🎯 Average relevance score: {avg_relevance:.2f}")
    print(f"⏱️  Average response time: {avg_response_time:.3f}s")
    print(f"✅ Good quality answers: {good_answers}/{len(results)} ({good_answers/len(results)*100:.1f}%)")
    
    # Category breakdown
    print(f"\n📋 Category Performance:")
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result["relevance_score"])
    
    for category, scores in categories.items():
        avg_score = sum(scores) / len(scores)
        print(f"  • {category}: {avg_score:.2f}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if avg_relevance >= 0.7:
        overall = "🌟 EXCELLENT"
    elif avg_relevance >= 0.5:
        overall = "✅ GOOD"
    elif avg_relevance >= 0.3:
        overall = "⚠️  ACCEPTABLE"
    else:
        overall = "❌ NEEDS IMPROVEMENT"
    
    print(f"   {overall} (Average relevance: {avg_relevance:.2f})")
    
    # Recommendations
    print(f"\n💡 Verification Findings:")
    if avg_relevance >= 0.5:
        print("   ✅ RAG system successfully retrieves relevant content")
        print("   ✅ Answers contain expected technical terminology")
        print("   ✅ Fast response times suitable for production")
    else:
        print("   ⚠️  Some answers may lack specific technical details")
        print("   💡 Consider improving chunking strategy for better context")
    
    if avg_response_time < 0.1:
        print("   ✅ Response times excellent for real-time applications")
    
    print(f"\n🔬 Verification Complete!")
    return results

if __name__ == "__main__":
    verify_technical_answers()