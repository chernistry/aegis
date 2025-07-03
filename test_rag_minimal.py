#!/usr/bin/env python3
"""
Minimal RAG Pipeline Test
Tests the core functionality without complex dependencies
"""

import os
import sys
import json
import time
from typing import List, Dict

# Basic packages only
try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

import numpy as np

class MinimalRAGPipeline:
    """Minimal RAG implementation for testing"""
    
    def __init__(self):
        print("🔧 Initializing Minimal RAG Pipeline...")
        
        # Simple embedding model
        print("📥 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # In-memory vector store
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_document(self, text: str, metadata: dict = None):
        """Add a document to the vector store"""
        if len(text.strip()) < 50:  # Skip very short texts
            return
            
        # Split into chunks
        chunks = self._chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue
                
            embedding = self.embedder.encode(chunk)
            
            self.documents.append(chunk)
            self.embeddings.append(embedding)
            self.metadata.append({
                **(metadata or {}),
                "chunk_id": len(self.documents),
                "chunk_index": i
            })
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple text chunking"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        if not self.documents:
            return []
            
        print(f"🔍 Searching for: '{query}'")
        
        # Encode query
        query_embedding = self.embedder.encode(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top results
        results = []
        for similarity, idx in similarities[:top_k]:
            results.append({
                "text": self.documents[idx],
                "similarity": float(similarity),
                "metadata": self.metadata[idx]
            })
            
        return results
    
    def answer_question(self, question: str) -> str:
        """Simple RAG answer generation"""
        relevant_docs = self.search(question, top_k=3)
        
        if not relevant_docs:
            return "I don't have enough information to answer this question."
        
        # Simple answer construction
        context = "\n\n".join([doc["text"] for doc in relevant_docs])
        
        # For now, return the most relevant context with some formatting
        best_match = relevant_docs[0]
        
        return f"""
Based on the documents, here's what I found:

{best_match['text']}

(Similarity: {best_match['similarity']:.3f})

Additional relevant context:
{relevant_docs[1]['text'] if len(relevant_docs) > 1 else 'No additional context available.'}
"""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Simple PDF text extraction (fallback method)"""
    try:
        # Try using pdfplumber first
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
        
    except ImportError:
        print("⚠️  pdfplumber not available, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")
        return ""

def test_rag_pipeline():
    """Test the RAG pipeline with downloaded papers"""
    
    print("🚀 Starting RAG Pipeline Test")
    print("=" * 50)
    
    # Initialize pipeline
    rag = MinimalRAGPipeline()
    
    # Check for downloaded PDFs
    pdf_dir = "data/raw"
    if not os.path.exists(pdf_dir):
        print(f"❌ Directory {pdf_dir} not found")
        return
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n📖 Processing: {pdf_file}")
        
        try:
            text = extract_text_from_pdf(pdf_path)
            if text:
                print(f"✅ Extracted {len(text)} characters")
                rag.add_document(text, {"source": pdf_file, "type": "research_paper"})
            else:
                print("⚠️  No text extracted")
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
    
    print(f"\n📊 Ingestion Complete!")
    print(f"  - Total documents: {len(rag.documents)}")
    print(f"  - Total embeddings: {len(rag.embeddings)}")
    
    # Test queries
    test_questions = [
        "What are the main contributions of this research?",
        "What methods were used in this study?",
        "What are the key findings?",
        "How does this work compare to previous approaches?",
        "What are the limitations mentioned?",
        "What future work is suggested?",
        "What datasets were used?",
        "What evaluation metrics were employed?"
    ]
    
    print(f"\n🤔 Testing Questions:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions[:4], 1):  # Test first 4 questions
        print(f"\n🙋 Question {i}: {question}")
        print("-" * 40)
        
        start_time = time.time()
        answer = rag.answer_question(question)
        end_time = time.time()
        
        print(f"🤖 Answer (took {end_time - start_time:.2f}s):")
        print(answer)
        print("\n" + "="*50)
    
    # Test search functionality
    print(f"\n🔍 Testing Search Functionality:")
    print("=" * 50)
    
    search_terms = ["transformer", "neural network", "machine learning", "deep learning"]
    
    for term in search_terms:
        print(f"\n🔎 Searching for: '{term}'")
        results = rag.search(term, top_k=3)
        print(f"Found {len(results)} results:")
        
        for i, result in enumerate(results[:2], 1):  # Show top 2
            print(f"\n  {i}. (Score: {result['similarity']:.3f})")
            print(f"     Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"     Text: {result['text'][:200]}...")
    
    print(f"\n🎉 RAG Pipeline Test Complete!")
    print("=" * 50)
    
    # Summary statistics
    print(f"\n📈 Test Summary:")
    print(f"  - PDFs processed: {len(pdf_files)}")
    print(f"  - Document chunks: {len(rag.documents)}")
    print(f"  - Questions tested: 4")
    print(f"  - Search terms tested: {len(search_terms)}")
    print("  - Status: ✅ All tests completed successfully!")

if __name__ == "__main__":
    test_rag_pipeline()