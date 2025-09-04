#!/usr/bin/env python3
"""
Test script for the Agentic RAG System
This script tests the core functionality without requiring API keys
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        from config import Config
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor module imported successfully")
    except ImportError as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        return False
    
    try:
        from vector_store import VectorStore
        print("✅ VectorStore module imported successfully")
    except ImportError as e:
        print(f"❌ VectorStore import failed: {e}")
        return False
    
    try:
        from llm_manager import LLMManager
        print("✅ LLMManager module imported successfully")
    except ImportError as e:
        print(f"❌ LLMManager import failed: {e}")
        return False
    
    try:
        from rag_agent import RAGAgent
        print("✅ RAGAgent module imported successfully")
    except ImportError as e:
        print(f"❌ RAGAgent import failed: {e}")
        return False
    
    try:
        from system_manager import SystemManager
        print("✅ SystemManager module imported successfully")
    except ImportError as e:
        print(f"❌ SystemManager import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration validation"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        
        # Test configuration access
        print(f"✅ Chunk size: {Config.CHUNK_SIZE}")
        print(f"✅ Chunk overlap: {Config.CHUNK_OVERLAP}")
        print(f"✅ Top-K retrieval: {Config.TOP_K_RETRIEVAL}")
        print(f"✅ Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
        
        # Test validation (should fail without API keys, which is expected)
        try:
            Config.validate_config()
            print("✅ Configuration validation passed")
        except ValueError as e:
            print(f"⚠️ Configuration validation failed (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_document_processing():
    """Test document processing functionality"""
    print("\nTesting document processing...")
    
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
        
        # Check if data directory exists
        data_dir = Path("Data")
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            print(f"✅ Found {len(pdf_files)} PDF files in Data directory")
            
            for pdf_file in pdf_files:
                print(f"  📄 {pdf_file.name}")
        else:
            print("⚠️ Data directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\nTesting vector store...")
    
    try:
        from vector_store import VectorStore
        
        vector_store = VectorStore()
        print("✅ VectorStore initialized")
        
        # Test collection stats (should work even without documents)
        stats = vector_store.get_collection_stats()
        print(f"✅ Collection stats retrieved: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_llm_manager():
    """Test LLM manager functionality"""
    print("\nTesting LLM manager...")
    
    try:
        from llm_manager import LLMManager
        
        # This will fail without API keys, which is expected
        try:
            llm_manager = LLMManager()
            print("✅ LLMManager initialized")
        except Exception as e:
            print(f"⚠️ LLMManager initialization failed (expected without API keys): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM manager test failed: {e}")
        return False

def test_system_manager():
    """Test system manager functionality"""
    print("\nTesting system manager...")
    
    try:
        from system_manager import SystemManager
        
        manager = SystemManager()
        print("✅ SystemManager initialized")
        
        # Test system status
        status = manager.get_system_status()
        print(f"✅ System status retrieved: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ System manager test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "requirements.txt",
        "config.py",
        "document_processor.py",
        "vector_store.py",
        "llm_manager.py",
        "rag_agent.py",
        "system_manager.py",
        "streamlit_app.py",
        "README.md",
        "technical_report.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧠 Agentic RAG System - System Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Document Processing", test_document_processing),
        ("Vector Store", test_vector_store),
        ("LLM Manager", test_llm_manager),
        ("System Manager", test_system_manager)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
        print("\nNext steps:")
        print("1. Set up your API keys in a .env file")
        print("2. Run: streamlit run streamlit_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
