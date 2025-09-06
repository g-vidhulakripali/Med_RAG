#!/usr/bin/env python3
"""
Test script to check Ollama availability and functionality
"""

import requests
import json
from langchain_ollama import ChatOllama

def test_ollama_connection():
    """Test if Ollama server is running and accessible"""
    try:
        # Test basic connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running and accessible")
            models = response.json()
            print(f"Available models: {[model['name'] for model in models.get('models', [])]}")
            return True
        else:
            print(f"❌ Ollama server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server at http://localhost:11434")
        print("   Make sure Ollama is installed and running")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False

def test_ollama_langchain():
    """Test if Ollama works with LangChain"""
    try:
        # Try to initialize ChatOllama
        llm = ChatOllama(
            model="mistral",
            base_url="http://localhost:11434"
        )
        print("✅ ChatOllama initialized successfully")
        
        # Test a simple query
        response = llm.invoke("Say 'Hello, Ollama is working!'")
        print(f"✅ Test response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Ollama with LangChain: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 Testing Ollama Integration...")
    print("=" * 50)
    
    # Test 1: Server connection
    print("\n1. Testing Ollama server connection...")
    server_ok = test_ollama_connection()
    
    # Test 2: LangChain integration
    print("\n2. Testing Ollama with LangChain...")
    langchain_ok = test_ollama_langchain()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Ollama Server: {'✅ Working' if server_ok else '❌ Not Working'}")
    print(f"   LangChain Integration: {'✅ Working' if langchain_ok else '❌ Not Working'}")
    
    if server_ok and langchain_ok:
        print("\n🎉 Ollama is fully functional and ready to use!")
        print("   You can now use it as a fallback LLM in your RAG system.")
    elif server_ok and not langchain_ok:
        print("\n⚠️  Ollama server is running but LangChain integration failed.")
        print("   Check your langchain-ollama installation.")
    else:
        print("\n❌ Ollama is not working properly.")
        print("   Please install and start Ollama first:")
        print("   1. Download from: https://ollama.ai/")
        print("   2. Install and run: ollama serve")
        print("   3. Pull Mistral model: ollama pull mistral")

if __name__ == "__main__":
    main()

