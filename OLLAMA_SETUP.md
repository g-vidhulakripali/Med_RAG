# 🦙 Ollama Integration Setup Guide

## Overview
This guide explains how to set up and use Ollama as a local LLM fallback in your Agentic RAG system.

## 🚀 What is Ollama?
Ollama is a local, open-source LLM server that allows you to run large language models on your own machine without needing API keys or internet connectivity.

## 📋 Prerequisites

### 1. Install Ollama
- **Windows**: Download from [https://ollama.ai/](https://ollama.ai/)
- **macOS**: `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`

### 2. Start Ollama Server
```bash
ollama serve
```

### 3. Pull the Mistral Model
```bash
ollama pull mistral
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in your project root:

```env
# API Keys (Optional - Ollama will work without them)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
```

### System Priority Order
1. **Google Gemini** (Primary - requires API key)
2. **OpenAI** (Fallback - requires API key)
3. **Ollama Mistral** (Local fallback - no API key needed)

## 🧪 Testing Ollama

### Test Script
Run the included test script to verify Ollama is working:

```bash
python test_ollama.py
```

Expected output:
```
✅ Ollama server is running and accessible
✅ ChatOllama initialized successfully
✅ Test response: Hello! Ollama is working!
```

### Manual Test
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
response = llm.invoke("Hello, how are you?")
print(response.content)
```

## 🌐 Accessing the System

### Streamlit App
1. Start the app: `streamlit run streamlit_app.py --server.port 8505`
2. Open browser: http://localhost:8505
3. Initialize the system
4. Ask questions - Ollama will automatically be used if other LLMs fail

### Direct Python Usage
```python
from system_manager import SystemManager

manager = SystemManager()
result = manager.process_query("Your question here")
print(result)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Server Not Running
```
❌ Cannot connect to Ollama server at http://localhost:11434
```
**Solution**: Start Ollama with `ollama serve`

#### 2. Model Not Found
```
❌ Model 'mistral' not found
```
**Solution**: Pull the model with `ollama pull mistral`

#### 3. Port Already in Use
```
❌ Port 11434 is already in use
```
**Solution**: Check if another Ollama instance is running

#### 4. Memory Issues
```
❌ Out of memory error
```
**Solution**: 
- Close other applications
- Use a smaller model: `ollama pull mistral:7b-instruct`
- Increase system swap space

### Performance Tips

1. **Model Selection**: Mistral 7B is a good balance of performance and resource usage
2. **Hardware**: 8GB+ RAM recommended for smooth operation
3. **GPU**: If available, Ollama will automatically use it for better performance

## 📊 System Status

Check which LLMs are available:

```python
from system_manager import SystemManager

manager = SystemManager()
status = manager.get_system_status()
print(status['llm_models'])
```

Expected output:
```json
{
  "google": {"model": "gemini-1.5-pro", "status": "available"},
  "ollama": {"model": "mistral", "status": "available", "base_url": "http://localhost:11434"}
}
```

## 🎯 Benefits of Ollama Integration

1. **No API Costs**: Run LLMs locally without usage fees
2. **Privacy**: Data stays on your machine
3. **Reliability**: No dependency on external services
4. **Customization**: Use any compatible model
5. **Offline**: Works without internet connection

## 🔄 Fallback Behavior

The system automatically falls back to Ollama when:
- Google Gemini API quota is exceeded
- OpenAI API fails or is unavailable
- Network connectivity issues occur
- API keys are invalid or missing

## 📚 Additional Models

You can use other models with Ollama:

```bash
# Smaller, faster models
ollama pull llama2:7b
ollama pull codellama:7b

# Larger, more capable models
ollama pull llama2:70b
ollama pull codellama:34b
```

Then update your config:
```python
OLLAMA_MODEL = "llama2:7b"  # or any other model name
```

## 🆘 Support

If you encounter issues:
1. Check Ollama server status: `ollama list`
2. Verify model availability: `ollama show mistral`
3. Check system logs for detailed error messages
4. Ensure sufficient system resources (RAM, disk space)

---

**Happy coding with your local LLM-powered RAG system! 🚀**

