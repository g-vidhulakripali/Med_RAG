# 🧠 Agentic RAG System with LangGraph

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangGraph that demonstrates autonomous decision-making, query routing, and self-correction capabilities for medical document analysis.

## 🎯 System Overview

This system processes medical PDF documents and provides intelligent, evidence-based responses to queries with the following key features:

- **Agentic Behavior**: Autonomous decision-making using LangGraph workflows
- **Multi-LLM Support**: Google AI Studio/Gemini (primary) and OpenAI (fallback) integration
- **Intelligent Retrieval**: Advanced document chunking and vector similarity search
- **Quality Assurance**: Self-evaluation and response improvement mechanisms
- **Source Citations**: Proper attribution and reference tracking
- **Modern UI**: Streamlit-based frontend for easy testing and interaction

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   System        │    │   RAG Agent     │
│   Frontend      │◄──►│   Manager       │◄──►│   (LangGraph)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Document      │    │   LLM Manager   │
                       │   Processor     │    │   (Gemini/OpenAI) │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Vector Store  │
                       │   (Chroma)      │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- API key from [Google AI Studio](https://makersuite.google.com/app/apikey) (recommended) or [OpenAI](https://platform.openai.com/)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Agentic-RAG-System

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env with your API keys
```

### 3. Configuration

Create a `.env` file with your API keys:

```bash
# Required: At least one API key
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize system behavior
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
CONFIDENCE_THRESHOLD=0.7
```

### 4. Run the System

```bash
# Start the Streamlit frontend
streamlit run streamlit_app.py
```

## 🔧 System Components

### Core Modules

- **`system_manager.py`**: Orchestrates the entire RAG pipeline
- **`rag_agent.py`**: LangGraph-based agent with autonomous workflows
- **`document_processor.py`**: PDF extraction, chunking, and preprocessing
- **`vector_store.py`**: Chroma-based vector storage and similarity search
- **`llm_manager.py`**: Multi-provider LLM management with fallback
- **`config.py`**: Centralized configuration management

### Key Features

1. **Autonomous Workflow**: The RAG agent follows a 5-step process:
   - Document retrieval
   - Query analysis
   - Response generation
   - Quality evaluation
   - Self-improvement (if needed)

2. **Intelligent Fallbacks**: Automatic switching between LLM providers
3. **Quality Assurance**: Built-in response evaluation and improvement
4. **Source Tracking**: Complete citation and reference management

## 📊 Usage Examples

### Sample Queries

The system comes with pre-loaded sample queries from the medical documents:

1. **Pain Observation Tool Study**: "What are the findings of this study Sensitivity and specificity of the critical-care pain observation tool for the detection of pain in intubated adults after cardiac surgery?"

2. **Evidence-Based Recommendation**: "Was ist die Evidenzbasierte Empfehlung 3. E 39, was sind ihre Empfehlungen?"

3. **Recommendation Analysis**: "Warum sehen die Autoren in diesem Absatz keinen Anlass, die Empfehlung 60 aufgrund der Kommentare von Thaler und Krüssel zu ändern?"

4. **AI in Medicine**: "Findest du, dass Künstliche Intelligenz in Zukunft Ärztinnen und Ärzte bei der Diagnose von seltenen Krankheiten besser unterstützen kann?"

### API Usage

```python
from system_manager import SystemManager

# Initialize the system
manager = SystemManager()
result = manager.initialize_system()

# Process a query
query_result = manager.process_query("Your medical query here")
print(query_result["response"])
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Size of document chunks in characters |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | 5 | Number of top documents to retrieve |
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence for response acceptance |
| `TEMPERATURE` | 0.1 | LLM response creativity (lower = more focused) |
| `MAX_TOKENS` | 4000 | Maximum tokens in LLM responses |

## 🔍 System Capabilities

### Document Processing
- **PDF Extraction**: Robust text extraction using PyPDF2
- **Smart Chunking**: Intelligent document segmentation with overlap
- **Metadata Preservation**: Source tracking and chunk identification

### Vector Storage
- **Chroma Integration**: High-performance vector database
- **Embedding Models**: Sentence transformers for semantic understanding
- **Similarity Search**: Advanced retrieval algorithms

### LLM Integration
- **Multi-Provider**: Google Gemini and OpenAI support
- **Automatic Fallback**: Seamless provider switching
- **Medical Specialization**: Domain-specific prompting and evaluation

### Agentic Behavior
- **Workflow Management**: LangGraph-based state management
- **Decision Making**: Autonomous query routing and processing
- **Self-Correction**: Quality evaluation and response improvement

## 📈 Performance & Limitations

### Strengths
- **High Accuracy**: Evidence-based responses with source citations
- **Scalability**: Efficient vector storage and retrieval
- **Reliability**: Multiple LLM providers and fallback mechanisms
- **Transparency**: Complete source tracking and confidence scoring

### Limitations
- **Processing Time**: Initial document processing can take several minutes
- **Memory Usage**: Vector embeddings require significant RAM
- **API Dependencies**: Requires external LLM API access
- **Language Support**: Optimized for English and German medical texts

## 🧪 Testing & Evaluation

### System Testing
```bash
# Run the Streamlit app
streamlit run streamlit_app.py

# Use the "Test System" button to verify functionality
# Check system status and document information
```

### Query Testing
1. Initialize the system
2. Use sample queries or enter custom queries
3. Review response quality and source citations
4. Adjust confidence thresholds and retrieval parameters

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure at least one API key is set in `.env`
2. **Document Processing Failures**: Check PDF file integrity and permissions
3. **Memory Issues**: Reduce chunk size or use smaller documents
4. **LLM Timeouts**: Increase timeout values or use faster models

### Debug Mode
Enable detailed logging by modifying the logging level in each module:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Technical Details

### Dependencies
- **LangGraph**: Workflow orchestration and agentic behavior
- **LangChain**: Document processing and LLM integration
- **Chroma**: Vector database for similarity search
- **Sentence Transformers**: Text embedding generation
- **Streamlit**: Web-based user interface
- **Google Generative AI**: Gemini model integration

### System Requirements
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB+ for vector embeddings
- **Network**: Stable internet for LLM API access

## 🤝 Contributing

Contributions are welcome! Please consider:
- Bug reports and feature requests
- Code improvements and optimizations
- Documentation enhancements
- Additional LLM provider integrations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangGraph team for the workflow framework
- Google AI Studio for providing access to Gemini models
- Medical document providers for test data
- Open-source community for supporting libraries

---

**Note**: This system is designed for research and educational purposes. Medical information should always be verified by qualified healthcare professionals.
