import streamlit as st
import time
import json
from pathlib import Path
from system_manager import SystemManager
from config import Config

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🧠 Agentic RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Medical Document Analysis with LangGraph & Autonomous AI")
    
    # Initialize session state
    if 'system_manager' not in st.session_state:
        st.session_state.system_manager = SystemManager()
        st.session_state.system_initialized = False
        st.session_state.query_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 System Control")
        
        # System initialization
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                result = st.session_state.system_manager.initialize_system()
                if result["success"]:
                    st.session_state.system_initialized = True
                    st.success("System initialized successfully!")
                else:
                    st.error(f"Initialization failed: {result.get('error', 'Unknown error')}")
        
        # Force rebuild option
        if st.checkbox("Force Rebuild Vector Store"):
            if st.button("🔄 Rebuild System"):
                with st.spinner("Rebuilding vector store..."):
                    result = st.session_state.system_manager.rebuild_vector_store()
                    if result["success"]:
                        st.success("Vector store rebuilt successfully!")
                    else:
                        st.error(f"Rebuild failed: {result.get('error', 'Unknown error')}")
        
        # System status
        st.markdown("## 📊 System Status")
        if st.session_state.system_initialized:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System Not Initialized")
        
        # Configuration info
        st.markdown("## ⚙️ Configuration")
        st.info(f"**Chunk Size:** {Config.CHUNK_SIZE}")
        st.info(f"**Chunk Overlap:** {Config.CHUNK_OVERLAP}")
        st.info(f"**Top-K Retrieval:** {Config.TOP_K_RETRIEVAL}")
        st.info(f"**Confidence Threshold:** {Config.CONFIDENCE_THRESHOLD}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">💬 Query Interface</h2>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Enter your medical query:",
            height=100,
            placeholder="e.g., What are the findings of the critical-care pain observation tool study?"
        )
        
        # Query options
        col1a, col1b = st.columns(2)
        with col1a:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=Config.CONFIDENCE_THRESHOLD,
                step=0.1
            )
        
        with col1b:
            top_k = st.slider(
                "Top-K Retrieval",
                min_value=1,
                max_value=10,
                value=Config.TOP_K_RETRIEVAL
            )
        
        # Process query button
        if st.button("🔍 Process Query", type="primary", disabled=not st.session_state.system_initialized):
            if query.strip():
                process_query(query, confidence_threshold, top_k)
            else:
                st.warning("Please enter a query first.")
        
        # Display results
        if 'current_result' in st.session_state and st.session_state.current_result:
            display_query_result(st.session_state.current_result)
    
    with col2:
        st.markdown('<h2 class="sub-header">📚 Sample Queries</h2>', unsafe_allow_html=True)
        
        sample_queries = [
            "What are the findings of this study Sensitivity and specificity of the critical-care pain observation tool for the detection of pain in intubated adults after cardiac surgery?",
            "Was ist die Evidenzbasierte Empfehlung 3. E 39, was sind ihre Empfehlungen?",
            "Warum sehen die Autoren in diesem Absatz keinen Anlass, die Empfehlung 60 aufgrund der Kommentare von Thaler und Krüssel zu ändern?",
            "Findest du, dass Künstliche Intelligenz in Zukunft Ärztinnen und Ärzte bei der Diagnose von seltenen Krankheiten besser unterstützen kann?"
        ]
        
        for i, sample_query in enumerate(sample_queries):
            if st.button(f"Query {i+1}", key=f"sample_{i}"):
                st.session_state.query_input = sample_query
                st.rerun()
        
        # Quick actions
        st.markdown('<h3 class="sub-header">⚡ Quick Actions</h3>', unsafe_allow_html=True)
        
        if st.button("🧪 Test System"):
            test_system()
        
        if st.button("📊 System Status"):
            show_system_status()
        
        if st.button("📁 Document Info"):
            show_document_info()

def process_query(query, confidence_threshold, top_k):
    """Process a user query and display results"""
    st.markdown("---")
    st.markdown(f"### 🔍 Processing Query: {query}")
    
    with st.spinner("Processing query through RAG system..."):
        # Update configuration temporarily
        Config.CONFIDENCE_THRESHOLD = confidence_threshold
        Config.TOP_K_RETRIEVAL = top_k
        
        # Process query
        result = st.session_state.system_manager.process_query(query)
        
        # Store result in session state
        st.session_state.current_result = result
        
        # Add to query history
        st.session_state.query_history.append({
            "query": query,
            "result": result,
            "timestamp": time.time()
        })
        
        # Display immediate feedback
        if result["success"]:
            st.success("✅ Query processed successfully!")
        else:
            st.error(f"❌ Query processing failed: {result.get('error', 'Unknown error')}")

def display_query_result(result):
    """Display the results of a processed query"""
    st.markdown("---")
    st.markdown("### 📋 Query Results")
    
    if not result["success"]:
        st.error(f"**Error:** {result.get('error', 'Unknown error')}")
        return
    
    # Response
    st.markdown("#### 💡 Generated Response")
    st.markdown(result["response"])
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence", f"{result['confidence']:.2f}")
    
    with col2:
        st.metric("Sources Used", len(result["sources"]))
    
    with col3:
        st.metric("Workflow Steps", result.get("workflow_steps", "N/A"))
    
    # Sources
    if result["sources"]:
        st.markdown("#### 📚 Source Documents")
        for source in result["sources"]:
            st.info(f"📄 {source}")
    
    # Evaluation details
    if result.get("evaluation"):
        st.markdown("#### 🔍 Quality Evaluation")
        with st.expander("View Evaluation Details"):
            for key, value in result["evaluation"].items():
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                st.markdown(value)
                st.markdown("---")

def test_system():
    """Run a system test"""
    st.markdown("---")
    st.markdown("### 🧪 System Test")
    
    with st.spinner("Running system test..."):
        result = st.session_state.system_manager.test_system()
        
        if "error" in result:
            st.error(f"Test failed: {result['error']}")
        else:
            st.success("✅ System test completed!")
            st.json(result)

def show_system_status():
    """Display comprehensive system status"""
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    with st.spinner("Gathering system information..."):
        status = st.session_state.system_manager.get_system_status()
        
        if "error" in status:
            st.error(f"Error getting status: {status['error']}")
        else:
            st.json(status)

def show_document_info():
    """Display information about processed documents"""
    st.markdown("---")
    st.markdown("### 📁 Document Information")
    
    with st.spinner("Gathering document information..."):
        doc_info = st.session_state.system_manager.get_document_info()
        
        if "error" in doc_info:
            st.error(f"Error getting document info: {doc_info['error']}")
        else:
            st.json(doc_info)

# Check if query input was set from sample queries
if 'query_input' in st.session_state:
    st.session_state.query_input = ""
    st.rerun()

if __name__ == "__main__":
    main()
