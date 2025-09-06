import os
import logging
from typing import Dict, Any, List
from pathlib import Path
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_agent import RAGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages the complete RAG system initialization and coordination"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.rag_agent = None
        self.is_initialized = False
        self.documents_processed = False
        
    def initialize_system(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the complete RAG system"""
        try:
            logger.info("Initializing RAG system...")
            
            # Validate configuration
            Config.validate_config()
            
            # Check if vector store already exists
            if not force_rebuild and self.vector_store.load_existing_vector_store():
                logger.info("Loaded existing vector store")
                self.documents_processed = True
            else:
                # Process documents and create vector store
                if force_rebuild:
                    logger.info("Force rebuild requested, reprocessing documents")
                
                success = self._process_documents()
                if not success:
                    raise Exception("Document processing failed")
                
                self.documents_processed = True
            
            # Initialize RAG agent
            self.rag_agent = RAGAgent()
            
            self.is_initialized = True
            logger.info("RAG system initialization completed successfully")
            
            return {
                "success": True,
                "message": "System initialized successfully",
                "documents_processed": self.documents_processed,
                "vector_store_loaded": True
            }
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "System initialization failed"
            }
    
    def _process_documents(self) -> bool:
        """Process PDF documents and create vector store"""
        try:
            logger.info("Processing PDF documents...")
            
            # Process all PDF documents
            documents = self.document_processor.process_pdf_documents()
            
            if not documents:
                logger.error("No documents found to process")
                return False
            
            # Get document summary
            summary = self.document_processor.get_document_summary(documents)
            logger.info(f"Document processing summary: {summary}")
            
            # Chunk documents
            chunked_docs = self.document_processor.chunk_documents(documents)
            
            if not chunked_docs:
                logger.error("Document chunking failed")
                return False
            
            # Create vector store
            self.vector_store.create_vector_store(chunked_docs)
            
            logger.info(f"Successfully processed {len(documents)} documents into {len(chunked_docs)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the RAG system"""
        if not self.is_initialized:
            return {
                "success": False,
                "error": "System not initialized",
                "message": "Please initialize the system first"
            }
        
        try:
            logger.info(f"Processing query: {query}")
            result = self.rag_agent.process_query(query)
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "message": "Query processing failed"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "system_initialized": self.is_initialized,
                "documents_processed": self.documents_processed,
                "config": {
                    "data_directory": Config.DATA_DIR,
                    "vector_store_directory": Config.VECTOR_STORE_DIR,
                    "chunk_size": Config.CHUNK_SIZE,
                    "chunk_overlap": Config.CHUNK_OVERLAP,
                    "top_k_retrieval": Config.TOP_K_RETRIEVAL,
                    "confidence_threshold": Config.CONFIDENCE_THRESHOLD
                }
            }
            
            if self.is_initialized and self.rag_agent:
                agent_status = self.rag_agent.get_system_status()
                status.update(agent_status)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def rebuild_vector_store(self) -> Dict[str, Any]:
        """Rebuild the vector store from scratch"""
        try:
            logger.info("Rebuilding vector store...")
            
            # Delete existing vector store
            if self.vector_store.vector_store:
                self.vector_store.delete_collection()
            
            # Reprocess documents
            success = self._process_documents()
            
            if success:
                # Reinitialize RAG agent
                self.rag_agent = RAGAgent()
                logger.info("Vector store rebuilt successfully")
                
                return {
                    "success": True,
                    "message": "Vector store rebuilt successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Document processing failed during rebuild"
                }
                
        except Exception as e:
            logger.error(f"Vector store rebuild failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about processed documents"""
        try:
            if not self.documents_processed:
                return {"error": "No documents processed yet"}
            
            # Get vector store statistics
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get document files info
            data_dir = Path(Config.DATA_DIR)
            pdf_files = list(data_dir.glob("*.pdf"))
            
            document_info = []
            for pdf_file in pdf_files:
                try:
                    stat = pdf_file.stat()
                    document_info.append({
                        "filename": pdf_file.name,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "last_modified": stat.st_mtime
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {pdf_file}: {e}")
            
            return {
                "total_pdf_files": len(pdf_files),
                "documents": document_info,
                "vector_store": vector_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {"error": str(e)}
    
    def test_system(self) -> Dict[str, Any]:
        """Run a test query to verify system functionality"""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        try:
            test_query = "What is the main topic of these documents?"
            logger.info("Running system test with sample query")
            
            result = self.process_query(test_query)
            
            return {
                "test_query": test_query,
                "result": result,
                "system_working": result.get("success", False)
            }
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            return {
                "error": str(e),
                "system_working": False
            }

    # pass  # Ensure the class is not empty; add your implementation here
