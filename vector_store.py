import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector storage and similarity search for documents"""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.collection_name = "medical_documents"
        
        # Ensure vector store directory exists
        os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create and populate the vector store with document embeddings"""
        try:
            logger.info("Creating vector store with document embeddings")
            
            # Create Chroma vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=Config.VECTOR_STORE_DIR,
                collection_name=self.collection_name
            )
            
            # Persist the vector store
            self.vector_store.persist()
            logger.info(f"Vector store created with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_existing_vector_store(self) -> bool:
        """Load existing vector store if available"""
        try:
            if os.path.exists(Config.VECTOR_STORE_DIR):
                # Check if the directory has actual data
                if os.path.exists(os.path.join(Config.VECTOR_STORE_DIR, "chroma.sqlite3")):
                    self.vector_store = Chroma(
                        persist_directory=Config.VECTOR_STORE_DIR,
                        embedding_function=self.embedding_model,
                        collection_name=self.collection_name
                    )
                    # Verify the vector store is working
                    try:
                        # Test if we can access the collection
                        _ = self.vector_store._collection.count()
                        logger.info("Loaded existing vector store successfully")
                        return True
                    except Exception as e:
                        logger.warning(f"Vector store loaded but collection access failed: {e}")
                        self.vector_store = None
                        return False
                else:
                    logger.info("Vector store directory exists but no data found")
                    return False
            return False
        except Exception as e:
            logger.error(f"Error loading existing vector store: {e}")
            self.vector_store = None
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search for a given query"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if k is None:
            k = Config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            logger.info(f"Retrieved {len(results)} relevant documents for query")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if k is None:
            k = Config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            logger.info(f"Retrieved {len(results)} relevant documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        if self.vector_store is None:
            return {"error": "Vector store not initialized"}
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_dimension": len(self.embedding_model.embed_query("test")),
                "embedding_model": Config.EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Added {len(documents)} new documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the entire collection (use with caution)"""
        if self.vector_store is None:
            return
        
        try:
            self.vector_store._collection.delete()
            logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a specific document by its ID"""
        if self.vector_store is None:
            return None
        
        try:
            # This is a simplified approach - in practice you might need to implement
            # a more sophisticated ID-based retrieval system
            results = self.vector_store.similarity_search("", k=1000)  # Get all docs
            for doc in results:
                if doc.metadata.get("chunk_id") == doc_id:
                    return doc
            return None
        except Exception as e:
            logger.error(f"Error retrieving document by ID: {e}")
            return None
