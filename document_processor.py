import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF document processing, extraction, and chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    
    def process_pdf_documents(self, data_dir: str = None) -> List[Document]:
        """Process all PDF documents in the data directory"""
        if data_dir is None:
            data_dir = Config.DATA_DIR
        
        documents = []
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}")
                text = self.extract_text_from_pdf(str(pdf_file))
                
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "file_path": str(pdf_file),
                        "file_size": pdf_file.stat().st_size,
                        "type": "pdf"
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for vector storage"""
        logger.info(f"Chunking {len(documents)} documents")
        
        chunked_docs = []
        for doc in documents:
            try:
                chunks = self.text_splitter.split_documents([doc])
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk.page_content)
                    })
                
                chunked_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.metadata.get('source', 'unknown')}: {e}")
                continue
        
        logger.info(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might interfere with processing
        text = text.replace('\x00', '')
        
        return text.strip()
    
    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate summary statistics for processed documents"""
        total_chunks = len(documents)
        total_text_length = sum(len(doc.page_content) for doc in documents)
        
        sources = {}
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        return {
            "total_documents": len(set(doc.metadata.get('source') for doc in documents)),
            "total_chunks": total_chunks,
            "total_text_length": total_text_length,
            "average_chunk_length": total_text_length / total_chunks if total_chunks > 0 else 0,
            "sources": sources
        }
