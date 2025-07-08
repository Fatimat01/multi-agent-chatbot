from typing import Dict, Any, List, Optional, Tuple
import os
import logging
import hashlib
from datetime import datetime
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from utils import Message, timer
import pandas as pd
from tqdm import tqdm


class DocumentLoaderAgent:
    """Agent responsible for document loading, chunking, and vector store management."""
    
    def __init__(self, 
                 data_path: str = "../data/complaints.csv",
                 vectorstore_path: str = "chroma_db",
                 chunk_size: int = 2000,
                 chunk_overlap: int = 500,
                 batch_size: int = 500):
        """Initialize document loader with configurable parameters.
        
        Args:
            data_path: Path to CSV data file
            vectorstore_path: Path to ChromaDB storage
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for vectorstore operations
        """
        self.data_path = data_path
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Initialize components
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len
        )
        
        # Track loading statistics
        self.stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "duplicates_skipped": 0,
            "load_time": 0,
            "last_update": None
        }
        
    @timer
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process document loading request."""
        command = state.get("loader_command", "check_status")
        
        if command == "load_data":
            result = self.load_and_chunk_data()
        elif command == "update_vectorstore":
            result = self.update_vectorstore()
        elif command == "check_status":
            result = self.check_vectorstore_status()
        elif command == "optimize_chunks":
            result = self.optimize_chunking_strategy()
        else:
            result = {"status": "error", "message": f"Unknown command: {command}"}
        
        # Add result to state
        msg = Message("DocumentLoaderAgent", "ResponseAgent", result)
        state["messages"].append(msg.to_dict())
        state["loader_result"] = result
        
        return state
    
    def load_and_chunk_data(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Load CSV data and create optimized chunks."""
        file_path = file_path or self.data_path
        
        try:
            logging.info(f"Loading data from {file_path}")
            
            # Load documents
            documents = self._csv_loader(file_path)
            self.stats["documents_loaded"] = len(documents)
            
            # Enhance documents with metadata
            enhanced_docs = self._enhance_document_metadata(documents)
            
            # Create chunks with optimized strategy
            chunks = self._create_optimized_chunks(enhanced_docs)
            self.stats["chunks_created"] = len(chunks)
            
            # Store in vectorstore
            vectorstore = self._get_or_create_vectorstore()
            self._add_chunks_to_vectorstore(vectorstore, chunks)
            
            return {
                "status": "success",
                "documents_loaded": self.stats["documents_loaded"],
                "chunks_created": self.stats["chunks_created"],
                "message": f"Successfully loaded {len(documents)} documents into {len(chunks)} chunks"
            }
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _csv_loader(self, file_path: str) -> List[Document]:
        """Load CSV file and return documents."""
        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8",
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': None  # Auto-detect from first row
            }
        )
        
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from {file_path}")
        
        return documents
    
    def _enhance_document_metadata(self, documents: List[Document]) -> List[Document]:
        """Enhance documents with additional metadata for better retrieval."""
        enhanced_docs = []
        
        for doc in tqdm(documents, desc="Enhancing metadata"):
            # Extract key fields from content
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Parse content to extract structured data
            lines = content.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Map common fields
                    if key in ['company', 'product', 'issue', 'state', 'zip_code']:
                        metadata[key] = value
                    elif key == 'date_received':
                        metadata['date'] = value
                        # Extract year and month for temporal queries
                        try:
                            date_obj = pd.to_datetime(value)
                            metadata['year'] = date_obj.year
                            metadata['month'] = date_obj.month
                        except:
                            pass
            
            # Add document hash for deduplication
            metadata['content_hash'] = hashlib.md5(content.encode()).hexdigest()
            
            # Create enhanced document
            enhanced_doc = Document(
                page_content=content,
                metadata=metadata
            )
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs
    
    def _create_optimized_chunks(self, documents: List[Document]) -> List[Document]:
        """Create optimized chunks for better semantic retrieval."""
        all_chunks = []
        
        for doc in tqdm(documents, desc="Creating chunks"):
            # For complaint data, we want to keep related information together
            content = doc.page_content
            
            # Check content length
            if len(content) <= self.chunk_size:
                # Small documents - keep as single chunk
                chunk = Document(
                    page_content=content,
                    metadata={**doc.metadata, "chunk_type": "full_document"}
                )
                all_chunks.append(chunk)
            else:
                # Large documents - intelligent splitting
                # Try to split on natural boundaries
                chunks = self.splitter.split_documents([doc])
                
                for i, chunk in enumerate(chunks):
                    # Add chunk-specific metadata
                    chunk.metadata.update({
                        **doc.metadata,
                        "chunk_type": "partial",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                
                all_chunks.extend(chunks)
        
        logging.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Deduplicate chunks
        unique_chunks = self._deduplicate_chunks(all_chunks)
        self.stats["duplicates_skipped"] = len(all_chunks) - len(unique_chunks)
        
        return unique_chunks
    
    def _deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove duplicate chunks based on content hash."""
        seen_hashes = set()
        unique_chunks = []
        
        for chunk in chunks:
            content_hash = chunk.metadata.get('content_hash')
            if not content_hash:
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
                chunk.metadata['content_hash'] = content_hash
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _get_or_create_vectorstore(self) -> Chroma:
        """Get existing vectorstore or create new one."""
        if os.path.exists(self.vectorstore_path):
            logging.info(f"Loading existing vectorstore from {self.vectorstore_path}")
            vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embedding
            )
        else:
            logging.info(f"Creating new vectorstore at {self.vectorstore_path}")
            os.makedirs(self.vectorstore_path, exist_ok=True)
            vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embedding
            )
        
        return vectorstore
    
    def _add_chunks_to_vectorstore(self, vectorstore: Chroma, chunks: List[Document]):
        """Add chunks to vectorstore in batches."""
        total_chunks = len(chunks)
        
        for i in tqdm(range(0, total_chunks, self.batch_size), desc="Adding to vectorstore"):
            batch = chunks[i:i + self.batch_size]
            
            # Extract texts and metadatas
            texts = [chunk.page_content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            
            # Add to vectorstore
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
        
        # Persist changes
        vectorstore.persist()
        logging.info(f"Added {total_chunks} chunks to vectorstore")
    
    def update_vectorstore(self) -> Dict[str, Any]:
        """Update vectorstore with new data."""
        try:
            # Check for new data
            if not os.path.exists(self.data_path):
                return {
                    "status": "error",
                    "message": f"Data file not found: {self.data_path}"
                }
            
            # Get file modification time
            file_mtime = os.path.getmtime(self.data_path)
            last_update = self.stats.get("last_update", 0)
            
            if file_mtime <= last_update:
                return {
                    "status": "no_update",
                    "message": "Data file has not been modified since last update"
                }
            
            # Load and update
            result = self.load_and_chunk_data()
            self.stats["last_update"] = file_mtime
            
            return result
            
        except Exception as e:
            logging.error(f"Error updating vectorstore: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def check_vectorstore_status(self) -> Dict[str, Any]:
        """Check current status of vectorstore."""
        try:
            if not os.path.exists(self.vectorstore_path):
                return {
                    "status": "not_initialized",
                    "message": "Vectorstore not found. Run load_data first."
                }
            
            vectorstore = self._get_or_create_vectorstore()
            collection = vectorstore._collection
            doc_count = collection.count()
            
            # Get sample metadata
            sample_metadata = {}
            if doc_count > 0:
                sample = collection.peek(1)
                if sample and sample.get('metadatas'):
                    sample_metadata = sample['metadatas'][0]
            
            return {
                "status": "ready",
                "document_count": doc_count,
                "vectorstore_path": self.vectorstore_path,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "sample_metadata_keys": list(sample_metadata.keys()),
                "stats": self.stats
            }
            
        except Exception as e:
            logging.error(f"Error checking vectorstore status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def optimize_chunking_strategy(self) -> Dict[str, Any]:
        """Analyze and optimize chunking strategy based on data characteristics."""
        try:
            # Load sample of documents
            documents = self._csv_loader(self.data_path)[:100]  # Sample first 100
            
            # Analyze document lengths
            lengths = [len(doc.page_content) for doc in documents]
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            
            # Recommend optimal chunk size
            if avg_length < 500:
                recommended_chunk_size = 1000
                recommended_overlap = 200
            elif avg_length < 2000:
                recommended_chunk_size = 2000
                recommended_overlap = 500
            else:
                recommended_chunk_size = 3000
                recommended_overlap = 750
            
            return {
                "status": "success",
                "analysis": {
                    "avg_doc_length": int(avg_length),
                    "max_doc_length": max_length,
                    "min_doc_length": min_length,
                    "current_chunk_size": self.chunk_size,
                    "recommended_chunk_size": recommended_chunk_size,
                    "recommended_overlap": recommended_overlap
                },
                "message": f"Recommended chunk size: {recommended_chunk_size} with {recommended_overlap} overlap"
            }
            
        except Exception as e:
            logging.error(f"Error optimizing chunking strategy: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
