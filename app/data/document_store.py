"""
Document storage and retrieval for the Ent_RAG system.
"""

import logging
import json
import os
import uuid
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from app.config import config
from app.data.vector_store import VectorStore

logger = logging.getLogger("ent_rag.data.document_store")


class Document:
    """
    Represents a document in the system.
    """
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        Initialize a document.
        
        Args:
            content: The document content
            metadata: Optional metadata
            doc_id: Optional document ID (generated if not provided)
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            "id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """
        Create a document from a dictionary.
        
        Args:
            data: Dictionary representation of the document
            
        Returns:
            Document instance
        """
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("id")
        )


class DocumentStore:
    """
    Manages document storage and retrieval.
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the document store.
        
        Args:
            vector_store: Optional vector store instance
        """
        self.vector_store = vector_store or VectorStore()
        self.documents = {}
        self.document_dir = Path(config.vector_db.chroma_persist_directory).parent / "documents"
        
        # Create document directory if it doesn't exist
        self.document_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing documents
        self._load_documents()
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        embed: bool = True
    ) -> str:
        """
        Add a document to the store.
        
        Args:
            content: The document content
            metadata: Optional metadata
            doc_id: Optional document ID (generated if not provided)
            embed: Whether to embed the document
            
        Returns:
            Document ID
        """
        # Create document
        document = Document(content=content, metadata=metadata, doc_id=doc_id)
        
        # Add timestamp if not present
        if "timestamp" not in document.metadata:
            document.metadata["timestamp"] = datetime.now().isoformat()
        
        # Store document
        self.documents[document.doc_id] = document
        
        # Save document to disk
        self._save_document(document)
        
        # Embed document if requested
        if embed:
            self.vector_store.add_texts(
                texts=[content],
                metadatas=[document.metadata],
                ids=[document.doc_id]
            )
        
        logger.info(f"Added document with ID: {document.doc_id}")
        return document.doc_id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document instance or None if not found
        """
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if the document was deleted, False otherwise
        """
        if doc_id not in self.documents:
            return False
        
        # Remove from memory
        del self.documents[doc_id]
        
        # Remove from disk
        doc_path = self.document_dir / f"{doc_id}.json"
        if doc_path.exists():
            doc_path.unlink()
        
        # Remove from vector store
        self.vector_store.delete(doc_id)
        
        logger.info(f"Deleted document with ID: {doc_id}")
        return True
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Document]:
        """
        Search for documents.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            List of matching documents
        """
        # Search in vector store
        results = self.vector_store.search(
            query=query,
            filters=filters,
            top_k=top_k
        )
        
        # Convert to Document instances
        documents = []
        for result in results:
            doc_id = result["id"]
            document = self.get_document(doc_id)
            if document:
                documents.append(document)
        
        return documents
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the store.
        
        Returns:
            List of all documents
        """
        return list(self.documents.values())
    
    def count_documents(self) -> int:
        """
        Count the number of documents in the store.
        
        Returns:
            Number of documents
        """
        return len(self.documents)
    
    def _save_document(self, document: Document) -> None:
        """
        Save a document to disk.
        
        Args:
            document: Document to save
        """
        doc_path = self.document_dir / f"{document.doc_id}.json"
        with open(doc_path, "w") as f:
            json.dump(document.to_dict(), f)
    
    def _load_documents(self) -> None:
        """
        Load documents from disk.
        """
        if not self.document_dir.exists():
            return
        
        for doc_path in self.document_dir.glob("*.json"):
            try:
                with open(doc_path, "r") as f:
                    data = json.load(f)
                
                document = Document.from_dict(data)
                self.documents[document.doc_id] = document
            except Exception as e:
                logger.error(f"Error loading document from {doc_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.documents)} documents from disk") 