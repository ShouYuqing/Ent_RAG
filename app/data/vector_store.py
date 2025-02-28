"""
Vector database interface for the Ent_RAG system.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

import numpy as np
from chromadb import PersistentClient, Collection
from chromadb.config import Settings

from app.config import config
from app.models.embeddings import EmbeddingManager

logger = logging.getLogger("ent_rag.data.vector_store")


class VectorStore:
    """
    Interface to vector database for storing and retrieving embeddings.
    """
    
    def __init__(
        self,
        collection_name: str = "ent_rag_documents",
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use
            embedding_manager: Optional embedding manager instance
        """
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()
        
        # Initialize ChromaDB client
        persist_directory = config.vector_db.chroma_persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            )
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def _embedding_function(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding function for ChromaDB.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        return self.embedding_manager.get_embeddings(texts)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Ensure metadatas is provided
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        # Add documents to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(texts)} documents to vector store")
        return ids
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        # Convert filters to ChromaDB format if provided
        where = self._convert_filters(filters) if filters else None
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results
    
    def delete(self, doc_id: str) -> None:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID to delete
        """
        self.collection.delete(ids=[doc_id])
        logger.info(f"Deleted document {doc_id} from vector store")
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        """
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
        
        # Recreate collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function
        )
        logger.info(f"Recreated collection: {self.collection_name}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary of collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count
        }
    
    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert filters to ChromaDB format.
        
        Args:
            filters: Filters in simplified format
            
        Returns:
            Filters in ChromaDB format
        """
        # Simple implementation - for more complex filters, this would need to be expanded
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict) and "$gt" in value:
                where_clause[key] = {"$gt": value["$gt"]}
            elif isinstance(value, dict) and "$lt" in value:
                where_clause[key] = {"$lt": value["$lt"]}
            else:
                where_clause[key] = {"$eq": value}
        
        return where_clause 