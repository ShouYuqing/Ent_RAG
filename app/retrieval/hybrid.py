"""
Hybrid retrieval implementation that combines vector search, keyword search, and metadata filtering.

This module implements a hybrid retrieval approach that leverages both semantic (vector)
and lexical (keyword) search methods, along with metadata filtering to provide
comprehensive and accurate document retrieval.

Author: yuqings
Created: February 2024
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize

from app.config import config
from app.data.document_store import DocumentStore
from app.models.embeddings import EmbeddingManager
from app.retrieval.base import BaseRetriever

logger = logging.getLogger("ent_rag.retrieval.hybrid")


class HybridRetriever(BaseRetriever):
    """
    Implements hybrid retrieval combining vector search, keyword search, and metadata filtering.
    
    This retriever uses a weighted combination of vector similarity and BM25 keyword matching
    to find the most relevant documents for a given query. It also supports metadata filtering
    to narrow down the search space based on document attributes.
    
    Attributes:
        document_store: Storage for documents and their metadata
        embedding_manager: Manager for embedding models
        hybrid_weight: Weight for balancing vector and keyword search (0-1)
    """
    
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        hybrid_weight: Optional[float] = None
    ):
        """
        Initialize the hybrid retriever with necessary components.
        
        Args:
            document_store: Storage for documents and their metadata
            embedding_manager: Manager for embedding models
            hybrid_weight: Weight for balancing vector and keyword search (0-1)
        """
        self.document_store = document_store or DocumentStore()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.hybrid_weight = hybrid_weight or config.retrieval.hybrid_search_weight
        logger.info(f"HybridRetriever initialized with weight: {self.hybrid_weight}")
    
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search combining vector and keyword search with metadata filtering.
        
        Args:
            query: The search query
            filters: Metadata filters to apply
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (if False, only vector search is used)
            
        Returns:
            List of retrieved documents with content and metadata
        """
        logger.debug(f"Retrieving documents for query: '{query}' with filters: {filters}")
        
        # Get expanded top_k for hybrid search
        expanded_k = top_k * 3 if use_hybrid else top_k
        
        # Perform vector search
        vector_results = self._vector_search(query, filters, expanded_k)
        
        if not use_hybrid:
            # If hybrid search is disabled, return vector search results directly
            return self._format_results(vector_results, query)
        
        # Perform keyword search
        keyword_results = self._keyword_search(query, filters, expanded_k)
        
        # Combine results with hybrid scoring
        combined_results = self._combine_results(
            query, vector_results, keyword_results, top_k
        )
        
        return self._format_results(combined_results, query)
    
    def _vector_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search using embeddings.
        
        Args:
            query: The search query
            filters: Metadata filters to apply
            top_k: Number of results to return
            
        Returns:
            List of vector search results
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Search vector store
        results = self.document_store.vector_store.search(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k
        )
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using BM25.
        
        Args:
            query: The search query
            filters: Metadata filters to apply
            top_k: Number of results to return
            
        Returns:
            List of keyword search results
        """
        # Get all documents that match filters
        documents = self.document_store.get_documents(filters=filters, limit=1000)
        
        if not documents:
            return []
        
        # Prepare corpus for BM25
        corpus = [doc.content for doc in documents]
        tokenized_corpus = [doc.content.split() for doc in documents]
        
        # Create BM25 model
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Tokenize query and get scores
        tokenized_query = query.split()
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top_k results
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        
        # Create results list
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:  # Only include results with non-zero scores
                results.append({
                    "id": documents[idx].id,
                    "content": documents[idx].content,
                    "metadata": documents[idx].metadata,
                    "score": float(doc_scores[idx])
                })
        
        return results
    
    def _combine_results(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results with weighted scoring.
        
        Args:
            query: The original query
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            top_k: Number of results to return
            
        Returns:
            Combined and reranked results
        """
        # Create a mapping of document IDs to results
        combined_dict = {}
        
        # Normalize vector scores (higher is better)
        if vector_results:
            vector_scores = np.array([r["score"] for r in vector_results])
            min_score, max_score = vector_scores.min(), vector_scores.max()
            score_range = max_score - min_score
            
            if score_range > 0:
                for r in vector_results:
                    normalized_score = (r["score"] - min_score) / score_range
                    combined_dict[r["id"]] = {
                        "id": r["id"],
                        "content": r["content"],
                        "metadata": r["metadata"],
                        "vector_score": normalized_score,
                        "keyword_score": 0.0,
                        "combined_score": normalized_score * self.hybrid_weight
                    }
        
        # Normalize keyword scores (higher is better)
        if keyword_results:
            keyword_scores = np.array([r["score"] for r in keyword_results])
            min_score, max_score = keyword_scores.min(), keyword_scores.max()
            score_range = max_score - min_score
            
            if score_range > 0:
                for r in keyword_results:
                    normalized_score = (r["score"] - min_score) / score_range
                    if r["id"] in combined_dict:
                        combined_dict[r["id"]]["keyword_score"] = normalized_score
                        combined_dict[r["id"]]["combined_score"] += normalized_score * (1 - self.hybrid_weight)
                    else:
                        combined_dict[r["id"]] = {
                            "id": r["id"],
                            "content": r["content"],
                            "metadata": r["metadata"],
                            "vector_score": 0.0,
                            "keyword_score": normalized_score,
                            "combined_score": normalized_score * (1 - self.hybrid_weight)
                        }
        
        # Convert dictionary to list and sort by combined score
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top_k results
        return combined_results[:top_k]
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Format search results for consistent output.
        
        Args:
            results: Search results to format
            query: The original query
            
        Returns:
            Formatted search results
        """
        formatted_results = []
        
        for result in results:
            # Ensure consistent result format
            formatted_result = {
                "id": result["id"],
                "content": result["content"],
                "metadata": result["metadata"],
                "score": result.get("combined_score", result.get("score", 0.0)),
                "relevance": self._calculate_relevance(query, result["content"])
            }
            
            # Add to formatted results
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate a simple relevance score based on term overlap.
        
        Args:
            query: The search query
            content: The document content
            
        Returns:
            Relevance score between 0 and 1
        """
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_terms.intersection(content_terms)
        union = query_terms.union(content_terms)
        
        return len(intersection) / len(query_terms) 