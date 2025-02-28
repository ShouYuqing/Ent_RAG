"""
Multi-stage retrieval implementation for the Ent_RAG system.

This module implements a multi-stage retrieval approach that combines pre-filtering
and reranking to improve retrieval accuracy and efficiency. It allows for a coarse-to-fine
approach where an initial broad retrieval is followed by more precise filtering and reranking.

Author: yuqings
Created: February 2024
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from collections import defaultdict

from app.config import config
from app.data.document_store import DocumentStore
from app.retrieval.base import BaseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranking import ReRanker

logger = logging.getLogger("ent_rag.retrieval.multi_stage")


class MultiStageRetriever(BaseRetriever):
    """
    Implements multi-stage retrieval with pre-filtering and reranking.
    
    This retriever uses a coarse-to-fine approach where an initial broad retrieval
    is followed by more precise filtering and reranking. This approach can improve
    both efficiency (by reducing the number of documents that need detailed processing)
    and accuracy (by applying more sophisticated ranking to a smaller set of candidates).
    
    Attributes:
        document_store: Storage for documents and their metadata
        base_retriever: Base retriever for initial document retrieval
        reranker: Reranker for improving relevance of retrieved documents
        pre_filter_multiplier: Multiplier for the number of documents to retrieve in the first stage
        use_metadata_filters: Whether to use metadata for filtering
        use_content_filters: Whether to use content-based filtering
    """
    
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        base_retriever: Optional[BaseRetriever] = None,
        reranker: Optional[ReRanker] = None,
        pre_filter_multiplier: int = 3,
        use_metadata_filters: bool = True,
        use_content_filters: bool = True
    ):
        """
        Initialize the multi-stage retriever with necessary components.
        
        Args:
            document_store: Storage for documents and their metadata
            base_retriever: Base retriever for initial document retrieval
            reranker: Reranker for improving relevance of retrieved documents
            pre_filter_multiplier: Multiplier for the number of documents to retrieve in the first stage
            use_metadata_filters: Whether to use metadata for filtering
            use_content_filters: Whether to use content-based filtering
        """
        self.document_store = document_store or DocumentStore()
        self.base_retriever = base_retriever or HybridRetriever(document_store=self.document_store)
        self.reranker = reranker or ReRanker()
        self.pre_filter_multiplier = pre_filter_multiplier
        self.use_metadata_filters = use_metadata_filters
        self.use_content_filters = use_content_filters
        
        logger.info(f"MultiStageRetriever initialized with pre_filter_multiplier: {pre_filter_multiplier}")
    
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using a multi-stage approach.
        
        This method:
        1. Performs an initial broad retrieval
        2. Applies pre-filtering based on metadata and content
        3. Reranks the filtered results
        
        Args:
            query: The search query
            filters: Metadata filters to apply
            top_k: Number of results to return
            **kwargs: Additional retriever-specific parameters
            
        Returns:
            List of retrieved documents with content and metadata
        """
        logger.debug(f"Multi-stage retrieval for query: '{query}' with filters: {filters}")
        
        # Stage 1: Initial broad retrieval
        initial_k = top_k * self.pre_filter_multiplier
        initial_results = self.base_retriever.retrieve(
            query=query,
            filters=filters,
            top_k=initial_k
        )
        
        logger.debug(f"Initial retrieval returned {len(initial_results)} documents")
        
        if not initial_results:
            return []
        
        # Stage 2: Pre-filtering
        filtered_results = self._apply_pre_filters(query, initial_results, filters)
        
        logger.debug(f"Pre-filtering reduced to {len(filtered_results)} documents")
        
        if not filtered_results:
            # If filtering removed all results, return initial results
            filtered_results = initial_results
        
        # Stage 3: Reranking
        reranked_results = self.reranker.rerank(
            query=query,
            documents=filtered_results,
            top_k=top_k
        )
        
        logger.debug(f"Reranking returned {len(reranked_results)} documents")
        
        return reranked_results
    
    def _apply_pre_filters(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply pre-filtering to the initial retrieval results.
        
        This method applies both metadata-based and content-based filtering
        to reduce the number of documents that need to be reranked.
        
        Args:
            query: The search query
            documents: Initial retrieval results
            filters: Metadata filters to apply
            
        Returns:
            Filtered list of documents
        """
        filtered_docs = documents
        
        # Apply metadata filters if enabled
        if self.use_metadata_filters:
            filtered_docs = self._apply_metadata_filters(filtered_docs, filters)
        
        # Apply content filters if enabled
        if self.use_content_filters:
            filtered_docs = self._apply_content_filters(query, filtered_docs)
        
        return filtered_docs
    
    def _apply_metadata_filters(
        self,
        documents: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata-based filtering to documents.
        
        This method filters documents based on their metadata attributes,
        such as source, author, date, etc.
        
        Args:
            documents: Documents to filter
            filters: Metadata filters to apply
            
        Returns:
            Filtered list of documents
        """
        if not filters:
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            include_doc = True
            
            for key, value in filters.items():
                # Skip special filter keys
                if key.startswith("_"):
                    continue
                
                # Check if metadata has the key
                if key not in metadata:
                    include_doc = False
                    break
                
                # Check if value matches
                if isinstance(value, list):
                    # List of possible values
                    if metadata[key] not in value:
                        include_doc = False
                        break
                elif isinstance(value, dict):
                    # Range or comparison
                    if "gt" in value and metadata[key] <= value["gt"]:
                        include_doc = False
                        break
                    if "gte" in value and metadata[key] < value["gte"]:
                        include_doc = False
                        break
                    if "lt" in value and metadata[key] >= value["lt"]:
                        include_doc = False
                        break
                    if "lte" in value and metadata[key] > value["lte"]:
                        include_doc = False
                        break
                else:
                    # Exact match
                    if metadata[key] != value:
                        include_doc = False
                        break
            
            if include_doc:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _apply_content_filters(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply content-based filtering to documents.
        
        This method filters documents based on their content,
        such as the presence of key terms, semantic similarity, etc.
        
        Args:
            query: The search query
            documents: Documents to filter
            
        Returns:
            Filtered list of documents
        """
        if not documents:
            return []
        
        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        
        if not key_terms:
            return documents
        
        # Score documents based on key term presence
        scored_docs = []
        
        for doc in documents:
            content = doc.get("content", "").lower()
            score = 0
            
            for term, weight in key_terms.items():
                if term in content:
                    # Increase score based on term frequency and weight
                    term_freq = content.count(term)
                    score += term_freq * weight
            
            # Only include documents with a positive score
            if score > 0:
                doc_copy = doc.copy()
                doc_copy["content_filter_score"] = score
                scored_docs.append(doc_copy)
        
        # If no documents passed the filter, return all documents
        if not scored_docs:
            return documents
        
        # Sort by content filter score
        scored_docs.sort(key=lambda x: x.get("content_filter_score", 0), reverse=True)
        
        return scored_docs
    
    def _extract_key_terms(self, query: str) -> Dict[str, float]:
        """
        Extract key terms from the query with their importance weights.
        
        This method identifies the most important terms in the query
        that should be used for content filtering.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary mapping key terms to their importance weights
        """
        # Simple implementation: split query into terms and assign equal weights
        terms = query.lower().split()
        stop_words = {"a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "to", "and", "or", "is", "are"}
        
        # Remove stop words and short terms
        filtered_terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        # Assign weights based on term length and position
        term_weights = {}
        
        for i, term in enumerate(filtered_terms):
            # Terms at the beginning are often more important
            position_weight = 1.0 - (i / len(filtered_terms)) * 0.5
            
            # Longer terms are often more specific and important
            length_weight = min(1.0, len(term) / 10.0)
            
            # Combine weights
            weight = position_weight * length_weight
            
            term_weights[term] = weight
        
        return term_weights
    
    def _diversify_results(
        self,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Diversify results to avoid redundancy.
        
        This method ensures that the final results cover a diverse range of topics
        rather than focusing on a single aspect of the query.
        
        Args:
            documents: Documents to diversify
            top_k: Number of results to return
            
        Returns:
            Diversified list of documents
        """
        if len(documents) <= top_k:
            return documents
        
        # Group documents by source/category
        groups = defaultdict(list)
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            # Use source or category as group key
            group_key = metadata.get("source", metadata.get("category", "default"))
            groups[group_key].append(doc)
        
        # Select documents from each group in a round-robin fashion
        diversified = []
        group_keys = list(groups.keys())
        
        while len(diversified) < top_k and group_keys:
            for key in list(group_keys):  # Use a copy to allow removal
                if groups[key]:
                    # Take the highest-ranked document from this group
                    diversified.append(groups[key].pop(0))
                    
                    if len(diversified) >= top_k:
                        break
                
                if not groups[key]:
                    # Remove empty groups
                    group_keys.remove(key)
        
        # If we still need more documents, take from the original list
        if len(diversified) < top_k:
            remaining = [doc for doc in documents if doc not in diversified]
            diversified.extend(remaining[:top_k - len(diversified)])
        
        return diversified 