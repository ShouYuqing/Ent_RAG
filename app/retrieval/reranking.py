"""
Reranking implementation for the Ent_RAG system.

This module implements document reranking to improve the relevance of retrieved documents
by applying more sophisticated relevance scoring after the initial retrieval.

Author: yuqings
Created: February 2024
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import config
from app.models.llm import LLMManager

logger = logging.getLogger("ent_rag.retrieval.reranking")


class ReRanker:
    """
    Implements document reranking to improve retrieval relevance.
    
    This class provides methods to rerank retrieved documents using various techniques,
    including TF-IDF similarity, cross-encoder models, or LLM-based reranking.
    
    Attributes:
        llm_manager: Manager for LLM interactions
        reranking_method: Method to use for reranking
    """
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        reranking_method: str = "tfidf"
    ):
        """
        Initialize the reranker with necessary components.
        
        Args:
            llm_manager: Manager for LLM interactions
            reranking_method: Method to use for reranking ('tfidf', 'cross_encoder', 'llm')
        """
        self.llm_manager = llm_manager or LLMManager()
        self.reranking_method = reranking_method
        logger.info(f"ReRanker initialized with method: {reranking_method}")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of documents to return after reranking
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Set default top_k if not provided
        if top_k is None:
            top_k = len(documents)
        
        # Choose reranking method
        if self.reranking_method == "tfidf":
            reranked_docs = self._rerank_tfidf(query, documents)
        elif self.reranking_method == "cross_encoder":
            reranked_docs = self._rerank_cross_encoder(query, documents)
        elif self.reranking_method == "llm":
            reranked_docs = self._rerank_llm(query, documents)
        else:
            logger.warning(f"Unknown reranking method: {self.reranking_method}, falling back to TF-IDF")
            reranked_docs = self._rerank_tfidf(query, documents)
        
        # Return top_k documents
        return reranked_docs[:top_k]
    
    def _rerank_tfidf(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using TF-IDF similarity.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        # Extract document contents
        doc_contents = [doc["content"] for doc in documents]
        
        # Fit and transform documents
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_contents + [query])
        except ValueError:
            # If vectorization fails, return documents in original order
            logger.warning("TF-IDF vectorization failed, returning documents in original order")
            return documents
        
        # Get query vector (last row in the matrix)
        query_vector = tfidf_matrix[-1]
        
        # Calculate similarity between query and each document
        similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
        
        # Create list of (index, similarity) pairs and sort by similarity
        doc_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Rerank documents
        reranked_docs = []
        for idx, sim in doc_similarities:
            doc = documents[idx].copy()
            doc["rerank_score"] = float(sim)
            doc["score"] = float(sim)  # Update the main score
            reranked_docs.append(doc)
        
        return reranked_docs
    
    def _rerank_cross_encoder(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using a cross-encoder model.
        
        This is a placeholder for cross-encoder implementation.
        In a production system, this would use a model like MS MARCO or BERT-based cross-encoder.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        logger.warning("Cross-encoder reranking not implemented, falling back to TF-IDF")
        return self._rerank_tfidf(query, documents)
    
    def _rerank_llm(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using LLM-based relevance assessment.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        # If too many documents, use TF-IDF to pre-filter
        if len(documents) > 10:
            logger.info(f"Too many documents ({len(documents)}) for LLM reranking, pre-filtering with TF-IDF")
            documents = self._rerank_tfidf(query, documents)[:10]
        
        # Prepare prompt for LLM
        prompt = self._create_reranking_prompt(query, documents)
        
        try:
            # Get LLM response
            response = self.llm_manager.generate(prompt=prompt, temperature=0.0)
            
            # Parse response to get document rankings
            reranked_docs = self._parse_llm_rankings(response, documents)
            
            # If parsing fails, fall back to TF-IDF
            if not reranked_docs:
                logger.warning("Failed to parse LLM rankings, falling back to TF-IDF")
                return self._rerank_tfidf(query, documents)
            
            return reranked_docs
        
        except Exception as e:
            logger.error(f"Error in LLM reranking: {str(e)}")
            return self._rerank_tfidf(query, documents)
    
    def _create_reranking_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Create a prompt for LLM-based reranking.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            Prompt for the LLM
        """
        prompt = f"""You are an expert at determining the relevance of documents to a query.
        
Query: {query}

Below are {len(documents)} documents. Rate each document's relevance to the query on a scale of 0-10, where 10 is most relevant.
For each document, provide a single line with the document ID and score in this format: "DOC_ID: SCORE"

"""
        
        for i, doc in enumerate(documents):
            # Truncate content if too long
            content = doc["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            
            prompt += f"\nDocument {i+1} (ID: {doc['id']}):\n{content}\n"
        
        prompt += "\nRelevance scores (one per line):\n"
        
        return prompt
    
    def _parse_llm_rankings(
        self,
        response: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract document rankings.
        
        Args:
            response: LLM response containing rankings
            documents: Original list of documents
            
        Returns:
            Reranked list of documents
        """
        # Create a mapping of document IDs to documents
        doc_map = {doc["id"]: doc for doc in documents}
        
        # Extract rankings from response
        rankings = []
        
        try:
            # Split response into lines
            lines = response.strip().split("\n")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to extract document ID and score
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                
                doc_id = parts[0].strip()
                score_str = parts[1].strip()
                
                # Try to convert score to float
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                
                # Check if document ID exists
                if doc_id in doc_map:
                    rankings.append((doc_id, score))
                # Check if it's a numeric index
                elif doc_id.isdigit() and int(doc_id) <= len(documents):
                    idx = int(doc_id) - 1
                    rankings.append((documents[idx]["id"], score))
            
            # If no valid rankings found, return empty list
            if not rankings:
                return []
            
            # Sort by score in descending order
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            # Create reranked document list
            reranked_docs = []
            for doc_id, score in rankings:
                doc = doc_map[doc_id].copy()
                doc["rerank_score"] = score
                doc["score"] = score / 10.0  # Normalize to 0-1 range
                reranked_docs.append(doc)
            
            # Add any documents that weren't ranked
            ranked_ids = {doc_id for doc_id, _ in rankings}
            for doc_id, doc in doc_map.items():
                if doc_id not in ranked_ids:
                    doc_copy = doc.copy()
                    doc_copy["rerank_score"] = 0.0
                    doc_copy["score"] = 0.0
                    reranked_docs.append(doc_copy)
            
            return reranked_docs
        
        except Exception as e:
            logger.error(f"Error parsing LLM rankings: {str(e)}")
            return [] 