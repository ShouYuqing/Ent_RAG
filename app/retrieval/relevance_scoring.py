"""
Custom relevance scoring algorithms for document ranking.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import config

logger = logging.getLogger("ent_rag.retrieval.relevance_scoring")


class RelevanceScorer:
    """
    Implements custom relevance scoring algorithms for document ranking.
    """
    
    def __init__(self):
        """Initialize the relevance scorer."""
        pass
    
    def score(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        method: str = "combined"
    ) -> List[Dict[str, Any]]:
        """
        Score documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to score
            method: Scoring method to use (combined, semantic, lexical, or recency)
            
        Returns:
            List of documents with updated scores
        """
        if not documents:
            return []
        
        logger.debug(f"Scoring {len(documents)} documents using {method} method")
        
        # Choose scoring method
        if method == "combined":
            scored_docs = self._combined_scoring(query, documents)
        elif method == "semantic":
            scored_docs = self._semantic_scoring(query, documents)
        elif method == "lexical":
            scored_docs = self._lexical_scoring(query, documents)
        elif method == "recency":
            scored_docs = self._recency_scoring(query, documents)
        else:
            logger.warning(f"Unknown scoring method: {method}, using combined scoring")
            scored_docs = self._combined_scoring(query, documents)
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_docs
    
    def _combined_scoring(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply combined scoring using multiple factors.
        
        Args:
            query: The search query
            documents: List of documents to score
            
        Returns:
            List of documents with updated scores
        """
        # Apply individual scoring methods
        semantic_docs = self._semantic_scoring(query, documents)
        lexical_docs = self._lexical_scoring(query, documents)
        
        # Create a mapping of document IDs to scores
        semantic_scores = {doc["id"]: doc["score"] for doc in semantic_docs}
        lexical_scores = {doc["id"]: doc["score"] for doc in lexical_docs}
        
        # Combine scores with weights
        combined_docs = []
        for doc in documents:
            doc_id = doc["id"]
            semantic_score = semantic_scores.get(doc_id, 0.0)
            lexical_score = lexical_scores.get(doc_id, 0.0)
            
            # Calculate combined score (70% semantic, 30% lexical)
            combined_score = 0.7 * semantic_score + 0.3 * lexical_score
            
            # Apply additional boosting factors
            combined_score = self._apply_boosting_factors(query, doc, combined_score)
            
            # Create scored document
            scored_doc = doc.copy()
            scored_doc["score"] = combined_score
            scored_doc["score_details"] = {
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "combined_score": combined_score
            }
            
            combined_docs.append(scored_doc)
        
        return combined_docs
    
    def _semantic_scoring(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score documents based on semantic similarity.
        
        Args:
            query: The search query
            documents: List of documents to score
            
        Returns:
            List of documents with updated scores
        """
        # For semantic scoring, we assume the documents already have a semantic score
        # from vector search or embedding similarity calculation
        scored_docs = []
        
        for doc in documents:
            # Get existing score or vector_score
            score = doc.get("score", doc.get("vector_score", 0.0))
            
            # Create scored document
            scored_doc = doc.copy()
            scored_doc["score"] = score
            
            scored_docs.append(scored_doc)
        
        return scored_docs
    
    def _lexical_scoring(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score documents based on lexical similarity (TF-IDF).
        
        Args:
            query: The search query
            documents: List of documents to score
            
        Returns:
            List of documents with updated scores
        """
        # Extract document contents
        contents = [doc.get("content", "") for doc in documents]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")
        
        # If we have only one document, we need special handling
        if len(contents) == 1:
            # Add the query as a second document to allow TF-IDF calculation
            tfidf_matrix = vectorizer.fit_transform(contents + [query])
            query_vector = tfidf_matrix[1]  # The query is the second document
            doc_vectors = tfidf_matrix[0:1]  # The first document
        else:
            # Normal case with multiple documents
            tfidf_matrix = vectorizer.fit_transform(contents)
            query_vector = vectorizer.transform([query])
            doc_vectors = tfidf_matrix
        
        # Calculate similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]
        
        # Create scored documents
        scored_docs = []
        for i, doc in enumerate(documents):
            # Create scored document
            scored_doc = doc.copy()
            scored_doc["score"] = float(similarity_scores[i])
            
            scored_docs.append(scored_doc)
        
        return scored_docs
    
    def _recency_scoring(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score documents based on recency.
        
        Args:
            query: The search query
            documents: List of documents to score
            
        Returns:
            List of documents with updated scores
        """
        from datetime import datetime
        
        # Get current time
        now = datetime.now()
        
        # Create scored documents
        scored_docs = []
        
        for doc in documents:
            # Get document creation time from metadata
            metadata = doc.get("metadata", {})
            created_at_str = metadata.get("created_at")
            
            if created_at_str:
                try:
                    # Parse creation time
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    
                    # Calculate age in days
                    age_days = (now - created_at).days
                    
                    # Calculate recency score (newer documents get higher scores)
                    # Score decays exponentially with age
                    recency_score = np.exp(-0.05 * age_days)
                except (ValueError, TypeError):
                    # If we can't parse the date, use a default score
                    recency_score = 0.5
            else:
                # If no creation time, use a default score
                recency_score = 0.5
            
            # Create scored document
            scored_doc = doc.copy()
            scored_doc["score"] = recency_score
            
            scored_docs.append(scored_doc)
        
        return scored_docs
    
    def _apply_boosting_factors(
        self,
        query: str,
        document: Dict[str, Any],
        base_score: float
    ) -> float:
        """
        Apply boosting factors to adjust the base score.
        
        Args:
            query: The search query
            document: The document to score
            base_score: The base relevance score
            
        Returns:
            Adjusted relevance score
        """
        score = base_score
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        # Boost factor 1: Exact phrase match
        if self._contains_exact_phrase(content, query):
            score *= 1.5
        
        # Boost factor 2: Title match
        title = metadata.get("title", "")
        if title and self._contains_query_terms(title, query):
            score *= 1.3
        
        # Boost factor 3: Document length (prefer medium-length documents)
        doc_length = len(content.split())
        if 200 <= doc_length <= 1000:
            score *= 1.2
        elif doc_length > 3000:
            score *= 0.8
        
        # Boost factor 4: Metadata quality
        metadata_quality = len(metadata) / 10  # Normalize by assuming 10 fields is complete
        metadata_boost = 1.0 + (0.2 * min(metadata_quality, 1.0))
        score *= metadata_boost
        
        # Boost factor 5: Document type preference
        doc_type = metadata.get("content_type", "").lower()
        if doc_type in ["pdf", "docx", "markdown"]:
            score *= 1.1
        
        return score
    
    def _contains_exact_phrase(self, text: str, phrase: str) -> bool:
        """
        Check if text contains the exact phrase.
        
        Args:
            text: The text to search in
            phrase: The phrase to search for
            
        Returns:
            True if the text contains the exact phrase, False otherwise
        """
        # Clean and normalize the phrase
        clean_phrase = re.sub(r'[^\w\s]', '', phrase.lower())
        
        # Clean and normalize the text
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Check for exact phrase match
        return clean_phrase in clean_text
    
    def _contains_query_terms(self, text: str, query: str) -> bool:
        """
        Check if text contains the query terms.
        
        Args:
            text: The text to search in
            query: The query containing terms to search for
            
        Returns:
            True if the text contains all query terms, False otherwise
        """
        # Clean and normalize the query
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        query_terms = clean_query.split()
        
        # Clean and normalize the text
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Check if all query terms are in the text
        return all(term in clean_text for term in query_terms) 