"""
Custom relevance scoring implementation for the Ent_RAG system.

This module implements sophisticated algorithms for determining document relevance
to a given query. It combines multiple scoring factors including semantic similarity,
term frequency, document freshness, and authority to provide a comprehensive relevance score.

Author: yuqings
Created: February 2024
License: MIT
"""

import logging
import math
import re
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import config
from app.models.embeddings import EmbeddingManager

logger = logging.getLogger("ent_rag.retrieval.relevance_scoring")


class RelevanceScorer:
    """
    Implements custom relevance scoring for retrieved documents.
    
    This class provides methods to score documents based on multiple factors,
    including semantic similarity, term frequency, document freshness, and authority.
    It allows for customizable weighting of these factors to tailor the scoring
    to specific use cases.
    
    Attributes:
        embedding_manager: Manager for embedding models
        semantic_weight: Weight for semantic similarity in the final score
        lexical_weight: Weight for lexical similarity in the final score
        freshness_weight: Weight for document freshness in the final score
        authority_weight: Weight for document authority in the final score
    """
    
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.3,
        freshness_weight: float = 0.1,
        authority_weight: float = 0.1
    ):
        """
        Initialize the relevance scorer with necessary components and weights.
        
        Args:
            embedding_manager: Manager for embedding models
            semantic_weight: Weight for semantic similarity in the final score
            lexical_weight: Weight for lexical similarity in the final score
            freshness_weight: Weight for document freshness in the final score
            authority_weight: Weight for document authority in the final score
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        
        # Weights for different scoring factors (should sum to 1.0)
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.freshness_weight = freshness_weight
        self.authority_weight = authority_weight
        
        # Initialize TF-IDF vectorizer for lexical scoring
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )
        
        logger.info(f"RelevanceScorer initialized with weights: semantic={semantic_weight}, "
                   f"lexical={lexical_weight}, freshness={freshness_weight}, authority={authority_weight}")
    
    def score(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score documents based on their relevance to the query.
        
        This method calculates a comprehensive relevance score for each document
        by combining multiple scoring factors with their respective weights.
        
        Args:
            query: The search query
            documents: Documents to score
            
        Returns:
            List of documents with updated relevance scores
        """
        if not documents:
            return []
        
        # Prepare query embedding for semantic scoring
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Calculate scores for each document
        scored_docs = []
        
        for doc in documents:
            # Calculate individual scores
            semantic_score = self._calculate_semantic_score(query_embedding, doc)
            lexical_score = self._calculate_lexical_score(query, doc)
            freshness_score = self._calculate_freshness_score(doc)
            authority_score = self._calculate_authority_score(doc)
            
            # Combine scores with weights
            combined_score = (
                semantic_score * self.semantic_weight +
                lexical_score * self.lexical_weight +
                freshness_score * self.freshness_weight +
                authority_score * self.authority_weight
            )
            
            # Create a copy of the document with the new score
            doc_copy = doc.copy()
            doc_copy["relevance_score"] = combined_score
            doc_copy["score"] = combined_score  # Update the main score
            
            # Add individual scores for debugging/analysis
            doc_copy["semantic_score"] = semantic_score
            doc_copy["lexical_score"] = lexical_score
            doc_copy["freshness_score"] = freshness_score
            doc_copy["authority_score"] = authority_score
            
            scored_docs.append(doc_copy)
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return scored_docs
    
    def _calculate_semantic_score(
        self,
        query_embedding: List[float],
        document: Dict[str, Any]
    ) -> float:
        """
        Calculate semantic similarity score between query and document.
        
        This method uses vector embeddings to measure the semantic similarity
        between the query and document content.
        
        Args:
            query_embedding: Embedding vector for the query
            document: Document to score
            
        Returns:
            Semantic similarity score (0-1)
        """
        # Check if document already has an embedding
        if "embedding" in document:
            doc_embedding = document["embedding"]
        else:
            # Get document content
            content = document.get("content", "")
            
            # Generate embedding for document content
            doc_embedding = self.embedding_manager.embed_text(content)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(query_embedding, doc_embedding)
        
        return similarity
    
    def _calculate_lexical_score(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> float:
        """
        Calculate lexical similarity score between query and document.
        
        This method uses TF-IDF and other lexical features to measure the
        similarity between the query and document content.
        
        Args:
            query: The search query
            document: Document to score
            
        Returns:
            Lexical similarity score (0-1)
        """
        # Get document content
        content = document.get("content", "")
        
        if not content:
            return 0.0
        
        # Calculate TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([query, content])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            # Fallback to simpler lexical scoring if TF-IDF fails
            similarity = self._simple_lexical_score(query, content)
        
        return float(similarity)
    
    def _simple_lexical_score(
        self,
        query: str,
        content: str
    ) -> float:
        """
        Calculate a simple lexical score based on term overlap.
        
        This method is used as a fallback when TF-IDF vectorization fails.
        
        Args:
            query: The search query
            content: Document content
            
        Returns:
            Simple lexical score (0-1)
        """
        # Normalize and tokenize
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_terms:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_terms.intersection(content_terms)
        
        # Calculate term frequency in content
        term_freq_score = 0.0
        if intersection:
            for term in intersection:
                # Count occurrences in content
                term_freq = content.lower().count(term)
                # Normalize by content length
                term_freq_norm = term_freq / (len(content) / 100)
                term_freq_score += term_freq_norm
            
            # Normalize by number of intersection terms
            term_freq_score /= len(intersection)
            # Scale to 0-1 range
            term_freq_score = min(1.0, term_freq_score / 5.0)
        
        # Combine Jaccard similarity with term frequency
        jaccard = len(intersection) / len(query_terms)
        
        return 0.7 * jaccard + 0.3 * term_freq_score
    
    def _calculate_freshness_score(
        self,
        document: Dict[str, Any]
    ) -> float:
        """
        Calculate freshness score based on document age.
        
        This method assigns higher scores to more recent documents.
        
        Args:
            document: Document to score
            
        Returns:
            Freshness score (0-1)
        """
        # Get document metadata
        metadata = document.get("metadata", {})
        
        # Try to get timestamp from metadata
        timestamp_str = metadata.get("timestamp", metadata.get("created_at", metadata.get("date", None)))
        
        if not timestamp_str:
            # No timestamp available, use neutral score
            return 0.5
        
        try:
            # Parse timestamp
            if isinstance(timestamp_str, str):
                # Try different formats
                for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # No format matched
                    return 0.5
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
            else:
                return 0.5
            
            # Calculate age in days
            age_days = (datetime.now() - timestamp).days
            
            # Calculate freshness score (newer documents get higher scores)
            # Use a decay function: score = exp(-age_days / half_life)
            half_life = 365  # Half-life of 1 year
            freshness_score = math.exp(-age_days / half_life)
            
            return freshness_score
        
        except Exception as e:
            logger.warning(f"Error calculating freshness score: {str(e)}")
            return 0.5
    
    def _calculate_authority_score(
        self,
        document: Dict[str, Any]
    ) -> float:
        """
        Calculate authority score based on document source and metadata.
        
        This method assigns higher scores to documents from authoritative sources
        or with indicators of high quality.
        
        Args:
            document: Document to score
            
        Returns:
            Authority score (0-1)
        """
        # Get document metadata
        metadata = document.get("metadata", {})
        
        # Initialize base score
        authority_score = 0.5
        
        # Factors that can increase authority
        authority_factors = {
            "verified": 0.2,  # Verified content
            "expert_reviewed": 0.2,  # Expert-reviewed content
            "citation_count": 0.1,  # Has citations
            "likes": 0.05,  # Has likes/upvotes
            "views": 0.05,  # Has views
            "official": 0.2,  # Official source
            "peer_reviewed": 0.2,  # Peer-reviewed content
        }
        
        # Check for authority factors in metadata
        for factor, weight in authority_factors.items():
            if factor in metadata:
                value = metadata[factor]
                
                if isinstance(value, bool) and value:
                    # Boolean flag is True
                    authority_score += weight
                elif isinstance(value, (int, float)) and value > 0:
                    # Numeric value is positive
                    # Scale based on magnitude (log scale)
                    if factor in ["citation_count", "likes", "views"]:
                        scaled_value = min(1.0, math.log10(value + 1) / 5.0)
                        authority_score += weight * scaled_value
                elif isinstance(value, str) and value.lower() in ["true", "yes", "1"]:
                    # String value indicates True
                    authority_score += weight
        
        # Check source authority
        source = metadata.get("source", "")
        if source:
            # List of authoritative sources (could be expanded or configured)
            authoritative_sources = [
                "official", "gov", "edu", "academic", "research", "journal",
                "university", "institute", "organization", "foundation"
            ]
            
            # Check if source contains any authoritative keywords
            for auth_source in authoritative_sources:
                if auth_source.lower() in source.lower():
                    authority_score += 0.1
                    break
        
        # Cap at 1.0
        return min(1.0, authority_score)
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Ensure result is in [0, 1] range
        return float(max(0.0, min(1.0, similarity)))


class ContextualRelevanceScorer(RelevanceScorer):
    """
    Extended relevance scorer that takes into account contextual information.
    
    This class extends the base RelevanceScorer with additional methods
    that consider the context of the user's query, such as previous queries,
    user preferences, and session information.
    
    Attributes:
        context_weight: Weight for contextual relevance in the final score
    """
    
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        semantic_weight: float = 0.4,
        lexical_weight: float = 0.2,
        freshness_weight: float = 0.1,
        authority_weight: float = 0.1,
        context_weight: float = 0.2
    ):
        """
        Initialize the contextual relevance scorer.
        
        Args:
            embedding_manager: Manager for embedding models
            semantic_weight: Weight for semantic similarity in the final score
            lexical_weight: Weight for lexical similarity in the final score
            freshness_weight: Weight for document freshness in the final score
            authority_weight: Weight for document authority in the final score
            context_weight: Weight for contextual relevance in the final score
        """
        super().__init__(
            embedding_manager=embedding_manager,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            freshness_weight=freshness_weight,
            authority_weight=authority_weight
        )
        
        self.context_weight = context_weight
        logger.info(f"ContextualRelevanceScorer initialized with context_weight={context_weight}")
    
    def score_with_context(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        user_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Score documents based on their relevance to the query and user context.
        
        This method extends the base scoring method by incorporating contextual
        information such as user preferences, previous queries, and session data.
        
        Args:
            query: The search query
            documents: Documents to score
            user_context: Contextual information about the user and session
            
        Returns:
            List of documents with updated relevance scores
        """
        # Get base scores
        scored_docs = super().score(query, documents)
        
        # Calculate contextual scores
        for doc in scored_docs:
            contextual_score = self._calculate_contextual_score(doc, user_context)
            
            # Adjust the final score with contextual relevance
            original_score = doc["relevance_score"]
            adjusted_score = (
                original_score * (1 - self.context_weight) +
                contextual_score * self.context_weight
            )
            
            # Update scores
            doc["contextual_score"] = contextual_score
            doc["relevance_score"] = adjusted_score
            doc["score"] = adjusted_score  # Update the main score
        
        # Re-sort by adjusted score
        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return scored_docs
    
    def _calculate_contextual_score(
        self,
        document: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> float:
        """
        Calculate contextual relevance score based on user context.
        
        This method considers factors such as user preferences, previous interactions,
        and session information to determine how relevant a document is to the
        current user context.
        
        Args:
            document: Document to score
            user_context: Contextual information about the user and session
            
        Returns:
            Contextual relevance score (0-1)
        """
        # Initialize score
        contextual_score = 0.5
        
        # Get document metadata
        doc_metadata = document.get("metadata", {})
        
        # Factor 1: User preferences
        preferences = user_context.get("preferences", {})
        for pref_key, pref_value in preferences.items():
            # Check if document metadata matches user preferences
            if pref_key in doc_metadata:
                if doc_metadata[pref_key] == pref_value:
                    contextual_score += 0.1
        
        # Factor 2: Previous interactions
        interactions = user_context.get("interactions", [])
        for interaction in interactions:
            # Check if user has interacted with this document before
            if interaction.get("document_id") == document.get("id"):
                # Positive interaction (e.g., clicked, saved)
                if interaction.get("type") in ["click", "save", "like"]:
                    contextual_score += 0.15
                # Negative interaction (e.g., dismissed)
                elif interaction.get("type") in ["dismiss", "dislike"]:
                    contextual_score -= 0.15
        
        # Factor 3: Topic continuity
        previous_queries = user_context.get("previous_queries", [])
        if previous_queries:
            # Check if document is related to previous queries
            doc_content = document.get("content", "").lower()
            for prev_query in previous_queries[-3:]:  # Consider last 3 queries
                # Simple check for term overlap
                query_terms = set(prev_query.lower().split())
                for term in query_terms:
                    if term in doc_content and len(term) > 3:  # Ignore short terms
                        contextual_score += 0.05
        
        # Factor 4: Session focus
        session_topics = user_context.get("session_topics", [])
        doc_topics = doc_metadata.get("topics", [])
        
        # Check for topic overlap
        if session_topics and doc_topics:
            overlap = set(session_topics).intersection(set(doc_topics))
            if overlap:
                contextual_score += 0.1 * (len(overlap) / len(session_topics))
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, contextual_score)) 