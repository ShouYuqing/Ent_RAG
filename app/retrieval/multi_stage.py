"""
Multi-stage retrieval with pre-filtering and reranking.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import config
from app.models.embeddings import EmbeddingModel
from app.models.llm import LLMManager

logger = logging.getLogger("ent_rag.retrieval.multi_stage")


class PreFilter:
    """
    Implements pre-filtering for multi-stage retrieval.
    """
    
    def __init__(self):
        """Initialize the pre-filter with necessary components."""
        self.embedding_model = EmbeddingModel()
    
    def filter(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
        max_documents: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Pre-filter documents based on metadata and basic relevance.
        
        Args:
            query: The search query
            documents: List of documents to filter
            filters: Metadata filters to apply
            max_documents: Maximum number of documents to return
            
        Returns:
            Filtered list of documents
        """
        logger.debug(f"Pre-filtering {len(documents)} documents")
        
        # Apply metadata filters if provided
        if filters:
            filtered_docs = self._apply_metadata_filters(documents, filters)
        else:
            filtered_docs = documents
        
        # If we have too many documents, apply basic relevance filtering
        if len(filtered_docs) > max_documents:
            filtered_docs = self._apply_relevance_filter(query, filtered_docs, max_documents)
        
        logger.debug(f"Pre-filtering returned {len(filtered_docs)} documents")
        return filtered_docs
    
    def _apply_metadata_filters(
        self,
        documents: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to documents.
        
        Args:
            documents: List of documents to filter
            filters: Metadata filters to apply
            
        Returns:
            Filtered list of documents
        """
        filtered_docs = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            match = True
            
            for key, value in filters.items():
                # Handle list values (OR condition)
                if isinstance(value, list):
                    if metadata.get(key) not in value:
                        match = False
                        break
                # Handle exact match
                elif metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _apply_relevance_filter(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_documents: int
    ) -> List[Dict[str, Any]]:
        """
        Apply basic relevance filtering using TF-IDF.
        
        Args:
            query: The search query
            documents: List of documents to filter
            max_documents: Maximum number of documents to return
            
        Returns:
            Filtered list of documents
        """
        # Extract document contents
        contents = [doc.get("content", "") for doc in documents]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(contents)
        
        # Transform query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # Get indices of top documents
        top_indices = np.argsort(similarity_scores)[-max_documents:][::-1]
        
        # Return top documents
        return [documents[i] for i in top_indices]


class Reranker:
    """
    Implements reranking for multi-stage retrieval.
    """
    
    def __init__(self):
        """Initialize the reranker with necessary components."""
        self.embedding_model = EmbeddingModel()
        self.llm_manager = LLMManager()
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        method: str = "cross_encoder",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            method: Reranking method to use (cross_encoder, semantic, or llm)
            top_k: Number of documents to return (None for all)
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        logger.debug(f"Reranking {len(documents)} documents using {method} method")
        
        # Choose reranking method
        if method == "cross_encoder":
            reranked_docs = self._rerank_cross_encoder(query, documents)
        elif method == "semantic":
            reranked_docs = self._rerank_semantic(query, documents)
        elif method == "llm":
            reranked_docs = self._rerank_llm(query, documents)
        else:
            logger.warning(f"Unknown reranking method: {method}, using semantic reranking")
            reranked_docs = self._rerank_semantic(query, documents)
        
        # Limit results if top_k is specified
        if top_k is not None and top_k < len(reranked_docs):
            reranked_docs = reranked_docs[:top_k]
        
        logger.debug(f"Reranking returned {len(reranked_docs)} documents")
        return reranked_docs
    
    def _rerank_cross_encoder(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using a cross-encoder model.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        try:
            # Import cross-encoder only when needed
            from sentence_transformers import CrossEncoder
            
            # Initialize cross-encoder model
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Prepare document pairs
            pairs = [(query, doc.get("content", "")) for doc in documents]
            
            # Get scores
            scores = model.predict(pairs)
            
            # Create reranked documents with updated scores
            reranked_docs = []
            for i, doc in enumerate(documents):
                reranked_doc = doc.copy()
                reranked_doc["score"] = float(scores[i])
                reranked_docs.append(reranked_doc)
            
            # Sort by score in descending order
            reranked_docs.sort(key=lambda x: x["score"], reverse=True)
            
            return reranked_docs
        
        except ImportError:
            logger.warning("CrossEncoder not available, falling back to semantic reranking")
            return self._rerank_semantic(query, documents)
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return self._rerank_semantic(query, documents)
    
    def _rerank_semantic(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using semantic similarity.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Create reranked documents with updated scores
        reranked_docs = []
        for doc in documents:
            # Get or generate document embedding
            if "embedding" in doc:
                doc_embedding = doc["embedding"]
            else:
                doc_content = doc.get("content", "")
                doc_embedding = self.embedding_model.embed_text(doc_content)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # Create reranked document
            reranked_doc = doc.copy()
            reranked_doc["score"] = float(similarity)
            reranked_docs.append(reranked_doc)
        
        # Sort by score in descending order
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked_docs
    
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
        # Limit the number of documents for LLM reranking to avoid excessive API calls
        max_docs_for_llm = min(len(documents), 10)
        docs_for_llm = documents[:max_docs_for_llm]
        
        # Create reranked documents
        reranked_docs = []
        
        for i, doc in enumerate(docs_for_llm):
            doc_content = doc.get("content", "")
            
            # Truncate content if too long
            if len(doc_content) > 1000:
                doc_content = doc_content[:1000] + "..."
            
            # Create prompt for relevance assessment
            prompt = f"""
            On a scale of 0 to 10, rate the relevance of the following document to the query.
            Return ONLY a number from 0 to 10, where 0 is completely irrelevant and 10 is perfectly relevant.
            
            Query: {query}
            
            Document:
            {doc_content}
            
            Relevance score (0-10):
            """
            
            try:
                # Get relevance score from LLM
                response = self.llm_manager.generate(
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.1
                ).strip()
                
                # Parse score
                try:
                    score = float(response)
                    # Normalize score to 0-1 range
                    normalized_score = score / 10.0
                except ValueError:
                    logger.warning(f"Could not parse LLM relevance score: {response}")
                    normalized_score = 0.5  # Default score
                
                # Create reranked document
                reranked_doc = doc.copy()
                reranked_doc["score"] = normalized_score
                reranked_docs.append(reranked_doc)
            
            except Exception as e:
                logger.error(f"Error in LLM reranking for document {i}: {str(e)}")
                # Keep original document and score
                reranked_docs.append(doc)
        
        # Add remaining documents with their original scores
        if max_docs_for_llm < len(documents):
            reranked_docs.extend(documents[max_docs_for_llm:])
        
        # Sort by score in descending order
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked_docs
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class MultiStageRetriever:
    """
    Implements multi-stage retrieval with pre-filtering and reranking.
    """
    
    def __init__(self):
        """Initialize the multi-stage retriever with necessary components."""
        self.pre_filter = PreFilter()
        self.reranker = Reranker()
    
    def retrieve(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        reranking_method: str = "cross_encoder"
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-stage retrieval with pre-filtering and reranking.
        
        Args:
            query: The search query
            documents: List of documents to search
            filters: Metadata filters to apply
            top_k: Number of results to return
            reranking_method: Method to use for reranking
            
        Returns:
            List of retrieved documents
        """
        logger.debug(f"Multi-stage retrieval for query: '{query}'")
        
        # Stage 1: Pre-filtering
        filtered_docs = self.pre_filter.filter(
            query=query,
            documents=documents,
            filters=filters,
            max_documents=min(100, len(documents))
        )
        
        # Stage 2: Reranking
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=filtered_docs,
            method=reranking_method,
            top_k=top_k
        )
        
        return reranked_docs 