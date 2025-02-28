"""
Context prioritization and filtering to remove irrelevant information.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import config
from app.models.embeddings import EmbeddingModel

logger = logging.getLogger("ent_rag.context.prioritization")


class ContextPrioritizer:
    """
    Implements context prioritization and filtering to focus on relevant information.
    """
    
    def __init__(self):
        """Initialize the context prioritizer with necessary components."""
        self.embedding_model = EmbeddingModel()
    
    def prioritize(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_tokens: int = 3500
    ) -> str:
        """
        Prioritize and filter context based on relevance to the query.
        
        Args:
            query: The user's query
            documents: List of retrieved documents
            max_tokens: Maximum number of tokens to include in the context
            
        Returns:
            Formatted context string with the most relevant information
        """
        logger.debug(f"Prioritizing context for query: '{query}' with {len(documents)} documents")
        
        if not documents:
            return ""
        
        # Rank documents by relevance
        ranked_docs = self._rank_by_relevance(query, documents)
        
        # Filter and format context
        context = self._filter_and_format(query, ranked_docs, max_tokens)
        
        return context
    
    def _rank_by_relevance(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by relevance to the query.
        
        Args:
            query: The user's query
            documents: List of retrieved documents
            
        Returns:
            List of documents ranked by relevance
        """
        # If documents already have scores, use them for ranking
        if all("score" in doc for doc in documents):
            ranked_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
            return ranked_docs
        
        # Otherwise, calculate relevance scores using embeddings
        query_embedding = self.embedding_model.embed_query(query)
        
        scored_docs = []
        for doc in documents:
            content = doc.get("content", "")
            
            # Get or generate document embedding
            if "embedding" in doc:
                doc_embedding = doc["embedding"]
            else:
                doc_embedding = self.embedding_model.embed_text(content)
            
            # Calculate similarity score
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # Create scored document
            scored_doc = doc.copy()
            scored_doc["score"] = float(similarity)
            
            scored_docs.append(scored_doc)
        
        # Sort by score in descending order
        ranked_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
        
        return ranked_docs
    
    def _filter_and_format(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_tokens: int
    ) -> str:
        """
        Filter and format context based on relevance and token limit.
        
        Args:
            query: The user's query
            documents: List of documents ranked by relevance
            max_tokens: Maximum number of tokens to include
            
        Returns:
            Formatted context string
        """
        # Initialize context parts and token count
        context_parts = []
        token_count = 0
        
        # Process documents in order of relevance
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Skip empty content
            if not content.strip():
                continue
            
            # Estimate token count (rough approximation: 4 chars per token)
            doc_tokens = len(content) // 4
            
            # If this document would exceed the token limit, skip it
            if token_count + doc_tokens > max_tokens and context_parts:
                continue
            
            # Format document with metadata
            formatted_doc = self._format_document(doc, i + 1)
            
            # Add to context
            context_parts.append(formatted_doc)
            token_count += doc_tokens
            
            # If we've reached the token limit, stop
            if token_count >= max_tokens:
                break
        
        # Combine context parts
        if context_parts:
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant information found."
        
        return context
    
    def _format_document(self, document: Dict[str, Any], index: int) -> str:
        """
        Format a document for inclusion in the context.
        
        Args:
            document: The document to format
            index: The document's index in the context
            
        Returns:
            Formatted document string
        """
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        # Extract metadata fields
        title = metadata.get("title", f"Document {index}")
        source = metadata.get("source", "Unknown source")
        
        # Format document
        formatted_doc = f"[Document {index}] {title} (Source: {source})\n\n{content}"
        
        return formatted_doc
    
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


class AdvancedContextPrioritizer(ContextPrioritizer):
    """
    Advanced context prioritization with more sophisticated filtering and formatting.
    """
    
    def __init__(self):
        """Initialize the advanced context prioritizer."""
        super().__init__()
    
    def prioritize(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_tokens: int = 3500
    ) -> str:
        """
        Prioritize and filter context with advanced techniques.
        
        Args:
            query: The user's query
            documents: List of retrieved documents
            max_tokens: Maximum number of tokens to include in the context
            
        Returns:
            Formatted context string with the most relevant information
        """
        logger.debug(f"Advanced prioritization for query: '{query}' with {len(documents)} documents")
        
        if not documents:
            return ""
        
        # Extract query keywords for focused filtering
        keywords = self._extract_keywords(query)
        
        # Rank documents by relevance
        ranked_docs = self._rank_by_relevance(query, documents)
        
        # Extract relevant passages from documents
        passages = self._extract_relevant_passages(query, ranked_docs, keywords)
        
        # Deduplicate and merge similar passages
        unique_passages = self._deduplicate_passages(passages)
        
        # Filter and format context
        context = self._filter_and_format_passages(query, unique_passages, max_tokens)
        
        return context
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the query.
        
        Args:
            query: The user's query
            
        Returns:
            List of important keywords
        """
        # Simple keyword extraction using TF-IDF
        try:
            # Create a small corpus with the query
            corpus = [query]
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for the first document (the query)
            scores = tfidf_matrix[0].toarray()[0]
            
            # Create a list of (word, score) tuples and sort by score
            word_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top keywords (up to 5)
            keywords = [word for word, score in word_scores[:5]]
            
            return keywords
        
        except Exception as e:
            logger.warning(f"Error extracting keywords: {str(e)}")
            # Fallback: just split the query into words and remove common stop words
            stop_words = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "by", "to", "and", "or", "is", "are"}
            return [word.lower() for word in query.split() if word.lower() not in stop_words]
    
    def _extract_relevant_passages(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant passages from documents.
        
        Args:
            query: The user's query
            documents: List of documents ranked by relevance
            keywords: List of important keywords from the query
            
        Returns:
            List of relevant passages
        """
        passages = []
        
        for doc_index, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Skip empty content
            if not content.strip():
                continue
            
            # Split content into sentences or paragraphs
            if len(content) > 1000:
                # For longer content, split into paragraphs
                parts = [p.strip() for p in content.split("\n\n") if p.strip()]
                if len(parts) == 1:
                    # If no paragraphs, try splitting by sentences
                    import nltk
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True)
                    from nltk.tokenize import sent_tokenize
                    parts = sent_tokenize(content)
            else:
                # For shorter content, use the whole document
                parts = [content]
            
            # Score each part based on keyword presence and position
            scored_parts = []
            for part_index, part in enumerate(parts):
                # Calculate base score from document relevance
                base_score = doc.get("score", 0.8) * (1.0 - 0.1 * doc_index)  # Slight penalty for later documents
                
                # Boost score based on keyword presence
                keyword_score = 0
                for keyword in keywords:
                    if keyword.lower() in part.lower():
                        keyword_score += 0.2
                
                # Boost score based on position (earlier parts get higher scores)
                position_score = 1.0 - (0.05 * part_index)
                
                # Calculate final score
                final_score = base_score + keyword_score + position_score
                
                # Create passage
                passage = {
                    "content": part,
                    "score": final_score,
                    "doc_index": doc_index,
                    "part_index": part_index,
                    "metadata": metadata
                }
                
                scored_parts.append(passage)
            
            # Add top-scoring parts to passages
            top_parts = sorted(scored_parts, key=lambda x: x["score"], reverse=True)
            passages.extend(top_parts[:3])  # Take up to 3 best passages per document
        
        # Sort all passages by score
        passages.sort(key=lambda x: x["score"], reverse=True)
        
        return passages
    
    def _deduplicate_passages(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate and merge similar passages.
        
        Args:
            passages: List of passages
            
        Returns:
            List of deduplicated passages
        """
        if not passages:
            return []
        
        # Create TF-IDF vectorizer
        contents = [p["content"] for p in passages]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(contents)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Track which passages to keep
        to_keep = [True] * len(passages)
        
        # Check each pair of passages
        for i in range(len(passages)):
            if not to_keep[i]:
                continue
                
            for j in range(i + 1, len(passages)):
                if not to_keep[j]:
                    continue
                    
                # If passages are very similar, mark the lower-scored one for removal
                if similarity_matrix[i, j] > 0.8:
                    if passages[i]["score"] >= passages[j]["score"]:
                        to_keep[j] = False
                    else:
                        to_keep[i] = False
                        break
        
        # Keep only non-duplicate passages
        unique_passages = [p for i, p in enumerate(passages) if to_keep[i]]
        
        return unique_passages
    
    def _filter_and_format_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        max_tokens: int
    ) -> str:
        """
        Filter and format passages based on relevance and token limit.
        
        Args:
            query: The user's query
            passages: List of passages ranked by relevance
            max_tokens: Maximum number of tokens to include
            
        Returns:
            Formatted context string
        """
        # Initialize context parts and token count
        context_parts = []
        token_count = 0
        
        # Track which documents we've included
        included_docs = set()
        
        # Process passages in order of relevance
        for i, passage in enumerate(passages):
            content = passage.get("content", "")
            metadata = passage.get("metadata", {})
            doc_index = passage.get("doc_index")
            
            # Skip empty content
            if not content.strip():
                continue
            
            # Estimate token count (rough approximation: 4 chars per token)
            passage_tokens = len(content) // 4
            
            # If this passage would exceed the token limit, skip it
            if token_count + passage_tokens > max_tokens and context_parts:
                continue
            
            # Format passage with metadata
            doc_num = doc_index + 1 if doc_index is not None else len(included_docs) + 1
            formatted_passage = self._format_passage(passage, doc_num)
            
            # Add to context
            context_parts.append(formatted_passage)
            token_count += passage_tokens
            
            # Track which document this came from
            if doc_index is not None:
                included_docs.add(doc_index)
            
            # If we've reached the token limit, stop
            if token_count >= max_tokens:
                break
        
        # Combine context parts
        if context_parts:
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant information found."
        
        return context
    
    def _format_passage(self, passage: Dict[str, Any], doc_num: int) -> str:
        """
        Format a passage for inclusion in the context.
        
        Args:
            passage: The passage to format
            doc_num: The document number
            
        Returns:
            Formatted passage string
        """
        content = passage.get("content", "")
        metadata = passage.get("metadata", {})
        
        # Extract metadata fields
        title = metadata.get("title", f"Document {doc_num}")
        source = metadata.get("source", "Unknown source")
        
        # Format passage
        formatted_passage = f"[Document {doc_num}] {title} (Source: {source})\n\n{content}"
        
        return formatted_passage 