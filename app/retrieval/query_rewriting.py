"""
Query rewriting and expansion to improve retrieval quality.
"""

import logging
from typing import List, Optional, Dict, Any
import re

from app.config import config
from app.models.llm import LLMManager

logger = logging.getLogger("ent_rag.retrieval.query_rewriting")


class QueryRewriter:
    """
    Implements query rewriting and expansion techniques to improve retrieval quality.
    """
    
    def __init__(self):
        """Initialize the query rewriter with necessary components."""
        self.llm_manager = LLMManager()
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query to improve retrieval quality.
        
        Args:
            query: The original query
            
        Returns:
            The rewritten query
        """
        # If the query is very short, expand it
        if len(query.split()) < 3:
            return self.expand_query(query)
        
        # If the query is a question, try to extract key concepts
        if self._is_question(query):
            return self.extract_key_concepts(query)
        
        # For longer queries, try to make them more specific
        if len(query.split()) > 10:
            return self.focus_query(query)
        
        # Default: return the original query
        return query
    
    def expand_query(self, query: str) -> str:
        """
        Expand a short query to include related terms.
        
        Args:
            query: The original query
            
        Returns:
            The expanded query
        """
        logger.debug(f"Expanding query: '{query}'")
        
        # Use LLM to expand the query
        prompt = f"""
        Your task is to expand the following short search query to improve document retrieval.
        Add relevant terms, synonyms, or related concepts that might appear in documents about this topic.
        Keep the expansion focused and relevant. Return ONLY the expanded query without explanation.
        
        Original query: {query}
        
        Expanded query:
        """
        
        try:
            expanded_query = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            ).strip()
            
            logger.debug(f"Expanded query: '{expanded_query}'")
            return expanded_query
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return query
    
    def extract_key_concepts(self, query: str) -> str:
        """
        Extract key concepts from a question to improve retrieval.
        
        Args:
            query: The original query (a question)
            
        Returns:
            A query focused on key concepts
        """
        logger.debug(f"Extracting key concepts from query: '{query}'")
        
        # Use LLM to extract key concepts
        prompt = f"""
        Your task is to extract the key concepts from the following question to improve document retrieval.
        Identify the main entities, topics, and important terms that documents would need to contain to answer this question.
        Return ONLY the key concepts as a search query without explanation.
        
        Question: {query}
        
        Key concepts for search:
        """
        
        try:
            key_concepts = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            ).strip()
            
            logger.debug(f"Extracted key concepts: '{key_concepts}'")
            return key_concepts
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
            return query
    
    def focus_query(self, query: str) -> str:
        """
        Focus a long query to make it more specific and targeted.
        
        Args:
            query: The original query
            
        Returns:
            A more focused query
        """
        logger.debug(f"Focusing long query: '{query}'")
        
        # Use LLM to focus the query
        prompt = f"""
        Your task is to focus the following long search query to make it more specific and targeted for document retrieval.
        Identify the core information need and express it concisely. Remove unnecessary details while preserving the main intent.
        Return ONLY the focused query without explanation.
        
        Original query: {query}
        
        Focused query:
        """
        
        try:
            focused_query = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            ).strip()
            
            logger.debug(f"Focused query: '{focused_query}'")
            return focused_query
        except Exception as e:
            logger.error(f"Error focusing query: {str(e)}")
            return query
    
    def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple variations of a query for ensemble retrieval.
        
        Args:
            query: The original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        logger.debug(f"Generating {num_variations} variations of query: '{query}'")
        
        # Use LLM to generate variations
        prompt = f"""
        Your task is to generate {num_variations} different variations of the following search query.
        Each variation should express the same information need but using different wording, phrasing, or perspective.
        Return ONLY the numbered list of variations without explanation.
        
        Original query: {query}
        
        Variations:
        1.
        """
        
        try:
            variations_text = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            ).strip()
            
            # Parse the variations
            variations = []
            for line in variations_text.split("\n"):
                # Remove numbering and whitespace
                cleaned_line = re.sub(r"^\d+\.\s*", "", line).strip()
                if cleaned_line and cleaned_line != query:
                    variations.append(cleaned_line)
            
            # Add original query if not enough variations
            if len(variations) < num_variations:
                variations.append(query)
            
            # Ensure we return the requested number of variations
            return variations[:num_variations]
        
        except Exception as e:
            logger.error(f"Error generating query variations: {str(e)}")
            return [query] * num_variations
    
    def _is_question(self, query: str) -> bool:
        """
        Check if a query is a question.
        
        Args:
            query: The query to check
            
        Returns:
            True if the query is a question, False otherwise
        """
        # Check if the query ends with a question mark
        if query.strip().endswith("?"):
            return True
        
        # Check if the query starts with a question word
        question_words = ["what", "who", "where", "when", "why", "how", "which", "can", "could", "would", "is", "are", "do", "does"]
        first_word = query.strip().split()[0].lower() if query.strip() else ""
        
        return first_word in question_words 