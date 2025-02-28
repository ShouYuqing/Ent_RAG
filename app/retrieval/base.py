"""
Base retriever interface for the Ent_RAG system.

This module defines the base interface that all retriever implementations must follow,
ensuring consistent behavior across different retrieval strategies.

Author: yuqings
Created: February 2024
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseRetriever(ABC):
    """
    Abstract base class for all retriever implementations.
    
    This class defines the interface that all retriever implementations must follow,
    ensuring consistent behavior across different retrieval strategies.
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the given query.
        
        Args:
            query: The search query
            filters: Optional metadata filters to apply
            top_k: Number of results to return
            **kwargs: Additional retriever-specific parameters
            
        Returns:
            List of retrieved documents with content and metadata
        """
        pass 