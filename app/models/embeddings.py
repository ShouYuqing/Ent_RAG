"""
Embedding models interface for the Ent_RAG system.
Provides a unified interface for generating embeddings from different providers.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union
import os

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import config

logger = logging.getLogger("ent_rag.models.embeddings")


class EmbeddingManager:
    """
    Manages embedding models from different providers.
    Provides a unified interface for generating embeddings.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Optional model name to override the default
        """
        self.model_name = model_name or config.llm.default_embedding_model
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        self.local_model = None  # Lazy initialization for local models
    
    def get_embeddings(
        self,
        texts: List[str],
        provider: str = "openai"
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            provider: Embedding provider to use
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if provider == "openai":
            return self._get_openai_embeddings(texts)
        elif provider == "local":
            return self._get_local_embeddings(texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI's API.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings
        """
        try:
            # Handle empty list
            if not texts:
                return []
            
            # Log request
            logger.debug(f"Sending embedding request to OpenAI API with model {self.model_name}")
            
            # Send request
            start_time = time.time()
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            end_time = time.time()
            
            # Log response time
            logger.debug(f"OpenAI API embedding response time: {end_time - start_time:.2f} seconds")
            
            # Extract and return embeddings
            embeddings = [data.embedding for data in response.data]
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            raise
    
    def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using a local model.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings
        """
        try:
            # Handle empty list
            if not texts:
                return []
            
            # Initialize model if not already initialized
            if self.local_model is None:
                logger.debug(f"Initializing local embedding model")
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Log request
            logger.debug(f"Generating embeddings with local model")
            
            # Generate embeddings
            start_time = time.time()
            embeddings = self.local_model.encode(texts)
            end_time = time.time()
            
            # Log response time
            logger.debug(f"Local embedding generation time: {end_time - start_time:.2f} seconds")
            
            # Convert numpy arrays to lists
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embeddings with local model: {str(e)}")
            raise
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2) 