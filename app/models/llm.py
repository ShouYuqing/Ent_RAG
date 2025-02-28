"""
LLM interface for the Ent_RAG system.
Provides a unified interface for interacting with different LLM providers.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
import os

from openai import OpenAI
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import config

logger = logging.getLogger("ent_rag.models.llm")


class LLMManager:
    """
    Manages interactions with LLM providers.
    Provides a unified interface for generating text and handling errors.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the LLM manager.
        
        Args:
            model_name: Optional model name to override the default
        """
        self.model_name = model_name or config.llm.default_model
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        self.langchain_llm = None  # Lazy initialization
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        provider: str = "openai"
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for chat models
            temperature: Optional temperature parameter
            max_tokens: Optional max tokens parameter
            stop_sequences: Optional list of stop sequences
            provider: LLM provider to use
            
        Returns:
            Generated text
        """
        if provider == "openai":
            return self._generate_openai(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for chat models
            temperature: Optional temperature parameter
            max_tokens: Optional max tokens parameter
            stop_sequences: Optional list of stop sequences
            
        Returns:
            Generated text
        """
        try:
            # Set default values if not provided
            temperature = temperature if temperature is not None else config.llm.temperature
            max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Log request
            logger.debug(f"Sending request to OpenAI API with model {self.model_name}")
            
            # Send request
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            end_time = time.time()
            
            # Log response time
            logger.debug(f"OpenAI API response time: {end_time - start_time:.2f} seconds")
            
            # Extract and return generated text
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
    
    def get_langchain_llm(self) -> ChatOpenAI:
        """
        Get a LangChain LLM instance.
        
        Returns:
            LangChain LLM instance
        """
        if self.langchain_llm is None:
            self.langchain_llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=config.llm.temperature,
                openai_api_key=config.llm.openai_api_key
            )
        
        return self.langchain_llm
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4 