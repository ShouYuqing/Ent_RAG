"""
Ent_RAG Core Component
======================

This module serves as the central orchestrator for the Ent_RAG system, integrating all components
into a cohesive retrieval-augmented generation pipeline.

The core component coordinates the flow of information between:
- Document ingestion and storage
- Query processing and retrieval
- Context optimization and prioritization
- Prompt generation and LLM interaction

It provides a unified interface for using the RAG system while abstracting away the complexity
of the individual components.

Author: yuqings
Created: February 2024
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import time
from datetime import datetime

from app.data.document_store import DocumentStore
from app.data.processor import DocumentProcessor
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranking import ReRanker
from app.context.prioritization import ContextPrioritizer
from app.context.metadata import MetadataEnricher
from app.context.token_management import TokenManager
from app.prompts.templates import PromptTemplateManager
from app.prompts.generator import PromptGenerator
from app.models.llm import LLMManager
from app.config import config

logger = logging.getLogger("ent_rag.core")


class EntRAG:
    """
    Enterprise Retrieval-Augmented Generation (EntRAG) System
    
    This class serves as the main entry point for the RAG system, orchestrating the interaction
    between various components to provide a seamless experience for retrieving information and
    generating responses based on that information.
    
    The system follows a pipeline architecture:
    1. Query Processing: Analyze and potentially rewrite the user query
    2. Retrieval: Find relevant documents from the knowledge base
    3. Context Processing: Prioritize, enrich, and optimize retrieved information
    4. Prompt Generation: Create an effective prompt incorporating the context
    5. Response Generation: Use an LLM to generate a response based on the prompt
    
    Attributes:
        document_store (DocumentStore): Storage for documents and their metadata
        document_processor (DocumentProcessor): Processes documents for ingestion
        retriever (HybridRetriever): Retrieves relevant documents for a query
        reranker (ReRanker): Reranks retrieved documents for better relevance
        context_prioritizer (ContextPrioritizer): Prioritizes context based on relevance
        metadata_enricher (MetadataEnricher): Enriches documents with metadata
        token_manager (TokenManager): Manages token usage for context optimization
        prompt_template_manager (PromptTemplateManager): Manages prompt templates
        prompt_generator (PromptGenerator): Generates prompts for the LLM
        llm_manager (LLMManager): Manages interactions with the LLM
    """
    
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        document_processor: Optional[DocumentProcessor] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[ReRanker] = None,
        context_prioritizer: Optional[ContextPrioritizer] = None,
        metadata_enricher: Optional[MetadataEnricher] = None,
        token_manager: Optional[TokenManager] = None,
        prompt_template_manager: Optional[PromptTemplateManager] = None,
        prompt_generator: Optional[PromptGenerator] = None,
        llm_manager: Optional[LLMManager] = None
    ):
        """
        Initialize the EntRAG system with its component parts.
        
        Components are created if not provided, allowing for both default initialization
        and customized component injection.
        
        Args:
            document_store: Storage for documents and their metadata
            document_processor: Processes documents for ingestion
            retriever: Retrieves relevant documents for a query
            reranker: Reranks retrieved documents for better relevance
            context_prioritizer: Prioritizes context based on relevance
            metadata_enricher: Enriches documents with metadata
            token_manager: Manages token usage for context optimization
            prompt_template_manager: Manages prompt templates
            prompt_generator: Generates prompts for the LLM
            llm_manager: Manages interactions with the LLM
        """
        # Initialize document storage and processing
        self.document_store = document_store or DocumentStore()
        self.document_processor = document_processor or DocumentProcessor(document_store=self.document_store)
        
        # Initialize retrieval components
        self.retriever = retriever or HybridRetriever(document_store=self.document_store)
        self.reranker = reranker or ReRanker()
        
        # Initialize context processing components
        self.context_prioritizer = context_prioritizer or ContextPrioritizer()
        self.metadata_enricher = metadata_enricher or MetadataEnricher()
        self.token_manager = token_manager or TokenManager()
        
        # Initialize prompt and LLM components
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
        self.prompt_generator = prompt_generator or PromptGenerator(template_manager=self.prompt_template_manager)
        self.llm_manager = llm_manager or LLMManager()
        
        logger.info("EntRAG system initialized")
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        chunk: bool = True,
        enrich_metadata: bool = True
    ) -> Union[str, List[str]]:
        """
        Add a document to the RAG system.
        
        This method processes the document, optionally chunks it, enriches its metadata,
        and adds it to the document store.
        
        Args:
            content: The document content
            metadata: Optional metadata for the document
            doc_id: Optional document ID (generated if not provided)
            chunk: Whether to chunk the document
            enrich_metadata: Whether to enrich the document's metadata
            
        Returns:
            Document ID or list of document IDs if chunked
        """
        logger.info(f"Adding document{' with chunking' if chunk else ''}")
        
        # Initialize metadata if not provided
        metadata = metadata or {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Enrich metadata if requested
        if enrich_metadata:
            metadata = self.metadata_enricher.enrich(content, metadata)
        
        # Process document
        if chunk:
            # Process and chunk document
            doc_ids = self.document_processor.process_text(content, metadata, chunk=True)
            logger.info(f"Document added and chunked into {len(doc_ids)} parts")
            return doc_ids
        else:
            # Add document without chunking
            doc_id = self.document_store.add_document(content, metadata, doc_id)
            logger.info(f"Document added with ID: {doc_id}")
            return doc_id
    
    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        rerank: bool = True,
        max_tokens: Optional[int] = None,
        template_name: str = "qa",
        temperature: Optional[float] = None,
        return_context: bool = False,
        return_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline and generate a response.
        
        This method:
        1. Retrieves relevant documents
        2. Optionally reranks them
        3. Prioritizes and formats the context
        4. Generates a prompt
        5. Obtains a response from the LLM
        
        Args:
            query: The user's query
            filters: Optional metadata filters for retrieval
            top_k: Number of documents to retrieve
            rerank: Whether to rerank retrieved documents
            max_tokens: Maximum tokens for context (default from config)
            template_name: Name of the prompt template to use
            temperature: Temperature for LLM generation
            return_context: Whether to include context in the response
            return_prompt: Whether to include the prompt in the response
            
        Returns:
            Dictionary containing the response and optionally context and prompt
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        # Set default max tokens if not provided
        if max_tokens is None:
            max_tokens = config.llm.max_context_tokens
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        documents = self.retriever.retrieve(query, filters=filters, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
        
        # Step 2: Rerank documents if requested
        if rerank and documents:
            rerank_start = time.time()
            documents = self.reranker.rerank(query, documents)
            rerank_time = time.time() - rerank_start
            logger.info(f"Reranked documents in {rerank_time:.2f}s")
        
        # Step 3: Prioritize and format context
        context_start = time.time()
        context = self.context_prioritizer.prioritize(query, documents, max_tokens=max_tokens)
        context_time = time.time() - context_start
        logger.info(f"Processed context in {context_time:.2f}s")
        
        # Step 4: Generate prompt
        prompt_start = time.time()
        prompt = self.prompt_generator.generate(
            query=query,
            context=context,
            template_name=template_name
        )
        prompt_time = time.time() - prompt_start
        logger.info(f"Generated prompt in {prompt_time:.2f}s")
        
        # Step 5: Generate response
        llm_start = time.time()
        response = self.llm_manager.generate(
            prompt=prompt,
            temperature=temperature
        )
        llm_time = time.time() - llm_start
        logger.info(f"Generated response in {llm_time:.2f}s")
        
        # Prepare result
        total_time = time.time() - start_time
        result = {
            "query": query,
            "response": response,
            "document_count": len(documents),
            "timing": {
                "total": total_time,
                "retrieval": retrieval_time,
                "reranking": rerank_time if rerank else 0,
                "context": context_time,
                "prompt": prompt_time,
                "llm": llm_time
            }
        }
        
        # Add context if requested
        if return_context:
            result["context"] = context
        
        # Add prompt if requested
        if return_prompt:
            result["prompt"] = prompt
        
        logger.info(f"Query processed in {total_time:.2f}s")
        return result
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document as a dictionary or None if not found
        """
        document = self.document_store.get_document(doc_id)
        if document:
            return document.to_dict()
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if the document was deleted, False otherwise
        """
        return self.document_store.delete_document(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary of system statistics
        """
        doc_count = self.document_store.count_documents()
        vector_stats = self.document_store.vector_store.get_collection_stats()
        
        return {
            "document_count": doc_count,
            "vector_store": vector_stats,
            "version": "1.0.0",
            "author": "yuqings"
        }


# Singleton instance for easy import
ent_rag = EntRAG() 