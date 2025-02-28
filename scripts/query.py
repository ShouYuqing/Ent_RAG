#!/usr/bin/env python
"""
Query script for the Ent_RAG system.

This script allows querying the RAG system from the command line.
"""

import argparse
import logging
import os
import sys
import json
from typing import Dict, List, Optional, Any, Union

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.document_store import DocumentStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranking import ReRanker
from app.context.prioritization import ContextPrioritizer
from app.context.token_management import TokenManager
from app.prompts.templates import PromptTemplateManager
from app.prompts.generator import PromptGenerator
from app.models.llm import LLMManager
from app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("query.log")
    ]
)

logger = logging.getLogger("ent_rag.query")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query the Ent_RAG system")
    
    # Query arguments
    parser.add_argument("query", type=str, help="Query to process")
    
    # Retrieval arguments
    parser.add_argument("--retriever", type=str, default="hybrid", choices=["hybrid", "bm25", "semantic"], help="Retriever to use")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking of results")
    
    # Context arguments
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum tokens for context")
    
    # Output arguments
    parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    parser.add_argument("--show-prompt", action="store_true", help="Show generated prompt")
    parser.add_argument("--output-file", type=str, help="Output file for response")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()


def query_system(
    query: str,
    retriever_type: str = "hybrid",
    top_k: int = 5,
    rerank: bool = False,
    max_tokens: int = 4000,
    show_context: bool = False,
    show_prompt: bool = False
) -> Dict[str, Any]:
    """
    Query the RAG system.
    
    Args:
        query: User query
        retriever_type: Type of retriever to use
        top_k: Number of documents to retrieve
        rerank: Whether to rerank results
        max_tokens: Maximum tokens for context
        show_context: Whether to show retrieved context
        show_prompt: Whether to show generated prompt
        
    Returns:
        Dictionary with query results
    """
    # Initialize components
    document_store = DocumentStore()
    
    # Select retriever
    if retriever_type == "hybrid":
        retriever = HybridRetriever(document_store=document_store)
    elif retriever_type == "bm25":
        from app.retrieval.keyword import KeywordRetriever
        retriever = KeywordRetriever(document_store=document_store)
    elif retriever_type == "semantic":
        from app.retrieval.semantic import SemanticRetriever
        retriever = SemanticRetriever(document_store=document_store)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    # Initialize other components
    reranker = ReRanker() if rerank else None
    context_prioritizer = ContextPrioritizer()
    token_manager = TokenManager()
    prompt_template_manager = PromptTemplateManager()
    prompt_generator = PromptGenerator()
    llm_manager = LLMManager()
    
    # Retrieve documents
    logger.info(f"Retrieving documents for query: {query}")
    documents = retriever.retrieve(query, top_k=top_k)
    
    # Rerank if requested
    if rerank and documents:
        logger.info("Reranking documents")
        documents = reranker.rerank(query, documents)
    
    # Prioritize context
    logger.info("Prioritizing context")
    context = context_prioritizer.prioritize(query, documents, max_tokens=max_tokens)
    
    # Generate prompt
    logger.info("Generating prompt")
    prompt = prompt_generator.generate(
        query=query,
        context=context,
        template_name="qa"
    )
    
    # Generate response
    logger.info("Generating response")
    response = llm_manager.generate(prompt)
    
    # Prepare result
    result = {
        "query": query,
        "response": response,
        "document_count": len(documents)
    }
    
    # Add context if requested
    if show_context:
        result["context"] = context
    
    # Add prompt if requested
    if show_prompt:
        result["prompt"] = prompt
    
    return result


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger("ent_rag").setLevel(logging.DEBUG)
    
    # Query system
    result = query_system(
        query=args.query,
        retriever_type=args.retriever,
        top_k=args.top_k,
        rerank=args.rerank,
        max_tokens=args.max_tokens,
        show_context=args.show_context,
        show_prompt=args.show_prompt
    )
    
    # Print response
    print("\n" + "=" * 80)
    print("QUERY:", result["query"])
    print("=" * 80)
    print("RESPONSE:")
    print(result["response"])
    print("=" * 80)
    
    # Print context if requested
    if args.show_context and "context" in result:
        print("CONTEXT:")
        print(result["context"])
        print("=" * 80)
    
    # Print prompt if requested
    if args.show_prompt and "prompt" in result:
        print("PROMPT:")
        print(result["prompt"])
        print("=" * 80)
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 