"""
Token management for optimizing context window usage.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
import tiktoken

from app.config import config

logger = logging.getLogger("ent_rag.context.token_management")


class TokenManager:
    """
    Implements token management for optimizing context window usage.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the token manager.
        
        Args:
            model_name: Name of the LLM model to use for token counting
        """
        self.model_name = model_name or config.llm.default_model
        self.encoding = self._get_encoding(self.model_name)
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        # Encode the text to get tokens
        tokens = self.encoding.encode(text)
        
        return len(tokens)
    
    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
        truncation_strategy: str = "end"
    ) -> str:
        """
        Truncate text to fit within a token limit.
        
        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed
            truncation_strategy: Strategy for truncation (start, end, or middle)
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        # Count tokens in the text
        tokens = self.encoding.encode(text)
        
        # If already within limit, return as is
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate based on strategy
        if truncation_strategy == "start":
            # Truncate from the start
            truncated_tokens = tokens[-max_tokens:]
        elif truncation_strategy == "middle":
            # Truncate from the middle
            half_max = max_tokens // 2
            truncated_tokens = tokens[:half_max] + tokens[-half_max:]
        else:
            # Default: truncate from the end
            truncated_tokens = tokens[:max_tokens]
        
        # Decode back to text
        truncated_text = self.encoding.decode(truncated_tokens)
        
        return truncated_text
    
    def optimize_context(
        self,
        query: str,
        context_parts: List[Dict[str, Any]],
        max_tokens: int,
        reserved_tokens: int = 500
    ) -> Tuple[str, int]:
        """
        Optimize context to fit within token limit while preserving most relevant information.
        
        Args:
            query: The user's query
            context_parts: List of context parts with content and metadata
            max_tokens: Maximum number of tokens for the entire context
            reserved_tokens: Number of tokens to reserve for the query and other parts
            
        Returns:
            Tuple of (optimized context string, token count)
        """
        logger.debug(f"Optimizing context with {len(context_parts)} parts to fit in {max_tokens} tokens")
        
        # Count tokens in the query
        query_tokens = self.count_tokens(query)
        
        # Calculate available tokens for context
        available_tokens = max_tokens - query_tokens - reserved_tokens
        if available_tokens <= 0:
            logger.warning(f"Not enough tokens available for context. Query uses {query_tokens} tokens, reserved {reserved_tokens}")
            return "", 0
        
        # Sort context parts by relevance (assuming they have a "score" field)
        sorted_parts = sorted(context_parts, key=lambda x: x.get("score", 0), reverse=True)
        
        # Initialize optimized context
        optimized_parts = []
        total_tokens = 0
        
        # Add parts until we reach the token limit
        for part in sorted_parts:
            content = part.get("content", "")
            if not content:
                continue
            
            # Format the part with metadata
            formatted_part = self._format_context_part(part)
            
            # Count tokens in this part
            part_tokens = self.count_tokens(formatted_part)
            
            # If adding this part would exceed the limit, try to truncate it
            if total_tokens + part_tokens > available_tokens:
                # Calculate remaining tokens
                remaining_tokens = available_tokens - total_tokens
                
                # If we have enough tokens for at least part of the content
                if remaining_tokens > 50:  # Ensure we have enough tokens for meaningful content
                    # Truncate the part to fit
                    truncated_part = self.truncate_to_token_limit(formatted_part, remaining_tokens)
                    optimized_parts.append(truncated_part)
                    total_tokens += self.count_tokens(truncated_part)
                
                # Break since we've reached the limit
                break
            
            # Otherwise, add the full part
            optimized_parts.append(formatted_part)
            total_tokens += part_tokens
        
        # Combine optimized parts
        optimized_context = "\n\n".join(optimized_parts)
        
        logger.debug(f"Optimized context to {total_tokens} tokens")
        return optimized_context, total_tokens
    
    def _format_context_part(self, part: Dict[str, Any]) -> str:
        """
        Format a context part with metadata.
        
        Args:
            part: Context part with content and metadata
            
        Returns:
            Formatted context part
        """
        content = part.get("content", "")
        metadata = part.get("metadata", {})
        
        # Extract metadata fields
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown source")
        
        # Format with minimal metadata to save tokens
        formatted_part = f"[{title}] {content}"
        
        return formatted_part
    
    def _get_encoding(self, model_name: str):
        """
        Get the appropriate encoding for a model.
        
        Args:
            model_name: Name of the LLM model
            
        Returns:
            Encoding for the model
        """
        try:
            # For newer models like GPT-4 and GPT-3.5-turbo
            if "gpt-4" in model_name or "gpt-3.5-turbo" in model_name:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            # For older models
            else:
                return tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo)
            return tiktoken.get_encoding("cl100k_base")


class AdvancedTokenManager(TokenManager):
    """
    Advanced token management with more sophisticated optimization strategies.
    """
    
    def __init__(self, model_name: str = None):
        """Initialize the advanced token manager."""
        super().__init__(model_name)
    
    def optimize_context(
        self,
        query: str,
        context_parts: List[Dict[str, Any]],
        max_tokens: int,
        reserved_tokens: int = 500
    ) -> Tuple[str, int]:
        """
        Optimize context with advanced strategies.
        
        Args:
            query: The user's query
            context_parts: List of context parts with content and metadata
            max_tokens: Maximum number of tokens for the entire context
            reserved_tokens: Number of tokens to reserve for the query and other parts
            
        Returns:
            Tuple of (optimized context string, token count)
        """
        logger.debug(f"Advanced optimization with {len(context_parts)} parts to fit in {max_tokens} tokens")
        
        # Count tokens in the query
        query_tokens = self.count_tokens(query)
        
        # Calculate available tokens for context
        available_tokens = max_tokens - query_tokens - reserved_tokens
        if available_tokens <= 0:
            logger.warning(f"Not enough tokens available for context. Query uses {query_tokens} tokens, reserved {reserved_tokens}")
            return "", 0
        
        # Extract query keywords for relevance assessment
        query_keywords = self._extract_keywords(query)
        
        # Score parts based on relevance to query keywords
        scored_parts = self._score_parts_by_keywords(context_parts, query_keywords)
        
        # Group parts by document to maintain coherence
        grouped_parts = self._group_parts_by_document(scored_parts)
        
        # Allocate tokens to groups based on relevance
        allocated_groups = self._allocate_tokens_to_groups(grouped_parts, available_tokens)
        
        # Format and combine parts
        optimized_parts = []
        total_tokens = 0
        
        for group in allocated_groups:
            group_parts = group["parts"]
            allocated_tokens = group["allocated_tokens"]
            
            # Format and truncate group parts
            formatted_group = self._format_group(group_parts, allocated_tokens)
            
            optimized_parts.append(formatted_group)
            total_tokens += self.count_tokens(formatted_group)
        
        # Combine optimized parts
        optimized_context = "\n\n".join(optimized_parts)
        
        logger.debug(f"Advanced optimization resulted in {total_tokens} tokens")
        return optimized_context, total_tokens
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction: remove stop words and get unique words
        stop_words = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "by", "to", "and", "or", "is", "are"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _score_parts_by_keywords(
        self,
        parts: List[Dict[str, Any]],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score parts based on keyword presence.
        
        Args:
            parts: List of context parts
            keywords: List of query keywords
            
        Returns:
            List of parts with keyword scores
        """
        scored_parts = []
        
        for part in parts:
            content = part.get("content", "").lower()
            
            # Calculate keyword score
            keyword_score = 0
            for keyword in keywords:
                if keyword in content:
                    # Count occurrences
                    occurrences = content.count(keyword)
                    # Add score based on occurrences (with diminishing returns)
                    keyword_score += min(occurrences, 3) * 0.2
            
            # Combine with existing score
            existing_score = part.get("score", 0.5)
            combined_score = 0.7 * existing_score + 0.3 * keyword_score
            
            # Create scored part
            scored_part = part.copy()
            scored_part["score"] = combined_score
            scored_part["keyword_score"] = keyword_score
            
            scored_parts.append(scored_part)
        
        # Sort by combined score
        scored_parts.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_parts
    
    def _group_parts_by_document(
        self,
        parts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group parts by source document.
        
        Args:
            parts: List of context parts
            
        Returns:
            List of groups, each containing parts from the same document
        """
        # Group parts by document ID
        groups = {}
        
        for part in parts:
            metadata = part.get("metadata", {})
            doc_id = metadata.get("document_id", metadata.get("source", "unknown"))
            
            if doc_id not in groups:
                groups[doc_id] = {
                    "doc_id": doc_id,
                    "parts": [],
                    "total_score": 0,
                    "max_score": 0
                }
            
            # Add part to group
            groups[doc_id]["parts"].append(part)
            
            # Update group scores
            score = part.get("score", 0)
            groups[doc_id]["total_score"] += score
            groups[doc_id]["max_score"] = max(groups[doc_id]["max_score"], score)
        
        # Convert to list and sort by max score
        grouped_list = list(groups.values())
        grouped_list.sort(key=lambda x: x["max_score"], reverse=True)
        
        return grouped_list
    
    def _allocate_tokens_to_groups(
        self,
        groups: List[Dict[str, Any]],
        available_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Allocate tokens to groups based on relevance.
        
        Args:
            groups: List of document groups
            available_tokens: Total available tokens
            
        Returns:
            List of groups with allocated tokens
        """
        # Calculate total score across all groups
        total_score = sum(group["max_score"] for group in groups)
        
        # Allocate tokens proportionally to group scores
        allocated_groups = []
        remaining_tokens = available_tokens
        
        for group in groups:
            # Skip empty groups
            if not group["parts"]:
                continue
            
            # Calculate token allocation based on score
            if total_score > 0:
                allocation_ratio = group["max_score"] / total_score
                allocated_tokens = int(available_tokens * allocation_ratio)
            else:
                # Equal allocation if no scores
                allocated_tokens = available_tokens // len(groups)
            
            # Ensure minimum allocation
            allocated_tokens = max(allocated_tokens, 100)
            
            # Ensure we don't exceed remaining tokens
            allocated_tokens = min(allocated_tokens, remaining_tokens)
            
            # Skip if allocation is too small
            if allocated_tokens < 50:
                continue
            
            # Create allocated group
            allocated_group = group.copy()
            allocated_group["allocated_tokens"] = allocated_tokens
            
            allocated_groups.append(allocated_group)
            remaining_tokens -= allocated_tokens
            
            # Stop if we've allocated all tokens
            if remaining_tokens <= 0:
                break
        
        # If we have remaining tokens and groups, redistribute
        if remaining_tokens > 0 and allocated_groups:
            # Add remaining tokens to the highest-scored group
            allocated_groups[0]["allocated_tokens"] += remaining_tokens
        
        return allocated_groups
    
    def _format_group(
        self,
        parts: List[Dict[str, Any]],
        allocated_tokens: int
    ) -> str:
        """
        Format a group of parts to fit within allocated tokens.
        
        Args:
            parts: List of parts in the group
            allocated_tokens: Tokens allocated to this group
            
        Returns:
            Formatted group text
        """
        # Sort parts by score
        sorted_parts = sorted(parts, key=lambda x: x.get("score", 0), reverse=True)
        
        # Format parts
        formatted_parts = []
        total_tokens = 0
        
        # Get metadata from first part for group header
        first_part = sorted_parts[0] if sorted_parts else {}
        metadata = first_part.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown source")
        
        # Create group header
        group_header = f"[{title}] (Source: {source})"
        header_tokens = self.count_tokens(group_header)
        
        # Adjust allocated tokens for header
        content_tokens = allocated_tokens - header_tokens
        
        # Add parts until we reach the token limit
        for part in sorted_parts:
            content = part.get("content", "")
            if not content:
                continue
            
            # Count tokens in this part
            part_tokens = self.count_tokens(content)
            
            # If adding this part would exceed the limit, try to truncate it
            if total_tokens + part_tokens > content_tokens:
                # Calculate remaining tokens
                remaining_tokens = content_tokens - total_tokens
                
                # If we have enough tokens for at least part of the content
                if remaining_tokens > 30:  # Ensure we have enough tokens for meaningful content
                    # Truncate the part to fit
                    truncated_content = self.truncate_to_token_limit(content, remaining_tokens)
                    formatted_parts.append(truncated_content)
                    total_tokens += self.count_tokens(truncated_content)
                
                # Break since we've reached the limit
                break
            
            # Otherwise, add the full part
            formatted_parts.append(content)
            total_tokens += part_tokens
        
        # Combine header and parts
        formatted_group = group_header + "\n\n" + "\n\n".join(formatted_parts)
        
        return formatted_group 