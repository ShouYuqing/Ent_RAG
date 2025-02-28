"""
Prompt generator that combines all prompt engineering components.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from app.config import config
from app.prompts.templates import PromptTemplateManager
from app.prompts.system_prompts import SystemPromptManager
from app.prompts.few_shot import FewShotExampleManager
from app.prompts.chain_of_thought import ChainOfThoughtPrompt

logger = logging.getLogger("ent_rag.prompts.generator")


class PromptGenerator:
    """
    Generates optimized prompts by combining templates, system prompts,
    few-shot examples, and chain-of-thought techniques.
    """
    
    def __init__(self):
        """Initialize the prompt generator with all prompt components."""
        self.template_manager = PromptTemplateManager()
        self.system_prompt_manager = SystemPromptManager()
        self.few_shot_manager = FewShotExampleManager()
        self.cot_prompt = ChainOfThoughtPrompt()
    
    def generate_prompt(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
        template_id: Optional[str] = None,
        use_few_shot: bool = True,
        use_cot: bool = True,
        num_examples: int = 2,
        example_format: str = "standard",
        max_tokens: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Generate an optimized prompt for the given query and context.
        
        Args:
            query: The user's query
            context: The retrieved context (string or list of documents)
            template_id: Optional template ID to use
            use_few_shot: Whether to include few-shot examples
            use_cot: Whether to include chain-of-thought instructions
            num_examples: Number of few-shot examples to include
            example_format: Format for few-shot examples (standard, concise, or detailed)
            max_tokens: Maximum tokens for the prompt
            
        Returns:
            Dictionary with 'system_prompt' and 'prompt' keys
        """
        # Format context if it's a list of documents
        formatted_context = self._format_context(context)
        
        # Get appropriate system prompt
        system_prompt = self.system_prompt_manager.get_system_prompt(query, template_id)
        
        # Get few-shot examples if requested
        few_shot_text = ""
        if use_few_shot:
            examples = self.few_shot_manager.get_examples(query, template_id, num_examples)
            few_shot_text = self.few_shot_manager.format_examples(examples, example_format)
        
        # Get chain-of-thought instructions if requested
        cot_text = ""
        if use_cot:
            query_type = self._determine_query_type(query)
            cot_text = self.cot_prompt.get_cot_prompt(query, query_type)
        
        # Generate the prompt using the template manager
        prompt_values = {
            "query": query,
            "context": formatted_context,
            "few_shot_examples": few_shot_text,
            "chain_of_thought": cot_text
        }
        
        prompt = self.template_manager.generate_prompt(
            template_id=template_id,
            query=query,
            values=prompt_values
        )
        
        # Truncate prompt if max_tokens is specified
        if max_tokens:
            prompt = self._truncate_prompt(prompt, max_tokens)
        
        return {
            "system_prompt": system_prompt,
            "prompt": prompt
        }
    
    def _format_context(self, context: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Format context for inclusion in a prompt.
        
        Args:
            context: The context as a string or list of documents
            
        Returns:
            Formatted context string
        """
        if isinstance(context, str):
            return context
        
        formatted = ""
        
        for i, doc in enumerate(context, 1):
            formatted += f"Document {i}:\n"
            
            # Add title if available
            if "title" in doc.get("metadata", {}):
                formatted += f"Title: {doc['metadata']['title']}\n"
            
            # Add content
            if "content" in doc:
                formatted += f"{doc['content']}\n"
            elif "text" in doc:
                formatted += f"{doc['text']}\n"
            
            # Add metadata if available
            if "metadata" in doc and doc["metadata"]:
                formatted += "Metadata:\n"
                for key, value in doc["metadata"].items():
                    if key != "title":  # Skip title as it's already added
                        formatted += f"- {key}: {value}\n"
            
            formatted += "\n"
        
        return formatted
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine the type of query.
        
        Args:
            query: The user's query
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        # Check for summarization queries
        if any(keyword in query_lower for keyword in ["summarize", "summary", "summarization", "brief overview"]):
            return "summarize"
        
        # Check for analysis queries
        if any(keyword in query_lower for keyword in ["analyze", "analysis", "examine", "evaluate", "assess"]):
            return "analyze"
        
        # Check for comparison queries
        if any(keyword in query_lower for keyword in ["compare", "comparison", "contrast", "difference", "similarities"]):
            return "compare"
        
        # Check for extraction queries
        if any(keyword in query_lower for keyword in ["extract", "find", "locate", "identify", "list"]):
            return "extract"
        
        # Check for reasoning queries
        if any(keyword in query_lower for keyword in ["why", "explain", "reason", "cause", "effect", "impact"]):
            return "reasoning"
        
        # Check for problem-solving queries
        if any(keyword in query_lower for keyword in ["solve", "calculate", "compute", "determine", "find solution"]):
            return "problem_solving"
        
        # Check for decision-making queries
        if any(keyword in query_lower for keyword in ["decide", "choose", "select", "recommend", "best option"]):
            return "decision_making"
        
        # Default to QA
        return "default"
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Truncate prompt to fit within max_tokens.
        
        Args:
            prompt: The prompt to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated prompt
        """
        # Simple approximation: 1 token ≈ 4 characters
        char_limit = max_tokens * 4
        
        if len(prompt) <= char_limit:
            return prompt
        
        # Find a good breaking point (end of a sentence)
        breaking_point = prompt[:char_limit].rfind(".")
        if breaking_point == -1:
            breaking_point = prompt[:char_limit].rfind("\n")
        if breaking_point == -1:
            breaking_point = char_limit
        
        truncated = prompt[:breaking_point + 1]
        
        # Add a note about truncation
        truncated += "\n\n[Note: The context has been truncated due to length constraints.]"
        
        return truncated 