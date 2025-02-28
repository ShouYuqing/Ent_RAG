"""
System prompts that define response format and constraints.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from app.config import config

logger = logging.getLogger("ent_rag.prompts.system_prompts")


class SystemPromptManager:
    """
    Manages system prompts for different types of queries and templates.
    """
    
    def __init__(self):
        """Initialize the system prompt manager."""
        self.system_prompts = self._load_system_prompts()
    
    def get_system_prompt(
        self,
        query: str,
        template_id: Optional[str] = None
    ) -> str:
        """
        Get an appropriate system prompt for a query and template.
        
        Args:
            query: The user's query
            template_id: Optional template ID to get a specific system prompt
            
        Returns:
            System prompt string
        """
        # If template ID is provided, try to get a template-specific system prompt
        if template_id and template_id in self.system_prompts:
            return self.system_prompts[template_id]
        
        # Otherwise, determine the type of query and return an appropriate system prompt
        query_type = self._determine_query_type(query)
        
        # Get system prompt for query type
        if query_type in self.system_prompts:
            return self.system_prompts[query_type]
        
        # Fallback to default system prompt
        return self.system_prompts["default"]
    
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
        
        # Default to QA
        return "default"
    
    def _load_system_prompts(self) -> Dict[str, str]:
        """
        Load system prompts from files or create default system prompts.
        
        Returns:
            Dictionary of prompt type to system prompt string
        """
        system_prompts = {}
        
        # Try to load system prompts from files
        prompts_dir = Path(__file__).parent / "data" / "system_prompts"
        if prompts_dir.exists():
            for file_path in prompts_dir.glob("*.txt"):
                try:
                    prompt_type = file_path.stem
                    with open(file_path, "r") as f:
                        system_prompts[prompt_type] = f.read().strip()
                except Exception as e:
                    logger.error(f"Error loading system prompt from {file_path}: {str(e)}")
        
        # If no system prompts were loaded, create default system prompts
        if not system_prompts:
            system_prompts = self._create_default_system_prompts()
        
        return system_prompts
    
    def _create_default_system_prompts(self) -> Dict[str, str]:
        """
        Create default system prompts.
        
        Returns:
            Dictionary of prompt type to system prompt string
        """
        system_prompts = {}
        
        # Default (QA) system prompt
        system_prompts["default"] = """
You are a helpful, accurate, and concise assistant. Your task is to answer questions based on the provided context.

Guidelines:
1. Base your answers solely on the information in the context provided.
2. If the answer is not in the context, say "I don't know" - do not make up information.
3. Keep your answers concise and to the point.
4. If the context contains conflicting information, acknowledge this in your answer.
5. If the context contains technical information, explain it in a clear and accessible way.
6. Use bullet points or numbered lists for clarity when appropriate.
7. Cite specific parts of the context to support your answer when relevant.
"""
        
        # Summarization system prompt
        system_prompts["summarize"] = """
You are a skilled summarizer. Your task is to create concise, accurate summaries of the provided information.

Guidelines:
1. Focus on the key points and main ideas in the context.
2. Maintain the original meaning and intent of the information.
3. Organize the summary in a logical and coherent structure.
4. Be objective and avoid adding your own opinions or interpretations.
5. Use clear and concise language.
6. Adjust the level of detail based on the complexity of the information.
7. Highlight important facts, figures, and conclusions.
8. Aim for approximately 10-20% of the original length unless otherwise specified.
"""
        
        # Analysis system prompt
        system_prompts["analyze"] = """
You are an analytical expert. Your task is to provide insightful analysis of the information in the context.

Guidelines:
1. Identify key patterns, trends, and relationships in the data or information.
2. Consider multiple perspectives and interpretations.
3. Evaluate the strengths and limitations of the information provided.
4. Support your analysis with specific evidence from the context.
5. Distinguish between facts, assumptions, and inferences.
6. Consider implications and potential consequences.
7. Organize your analysis in a structured and logical manner.
8. Be objective and balanced in your assessment.
"""
        
        # Comparison system prompt
        system_prompts["compare"] = """
You are a comparison specialist. Your task is to compare and contrast different elements based on the provided context.

Guidelines:
1. Identify the key elements to be compared.
2. Establish clear criteria for comparison.
3. Highlight both similarities and differences.
4. Organize your comparison in a structured format (e.g., point-by-point or subject-by-subject).
5. Use specific examples from the context to support your comparisons.
6. Be balanced and objective in your assessment.
7. Consider the significance of the similarities and differences.
8. Summarize the most important points of comparison in a conclusion.
"""
        
        # Extraction system prompt
        system_prompts["extract"] = """
You are an information extraction expert. Your task is to extract specific information from the provided context.

Guidelines:
1. Focus only on extracting the information requested in the query.
2. Present the extracted information in a clear, structured format.
3. Use bullet points, tables, or lists when appropriate.
4. Include all relevant details related to the requested information.
5. Maintain accuracy and avoid paraphrasing that might change the meaning.
6. If the requested information is not in the context, clearly state this.
7. Organize the extracted information in a logical order.
8. Be precise and avoid unnecessary elaboration.
"""
        
        return system_prompts 