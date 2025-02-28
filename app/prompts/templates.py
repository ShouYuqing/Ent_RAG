"""
Prompt template system with carefully designed templates and clear instructions.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from app.config import config
from app.prompts.system_prompts import SystemPromptManager
from app.prompts.few_shot import FewShotExampleManager
from app.prompts.chain_of_thought import ChainOfThoughtManager

logger = logging.getLogger("ent_rag.prompts.templates")


class PromptTemplate:
    """
    Represents a prompt template with placeholders for dynamic content.
    """
    
    def __init__(
        self,
        template_id: str,
        template: str,
        system_prompt: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a prompt template.
        
        Args:
            template_id: Unique identifier for the template
            template: The template string with placeholders
            system_prompt: Optional system prompt to use with this template
            description: Optional description of the template
            metadata: Optional metadata about the template
        """
        self.template_id = template_id
        self.template = template
        self.system_prompt = system_prompt
        self.description = description or ""
        self.metadata = metadata or {}
    
    def format(self, **kwargs) -> str:
        """
        Format the template by replacing placeholders with values.
        
        Args:
            **kwargs: Keyword arguments for placeholder replacement
            
        Returns:
            Formatted prompt string
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing placeholder in template {self.template_id}: {str(e)}")
            # Return template with missing placeholders marked
            return self.template.format(**{
                **kwargs,
                **{k: f"[MISSING: {k}]" for k in self._get_placeholders() if k not in kwargs}
            })
    
    def _get_placeholders(self) -> List[str]:
        """
        Extract placeholder names from the template.
        
        Returns:
            List of placeholder names
        """
        import re
        # Find all {placeholder} patterns in the template
        placeholders = re.findall(r'\{([^{}]*)\}', self.template)
        return placeholders
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            "template_id": self.template_id,
            "template": self.template,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """
        Create a template from a dictionary.
        
        Args:
            data: Dictionary representation of the template
            
        Returns:
            PromptTemplate instance
        """
        return cls(
            template_id=data["template_id"],
            template=data["template"],
            system_prompt=data.get("system_prompt"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )


class PromptTemplateManager:
    """
    Manages prompt templates and their generation.
    """
    
    def __init__(self):
        """Initialize the prompt template manager with necessary components."""
        self.templates = self._load_templates()
        self.system_prompt_manager = SystemPromptManager()
        self.few_shot_manager = FewShotExampleManager()
        self.chain_of_thought_manager = ChainOfThoughtManager()
    
    def generate_prompt(
        self,
        query: str,
        context: str,
        template_name: Optional[str] = None,
        use_few_shot: bool = True,
        use_chain_of_thought: bool = True
    ) -> str:
        """
        Generate a prompt using a template and context.
        
        Args:
            query: The user's query
            context: The retrieved context
            template_name: Name of the template to use (or None for default)
            use_few_shot: Whether to include few-shot examples
            use_chain_of_thought: Whether to include chain-of-thought instructions
            
        Returns:
            Generated prompt string
        """
        # Select template
        template = self._select_template(template_name, query)
        
        # Get system prompt
        system_prompt = self._get_system_prompt(template, query)
        
        # Get few-shot examples if requested
        few_shot_examples = ""
        if use_few_shot:
            few_shot_examples = self.few_shot_manager.get_examples(query, template.template_id)
        
        # Get chain-of-thought instructions if requested
        cot_instructions = ""
        if use_chain_of_thought:
            cot_instructions = self.chain_of_thought_manager.get_instructions(query, template.template_id)
        
        # Format template
        prompt = template.format(
            query=query,
            context=context,
            few_shot_examples=few_shot_examples,
            chain_of_thought=cot_instructions
        )
        
        # Combine system prompt and prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        return full_prompt
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of template information dictionaries
        """
        return [
            {
                "id": template.template_id,
                "description": template.description,
                "metadata": template.metadata
            }
            for template in self.templates.values()
        ]
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a template by ID.
        
        Args:
            template_id: ID of the template to get
            
        Returns:
            PromptTemplate instance or None if not found
        """
        return self.templates.get(template_id)
    
    def _select_template(
        self,
        template_name: Optional[str],
        query: str
    ) -> PromptTemplate:
        """
        Select an appropriate template based on name or query.
        
        Args:
            template_name: Name of the template to use (or None for automatic selection)
            query: The user's query
            
        Returns:
            Selected PromptTemplate instance
        """
        # If template name is provided and exists, use it
        if template_name and template_name in self.templates:
            return self.templates[template_name]
        
        # If template name is provided but doesn't exist, log warning
        if template_name:
            logger.warning(f"Template '{template_name}' not found, using default")
        
        # Automatic template selection based on query
        # For now, just use the default template
        return self.templates["default"]
    
    def _get_system_prompt(
        self,
        template: PromptTemplate,
        query: str
    ) -> str:
        """
        Get an appropriate system prompt for the template and query.
        
        Args:
            template: The selected template
            query: The user's query
            
        Returns:
            System prompt string
        """
        # If template has a specific system prompt, use it
        if template.system_prompt:
            return template.system_prompt
        
        # Otherwise, get a system prompt from the manager
        return self.system_prompt_manager.get_system_prompt(query, template.template_id)
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """
        Load templates from files or create default templates.
        
        Returns:
            Dictionary of template ID to PromptTemplate instance
        """
        templates = {}
        
        # Try to load templates from files
        templates_dir = Path(__file__).parent / "data" / "templates"
        if templates_dir.exists():
            for file_path in templates_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        template_data = json.load(f)
                        template = PromptTemplate.from_dict(template_data)
                        templates[template.template_id] = template
                except Exception as e:
                    logger.error(f"Error loading template from {file_path}: {str(e)}")
        
        # If no templates were loaded, create default templates
        if not templates:
            templates = self._create_default_templates()
        
        return templates
    
    def _create_default_templates(self) -> Dict[str, PromptTemplate]:
        """
        Create default prompt templates.
        
        Returns:
            Dictionary of template ID to PromptTemplate instance
        """
        templates = {}
        
        # Default template
        default_template = PromptTemplate(
            template_id="default",
            template="""
You are a helpful AI assistant. Answer the following question based on the provided context.
If the answer cannot be determined from the context, say "I don't know" and suggest where the information might be found.

Context:
{context}

{few_shot_examples}

Question: {query}

{chain_of_thought}

Answer:
""",
            description="General-purpose question answering template",
            metadata={"type": "qa", "version": "1.0"}
        )
        templates[default_template.template_id] = default_template
        
        # Summarization template
        summarization_template = PromptTemplate(
            template_id="summarize",
            template="""
Summarize the following information in a concise and informative way.
Focus on the key points and main ideas.

Context:
{context}

{few_shot_examples}

Question/Topic: {query}

{chain_of_thought}

Summary:
""",
            description="Template for summarizing information",
            metadata={"type": "summarization", "version": "1.0"}
        )
        templates[summarization_template.template_id] = summarization_template
        
        # Analysis template
        analysis_template = PromptTemplate(
            template_id="analyze",
            template="""
Analyze the following information and provide insights, patterns, and implications.
Be thorough and consider multiple perspectives.

Context:
{context}

{few_shot_examples}

Question/Topic for Analysis: {query}

{chain_of_thought}

Analysis:
""",
            description="Template for in-depth analysis",
            metadata={"type": "analysis", "version": "1.0"}
        )
        templates[analysis_template.template_id] = analysis_template
        
        # Comparison template
        comparison_template = PromptTemplate(
            template_id="compare",
            template="""
Compare and contrast the following information, highlighting similarities and differences.
Organize your response in a structured way.

Context:
{context}

{few_shot_examples}

Comparison Topic: {query}

{chain_of_thought}

Comparison:
""",
            description="Template for comparing information",
            metadata={"type": "comparison", "version": "1.0"}
        )
        templates[comparison_template.template_id] = comparison_template
        
        # Extraction template
        extraction_template = PromptTemplate(
            template_id="extract",
            template="""
Extract specific information from the context based on the query.
Present the extracted information in a structured format.

Context:
{context}

{few_shot_examples}

Information to Extract: {query}

{chain_of_thought}

Extracted Information:
""",
            description="Template for extracting specific information",
            metadata={"type": "extraction", "version": "1.0"}
        )
        templates[extraction_template.template_id] = extraction_template
        
        return templates 