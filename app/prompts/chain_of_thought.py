"""
Chain-of-thought prompting for step-by-step reasoning.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union

from app.config import config

logger = logging.getLogger("ent_rag.prompts.chain_of_thought")


class ChainOfThoughtPrompt:
    """
    Implements chain-of-thought prompting techniques for step-by-step reasoning.
    """
    
    def __init__(self):
        """Initialize the chain-of-thought prompt generator."""
        self.cot_instructions = self._load_cot_instructions()
    
    def get_cot_prompt(
        self,
        query: str,
        query_type: Optional[str] = None
    ) -> str:
        """
        Get chain-of-thought instructions for a query.
        
        Args:
            query: The user's query
            query_type: Optional query type to get specific instructions
            
        Returns:
            Chain-of-thought instructions string
        """
        # If query type is not provided, determine it
        if not query_type:
            query_type = self._determine_query_type(query)
        
        # Get instructions for query type
        if query_type in self.cot_instructions:
            return self.cot_instructions[query_type]
        
        # Fallback to default instructions
        return self.cot_instructions["default"]
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine the type of query for chain-of-thought prompting.
        
        Args:
            query: The user's query
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        # Check for complex reasoning queries
        if any(keyword in query_lower for keyword in ["why", "explain", "reason", "cause", "effect", "impact"]):
            return "reasoning"
        
        # Check for problem-solving queries
        if any(keyword in query_lower for keyword in ["solve", "calculate", "compute", "determine", "find solution"]):
            return "problem_solving"
        
        # Check for decision-making queries
        if any(keyword in query_lower for keyword in ["decide", "choose", "select", "recommend", "best option"]):
            return "decision_making"
        
        # Check for analysis queries
        if any(keyword in query_lower for keyword in ["analyze", "analysis", "examine", "evaluate", "assess"]):
            return "analysis"
        
        # Default to standard chain-of-thought
        return "default"
    
    def _load_cot_instructions(self) -> Dict[str, str]:
        """
        Load chain-of-thought instructions.
        
        Returns:
            Dictionary of query type to instruction string
        """
        cot_instructions = {}
        
        # Default chain-of-thought instructions
        cot_instructions["default"] = """
Before providing your final answer, think step-by-step through the problem:

1. Break down the question into its key components
2. Consider what information from the context is relevant to each component
3. Reason through each part systematically
4. Synthesize your findings into a coherent answer
"""
        
        # Reasoning chain-of-thought instructions
        cot_instructions["reasoning"] = """
To answer this question, I'll think through it step-by-step:

1. First, I'll identify the key concepts and relationships in the question
2. Next, I'll examine the relevant information from the context
3. I'll analyze potential causes and effects
4. I'll consider alternative explanations
5. Finally, I'll synthesize a comprehensive explanation based on the evidence
"""
        
        # Problem-solving chain-of-thought instructions
        cot_instructions["problem_solving"] = """
I'll solve this problem step-by-step:

1. First, I'll clearly define what we're trying to solve
2. I'll identify the relevant information and constraints from the context
3. I'll determine which approach or formula is appropriate
4. I'll work through the solution methodically, showing each step
5. I'll verify my answer by checking if it makes sense in context
"""
        
        # Decision-making chain-of-thought instructions
        cot_instructions["decision_making"] = """
To make this decision, I'll follow a structured approach:

1. First, I'll identify all the available options
2. For each option, I'll list the potential benefits and drawbacks based on the context
3. I'll evaluate each option against relevant criteria
4. I'll consider any constraints or limitations
5. Based on this analysis, I'll recommend the most appropriate option with justification
"""
        
        # Analysis chain-of-thought instructions
        cot_instructions["analysis"] = """
I'll analyze this systematically:

1. First, I'll identify the key elements that need to be analyzed
2. I'll examine the relevant data and information from the context
3. I'll look for patterns, trends, and relationships
4. I'll consider different perspectives and interpretations
5. I'll evaluate the strengths and limitations of the evidence
6. Finally, I'll synthesize my findings into a comprehensive analysis
"""
        
        return cot_instructions
    
    def apply_cot_to_examples(
        self,
        examples: List[Dict[str, str]],
        query_type: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Apply chain-of-thought reasoning to example responses.
        
        Args:
            examples: List of example dictionaries
            query_type: Optional query type to get specific instructions
            
        Returns:
            List of examples with chain-of-thought reasoning added
        """
        cot_examples = []
        
        for example in examples:
            # Create a copy of the example
            cot_example = example.copy()
            
            # Determine query type if not provided
            example_query_type = query_type or self._determine_query_type(example["query"])
            
            # Get chain-of-thought instructions
            cot_instructions = self.get_cot_prompt(example["query"], example_query_type)
            
            # Add chain-of-thought reasoning to the response
            cot_example["response"] = self._add_cot_to_response(
                example["response"],
                cot_instructions,
                example_query_type
            )
            
            cot_examples.append(cot_example)
        
        return cot_examples
    
    def _add_cot_to_response(
        self,
        response: str,
        cot_instructions: str,
        query_type: str
    ) -> str:
        """
        Add chain-of-thought reasoning to a response.
        
        Args:
            response: The original response
            cot_instructions: Chain-of-thought instructions
            query_type: Query type
            
        Returns:
            Response with chain-of-thought reasoning added
        """
        # Create a chain-of-thought reasoning section based on query type
        if query_type == "reasoning":
            cot_reasoning = self._create_reasoning_cot(response)
        elif query_type == "problem_solving":
            cot_reasoning = self._create_problem_solving_cot(response)
        elif query_type == "decision_making":
            cot_reasoning = self._create_decision_making_cot(response)
        elif query_type == "analysis":
            cot_reasoning = self._create_analysis_cot(response)
        else:
            cot_reasoning = self._create_default_cot(response)
        
        # Combine the chain-of-thought reasoning with the original response
        return f"Thinking step-by-step:\n{cot_reasoning}\n\nFinal answer:\n{response}"
    
    def _create_reasoning_cot(self, response: str) -> str:
        """Create chain-of-thought reasoning for reasoning queries."""
        # Extract key concepts from the response
        concepts = self._extract_key_concepts(response)
        
        cot = "1. Key concepts in this question: " + ", ".join(concepts) + "\n\n"
        
        # Add analysis of relationships
        cot += "2. Analyzing the relationships between these concepts:\n"
        for i, concept in enumerate(concepts[:3], 1):
            cot += f"   - {concept}: This relates to the question because it's a central element in understanding the cause and effect relationship.\n"
        cot += "\n"
        
        # Add consideration of evidence
        cot += "3. Evidence from the context:\n"
        sentences = self._extract_sentences(response)
        for i, sentence in enumerate(sentences[:3], 1):
            cot += f"   - {sentence}\n"
        cot += "\n"
        
        # Add synthesis
        cot += "4. Synthesizing the information:\n"
        cot += "   Based on the evidence, I can see that " + self._extract_main_point(response) + "\n\n"
        
        # Add conclusion
        cot += "5. Therefore, the most logical explanation is that " + self._extract_conclusion(response)
        
        return cot
    
    def _create_problem_solving_cot(self, response: str) -> str:
        """Create chain-of-thought reasoning for problem-solving queries."""
        cot = "1. Problem definition:\n"
        cot += "   " + self._extract_main_point(response) + "\n\n"
        
        # Add relevant information
        cot += "2. Relevant information and constraints:\n"
        sentences = self._extract_sentences(response)
        for i, sentence in enumerate(sentences[:3], 1):
            if any(keyword in sentence.lower() for keyword in ["is", "are", "was", "were", "has", "have", "given"]):
                cot += f"   - {sentence}\n"
        cot += "\n"
        
        # Add approach
        cot += "3. Approach:\n"
        cot += "   To solve this problem, I'll " + self._extract_approach(response) + "\n\n"
        
        # Add solution steps
        cot += "4. Solution steps:\n"
        steps = self._extract_steps(response)
        for i, step in enumerate(steps, 1):
            cot += f"   - Step {i}: {step}\n"
        cot += "\n"
        
        # Add verification
        cot += "5. Verification:\n"
        cot += "   This solution makes sense because " + self._extract_conclusion(response)
        
        return cot
    
    def _create_decision_making_cot(self, response: str) -> str:
        """Create chain-of-thought reasoning for decision-making queries."""
        cot = "1. Available options:\n"
        options = self._extract_options(response)
        for i, option in enumerate(options, 1):
            cot += f"   - Option {i}: {option}\n"
        cot += "\n"
        
        # Add evaluation of options
        cot += "2. Evaluation of each option:\n"
        for i, option in enumerate(options, 1):
            cot += f"   - Option {i} ({option}):\n"
            cot += f"     * Benefits: {self._generate_benefits(option)}\n"
            cot += f"     * Drawbacks: {self._generate_drawbacks(option)}\n"
        cot += "\n"
        
        # Add criteria
        cot += "3. Key decision criteria:\n"
        criteria = self._extract_criteria(response)
        for i, criterion in enumerate(criteria, 1):
            cot += f"   - {criterion}\n"
        cot += "\n"
        
        # Add constraints
        cot += "4. Constraints and limitations:\n"
        cot += "   " + self._extract_constraints(response) + "\n\n"
        
        # Add recommendation
        cot += "5. Recommendation:\n"
        cot += "   Based on this analysis, " + self._extract_conclusion(response)
        
        return cot
    
    def _create_analysis_cot(self, response: str) -> str:
        """Create chain-of-thought reasoning for analysis queries."""
        cot = "1. Key elements to analyze:\n"
        elements = self._extract_key_concepts(response)
        for i, element in enumerate(elements, 1):
            cot += f"   - {element}\n"
        cot += "\n"
        
        # Add data examination
        cot += "2. Relevant information from the context:\n"
        sentences = self._extract_sentences(response)
        for i, sentence in enumerate(sentences[:3], 1):
            if any(keyword in sentence.lower() for keyword in ["data", "information", "evidence", "shows", "indicates"]):
                cot += f"   - {sentence}\n"
        cot += "\n"
        
        # Add patterns and trends
        cot += "3. Patterns and relationships identified:\n"
        patterns = self._extract_patterns(response)
        for i, pattern in enumerate(patterns, 1):
            cot += f"   - {pattern}\n"
        cot += "\n"
        
        # Add perspectives
        cot += "4. Different perspectives to consider:\n"
        perspectives = self._extract_perspectives(response)
        for i, perspective in enumerate(perspectives, 1):
            cot += f"   - {perspective}\n"
        cot += "\n"
        
        # Add evaluation
        cot += "5. Evaluation of evidence:\n"
        cot += "   " + self._extract_evaluation(response) + "\n\n"
        
        # Add synthesis
        cot += "6. Synthesis of findings:\n"
        cot += "   " + self._extract_conclusion(response)
        
        return cot
    
    def _create_default_cot(self, response: str) -> str:
        """Create default chain-of-thought reasoning."""
        cot = "1. Key components of the question:\n"
        concepts = self._extract_key_concepts(response)
        for i, concept in enumerate(concepts, 1):
            cot += f"   - {concept}\n"
        cot += "\n"
        
        # Add relevant information
        cot += "2. Relevant information from the context:\n"
        sentences = self._extract_sentences(response)
        for i, sentence in enumerate(sentences[:3], 1):
            cot += f"   - {sentence}\n"
        cot += "\n"
        
        # Add reasoning
        cot += "3. Reasoning through each part:\n"
        parts = self._extract_parts(response)
        for i, part in enumerate(parts, 1):
            cot += f"   - Part {i}: {part}\n"
        cot += "\n"
        
        # Add synthesis
        cot += "4. Synthesizing the findings:\n"
        cot += "   " + self._extract_conclusion(response)
        
        return cot
    
    # Helper methods for extracting information from responses
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple extraction based on nouns and noun phrases
        words = re.findall(r'\b[A-Za-z][A-Za-z\-]+\b', text)
        concepts = []
        
        # Filter for likely concepts (longer words, capitalized words)
        for word in words:
            if len(word) > 5 or word[0].isupper():
                if word.lower() not in ["therefore", "however", "although", "because"]:
                    concepts.append(word)
        
        # Deduplicate and limit
        unique_concepts = []
        for concept in concepts:
            if concept not in unique_concepts:
                unique_concepts.append(concept)
                if len(unique_concepts) >= 5:
                    break
        
        return unique_concepts if unique_concepts else ["Main concept", "Secondary concept", "Related factor"]
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_main_point(self, text: str) -> str:
        """Extract the main point from text."""
        sentences = self._extract_sentences(text)
        if sentences:
            return sentences[0]
        return "the main point relates to the key concepts identified"
    
    def _extract_conclusion(self, text: str) -> str:
        """Extract the conclusion from text."""
        sentences = self._extract_sentences(text)
        if sentences and len(sentences) > 1:
            return sentences[-1]
        return "the evidence supports the main conclusion"
    
    def _extract_approach(self, text: str) -> str:
        """Extract the approach from text."""
        for sentence in self._extract_sentences(text):
            if any(keyword in sentence.lower() for keyword in ["approach", "method", "strategy", "using", "apply"]):
                return sentence
        return "use a systematic approach based on the available information"
    
    def _extract_steps(self, text: str) -> List[str]:
        """Extract steps from text."""
        steps = []
        
        # Look for numbered steps
        numbered_steps = re.findall(r'\d+\.\s+([^.!?]+[.!?])', text)
        if numbered_steps:
            return numbered_steps[:4]
        
        # Look for sentences with step-like keywords
        sentences = self._extract_sentences(text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["first", "then", "next", "finally", "lastly"]):
                steps.append(sentence)
        
        # If no steps found, create generic ones
        if not steps:
            steps = [
                "Identify the key variables and constraints",
                "Apply the appropriate formula or method",
                "Calculate the result step by step",
                "Verify the solution makes sense in context"
            ]
        
        return steps[:4]
    
    def _extract_options(self, text: str) -> List[str]:
        """Extract options from text."""
        options = []
        
        # Look for options in bullet points or numbered lists
        bullet_options = re.findall(r'[•\-*]\s+([^.!?]+[.!?])', text)
        if bullet_options:
            return bullet_options[:4]
        
        numbered_options = re.findall(r'\d+\.\s+([^.!?]+[.!?])', text)
        if numbered_options:
            return numbered_options[:4]
        
        # Look for sentences with option-like keywords
        sentences = self._extract_sentences(text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["option", "alternative", "choice", "possibility"]):
                options.append(sentence)
        
        # If no options found, create generic ones
        if not options:
            options = [
                "Option A with its specific characteristics",
                "Option B with different advantages",
                "Option C with alternative trade-offs"
            ]
        
        return options[:4]
    
    def _generate_benefits(self, option: str) -> str:
        """Generate benefits for an option."""
        # Simple heuristic to extract potential benefits
        words = option.split()
        for word in words:
            if word.lower() in ["effective", "efficient", "better", "improved", "advantage", "benefit"]:
                return f"Provides {word.lower()} results based on the context"
        
        return "Aligns well with the requirements based on the evidence"
    
    def _generate_drawbacks(self, option: str) -> str:
        """Generate drawbacks for an option."""
        # Simple heuristic to extract potential drawbacks
        words = option.split()
        for word in words:
            if word.lower() in ["challenge", "difficult", "complex", "costly", "expensive", "time-consuming"]:
                return f"May be {word.lower()} to implement based on the context"
        
        return "May have limitations in certain scenarios as suggested by the context"
    
    def _extract_criteria(self, text: str) -> List[str]:
        """Extract decision criteria from text."""
        criteria = []
        
        # Look for criteria-like phrases
        sentences = self._extract_sentences(text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["criteria", "factor", "consideration", "important", "significant"]):
                criteria.append(sentence)
        
        # If no criteria found, create generic ones
        if not criteria:
            criteria = [
                "Effectiveness in addressing the core problem",
                "Resource efficiency and cost considerations",
                "Long-term sustainability and scalability"
            ]
        
        return criteria[:3]
    
    def _extract_constraints(self, text: str) -> str:
        """Extract constraints from text."""
        for sentence in self._extract_sentences(text):
            if any(keyword in sentence.lower() for keyword in ["constraint", "limitation", "restricted", "boundary", "cannot", "must not"]):
                return sentence
        
        return "There are practical limitations to consider based on the available resources and context"
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract patterns from text."""
        patterns = []
        
        # Look for pattern-like phrases
        sentences = self._extract_sentences(text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["pattern", "trend", "correlation", "relationship", "connection", "linked", "associated"]):
                patterns.append(sentence)
        
        # If no patterns found, create generic ones
        if not patterns:
            patterns = [
                "There appears to be a correlation between key factors mentioned in the context",
                "A trend emerges when examining the information chronologically",
                "Several elements show interconnected relationships"
            ]
        
        return patterns[:3]
    
    def _extract_perspectives(self, text: str) -> List[str]:
        """Extract perspectives from text."""
        perspectives = []
        
        # Look for perspective-like phrases
        sentences = self._extract_sentences(text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["perspective", "viewpoint", "angle", "approach", "theory", "framework"]):
                perspectives.append(sentence)
        
        # If no perspectives found, create generic ones
        if not perspectives:
            perspectives = [
                "From one perspective, the evidence suggests a primary causal relationship",
                "An alternative viewpoint emphasizes different factors",
                "A balanced approach considers multiple contributing elements"
            ]
        
        return perspectives[:3]
    
    def _extract_evaluation(self, text: str) -> str:
        """Extract evaluation from text."""
        for sentence in self._extract_sentences(text):
            if any(keyword in sentence.lower() for keyword in ["evidence", "support", "strong", "weak", "suggest", "indicate", "demonstrate"]):
                return sentence
        
        return "The evidence provides strong support for certain conclusions while leaving other aspects less certain"
    
    def _extract_parts(self, text: str) -> List[str]:
        """Extract parts for default reasoning."""
        parts = []
        
        # Look for key sentences that represent different parts of the reasoning
        sentences = self._extract_sentences(text)
        
        # Take a sampling of sentences from beginning, middle, and end
        if len(sentences) >= 3:
            parts.append(sentences[0])
            parts.append(sentences[len(sentences) // 2])
            parts.append(sentences[-2])
        else:
            parts = sentences
        
        # If still not enough parts, create generic ones
        if len(parts) < 3:
            additional_parts = [
                "The primary aspect relates to the main concepts identified",
                "Secondary factors contribute to a more nuanced understanding",
                "Considering the context provides important qualifications"
            ]
            parts.extend(additional_parts[:(3 - len(parts))])
        
        return parts[:3] 