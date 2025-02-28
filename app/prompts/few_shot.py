"""
Few-shot examples for in-context learning.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from app.config import config

logger = logging.getLogger("ent_rag.prompts.few_shot")


class FewShotExampleManager:
    """
    Manages few-shot examples for different types of queries and templates.
    """
    
    def __init__(self):
        """Initialize the few-shot example manager."""
        self.examples = self._load_examples()
    
    def get_examples(
        self,
        query: str,
        template_id: Optional[str] = None,
        num_examples: int = 2
    ) -> List[Dict[str, str]]:
        """
        Get appropriate few-shot examples for a query and template.
        
        Args:
            query: The user's query
            template_id: Optional template ID to get specific examples
            num_examples: Number of examples to return
            
        Returns:
            List of example dictionaries with 'query', 'context', and 'response' keys
        """
        # If template ID is provided, try to get template-specific examples
        if template_id and template_id in self.examples:
            examples = self.examples[template_id]
            return examples[:min(num_examples, len(examples))]
        
        # Otherwise, determine the type of query and return appropriate examples
        query_type = self._determine_query_type(query)
        
        # Get examples for query type
        if query_type in self.examples:
            examples = self.examples[query_type]
            return examples[:min(num_examples, len(examples))]
        
        # Fallback to default examples
        if "default" in self.examples:
            examples = self.examples["default"]
            return examples[:min(num_examples, len(examples))]
        
        # If no examples are available, return an empty list
        return []
    
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
    
    def _load_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Load few-shot examples from files or create default examples.
        
        Returns:
            Dictionary of query type to list of example dictionaries
        """
        examples = {}
        
        # Try to load examples from files
        examples_dir = Path(__file__).parent / "data" / "few_shot"
        if examples_dir.exists():
            for file_path in examples_dir.glob("*.json"):
                try:
                    example_type = file_path.stem
                    with open(file_path, "r") as f:
                        examples[example_type] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading few-shot examples from {file_path}: {str(e)}")
        
        # If no examples were loaded, create default examples
        if not examples:
            examples = self._create_default_examples()
        
        return examples
    
    def _create_default_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Create default few-shot examples.
        
        Returns:
            Dictionary of query type to list of example dictionaries
        """
        examples = {}
        
        # Default (QA) examples
        examples["default"] = [
            {
                "query": "What is the capital of France?",
                "context": "France is a country in Western Europe. Its capital is Paris, which is known for landmarks such as the Eiffel Tower and the Louvre Museum. France has a population of approximately 67 million people.",
                "response": "The capital of France is Paris."
            },
            {
                "query": "Who invented the telephone?",
                "context": "The telephone was invented by Alexander Graham Bell, who was granted a patent for the device on March 7, 1876. Bell's invention came after years of work on the harmonic telegraph. His main rival was Elisha Gray, who filed a caveat for a similar invention on the same day Bell filed his patent application.",
                "response": "Alexander Graham Bell invented the telephone and was granted a patent for it on March 7, 1876."
            }
        ]
        
        # Summarization examples
        examples["summarize"] = [
            {
                "query": "Summarize the key points about climate change.",
                "context": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases. The main greenhouse gases that contribute to climate change include carbon dioxide, methane, nitrous oxide, and water vapor. The effects of climate change include rising global temperatures, more frequent and severe weather events, rising sea levels, and disruptions to ecosystems. According to the Intergovernmental Panel on Climate Change (IPCC), limiting global warming to 1.5°C would require 'rapid, far-reaching, and unprecedented changes in all aspects of society.'",
                "response": "Climate change involves long-term shifts in temperatures and weather patterns, primarily driven by human activities since the 1800s, especially fossil fuel burning that produces greenhouse gases. Key effects include rising global temperatures, severe weather events, rising sea levels, and ecosystem disruptions. The IPCC states that limiting warming to 1.5°C requires unprecedented societal changes."
            },
            {
                "query": "Provide a summary of quantum computing.",
                "context": "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers. Though current quantum computers are too small to outperform conventional computers for practical applications, they are believed to be capable of solving certain computational problems, such as integer factorization, substantially faster than classical computers. Quantum computing began in the early 1980s when physicist Paul Benioff proposed a quantum mechanical model of the Turing machine. Richard Feynman and Yuri Manin later suggested that a quantum computer had the potential to simulate things that a classical computer could not. In 1994, Peter Shor developed a quantum algorithm for factoring integers that has the potential to decrypt RSA-encrypted communications.",
                "response": "Quantum computing uses quantum states (superposition, interference, and entanglement) to perform calculations. While current quantum computers are limited in practical applications, they potentially can solve certain problems faster than classical computers, such as integer factorization. The field began in the 1980s with Paul Benioff's quantum mechanical Turing machine model, followed by Feynman and Manin's suggestions about quantum simulation capabilities. In 1994, Peter Shor developed a quantum algorithm that could potentially decrypt RSA encryption."
            }
        ]
        
        # Analysis examples
        examples["analyze"] = [
            {
                "query": "Analyze the causes of the 2008 financial crisis.",
                "context": "The 2008 financial crisis was triggered by a complex interplay of factors in the housing and financial markets. The crisis began with the bursting of the U.S. housing bubble, which peaked in 2006-2007. In the years leading up to the crisis, financial institutions had engaged in excessive risk-taking through the creation and trading of complex mortgage-backed securities and derivatives. Subprime lending, where loans were given to borrowers with poor credit histories, increased dramatically. When housing prices began to decline, many borrowers defaulted on their mortgages, causing significant losses for financial institutions. The collapse of Lehman Brothers in September 2008 marked a critical point in the crisis, leading to a severe credit crunch and a global economic downturn. Regulatory failures and the lack of oversight in the financial industry also contributed to the crisis. The Federal Reserve and other central banks responded with unprecedented monetary policy measures, including near-zero interest rates and quantitative easing.",
                "response": "The 2008 financial crisis resulted from multiple interconnected factors:\n\n1. Housing market bubble: The crisis began when the U.S. housing bubble burst in 2006-2007, following years of unsustainable price growth.\n\n2. Excessive risk-taking: Financial institutions created and traded complex mortgage-backed securities and derivatives without adequate risk assessment.\n\n3. Subprime lending: Loans were increasingly given to borrowers with poor credit histories, who later defaulted when housing prices declined.\n\n4. Institutional failure: The collapse of Lehman Brothers in September 2008 triggered a severe credit crunch and global economic downturn.\n\n5. Regulatory failures: Inadequate oversight of the financial industry allowed risky practices to continue unchecked.\n\nThe crisis prompted unprecedented responses from central banks, including near-zero interest rates and quantitative easing to stabilize markets and stimulate economic recovery."
            },
            {
                "query": "Analyze the impact of social media on political discourse.",
                "context": "Social media has fundamentally transformed political discourse in the 21st century. Platforms like Twitter, Facebook, and YouTube have democratized information sharing, allowing politicians to communicate directly with constituents and citizens to organize politically outside traditional channels. However, these platforms have also contributed to political polarization through algorithmic content curation that creates 'filter bubbles' where users primarily encounter information that confirms their existing beliefs. The spread of misinformation and 'fake news' has become a significant concern, particularly during election periods. A 2018 MIT study found that false news stories are 70% more likely to be retweeted than true stories. Social media has also enabled foreign interference in domestic politics, as seen in the 2016 U.S. presidential election when Russian operatives used platforms to spread divisive content. The rapid pace of social media has accelerated news cycles, often prioritizing speed over accuracy and nuance. While social media has increased political participation among younger demographics, the quality of political discourse has been questioned due to the platforms' tendency to reward emotional and provocative content over substantive policy discussion.",
                "response": "Social media has transformed political discourse in several significant ways:\n\n1. Democratization of communication: Platforms enable direct politician-constituent communication and grassroots political organization outside traditional channels.\n\n2. Increased polarization: Algorithmic content curation creates 'filter bubbles' that reinforce existing beliefs and limit exposure to diverse perspectives.\n\n3. Misinformation proliferation: The MIT study cited shows false news spreads 70% more readily than accurate information, undermining informed discourse.\n\n4. Vulnerability to interference: Foreign actors can exploit these platforms to influence domestic politics, as demonstrated in the 2016 U.S. election.\n\n5. Accelerated news cycles: The rapid pace prioritizes speed over accuracy and nuance in reporting and discussion.\n\n6. Changed participation patterns: While youth engagement has increased, discourse quality has declined as emotional and provocative content is rewarded over substantive policy discussion.\n\nThese impacts reveal a complex relationship between social media and political discourse, with both democratizing benefits and concerning challenges to democratic processes."
            }
        ]
        
        # Comparison examples
        examples["compare"] = [
            {
                "query": "Compare renewable and non-renewable energy sources.",
                "context": "Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass. These sources are naturally replenished and generally produce lower greenhouse gas emissions. Solar energy harnesses radiation from the sun using photovoltaic cells or solar thermal collectors. Wind energy converts kinetic energy from wind into electricity using turbines. Hydroelectric power generates electricity from flowing water, typically using dams. Non-renewable energy sources include fossil fuels (coal, oil, natural gas) and nuclear energy. Fossil fuels formed over millions of years from the remains of plants and animals and release carbon dioxide when burned, contributing to climate change. Nuclear energy is generated through fission, where atoms are split to release energy, producing radioactive waste that requires careful management. Renewable energy has seen decreasing costs in recent years but can be intermittent and location-dependent. Non-renewable sources provide consistent power output but have environmental impacts and finite supplies. The global energy mix is gradually shifting toward renewables, with many countries setting targets to reduce reliance on fossil fuels.",
                "response": "Comparison of Renewable and Non-renewable Energy Sources:\n\nSources:\n- Renewable: Solar, wind, hydroelectric, geothermal, and biomass\n- Non-renewable: Fossil fuels (coal, oil, natural gas) and nuclear energy\n\nFormation and Availability:\n- Renewable: Naturally replenished and theoretically infinite\n- Non-renewable: Formed over millions of years (fossil fuels) or using finite uranium (nuclear); limited supply\n\nEnvironmental Impact:\n- Renewable: Generally lower greenhouse gas emissions and pollution\n- Non-renewable: Fossil fuels release carbon dioxide contributing to climate change; nuclear produces radioactive waste\n\nReliability:\n- Renewable: Often intermittent and location-dependent (e.g., solar, wind)\n- Non-renewable: Provide consistent, reliable power output\n\nCost Trends:\n- Renewable: Decreasing costs in recent years, becoming more competitive\n- Non-renewable: Established infrastructure but increasing environmental compliance costs\n\nFuture Outlook:\n- The global energy mix is gradually shifting toward renewables as countries set targets to reduce fossil fuel dependence"
            },
            {
                "query": "Compare machine learning and traditional programming approaches.",
                "context": "Traditional programming involves explicitly coding rules and logic for a computer to follow. Programmers must understand the problem domain thoroughly and translate their understanding into explicit instructions. The process is deterministic, with the same input always producing the same output if the code is correct. Debugging involves examining the code logic to find errors. Machine learning, in contrast, is a programming approach where algorithms learn patterns from data without being explicitly programmed for specific tasks. ML systems develop their own rules based on training data and can improve with more data or feedback. The process is probabilistic rather than deterministic, with predictions having confidence levels rather than absolute certainty. ML excels at tasks that are difficult to define with explicit rules, such as image recognition, natural language processing, and recommendation systems. However, ML systems require large amounts of quality data for training, can be computationally intensive, and may produce results that are difficult to interpret or explain. Traditional programming is typically more appropriate for problems with well-defined rules and logic, while machine learning is better suited for complex pattern recognition tasks where rules are difficult to define explicitly.",
                "response": "Comparison of Machine Learning and Traditional Programming:\n\nApproach:\n- Traditional: Explicitly coded rules and logic; deterministic process\n- ML: Algorithms learn patterns from data; probabilistic process with confidence levels\n\nDevelopment Process:\n- Traditional: Programmer must understand and code all rules explicitly\n- ML: System develops its own rules based on training data and can improve over time\n\nStrengths:\n- Traditional: Well-suited for problems with clear, definable rules; more interpretable results\n- ML: Excels at complex pattern recognition (image recognition, NLP, recommendations) where explicit rules are difficult to define\n\nRequirements:\n- Traditional: Detailed understanding of problem domain and logic\n- ML: Large amounts of quality training data; often computationally intensive\n\nDebugging/Maintenance:\n- Traditional: Examining code logic to find errors\n- ML: May produce results that are difficult to interpret or explain (black box problem)\n\nIdeal Applications:\n- Traditional: Well-defined logical problems, critical systems requiring deterministic behavior\n- ML: Complex pattern recognition, adaptable systems, problems with many variables"
            }
        ]
        
        # Extraction examples
        examples["extract"] = [
            {
                "query": "Extract the key dates from this historical text.",
                "context": "The American Revolutionary War began with the Battles of Lexington and Concord on April 19, 1775. The Second Continental Congress appointed George Washington as commander-in-chief of the Continental Army on June 15, 1775. The Declaration of Independence was adopted by the Continental Congress on July 4, 1776, though it wasn't signed by most delegates until August 2, 1776. The British surrendered at Yorktown on October 19, 1781, effectively ending major combat operations. The Treaty of Paris, which formally ended the war, was signed on September 3, 1783. The United States Constitution was written during the Constitutional Convention, which took place from May 25 to September 17, 1787. George Washington was inaugurated as the first President of the United States on April 30, 1789.",
                "response": "Key Dates in American Revolutionary History:\n\n• April 19, 1775: Battles of Lexington and Concord begin the American Revolutionary War\n• June 15, 1775: George Washington appointed commander-in-chief of the Continental Army\n• July 4, 1776: Declaration of Independence adopted by the Continental Congress\n• August 2, 1776: Most delegates sign the Declaration of Independence\n• October 19, 1781: British surrender at Yorktown, effectively ending major combat\n• September 3, 1783: Treaty of Paris signed, formally ending the war\n• May 25 - September 17, 1787: Constitutional Convention where the U.S. Constitution was written\n• April 30, 1789: George Washington inaugurated as first U.S. President"
            },
            {
                "query": "List all the programming languages mentioned in this text.",
                "context": "Modern software development involves a variety of programming languages, each with its own strengths. Python has become popular for data science and machine learning due to its readability and extensive libraries. JavaScript remains essential for web development, with frameworks like React and Angular extending its capabilities. Java continues to be widely used in enterprise applications and Android development. C++ is valued in performance-critical applications such as game development and system programming. Swift has become the preferred language for iOS app development since its introduction by Apple in 2014. Rust is gaining popularity for systems programming due to its memory safety guarantees without a garbage collector. SQL, while not a general-purpose programming language, remains crucial for database operations. Ruby, particularly with the Ruby on Rails framework, is known for its developer productivity in web applications. Go, developed at Google, is designed for simplicity and efficiency in networked and multicore computing environments.",
                "response": "Programming Languages Mentioned:\n\n1. Python\n2. JavaScript\n3. Java\n4. C++\n5. Swift\n6. Rust\n7. SQL\n8. Ruby\n9. Go"
            }
        ]
        
        return examples
    
    def format_examples(
        self,
        examples: List[Dict[str, str]],
        format_type: str = "standard"
    ) -> str:
        """
        Format few-shot examples for inclusion in a prompt.
        
        Args:
            examples: List of example dictionaries
            format_type: Format type (standard, concise, or detailed)
            
        Returns:
            Formatted examples string
        """
        if not examples:
            return ""
        
        formatted = "\n\nHere are some examples of how to respond:\n\n"
        
        if format_type == "concise":
            for i, example in enumerate(examples, 1):
                formatted += f"Example {i}:\n"
                formatted += f"Question: {example['query']}\n"
                formatted += f"Answer: {example['response']}\n\n"
        
        elif format_type == "detailed":
            for i, example in enumerate(examples, 1):
                formatted += f"Example {i}:\n"
                formatted += f"Question: {example['query']}\n"
                formatted += f"Context: {example['context']}\n"
                formatted += f"Answer: {example['response']}\n\n"
        
        else:  # standard format
            for i, example in enumerate(examples, 1):
                formatted += f"Example {i}:\n"
                formatted += f"Question: {example['query']}\n"
                formatted += f"Answer: {example['response']}\n\n"
        
        return formatted 