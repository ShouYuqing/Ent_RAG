"""
Metadata enrichment for documents to enhance context quality.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize

from app.config import config
from app.models.llm import LLMManager

logger = logging.getLogger("ent_rag.context.metadata")

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class MetadataEnricher:
    """
    Implements metadata enrichment for documents to enhance context quality.
    """
    
    def __init__(self):
        """Initialize the metadata enricher with necessary components."""
        self.llm_manager = LLMManager()
    
    def enrich(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enrich document metadata with additional information.
        
        Args:
            content: The document content
            metadata: Existing metadata (if any)
            
        Returns:
            Enriched metadata dictionary
        """
        logger.debug("Enriching metadata for document")
        
        # Initialize metadata if not provided
        enriched_metadata = metadata.copy() if metadata else {}
        
        # Add basic metadata if not already present
        enriched_metadata = self._add_basic_metadata(content, enriched_metadata)
        
        # Extract entities if not already present
        if "entities" not in enriched_metadata:
            enriched_metadata["entities"] = self._extract_entities(content)
        
        # Extract keywords if not already present
        if "keywords" not in enriched_metadata:
            enriched_metadata["keywords"] = self._extract_keywords(content)
        
        # Generate summary if not already present
        if "summary" not in enriched_metadata:
            enriched_metadata["summary"] = self._generate_summary(content)
        
        # Categorize content if not already present
        if "category" not in enriched_metadata:
            enriched_metadata["category"] = self._categorize_content(content)
        
        # Determine reading level if not already present
        if "reading_level" not in enriched_metadata:
            enriched_metadata["reading_level"] = self._determine_reading_level(content)
        
        # Add timestamp if not already present
        if "processed_at" not in enriched_metadata:
            enriched_metadata["processed_at"] = datetime.now().isoformat()
        
        return enriched_metadata
    
    def _add_basic_metadata(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add basic metadata fields.
        
        Args:
            content: The document content
            metadata: Existing metadata
            
        Returns:
            Metadata with basic fields added
        """
        # Copy metadata to avoid modifying the original
        enriched = metadata.copy()
        
        # Add character count if not present
        if "char_count" not in enriched:
            enriched["char_count"] = len(content)
        
        # Add word count if not present
        if "word_count" not in enriched:
            enriched["word_count"] = len(content.split())
        
        # Add sentence count if not present
        if "sentence_count" not in enriched:
            sentences = sent_tokenize(content)
            enriched["sentence_count"] = len(sentences)
        
        # Add language if not present (assuming English for now)
        if "language" not in enriched:
            enriched["language"] = "en"
        
        # Try to extract title if not present
        if "title" not in enriched:
            # Look for a title in the first few lines
            lines = content.split("\n")
            for line in lines[:5]:
                line = line.strip()
                # If line is short, capitalized, and doesn't end with a period, it might be a title
                if 3 < len(line) < 100 and not line.endswith(".") and line.upper() != line:
                    enriched["title"] = line
                    break
        
        return enriched
    
    def _extract_entities(self, content: str) -> List[Dict[str, str]]:
        """
        Extract named entities from content.
        
        Args:
            content: The document content
            
        Returns:
            List of extracted entities with type
        """
        # For a simple implementation, use regex patterns to extract common entities
        entities = []
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        for email in emails:
            entities.append({"text": email, "type": "EMAIL"})
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, content)
        for url in urls:
            entities.append({"text": url, "type": "URL"})
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, content)
        for date in dates:
            entities.append({"text": date, "type": "DATE"})
        
        # For more sophisticated entity extraction, we could use spaCy or an LLM
        # If the content is not too long, use LLM for better entity extraction
        if len(content) < 5000:
            llm_entities = self._extract_entities_with_llm(content)
            entities.extend(llm_entities)
        
        # Remove duplicates
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity["text"], entity["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_entities_with_llm(self, content: str) -> List[Dict[str, str]]:
        """
        Extract entities using an LLM.
        
        Args:
            content: The document content
            
        Returns:
            List of extracted entities with type
        """
        # Truncate content if too long
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        # Create prompt for entity extraction
        prompt = f"""
        Extract named entities from the following text. Return the results as a JSON array of objects,
        where each object has "text" and "type" fields. Entity types should be one of: PERSON, ORGANIZATION,
        LOCATION, DATE, PRODUCT, EVENT, or OTHER.
        
        Text:
        {content}
        
        Entities (JSON array):
        """
        
        try:
            # Generate response from LLM
            response = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            # Find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                entities = json.loads(json_str)
                return entities
            else:
                logger.warning("Could not find JSON array in LLM response for entity extraction")
                return []
        
        except Exception as e:
            logger.error(f"Error extracting entities with LLM: {str(e)}")
            return []
    
    def _extract_keywords(self, content: str) -> List[str]:
        """
        Extract keywords from content.
        
        Args:
            content: The document content
            
        Returns:
            List of extracted keywords
        """
        # For a simple implementation, use TF-IDF to extract keywords
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create a small corpus with just this document
            corpus = [content]
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for the first document
            scores = tfidf_matrix[0].toarray()[0]
            
            # Create a list of (word, score) tuples and sort by score
            word_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top keywords
            keywords = [word for word, score in word_scores]
            
            return keywords
        
        except Exception as e:
            logger.warning(f"Error extracting keywords with TF-IDF: {str(e)}")
            
            # Fallback: use LLM for keyword extraction if content is not too long
            if len(content) < 5000:
                return self._extract_keywords_with_llm(content)
            else:
                # Simple fallback: just take the most common words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
                word_counts = {}
                for word in words:
                    if word not in word_counts:
                        word_counts[word] = 0
                    word_counts[word] += 1
                
                # Sort by count and take top 10
                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                return [word for word, count in sorted_words[:10]]
    
    def _extract_keywords_with_llm(self, content: str) -> List[str]:
        """
        Extract keywords using an LLM.
        
        Args:
            content: The document content
            
        Returns:
            List of extracted keywords
        """
        # Truncate content if too long
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        # Create prompt for keyword extraction
        prompt = f"""
        Extract the 5-10 most important keywords or key phrases from the following text.
        Return the results as a comma-separated list of keywords, without numbering or explanation.
        
        Text:
        {content}
        
        Keywords:
        """
        
        try:
            # Generate response from LLM
            response = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse response
            keywords = [kw.strip() for kw in response.split(",")]
            
            # Clean up keywords
            keywords = [kw for kw in keywords if kw]
            
            return keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords with LLM: {str(e)}")
            return []
    
    def _generate_summary(self, content: str) -> str:
        """
        Generate a summary of the content.
        
        Args:
            content: The document content
            
        Returns:
            Generated summary
        """
        # For a simple implementation, use the first few sentences as a summary
        sentences = sent_tokenize(content)
        if len(sentences) <= 3:
            return content
        
        # Take first 3 sentences as summary
        simple_summary = " ".join(sentences[:3])
        
        # If content is not too long, use LLM for better summarization
        if len(content) < 5000:
            return self._generate_summary_with_llm(content)
        else:
            return simple_summary
    
    def _generate_summary_with_llm(self, content: str) -> str:
        """
        Generate a summary using an LLM.
        
        Args:
            content: The document content
            
        Returns:
            Generated summary
        """
        # Truncate content if too long
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        # Create prompt for summarization
        prompt = f"""
        Summarize the following text in 2-3 sentences, capturing the main points.
        
        Text:
        {content}
        
        Summary:
        """
        
        try:
            # Generate response from LLM
            summary = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.3
            ).strip()
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary with LLM: {str(e)}")
            
            # Fallback to simple summary
            sentences = sent_tokenize(content)
            return " ".join(sentences[:3])
    
    def _categorize_content(self, content: str) -> str:
        """
        Categorize the content.
        
        Args:
            content: The document content
            
        Returns:
            Content category
        """
        # Define common categories
        categories = [
            "Technology", "Business", "Science", "Health", "Politics",
            "Entertainment", "Sports", "Education", "Finance", "Travel",
            "Food", "Art", "History", "Environment", "Other"
        ]
        
        # For a simple implementation, use keyword matching
        category_keywords = {
            "Technology": ["software", "hardware", "computer", "algorithm", "programming", "code", "tech", "digital", "internet", "app"],
            "Business": ["company", "market", "industry", "startup", "entrepreneur", "business", "corporate", "CEO", "strategy", "management"],
            "Science": ["research", "experiment", "scientist", "laboratory", "discovery", "physics", "chemistry", "biology", "scientific", "theory"],
            "Health": ["medical", "doctor", "patient", "hospital", "disease", "treatment", "health", "medicine", "symptom", "diagnosis"],
            "Politics": ["government", "policy", "election", "politician", "vote", "democracy", "president", "congress", "political", "law"],
            "Entertainment": ["movie", "music", "celebrity", "film", "actor", "actress", "song", "concert", "TV", "show"],
            "Sports": ["game", "player", "team", "coach", "championship", "tournament", "athlete", "score", "win", "league"],
            "Education": ["school", "student", "teacher", "university", "college", "learning", "education", "academic", "course", "study"],
            "Finance": ["money", "investment", "bank", "stock", "financial", "economy", "fund", "budget", "profit", "revenue"],
            "Travel": ["destination", "hotel", "flight", "tourism", "vacation", "travel", "tourist", "trip", "journey", "adventure"]
        }
        
        # Count keyword matches for each category
        category_scores = {category: 0 for category in categories}
        content_lower = content.lower()
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    category_scores[category] += 1
        
        # Find category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        # If no clear category, use LLM for better categorization
        if best_category[1] == 0 and len(content) < 5000:
            return self._categorize_content_with_llm(content, categories)
        elif best_category[1] == 0:
            return "Other"
        else:
            return best_category[0]
    
    def _categorize_content_with_llm(self, content: str, categories: List[str]) -> str:
        """
        Categorize content using an LLM.
        
        Args:
            content: The document content
            categories: List of possible categories
            
        Returns:
            Content category
        """
        # Truncate content if too long
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        # Create prompt for categorization
        categories_str = ", ".join(categories)
        prompt = f"""
        Categorize the following text into exactly one of these categories: {categories_str}.
        Return ONLY the category name without explanation.
        
        Text:
        {content}
        
        Category:
        """
        
        try:
            # Generate response from LLM
            category = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.1
            ).strip()
            
            # Check if category is in the list
            for valid_category in categories:
                if valid_category.lower() in category.lower():
                    return valid_category
            
            # If not found, return Other
            return "Other"
        
        except Exception as e:
            logger.error(f"Error categorizing content with LLM: {str(e)}")
            return "Other"
    
    def _determine_reading_level(self, content: str) -> str:
        """
        Determine the reading level of the content.
        
        Args:
            content: The document content
            
        Returns:
            Reading level (Elementary, Intermediate, Advanced, Technical)
        """
        # For a simple implementation, use average sentence and word length
        sentences = sent_tokenize(content)
        if not sentences:
            return "Intermediate"
        
        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Calculate average word length
        words = content.split()
        if not words:
            return "Intermediate"
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Determine reading level based on these metrics
        if avg_sentence_length < 10 and avg_word_length < 4.5:
            return "Elementary"
        elif avg_sentence_length < 15 and avg_word_length < 5.0:
            return "Intermediate"
        elif avg_sentence_length < 20 and avg_word_length < 5.5:
            return "Advanced"
        else:
            return "Technical" 