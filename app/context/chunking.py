"""
Intelligent document chunking strategies based on content semantics.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Callable
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from app.config import config

logger = logging.getLogger("ent_rag.context.chunking")

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk the text into smaller pieces.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata about the text
            
        Returns:
            List of chunks with content and metadata
        """
        raise NotImplementedError("Subclasses must implement chunk method")


class FixedSizeChunker(ChunkingStrategy):
    """Chunks text into fixed-size chunks with overlap."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the fixed-size chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk the text into fixed-size chunks with overlap.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata about the text
            
        Returns:
            List of chunks with content and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # If this is not the last chunk, try to find a good break point
            if end < text_length:
                # Look for a period, question mark, or exclamation point followed by a space or newline
                match = re.search(r'[.!?][\s\n]', text[end - 100:end])
                if match:
                    end = end - 100 + match.end()
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            # Create chunk with metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "chunk_start_char": start,
                "chunk_end_char": end
            })
            
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
            
            # Move start position for next chunk
            start = end - self.chunk_overlap
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """Chunks text based on semantic boundaries like paragraphs and sections."""
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk the text based on semantic boundaries.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata about the text
            
        Returns:
            List of chunks with content and metadata
        """
        if not text:
            return []
        
        # Split text into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If adding this paragraph would exceed max_chunk_size and we already have content,
            # finalize the current chunk and start a new one
            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                
                # Create chunk with metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "paragraph_count": len(current_chunk)
                })
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                
                current_chunk = []
                current_size = 0
            
            # If paragraph is larger than max_chunk_size, split it into sentences
            if para_size > self.max_chunk_size:
                sentences = sent_tokenize(para)
                sentence_chunks = self._group_sentences(sentences, metadata, len(chunks))
                chunks.extend(sentence_chunks)
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            
            # Create chunk with metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "paragraph_count": len(current_chunk)
            })
            
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: The text to split
            
        Returns:
            List of paragraphs
        """
        # Split on double newlines (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs and strip whitespace
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _group_sentences(
        self,
        sentences: List[str],
        metadata: Optional[Dict[str, Any]],
        start_index: int
    ) -> List[Dict[str, Any]]:
        """
        Group sentences into chunks.
        
        Args:
            sentences: List of sentences to group
            metadata: Optional metadata about the text
            start_index: Starting index for chunk numbering
            
        Returns:
            List of chunks with content and metadata
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max_chunk_size and we already have content,
            # finalize the current chunk and start a new one
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                
                # Create chunk with metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "chunk_index": start_index + len(chunks),
                    "sentence_count": len(current_chunk)
                })
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                
                current_chunk = []
                current_size = 0
            
            # If sentence is larger than max_chunk_size, split it into smaller pieces
            if sentence_size > self.max_chunk_size:
                words = word_tokenize(sentence)
                current_piece = []
                current_piece_size = 0
                
                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    
                    if current_piece_size + word_size > self.max_chunk_size and current_piece:
                        piece_text = " ".join(current_piece)
                        
                        # Create chunk with metadata
                        chunk_metadata = metadata.copy() if metadata else {}
                        chunk_metadata.update({
                            "chunk_index": start_index + len(chunks),
                            "is_partial_sentence": True
                        })
                        
                        chunks.append({
                            "content": piece_text,
                            "metadata": chunk_metadata
                        })
                        
                        current_piece = []
                        current_piece_size = 0
                    
                    current_piece.append(word)
                    current_piece_size += word_size
                
                # Add the last piece if there's anything left
                if current_piece:
                    piece_text = " ".join(current_piece)
                    
                    # Create chunk with metadata
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        "chunk_index": start_index + len(chunks),
                        "is_partial_sentence": True
                    })
                    
                    chunks.append({
                        "content": piece_text,
                        "metadata": chunk_metadata
                    })
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            
            # Create chunk with metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": start_index + len(chunks),
                "sentence_count": len(current_chunk)
            })
            
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks


class StructuredDocumentChunker(ChunkingStrategy):
    """Chunks structured documents (like academic papers or legal documents) based on their structure."""
    
    def __init__(
        self,
        max_section_size: int = 2000,
        max_chunk_size: int = 1000
    ):
        """
        Initialize the structured document chunker.
        
        Args:
            max_section_size: Maximum size of a section in characters
            max_chunk_size: Maximum size of a chunk in characters
        """
        self.max_section_size = max_section_size
        self.max_chunk_size = max_chunk_size
        self.fixed_size_chunker = FixedSizeChunker(
            chunk_size=max_chunk_size,
            chunk_overlap=int(max_chunk_size * 0.1)
        )
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk the structured document based on its structure.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata about the text
            
        Returns:
            List of chunks with content and metadata
        """
        if not text:
            return []
        
        # Extract sections from the document
        sections = self._extract_sections(text)
        
        # Process each section
        chunks = []
        for section_title, section_content in sections:
            # Create section metadata
            section_metadata = metadata.copy() if metadata else {}
            section_metadata["section_title"] = section_title
            
            # If section is small enough, keep it as a single chunk
            if len(section_content) <= self.max_section_size:
                section_metadata.update({
                    "chunk_index": len(chunks),
                    "is_complete_section": True
                })
                
                chunks.append({
                    "content": section_content,
                    "metadata": section_metadata
                })
            else:
                # Otherwise, chunk the section using fixed-size chunker
                section_chunks = self.fixed_size_chunker.chunk(section_content, section_metadata)
                chunks.extend(section_chunks)
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[tuple]:
        """
        Extract sections from a structured document.
        
        Args:
            text: The document text
            
        Returns:
            List of (section_title, section_content) tuples
        """
        # Look for common section headers
        section_patterns = [
            # Academic paper sections
            r'^(Abstract|Introduction|Background|Related Work|Methodology|Methods|Experiments|Results|Discussion|Conclusion|References)[\s]*\n',
            # Numbered sections
            r'^(\d+\.[\d\.]*\s+[A-Z][^\n]+)[\s]*\n',
            # Legal document sections
            r'^(Article \d+|Section \d+|ARTICLE \d+|SECTION \d+)[\s]*\n'
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(f'({p})' for p in section_patterns)
        
        # Find all section headers
        matches = list(re.finditer(combined_pattern, text, re.MULTILINE))
        
        # If no sections found, treat the whole document as one section
        if not matches:
            return [("Document", text)]
        
        # Extract sections
        sections = []
        for i, match in enumerate(matches):
            # Get section title
            section_title = match.group(0).strip()
            
            # Get section content
            start_pos = match.end()
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            section_content = text[start_pos:end_pos].strip()
            
            sections.append((section_title, section_content))
        
        return sections


class IntelligentChunker:
    """
    Intelligent chunking system that selects the appropriate chunking strategy
    based on document type and content.
    """
    
    def __init__(self):
        """Initialize the intelligent chunker with various chunking strategies."""
        self.fixed_size_chunker = FixedSizeChunker(
            chunk_size=config.context.max_chunk_size,
            chunk_overlap=config.context.max_chunk_overlap
        )
        self.semantic_chunker = SemanticChunker(
            max_chunk_size=config.context.max_chunk_size,
            min_chunk_size=int(config.context.max_chunk_size * 0.2)
        )
        self.structured_chunker = StructuredDocumentChunker(
            max_section_size=config.context.max_chunk_size * 2,
            max_chunk_size=config.context.max_chunk_size
        )
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Intelligently chunk the text based on content type and structure.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata about the text
            
        Returns:
            List of chunks with content and metadata
        """
        if not text:
            return []
        
        # Determine document type from metadata
        doc_type = self._determine_document_type(text, metadata)
        
        # Select chunking strategy based on document type
        if doc_type == "structured":
            logger.debug("Using structured document chunker")
            return self.structured_chunker.chunk(text, metadata)
        elif doc_type == "semantic":
            logger.debug("Using semantic chunker")
            return self.semantic_chunker.chunk(text, metadata)
        else:
            logger.debug("Using fixed-size chunker")
            return self.fixed_size_chunker.chunk(text, metadata)
    
    def _determine_document_type(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """
        Determine the document type based on content and metadata.
        
        Args:
            text: The document text
            metadata: Optional metadata about the text
            
        Returns:
            Document type: "structured", "semantic", or "default"
        """
        # Check if semantic chunking is disabled in config
        if not config.context.chunk_by_semantics:
            return "default"
        
        # Check metadata for document type hints
        if metadata:
            content_type = metadata.get("content_type", "").lower()
            
            # Academic papers, legal documents, etc.
            if content_type in ["pdf", "docx", "doc"] and len(text) > 3000:
                # Check for structured document patterns
                if re.search(r'^(Abstract|Introduction|Section \d+|Article \d+)', text, re.MULTILINE):
                    return "structured"
            
            # Markdown, text files, etc.
            if content_type in ["markdown", "md", "txt"]:
                return "semantic"
        
        # Check text structure
        # If text has multiple paragraphs, use semantic chunking
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) > 5:
            return "semantic"
        
        # Default to fixed-size chunking
        return "default" 