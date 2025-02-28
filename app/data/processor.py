"""
Document processing utilities for the Ent_RAG system.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import os
from pathlib import Path

from app.config import config
from app.data.document_store import Document, DocumentStore

logger = logging.getLogger("ent_rag.data.processor")


class TextSplitter:
    """
    Splits text into chunks for processing and embedding.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n"
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separator: Separator to use for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Handle empty or very short text
        if not text or len(text) <= self.chunk_size:
            return [text]
        
        # Split text by separator
        splits = text.split(self.separator)
        
        # Initialize chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Process each split
        for split in splits:
            # Skip empty splits
            if not split:
                continue
            
            # If adding this split would exceed chunk size, finalize current chunk
            if current_length + len(split) > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))
                
                # Keep overlap from previous chunk
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk) + len(self.separator) * (len(current_chunk) - 1)
            
            # Add split to current chunk
            current_chunk.append(split)
            current_length += len(split) + len(self.separator)
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Splits text recursively by different separators.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the recursive text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to use for splitting text, in order of priority
        """
        super().__init__(chunk_size, chunk_overlap, "\n")
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively using multiple separators.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Handle empty or very short text
        if not text or len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # If we're at the character level, split by chunk size
                return [
                    text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
                ]
            
            # Set current separator and try splitting
            self.separator = separator
            chunks = super().split_text(text)
            
            # If we got multiple chunks, return them
            if len(chunks) > 1:
                return chunks
        
        # Fallback: split by characters
        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]


class DocumentProcessor:
    """
    Processes documents for ingestion into the RAG system.
    """
    
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        text_splitter: Optional[TextSplitter] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            document_store: Optional document store instance
            text_splitter: Optional text splitter instance
        """
        self.document_store = document_store or DocumentStore()
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter()
    
    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Process text and add to document store.
        
        Args:
            text: Text to process
            metadata: Optional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        metadata = metadata or {}
        
        # Split text into chunks if requested
        if chunk:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
        else:
            chunks = [text]
        
        # Add chunks to document store
        doc_ids = []
        for i, chunk_text in enumerate(chunks):
            # Create chunk-specific metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(chunks)
            
            # Add document to store
            doc_id = self.document_store.add_document(
                content=chunk_text,
                metadata=chunk_metadata
            )
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def process_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Process a file and add to document store.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        # Read file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        metadata["source"] = str(file_path)
        metadata["filename"] = file_path.name
        metadata["file_extension"] = file_path.suffix.lower()
        
        # Process text
        return self.process_text(text, metadata, chunk)
    
    def process_directory(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "**/*.*",
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True,
        file_filter: Optional[Callable[[Path], bool]] = None
    ) -> Dict[str, List[str]]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to directory
            glob_pattern: Glob pattern for finding files
            metadata: Optional metadata to apply to all files
            chunk: Whether to chunk the text
            file_filter: Optional function to filter files
            
        Returns:
            Dictionary mapping file paths to document IDs
        """
        directory_path = Path(directory_path)
        
        # Check if directory exists
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return {}
        
        # Find files
        files = list(directory_path.glob(glob_pattern))
        
        # Apply filter if provided
        if file_filter:
            files = [f for f in files if file_filter(f)]
        
        # Process files
        results = {}
        for file_path in files:
            if file_path.is_file():
                # Create file-specific metadata
                file_metadata = metadata.copy() if metadata else {}
                file_metadata["directory"] = str(file_path.parent)
                
                # Process file
                doc_ids = self.process_file(file_path, file_metadata, chunk)
                
                if doc_ids:
                    results[str(file_path)] = doc_ids
        
        logger.info(f"Processed {len(results)} files from {directory_path}")
        return results 