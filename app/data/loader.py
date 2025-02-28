"""
Data loaders for the Ent_RAG system.
"""

import logging
import os
import json
import csv
import re
from typing import Dict, List, Optional, Any, Union, Callable, Iterator
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx

from app.config import config
from app.data.processor import DocumentProcessor

logger = logging.getLogger("ent_rag.data.loader")


class BaseLoader:
    """
    Base class for data loaders.
    """
    
    def __init__(self, processor: Optional[DocumentProcessor] = None):
        """
        Initialize the loader.
        
        Args:
            processor: Optional document processor instance
        """
        self.processor = processor or DocumentProcessor()
    
    def load(self, source: Any, **kwargs) -> List[str]:
        """
        Load data from a source.
        
        Args:
            source: Data source
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        raise NotImplementedError("Subclasses must implement this method")


class TextLoader(BaseLoader):
    """
    Loader for plain text files.
    """
    
    def load(
        self,
        source: Union[str, Path],
        encoding: str = "utf-8",
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Load text from a file.
        
        Args:
            source: Path to text file
            encoding: File encoding
            metadata: Optional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        source_path = Path(source)
        
        # Check if file exists
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            return []
        
        # Read file
        try:
            with open(source_path, "r", encoding=encoding) as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading file {source_path}: {str(e)}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        metadata["source"] = str(source_path)
        metadata["filename"] = source_path.name
        metadata["file_type"] = "text"
        
        # Process text
        return self.processor.process_text(text, metadata, chunk)


class PDFLoader(BaseLoader):
    """
    Loader for PDF files.
    """
    
    def load(
        self,
        source: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Load text from a PDF file.
        
        Args:
            source: Path to PDF file
            metadata: Optional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        source_path = Path(source)
        
        # Check if file exists
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            return []
        
        # Read PDF
        try:
            text = ""
            with open(source_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
        except Exception as e:
            logger.error(f"Error reading PDF {source_path}: {str(e)}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        metadata["source"] = str(source_path)
        metadata["filename"] = source_path.name
        metadata["file_type"] = "pdf"
        
        # Process text
        return self.processor.process_text(text, metadata, chunk)


class DocxLoader(BaseLoader):
    """
    Loader for DOCX files.
    """
    
    def load(
        self,
        source: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Load text from a DOCX file.
        
        Args:
            source: Path to DOCX file
            metadata: Optional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        source_path = Path(source)
        
        # Check if file exists
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            return []
        
        # Read DOCX
        try:
            doc = docx.Document(source_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error reading DOCX {source_path}: {str(e)}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        metadata["source"] = str(source_path)
        metadata["filename"] = source_path.name
        metadata["file_type"] = "docx"
        
        # Process text
        return self.processor.process_text(text, metadata, chunk)


class JSONLoader(BaseLoader):
    """
    Loader for JSON files.
    """
    
    def load(
        self,
        source: Union[str, Path],
        jq_filter: str = ".",
        text_content_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Load text from a JSON file.
        
        Args:
            source: Path to JSON file
            jq_filter: JQ-style filter to extract data
            text_content_key: Key for text content
            metadata_keys: Keys to extract as metadata
            metadata: Optional additional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        source_path = Path(source)
        
        # Check if file exists
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            return []
        
        # Read JSON
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON {source_path}: {str(e)}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        metadata["source"] = str(source_path)
        metadata["filename"] = source_path.name
        metadata["file_type"] = "json"
        
        # Extract text and metadata
        doc_ids = []
        
        # Handle different data structures
        if isinstance(data, list):
            for item in data:
                item_text = self._extract_text(item, text_content_key)
                item_metadata = self._extract_metadata(item, metadata_keys, metadata)
                
                if item_text:
                    doc_ids.extend(self.processor.process_text(item_text, item_metadata, chunk))
        else:
            text = self._extract_text(data, text_content_key)
            item_metadata = self._extract_metadata(data, metadata_keys, metadata)
            
            if text:
                doc_ids.extend(self.processor.process_text(text, item_metadata, chunk))
        
        return doc_ids
    
    def _extract_text(self, data: Dict[str, Any], text_content_key: Optional[str]) -> str:
        """
        Extract text from JSON data.
        
        Args:
            data: JSON data
            text_content_key: Key for text content
            
        Returns:
            Extracted text
        """
        if text_content_key:
            return str(data.get(text_content_key, ""))
        
        # If no key specified, convert the entire object to string
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _extract_metadata(
        self,
        data: Dict[str, Any],
        metadata_keys: Optional[List[str]],
        base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata from JSON data.
        
        Args:
            data: JSON data
            metadata_keys: Keys to extract as metadata
            base_metadata: Base metadata
            
        Returns:
            Extracted metadata
        """
        result = base_metadata.copy()
        
        if metadata_keys:
            for key in metadata_keys:
                if key in data:
                    result[key] = data[key]
        
        return result


class CSVLoader(BaseLoader):
    """
    Loader for CSV files.
    """
    
    def load(
        self,
        source: Union[str, Path],
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ",",
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Load text from a CSV file.
        
        Args:
            source: Path to CSV file
            content_columns: Columns to use as content
            metadata_columns: Columns to use as metadata
            delimiter: CSV delimiter
            metadata: Optional additional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        source_path = Path(source)
        
        # Check if file exists
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            return []
        
        # Read CSV
        try:
            rows = []
            with open(source_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            logger.error(f"Error reading CSV {source_path}: {str(e)}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        metadata["source"] = str(source_path)
        metadata["filename"] = source_path.name
        metadata["file_type"] = "csv"
        
        # Process rows
        doc_ids = []
        for i, row in enumerate(rows):
            # Extract content
            if content_columns:
                content_parts = [str(row.get(col, "")) for col in content_columns]
                content = "\n".join(content_parts)
            else:
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
            
            # Extract metadata
            row_metadata = metadata.copy()
            row_metadata["row_index"] = i
            
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        row_metadata[col] = row[col]
            
            # Process content
            if content:
                doc_ids.extend(self.processor.process_text(content, row_metadata, chunk))
        
        return doc_ids


class WebLoader(BaseLoader):
    """
    Loader for web pages.
    """
    
    def load(
        self,
        source: str,
        headers: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Load text from a web page.
        
        Args:
            source: URL to load
            headers: Optional HTTP headers
            metadata: Optional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        # Check if URL is valid
        if not source.startswith(("http://", "https://")):
            logger.error(f"Invalid URL: {source}")
            return []
        
        # Fetch web page
        try:
            response = requests.get(source, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error fetching URL {source}: {str(e)}")
            return []
        
        # Parse HTML
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator="\n")
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            logger.error(f"Error parsing HTML from {source}: {str(e)}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add web metadata
        metadata["source"] = source
        metadata["file_type"] = "web"
        
        # Try to extract title
        try:
            title = soup.title.string
            if title:
                metadata["title"] = title
        except:
            pass
        
        # Process text
        return self.processor.process_text(text, metadata, chunk)


class DirectoryLoader(BaseLoader):
    """
    Loader for directories of files.
    """
    
    def __init__(
        self,
        processor: Optional[DocumentProcessor] = None,
        glob_pattern: str = "**/*.*",
        exclude_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the directory loader.
        
        Args:
            processor: Optional document processor instance
            glob_pattern: Glob pattern for finding files
            exclude_patterns: Patterns to exclude
        """
        super().__init__(processor)
        self.glob_pattern = glob_pattern
        self.exclude_patterns = exclude_patterns or []
        
        # Initialize loaders for different file types
        self.loaders = {
            ".txt": TextLoader(processor),
            ".pdf": PDFLoader(processor),
            ".docx": DocxLoader(processor),
            ".json": JSONLoader(processor),
            ".csv": CSVLoader(processor)
        }
    
    def load(
        self,
        source: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Load files from a directory.
        
        Args:
            source: Path to directory
            metadata: Optional metadata
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        source_path = Path(source)
        
        # Check if directory exists
        if not source_path.exists() or not source_path.is_dir():
            logger.error(f"Directory not found: {source_path}")
            return []
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add directory metadata
        metadata["source_directory"] = str(source_path)
        
        # Find files
        files = list(source_path.glob(self.glob_pattern))
        
        # Filter files
        files = [
            f for f in files 
            if f.is_file() and not any(re.match(pattern, str(f)) for pattern in self.exclude_patterns)
        ]
        
        # Process files
        doc_ids = []
        for file_path in files:
            # Get file extension
            ext = file_path.suffix.lower()
            
            # Select appropriate loader
            loader = self.loaders.get(ext)
            
            if loader:
                # Create file-specific metadata
                file_metadata = metadata.copy()
                file_metadata["relative_path"] = str(file_path.relative_to(source_path))
                
                # Load file
                file_doc_ids = loader.load(file_path, metadata=file_metadata, chunk=chunk)
                doc_ids.extend(file_doc_ids)
            else:
                logger.warning(f"No loader available for file type: {ext}")
        
        logger.info(f"Loaded {len(doc_ids)} documents from {len(files)} files in {source_path}")
        return doc_ids 