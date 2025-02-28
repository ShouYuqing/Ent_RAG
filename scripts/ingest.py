#!/usr/bin/env python
"""
Data ingestion script for the Ent_RAG system.

This script loads documents from various sources and ingests them into the system.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.loader import (
    TextLoader, PDFLoader, DocxLoader, JSONLoader, CSVLoader, WebLoader, DirectoryLoader
)
from app.data.document_store import DocumentStore
from app.data.processor import DocumentProcessor
from app.context.metadata import MetadataEnricher
from app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingest.log")
    ]
)

logger = logging.getLogger("ent_rag.ingest")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into the Ent_RAG system")
    
    # Source arguments
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file", type=str, help="Path to a file to ingest")
    source_group.add_argument("--directory", type=str, help="Path to a directory to ingest")
    source_group.add_argument("--url", type=str, help="URL to ingest")
    source_group.add_argument("--urls-file", type=str, help="Path to a file containing URLs to ingest")
    
    # Processing arguments
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for text splitting")
    parser.add_argument("--no-chunk", action="store_true", help="Disable text chunking")
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata enrichment")
    
    # File type arguments
    parser.add_argument("--glob", type=str, default="**/*.*", help="Glob pattern for directory ingestion")
    parser.add_argument("--exclude", type=str, nargs="*", default=[], help="Patterns to exclude from directory ingestion")
    
    # JSON/CSV specific arguments
    parser.add_argument("--content-columns", type=str, nargs="*", help="Columns to use as content for CSV files")
    parser.add_argument("--metadata-columns", type=str, nargs="*", help="Columns to use as metadata for CSV files")
    parser.add_argument("--json-content-key", type=str, help="Key for text content in JSON files")
    parser.add_argument("--json-metadata-keys", type=str, nargs="*", help="Keys to extract as metadata from JSON files")
    
    # Output arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()


def ingest_file(
    file_path: str,
    processor: DocumentProcessor,
    metadata_enricher: Optional[MetadataEnricher] = None,
    chunk: bool = True,
    **kwargs
) -> List[str]:
    """
    Ingest a single file.
    
    Args:
        file_path: Path to the file
        processor: Document processor
        metadata_enricher: Optional metadata enricher
        chunk: Whether to chunk the text
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        List of document IDs
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    # Select loader based on file extension
    ext = file_path.suffix.lower()
    
    if ext == ".txt":
        loader = TextLoader(processor)
        doc_ids = loader.load(file_path, chunk=chunk)
    elif ext == ".pdf":
        loader = PDFLoader(processor)
        doc_ids = loader.load(file_path, chunk=chunk)
    elif ext == ".docx":
        loader = DocxLoader(processor)
        doc_ids = loader.load(file_path, chunk=chunk)
    elif ext == ".json":
        loader = JSONLoader(processor)
        doc_ids = loader.load(
            file_path,
            text_content_key=kwargs.get("json_content_key"),
            metadata_keys=kwargs.get("json_metadata_keys"),
            chunk=chunk
        )
    elif ext == ".csv":
        loader = CSVLoader(processor)
        doc_ids = loader.load(
            file_path,
            content_columns=kwargs.get("content_columns"),
            metadata_columns=kwargs.get("metadata_columns"),
            chunk=chunk
        )
    else:
        logger.error(f"Unsupported file type: {ext}")
        return []
    
    # Enrich metadata if requested
    if metadata_enricher and doc_ids:
        document_store = processor.document_store
        for doc_id in doc_ids:
            document = document_store.get_document(doc_id)
            if document:
                enriched_metadata = metadata_enricher.enrich(document.content, document.metadata)
                document.metadata = enriched_metadata
                # Save updated document
                document_store._save_document(document)
    
    return doc_ids


def ingest_directory(
    directory_path: str,
    processor: DocumentProcessor,
    metadata_enricher: Optional[MetadataEnricher] = None,
    glob_pattern: str = "**/*.*",
    exclude_patterns: Optional[List[str]] = None,
    chunk: bool = True,
    **kwargs
) -> List[str]:
    """
    Ingest a directory of files.
    
    Args:
        directory_path: Path to the directory
        processor: Document processor
        metadata_enricher: Optional metadata enricher
        glob_pattern: Glob pattern for finding files
        exclude_patterns: Patterns to exclude
        chunk: Whether to chunk the text
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        List of document IDs
    """
    loader = DirectoryLoader(
        processor=processor,
        glob_pattern=glob_pattern,
        exclude_patterns=exclude_patterns
    )
    
    doc_ids = loader.load(directory_path, chunk=chunk)
    
    # Enrich metadata if requested
    if metadata_enricher and doc_ids:
        document_store = processor.document_store
        for doc_id in doc_ids:
            document = document_store.get_document(doc_id)
            if document:
                enriched_metadata = metadata_enricher.enrich(document.content, document.metadata)
                document.metadata = enriched_metadata
                # Save updated document
                document_store._save_document(document)
    
    return doc_ids


def ingest_url(
    url: str,
    processor: DocumentProcessor,
    metadata_enricher: Optional[MetadataEnricher] = None,
    chunk: bool = True
) -> List[str]:
    """
    Ingest a web page.
    
    Args:
        url: URL to ingest
        processor: Document processor
        metadata_enricher: Optional metadata enricher
        chunk: Whether to chunk the text
        
    Returns:
        List of document IDs
    """
    loader = WebLoader(processor)
    doc_ids = loader.load(url, chunk=chunk)
    
    # Enrich metadata if requested
    if metadata_enricher and doc_ids:
        document_store = processor.document_store
        for doc_id in doc_ids:
            document = document_store.get_document(doc_id)
            if document:
                enriched_metadata = metadata_enricher.enrich(document.content, document.metadata)
                document.metadata = enriched_metadata
                # Save updated document
                document_store._save_document(document)
    
    return doc_ids


def ingest_urls_from_file(
    file_path: str,
    processor: DocumentProcessor,
    metadata_enricher: Optional[MetadataEnricher] = None,
    chunk: bool = True
) -> List[str]:
    """
    Ingest URLs from a file.
    
    Args:
        file_path: Path to file containing URLs
        processor: Document processor
        metadata_enricher: Optional metadata enricher
        chunk: Whether to chunk the text
        
    Returns:
        List of document IDs
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    # Read URLs from file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading URLs from {file_path}: {str(e)}")
        return []
    
    # Ingest each URL
    all_doc_ids = []
    for url in urls:
        doc_ids = ingest_url(url, processor, metadata_enricher, chunk)
        all_doc_ids.extend(doc_ids)
    
    return all_doc_ids


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger("ent_rag").setLevel(logging.DEBUG)
    
    # Initialize components
    document_store = DocumentStore()
    processor = DocumentProcessor(
        document_store=document_store,
        text_splitter=None  # Will use default
    )
    
    # Initialize metadata enricher if requested
    metadata_enricher = None if args.no_metadata else MetadataEnricher()
    
    # Prepare kwargs for loaders
    loader_kwargs = {
        "json_content_key": args.json_content_key,
        "json_metadata_keys": args.json_metadata_keys,
        "content_columns": args.content_columns,
        "metadata_columns": args.metadata_columns
    }
    
    # Ingest documents
    doc_ids = []
    
    if args.file:
        logger.info(f"Ingesting file: {args.file}")
        doc_ids = ingest_file(
            args.file,
            processor,
            metadata_enricher,
            not args.no_chunk,
            **loader_kwargs
        )
    
    elif args.directory:
        logger.info(f"Ingesting directory: {args.directory}")
        doc_ids = ingest_directory(
            args.directory,
            processor,
            metadata_enricher,
            args.glob,
            args.exclude,
            not args.no_chunk,
            **loader_kwargs
        )
    
    elif args.url:
        logger.info(f"Ingesting URL: {args.url}")
        doc_ids = ingest_url(
            args.url,
            processor,
            metadata_enricher,
            not args.no_chunk
        )
    
    elif args.urls_file:
        logger.info(f"Ingesting URLs from file: {args.urls_file}")
        doc_ids = ingest_urls_from_file(
            args.urls_file,
            processor,
            metadata_enricher,
            not args.no_chunk
        )
    
    # Print summary
    logger.info(f"Ingestion complete. Added {len(doc_ids)} documents.")
    
    # Print document store stats
    stats = document_store.count_documents()
    logger.info(f"Document store now contains {stats} documents.")


if __name__ == "__main__":
    main() 