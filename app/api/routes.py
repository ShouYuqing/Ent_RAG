"""
API routes for the Ent_RAG system.
Defines endpoints for querying, document ingestion, and system management.

Author: yuqings
Created: February 2024
License: MIT
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Body
from pydantic import BaseModel, Field

from app.config import config
from app.core import ent_rag  # Import the EntRAG singleton instance

# Set up logger
logger = logging.getLogger("ent_rag.api")

# Create router
router = APIRouter()


# Define request and response models
class QueryOptions(BaseModel):
    """Options for query processing."""
    use_hybrid_search: bool = Field(default=True, description="Whether to use hybrid search")
    rerank_results: bool = Field(default=True, description="Whether to rerank search results")
    rewrite_query: bool = Field(default=True, description="Whether to rewrite the query")
    max_tokens: int = Field(default=1000, description="Maximum tokens in the response")
    temperature: float = Field(default=0.1, description="Temperature for LLM generation")
    prompt_template: Optional[str] = Field(default=None, description="Name of prompt template to use")
    use_few_shot: bool = Field(default=True, description="Whether to use few-shot examples")
    use_chain_of_thought: bool = Field(default=True, description="Whether to use chain-of-thought reasoning")


class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query: str = Field(..., description="The user's query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for document retrieval")
    options: Optional[QueryOptions] = Field(default=None, description="Query processing options")


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict[str, Any]] = Field(default=[], description="Sources used to generate the answer")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata about the response")


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: Optional[str] = Field(default=None, description="Source of the document")
    author: Optional[str] = Field(default=None, description="Author of the document")
    created_at: Optional[str] = Field(default=None, description="Creation date of the document")
    category: Optional[str] = Field(default=None, description="Category of the document")
    tags: Optional[List[str]] = Field(default=None, description="Tags for the document")
    custom_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata fields")


class IngestResponse(BaseModel):
    """Response model for the ingest endpoint."""
    success: bool = Field(..., description="Whether the ingestion was successful")
    document_id: str = Field(..., description="ID of the ingested document")
    message: str = Field(..., description="Status message")


class SystemInfoResponse(BaseModel):
    """Response model for the system info endpoint."""
    version: str = Field(..., description="Version of the Ent_RAG system")
    document_count: int = Field(..., description="Number of documents in the system")
    vector_db_type: str = Field(..., description="Type of vector database being used")
    llm_model: str = Field(..., description="LLM model being used")
    embedding_model: str = Field(..., description="Embedding model being used")
    author: str = Field(..., description="Author of the system")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query and generate a response using the RAG system.
    
    This endpoint:
    1. Takes a user query and optional filters/options
    2. Retrieves relevant documents
    3. Generates a response based on the retrieved context
    4. Returns the response with source information
    
    Args:
        request: Query request containing the query, filters, and options
        
    Returns:
        QueryResponse with the answer, sources, and metadata
    """
    try:
        # Extract query and filters
        query_text = request.query
        filters = request.filters
        
        # Set default options if not provided
        options = request.options or QueryOptions()
        
        # Process query through the EntRAG system
        result = ent_rag.query(
            query=query_text,
            filters=filters,
            top_k=10,  # Retrieve more documents than needed for better reranking
            rerank=options.rerank_results,
            max_tokens=options.max_tokens,
            template_name=options.prompt_template or "qa",
            temperature=options.temperature,
            return_context=True  # We need context to provide sources
        )
        
        # Extract answer and context
        answer = result["response"]
        context = result.get("context", "")
        
        # Extract source documents
        sources = []
        for doc_id in result.get("document_ids", []):
            doc = ent_rag.get_document(doc_id)
            if doc:
                sources.append({
                    "id": doc["id"],
                    "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "metadata": doc["metadata"]
                })
        
        # Prepare metadata
        metadata = {
            "processing_time": result["timing"]["total"],
            "document_count": result["document_count"],
            "model": config.llm.default_model
        }
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Ingest a file into the RAG system.
    
    This endpoint:
    1. Receives a file upload
    2. Processes the file content
    3. Adds the document to the system
    
    Args:
        file: Uploaded file
        metadata: Optional JSON string with metadata
        
    Returns:
        IngestResponse with status and document ID
    """
    try:
        # Read file content
        content = await file.read()
        
        # Try to decode as text
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be a text file")
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Add file metadata
        doc_metadata["filename"] = file.filename
        doc_metadata["content_type"] = file.content_type
        
        # Add document to the system
        doc_ids = ent_rag.add_document(
            content=text_content,
            metadata=doc_metadata,
            chunk=True,
            enrich_metadata=True
        )
        
        # Return response
        return IngestResponse(
            success=True,
            document_id=doc_ids[0] if isinstance(doc_ids, list) else doc_ids,
            message=f"Document ingested successfully with {len(doc_ids) if isinstance(doc_ids, list) else 1} chunks"
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error ingesting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    text: str = Body(..., embed=True),
    title: str = Body(..., embed=True),
    metadata: Optional[DocumentMetadata] = Body(None, embed=True)
):
    """
    Ingest text content into the RAG system.
    
    This endpoint:
    1. Receives text content and metadata
    2. Processes the text
    3. Adds the document to the system
    
    Args:
        text: Text content to ingest
        title: Title for the document
        metadata: Optional metadata for the document
        
    Returns:
        IngestResponse with status and document ID
    """
    try:
        # Prepare metadata
        doc_metadata = {}
        if metadata:
            doc_metadata = metadata.dict(exclude_none=True)
        
        # Add title to metadata
        doc_metadata["title"] = title
        
        # Add document to the system
        doc_ids = ent_rag.add_document(
            content=text,
            metadata=doc_metadata,
            chunk=True,
            enrich_metadata=True
        )
        
        # Return response
        return IngestResponse(
            success=True,
            document_id=doc_ids[0] if isinstance(doc_ids, list) else doc_ids,
            message=f"Document ingested successfully with {len(doc_ids) if isinstance(doc_ids, list) else 1} chunks"
        )
    
    except Exception as e:
        logger.error(f"Error ingesting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting text: {str(e)}")


@router.get("/system/info", response_model=SystemInfoResponse)
async def system_info():
    """
    Get information about the RAG system.
    
    This endpoint provides details about:
    - System version
    - Document count
    - Vector database type
    - LLM model
    - Embedding model
    
    Returns:
        SystemInfoResponse with system information
    """
    try:
        # Get system stats
        stats = ent_rag.get_stats()
        
        return SystemInfoResponse(
            version=stats.get("version", "1.0.0"),
            document_count=stats.get("document_count", 0),
            vector_db_type="ChromaDB",
            llm_model=config.llm.default_model,
            embedding_model=config.llm.default_embedding_model,
            author=stats.get("author", "yuqings")
        )
    
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system info: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the system.
    
    Args:
        document_id: ID of the document to delete
        
    Returns:
        Status message
    """
    try:
        # Delete document
        success = ent_rag.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        return {"success": True, "message": f"Document {document_id} deleted successfully"}
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.get("/templates")
async def list_templates():
    """
    List available prompt templates.
    
    Returns:
        List of available templates with their descriptions
    """
    try:
        # Get templates from the prompt template manager
        templates = ent_rag.prompt_template_manager.list_templates()
        
        return {"templates": templates}
    
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing templates: {str(e)}") 