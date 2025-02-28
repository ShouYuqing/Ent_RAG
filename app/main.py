"""
Main entry point for the Ent_RAG application.
Initializes and runs the FastAPI server.

This module sets up the FastAPI application with appropriate middleware,
routes, and configuration. It serves as the entry point for the API server
that provides access to the RAG system functionality.

Author: yuqings
Created: February 2024
License: MIT
"""

import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import config
from app.api.routes import router as api_router
from app.api.middleware import APIKeyMiddleware
from app.core import ent_rag  # Import the EntRAG singleton instance

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.api.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ent_rag")

# Create FastAPI application
app = FastAPI(
    title="Ent_RAG API",
    description="Enterprise-Grade Retrieval Augmented Generation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "yuqings",
        "email": "yuqings@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API key middleware if required
if config.api.api_key_required:
    app.add_middleware(APIKeyMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """
    Root endpoint that returns basic information about the API.
    
    This endpoint serves as a simple health check and provides
    basic information about the API, including version and
    documentation links.
    
    Returns:
        Dict: Basic information about the API
    """
    return {
        "name": "Ent_RAG API",
        "version": "1.0.0",
        "description": "Enterprise-Grade Retrieval Augmented Generation System",
        "docs": "/docs",
        "author": "yuqings",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    This endpoint verifies that the API server is running and
    can connect to its dependencies. It checks:
    - API server status
    - Database connectivity
    - LLM service availability
    
    Returns:
        Dict: Health status information
    """
    # Get system stats to verify connectivity
    try:
        stats = ent_rag.get_stats()
        status = "healthy"
        details = {
            "document_count": stats["document_count"],
            "vector_store": stats["vector_store"]["collection_name"],
        }
    except Exception as e:
        status = "unhealthy"
        details = {"error": str(e)}
    
    return {
        "status": status,
        "details": details,
        "version": "1.0.0",
    }


def start():
    """
    Start the FastAPI server.
    
    This function initializes and starts the Uvicorn server
    with the configured host, port, and other settings.
    """
    logger.info(f"Starting Ent_RAG API server on {config.api.host}:{config.api.port}")
    logger.info(f"Documentation available at http://{config.api.host}:{config.api.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug_mode,
        log_level=config.api.log_level.lower(),
    )


if __name__ == "__main__":
    start() 