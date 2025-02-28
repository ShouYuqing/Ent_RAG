"""
Configuration management for the Ent_RAG system.
Loads environment variables and provides configuration settings for all components.
"""

import os
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseSettings):
    """Configuration for LLM models."""
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    cohere_api_key: Optional[str] = Field(default=os.getenv("COHERE_API_KEY", ""))
    default_model: str = Field(default=os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo-preview"))
    default_embedding_model: str = Field(default=os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002"))
    temperature: float = Field(default=float(os.getenv("TEMPERATURE", "0.1")))
    max_tokens: int = Field(default=int(os.getenv("MAX_TOKENS", "1000")))


class VectorDBConfig(BaseSettings):
    """Configuration for vector databases."""
    db_type: str = Field(default=os.getenv("VECTOR_DB_TYPE", "chroma"))
    chroma_persist_directory: str = Field(default=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"))
    qdrant_url: str = Field(default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection_name: str = Field(default=os.getenv("QDRANT_COLLECTION_NAME", "ent_rag_collection"))
    milvus_host: str = Field(default=os.getenv("MILVUS_HOST", "localhost"))
    milvus_port: int = Field(default=int(os.getenv("MILVUS_PORT", "19530")))
    milvus_collection: str = Field(default=os.getenv("MILVUS_COLLECTION", "ent_rag_collection"))


class RetrievalConfig(BaseSettings):
    """Configuration for retrieval components."""
    default_top_k: int = Field(default=int(os.getenv("DEFAULT_TOP_K", "5")))
    hybrid_search_weight: float = Field(default=float(os.getenv("HYBRID_SEARCH_WEIGHT", "0.7")))
    reranking_enabled: bool = Field(default=os.getenv("RERANKING_ENABLED", "true").lower() == "true")
    query_rewriting_enabled: bool = Field(default=os.getenv("QUERY_REWRITING_ENABLED", "true").lower() == "true")


class ContextConfig(BaseSettings):
    """Configuration for context processing."""
    max_chunk_size: int = Field(default=int(os.getenv("MAX_CHUNK_SIZE", "1000")))
    max_chunk_overlap: int = Field(default=int(os.getenv("MAX_CHUNK_OVERLAP", "200")))
    chunk_by_semantics: bool = Field(default=os.getenv("CHUNK_BY_SEMANTICS", "true").lower() == "true")
    max_context_window_tokens: int = Field(default=int(os.getenv("MAX_CONTEXT_WINDOW_TOKENS", "3500")))


class APIConfig(BaseSettings):
    """Configuration for API server."""
    host: str = Field(default=os.getenv("API_HOST", "0.0.0.0"))
    port: int = Field(default=int(os.getenv("API_PORT", "8000")))
    debug_mode: bool = Field(default=os.getenv("DEBUG_MODE", "false").lower() == "true")
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    api_key_required: bool = Field(default=os.getenv("API_KEY_REQUIRED", "false").lower() == "true")
    api_key: Optional[str] = Field(default=os.getenv("API_KEY", ""))
    cors_origins: List[str] = Field(default=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","))


class Config:
    """Main configuration class that combines all config components."""
    llm: LLMConfig = LLMConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    context: ContextConfig = ContextConfig()
    api: APIConfig = APIConfig()

    @classmethod
    def get_config(cls) -> 'Config':
        """Get the singleton config instance."""
        if not hasattr(cls, '_instance'):
            cls._instance = Config()
        return cls._instance


# Singleton instance
config = Config.get_config() 