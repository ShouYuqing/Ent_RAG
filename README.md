# Ent_RAG: Enterprise-Grade Retrieval Augmented Generation System

Ent_RAG is a sophisticated Retrieval Augmented Generation (RAG) system designed for enterprise applications. It combines advanced retrieval techniques, intelligent context processing, and robust prompt engineering to deliver high-quality AI-generated responses based on your data.

**Author:** yuqings  
**License:** MIT

## Features

### Advanced Retrieval
- **Hybrid Search**: Combines vector, keyword, and metadata filtering for comprehensive search
- **Query Rewriting**: Automatically expands and rewrites queries to improve retrieval quality
- **Multi-stage Retrieval**: Implements pre-filtering and reranking for more accurate results
- **Custom Relevance Scoring**: Uses sophisticated algorithms to determine document relevance

### Sophisticated Context Processing
- **Intelligent Chunking**: Adapts chunking strategies based on content semantics
- **Context Prioritization**: Filters and prioritizes information to focus on relevant content
- **Metadata Enrichment**: Enhances documents with additional structured information
- **Token Management**: Optimizes context window usage for LLM interactions

### Robust Prompt Engineering
- **Template System**: Includes carefully designed prompt templates with clear instructions
- **System Prompts**: Defines response format and constraints for consistent outputs
- **Few-shot Learning**: Provides examples to guide model behavior
- **Chain-of-thought**: Implements techniques for complex reasoning tasks

### Unified Core Component
- **Centralized Orchestration**: Coordinates all components through a single interface
- **Simplified Integration**: Provides a clean API for application development
- **Comprehensive Logging**: Tracks performance metrics across the entire pipeline
- **Dependency Injection**: Allows for easy customization of individual components

## Project Structure

```
Ent_RAG/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .env.example                  # Example environment variables
├── app/                          # Main application code
│   ├── __init__.py
│   ├── main.py                   # Application entry point
│   ├── config.py                 # Configuration management
│   ├── core.py                   # Core orchestration component
│   ├── retrieval/                # Advanced retrieval components
│   │   ├── __init__.py
│   │   ├── base.py               # Base retriever interface
│   │   ├── semantic.py           # Vector-based retrieval
│   │   ├── keyword.py            # Keyword-based retrieval
│   │   ├── hybrid.py             # Combined retrieval approach
│   │   ├── query_rewriting.py    # Query expansion and rewriting
│   │   └── reranking.py          # Result reranking
│   ├── context/                  # Context processing components
│   │   ├── __init__.py
│   │   ├── prioritization.py     # Context filtering and prioritization
│   │   ├── metadata.py           # Metadata enrichment
│   │   └── token_management.py   # Context window optimization
│   ├── prompts/                  # Prompt engineering components
│   │   ├── __init__.py
│   │   ├── templates.py          # Prompt template system
│   │   ├── generator.py          # Prompt generation
│   │   ├── few_shot.py           # Few-shot example management
│   │   └── chain_of_thought.py   # Chain-of-thought techniques
│   ├── models/                   # Model integration
│   │   ├── __init__.py
│   │   ├── llm.py                # LLM interface
│   │   └── embeddings.py         # Embedding models
│   ├── data/                     # Data management
│   │   ├── __init__.py
│   │   ├── document_store.py     # Document storage and retrieval
│   │   ├── vector_store.py       # Vector database interface
│   │   ├── processor.py          # Document processing
│   │   └── loader.py             # Data loading utilities
│   └── api/                      # API layer
│       ├── __init__.py
│       ├── routes.py             # API endpoints
│       └── middleware.py         # API middleware
├── scripts/                      # Utility scripts
│   ├── ingest.py                 # Data ingestion script
│   ├── query.py                  # Command-line query interface
│   └── evaluate.py               # Evaluation script
├── data/                         # Sample data
│   ├── sample_document.txt       # Example document for testing
│   └── sample_questions.json     # Example questions for evaluation
└── tests/                        # Test suite
    ├── __init__.py
    ├── test_document_store.py    # Tests for document storage
    ├── test_retrieval.py         # Tests for retrieval components
    ├── test_context.py           # Tests for context processing
    ├── test_prompts.py           # Tests for prompt engineering
    └── test_integration.py       # End-to-end tests
```

## Core Component

The `EntRAG` class in `app/core.py` serves as the central orchestrator for the entire system. It provides a unified interface for:

- **Document Management**: Adding, retrieving, and deleting documents
- **Query Processing**: Handling user queries through the complete RAG pipeline
- **System Monitoring**: Tracking performance metrics and system statistics

Example usage:

```python
from app.core import ent_rag

# Add a document
doc_ids = ent_rag.add_document(
    content="Paris is the capital of France.",
    metadata={"category": "geography", "source": "world_facts.txt"}
)

# Query the system
result = ent_rag.query(
    query="What is the capital of France?",
    filters={"category": "geography"},
    rerank=True,
    max_tokens=500,
    template_name="qa"
)

print(result["response"])
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Access to an OpenAI API key or other LLM provider

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yuqings/Ent_RAG.git
   cd Ent_RAG
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Running the Application

Start the API server:
```bash
python -m app.main
```

The API will be available at http://localhost:8000.

### Ingesting Documents

To ingest documents into the system:
```bash
python scripts/ingest.py --file /path/to/document.txt
```

Or ingest an entire directory:
```bash
python scripts/ingest.py --directory /path/to/documents
```

### Querying from Command Line

Use the query script for command-line interaction:
```bash
python scripts/query.py "What is the capital of France?"
```

With additional options:
```bash
python scripts/query.py "What is the capital of France?" --rerank --max-tokens 1000 --show-context
```

### Using the API

Example API request:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "What is the capital of France?",
        "filters": {"category": "geography"},
        "options": {
            "use_hybrid_search": True,
            "rerank_results": True,
            "max_tokens": 500
        }
    }
)

print(response.json())
```

## Customization

### Extending the Core Component

You can create a custom version of the `EntRAG` class with specialized behavior:

```python
from app.core import EntRAG
from app.retrieval.custom import CustomRetriever

class CustomEntRAG(EntRAG):
    def __init__(self):
        super().__init__(retriever=CustomRetriever())
        
    def custom_query_method(self, query):
        # Implement custom query processing
        pass
```

### Adding Custom Retrieval Methods

Extend the retrieval system by implementing new retriever classes that inherit from `BaseRetriever`.

### Creating Custom Context Processing

Implement domain-specific context processing by extending the context processing classes.

### Designing New Prompt Templates

Add specialized prompt templates in `app/prompts/templates.py` for different use cases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 