"""
Tests for the document store component.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.document_store import Document, DocumentStore
from app.data.vector_store import VectorStore


class TestDocument(unittest.TestCase):
    """Test the Document class."""
    
    def test_init(self):
        """Test document initialization."""
        # Test with minimal arguments
        doc = Document(content="Test content")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata, {})
        self.assertIsNotNone(doc.doc_id)
        
        # Test with all arguments
        doc = Document(content="Test content", metadata={"key": "value"}, doc_id="test-id")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata, {"key": "value"})
        self.assertEqual(doc.doc_id, "test-id")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = Document(content="Test content", metadata={"key": "value"}, doc_id="test-id")
        doc_dict = doc.to_dict()
        
        self.assertEqual(doc_dict["id"], "test-id")
        self.assertEqual(doc_dict["content"], "Test content")
        self.assertEqual(doc_dict["metadata"], {"key": "value"})
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        doc_dict = {
            "id": "test-id",
            "content": "Test content",
            "metadata": {"key": "value"}
        }
        
        doc = Document.from_dict(doc_dict)
        
        self.assertEqual(doc.doc_id, "test-id")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata, {"key": "value"})


class TestDocumentStore(unittest.TestCase):
    """Test the DocumentStore class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for document store
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock vector store
        self.mock_vector_store = MockVectorStore()
        
        # Initialize document store with mock vector store
        self.document_store = DocumentStore(vector_store=self.mock_vector_store)
        self.document_store.document_dir = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_add_document(self):
        """Test adding a document."""
        # Add document
        doc_id = self.document_store.add_document(
            content="Test content",
            metadata={"key": "value"}
        )
        
        # Check document was added
        self.assertIn(doc_id, self.document_store.documents)
        
        # Check document content
        doc = self.document_store.get_document(doc_id)
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["key"], "value")
        
        # Check document was saved to disk
        doc_path = self.document_store.document_dir / f"{doc_id}.json"
        self.assertTrue(doc_path.exists())
        
        # Check document was added to vector store
        self.assertEqual(len(self.mock_vector_store.texts), 1)
        self.assertEqual(self.mock_vector_store.texts[0], "Test content")
    
    def test_get_document(self):
        """Test getting a document."""
        # Add document
        doc_id = self.document_store.add_document(content="Test content")
        
        # Get document
        doc = self.document_store.get_document(doc_id)
        
        # Check document
        self.assertIsNotNone(doc)
        self.assertEqual(doc.content, "Test content")
        
        # Try to get non-existent document
        doc = self.document_store.get_document("non-existent")
        self.assertIsNone(doc)
    
    def test_delete_document(self):
        """Test deleting a document."""
        # Add document
        doc_id = self.document_store.add_document(content="Test content")
        
        # Check document exists
        self.assertIn(doc_id, self.document_store.documents)
        
        # Delete document
        result = self.document_store.delete_document(doc_id)
        
        # Check result
        self.assertTrue(result)
        
        # Check document was removed
        self.assertNotIn(doc_id, self.document_store.documents)
        
        # Check document was removed from disk
        doc_path = self.document_store.document_dir / f"{doc_id}.json"
        self.assertFalse(doc_path.exists())
        
        # Check document was removed from vector store
        self.assertEqual(len(self.mock_vector_store.deleted_ids), 1)
        self.assertEqual(self.mock_vector_store.deleted_ids[0], doc_id)
        
        # Try to delete non-existent document
        result = self.document_store.delete_document("non-existent")
        self.assertFalse(result)
    
    def test_search(self):
        """Test searching for documents."""
        # Add documents
        doc_id1 = self.document_store.add_document(content="First document")
        doc_id2 = self.document_store.add_document(content="Second document")
        
        # Set up mock search results
        self.mock_vector_store.search_results = [
            {"id": doc_id1, "content": "First document", "metadata": {}, "score": 0.9},
            {"id": doc_id2, "content": "Second document", "metadata": {}, "score": 0.7}
        ]
        
        # Search for documents
        results = self.document_store.search("test query")
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].doc_id, doc_id1)
        self.assertEqual(results[1].doc_id, doc_id2)
    
    def test_count_documents(self):
        """Test counting documents."""
        # Check initial count
        self.assertEqual(self.document_store.count_documents(), 0)
        
        # Add documents
        self.document_store.add_document(content="First document")
        self.document_store.add_document(content="Second document")
        
        # Check count
        self.assertEqual(self.document_store.count_documents(), 2)
    
    def test_get_all_documents(self):
        """Test getting all documents."""
        # Check initial documents
        self.assertEqual(len(self.document_store.get_all_documents()), 0)
        
        # Add documents
        self.document_store.add_document(content="First document")
        self.document_store.add_document(content="Second document")
        
        # Check documents
        documents = self.document_store.get_all_documents()
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].content, "First document")
        self.assertEqual(documents[1].content, "Second document")


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self):
        """Initialize mock vector store."""
        self.texts = []
        self.metadatas = []
        self.ids = []
        self.deleted_ids = []
        self.search_results = []
    
    def add_texts(self, texts, metadatas, ids):
        """Mock adding texts."""
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        return ids
    
    def search(self, query, filters=None, top_k=5):
        """Mock search."""
        return self.search_results
    
    def delete(self, doc_id):
        """Mock delete."""
        self.deleted_ids.append(doc_id)


if __name__ == "__main__":
    unittest.main() 