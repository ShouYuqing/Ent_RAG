"""
Tests for the retrieval components.
"""

import unittest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.document_store import Document, DocumentStore
from app.retrieval.base import BaseRetriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranking import ReRanker


class TestBaseRetriever(unittest.TestCase):
    """Test the BaseRetriever class."""
    
    def setUp(self):
        """Set up test environment."""
        self.document_store = MagicMock(spec=DocumentStore)
        self.retriever = BaseRetriever(document_store=self.document_store)
    
    def test_init(self):
        """Test retriever initialization."""
        self.assertEqual(self.retriever.document_store, self.document_store)
    
    def test_retrieve_not_implemented(self):
        """Test retrieve method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.retriever.retrieve("test query")


class TestSemanticRetriever(unittest.TestCase):
    """Test the SemanticRetriever class."""
    
    def setUp(self):
        """Set up test environment."""
        self.document_store = MagicMock(spec=DocumentStore)
        self.retriever = SemanticRetriever(document_store=self.document_store)
    
    def test_init(self):
        """Test retriever initialization."""
        self.assertEqual(self.retriever.document_store, self.document_store)
        self.assertEqual(self.retriever.top_k, 5)
    
    def test_retrieve(self):
        """Test retrieve method."""
        # Set up mock documents
        mock_docs = [
            MagicMock(spec=Document),
            MagicMock(spec=Document)
        ]
        self.document_store.search.return_value = mock_docs
        
        # Call retrieve
        result = self.retriever.retrieve("test query", top_k=3)
        
        # Check result
        self.assertEqual(result, mock_docs)
        
        # Check document_store.search was called correctly
        self.document_store.search.assert_called_once_with(
            query="test query",
            filters=None,
            top_k=3
        )


class TestKeywordRetriever(unittest.TestCase):
    """Test the KeywordRetriever class."""
    
    def setUp(self):
        """Set up test environment."""
        self.document_store = MagicMock(spec=DocumentStore)
        self.retriever = KeywordRetriever(document_store=self.document_store)
    
    def test_init(self):
        """Test retriever initialization."""
        self.assertEqual(self.retriever.document_store, self.document_store)
        self.assertEqual(self.retriever.top_k, 5)
    
    @patch('app.retrieval.keyword.KeywordRetriever._search_bm25')
    def test_retrieve(self, mock_search_bm25):
        """Test retrieve method."""
        # Set up mock documents
        mock_docs = [
            MagicMock(spec=Document),
            MagicMock(spec=Document)
        ]
        mock_search_bm25.return_value = mock_docs
        
        # Call retrieve
        result = self.retriever.retrieve("test query", top_k=3)
        
        # Check result
        self.assertEqual(result, mock_docs)
        
        # Check _search_bm25 was called correctly
        mock_search_bm25.assert_called_once_with("test query", None, 3)


class TestHybridRetriever(unittest.TestCase):
    """Test the HybridRetriever class."""
    
    def setUp(self):
        """Set up test environment."""
        self.document_store = MagicMock(spec=DocumentStore)
        
        # Create mock retrievers
        self.semantic_retriever = MagicMock(spec=SemanticRetriever)
        self.keyword_retriever = MagicMock(spec=KeywordRetriever)
        
        # Initialize hybrid retriever with mocks
        self.retriever = HybridRetriever(
            document_store=self.document_store,
            semantic_retriever=self.semantic_retriever,
            keyword_retriever=self.keyword_retriever
        )
    
    def test_init(self):
        """Test retriever initialization."""
        self.assertEqual(self.retriever.document_store, self.document_store)
        self.assertEqual(self.retriever.semantic_retriever, self.semantic_retriever)
        self.assertEqual(self.retriever.keyword_retriever, self.keyword_retriever)
        self.assertEqual(self.retriever.top_k, 5)
        self.assertEqual(self.retriever.semantic_weight, 0.7)
        self.assertEqual(self.retriever.keyword_weight, 0.3)
    
    def test_retrieve(self):
        """Test retrieve method."""
        # Set up mock documents
        doc1 = MagicMock(spec=Document)
        doc1.doc_id = "doc1"
        doc1.content = "Document 1"
        
        doc2 = MagicMock(spec=Document)
        doc2.doc_id = "doc2"
        doc2.content = "Document 2"
        
        doc3 = MagicMock(spec=Document)
        doc3.doc_id = "doc3"
        doc3.content = "Document 3"
        
        # Set up mock retriever results
        self.semantic_retriever.retrieve.return_value = [doc1, doc2]
        self.keyword_retriever.retrieve.return_value = [doc2, doc3]
        
        # Call retrieve
        result = self.retriever.retrieve("test query", top_k=3)
        
        # Check result
        self.assertEqual(len(result), 3)
        self.assertIn(doc1, result)
        self.assertIn(doc2, result)
        self.assertIn(doc3, result)
        
        # Check retrievers were called correctly
        self.semantic_retriever.retrieve.assert_called_once_with("test query", None, 3)
        self.keyword_retriever.retrieve.assert_called_once_with("test query", None, 3)


class TestReRanker(unittest.TestCase):
    """Test the ReRanker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.reranker = ReRanker()
    
    def test_init(self):
        """Test reranker initialization."""
        self.assertIsNotNone(self.reranker)
    
    @patch('app.retrieval.reranking.ReRanker._compute_cross_encoder_scores')
    def test_rerank(self, mock_compute_scores):
        """Test rerank method."""
        # Set up mock documents
        doc1 = MagicMock(spec=Document)
        doc1.doc_id = "doc1"
        doc1.content = "Document 1"
        
        doc2 = MagicMock(spec=Document)
        doc2.doc_id = "doc2"
        doc2.content = "Document 2"
        
        documents = [doc1, doc2]
        
        # Set up mock scores
        mock_compute_scores.return_value = [0.3, 0.7]
        
        # Call rerank
        result = self.reranker.rerank("test query", documents)
        
        # Check result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], doc2)  # Higher score should be first
        self.assertEqual(result[1], doc1)
        
        # Check _compute_cross_encoder_scores was called correctly
        mock_compute_scores.assert_called_once()
        args = mock_compute_scores.call_args[0]
        self.assertEqual(args[0], "test query")
        self.assertEqual(args[1], ["Document 1", "Document 2"])


if __name__ == "__main__":
    unittest.main() 