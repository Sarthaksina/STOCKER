"""
Unit tests for the RAG system components.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock
import json
from typing import List, Dict, Any

import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rag.chroma_db import ChromaDBManager
from src.rag.news_collector import NewsCollector
from src.rag.chunking import ChunkingPipeline, EmbeddingPipeline
from src.rag.query_formulation import QueryFormatter, QueryGenerator, optimize_query
from src.rag.response_generator import ContextBuilder, TemplateManager, ResponseGenerator


class TestChromaDBManager(unittest.TestCase):
    """Tests for the ChromaDBManager class."""
    
    @patch("src.rag.chroma_db.chromadb.PersistentClient")
    @patch("src.rag.chroma_db.embedding_functions.HuggingFaceEmbeddingFunction")
    def test_init(self, mock_embedding_fn, mock_client):
        """Test initialization of ChromaDBManager."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Create instance
        chroma_db = ChromaDBManager(collection_name="test_collection")
        
        # Assert
        mock_client.assert_called_once()
        mock_embedding_fn.assert_called_once()
        mock_client_instance.get_collection.assert_called_once()
        self.assertEqual(chroma_db.collection, mock_collection)
    
    @patch("src.rag.chroma_db.chromadb.PersistentClient")
    @patch("src.rag.chroma_db.embedding_functions.HuggingFaceEmbeddingFunction")
    def test_add_documents(self, mock_embedding_fn, mock_client):
        """Test adding documents to ChromaDB."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Create instance
        chroma_db = ChromaDBManager(collection_name="test_collection")
        
        # Test data
        documents = ["test document 1", "test document 2"]
        ids = ["id1", "id2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        
        # Call method
        chroma_db.add_documents(documents=documents, ids=ids, metadatas=metadatas)
        
        # Assert
        mock_collection.add.assert_called_once_with(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    
    @patch("src.rag.chroma_db.chromadb.PersistentClient")
    @patch("src.rag.chroma_db.embedding_functions.HuggingFaceEmbeddingFunction")
    def test_search(self, mock_embedding_fn, mock_client):
        """Test searching in ChromaDB."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock query results
        mock_results = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.2]]
        }
        mock_collection.query.return_value = mock_results
        
        # Create instance
        chroma_db = ChromaDBManager(collection_name="test_collection")
        
        # Call method
        results = chroma_db.search(query="test query", n_results=2)
        
        # Assert
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=2,
            where=None
        )
        self.assertEqual(results, mock_results)


class TestChunkingPipeline(unittest.TestCase):
    """Tests for the ChunkingPipeline class."""
    
    def test_init(self):
        """Test initialization with different strategies."""
        # Test default
        pipeline = ChunkingPipeline()
        self.assertEqual(pipeline.chunk_size, 512)
        self.assertEqual(pipeline.chunk_overlap, 128)
        self.assertEqual(pipeline.strategy, "sentence")
        
        # Test custom values
        pipeline = ChunkingPipeline(chunk_size=1024, chunk_overlap=256, strategy="paragraph")
        self.assertEqual(pipeline.chunk_size, 1024)
        self.assertEqual(pipeline.chunk_overlap, 256)
        self.assertEqual(pipeline.strategy, "paragraph")
        
        # Test invalid strategy
        pipeline = ChunkingPipeline(strategy="invalid")
        self.assertEqual(pipeline.strategy, "sentence")  # Should default to sentence
    
    def test_split_by_fixed_size(self):
        """Test splitting text by fixed size."""
        pipeline = ChunkingPipeline(chunk_size=10, chunk_overlap=2, strategy="fixed")
        text = "This is a test text to split into chunks"
        chunks = pipeline._split_by_fixed_size(text)
        
        # Expected chunks with size 10 and overlap 2
        expected = [
            "This is a ", 
            "a test tex", 
            "ext to spl", 
            "plit into ", 
            "o chunks"
        ]
        
        self.assertEqual(chunks, expected)
    
    def test_split_by_sentences(self):
        """Test splitting text by sentences."""
        pipeline = ChunkingPipeline(chunk_size=20, chunk_overlap=5, strategy="sentence")
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = pipeline._split_by_sentences(text)
        
        # Check that we have multiple chunks
        self.assertGreaterEqual(len(chunks), 2)
        
        # Check that each chunk respects size limit (with some tolerance for overlaps)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)  # Allow some tolerance
    
    def test_create_chunks(self):
        """Test creating chunks with metadata."""
        pipeline = ChunkingPipeline(chunk_size=50, chunk_overlap=10, strategy="fixed")
        text = "This is a test text that should be split into chunks with metadata."
        chunks = pipeline.create_chunks(text)
        
        # Check that we have chunks
        self.assertGreater(len(chunks), 0)
        
        # Check structure of first chunk
        first_chunk = chunks[0]
        self.assertIn("id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("metadata", first_chunk)
        
        # Check metadata
        metadata = first_chunk["metadata"]
        self.assertIn("chunk_index", metadata)
        self.assertIn("chunk_count", metadata)
        self.assertIn("chunk_strategy", metadata)
        self.assertEqual(metadata["chunk_strategy"], "fixed")


class TestQueryFormulation(unittest.TestCase):
    """Tests for query formulation components."""
    
    def test_query_formatter_init(self):
        """Test initialization of QueryFormatter."""
        formatter = QueryFormatter()
        self.assertIsNotNone(formatter.stock_symbols)
        self.assertIsNotNone(formatter.financial_terms)
        
        # Check that default symbols were loaded
        self.assertIn("AAPL", formatter.stock_symbols)
        
        # Check that default terms were loaded
        self.assertIn("eps", formatter.financial_terms)
    
    def test_identify_entities(self):
        """Test identifying financial entities in queries."""
        formatter = QueryFormatter()
        
        # Test query with stock symbol
        query = "What is the current performance of AAPL?"
        entities = formatter.identify_entities(query)
        self.assertIn("AAPL", entities["stock_symbols"])
        self.assertIn("Apple Inc.", entities["company_names"])
        
        # Test query with financial term
        query = "Explain the impact of inflation on eps"
        entities = formatter.identify_entities(query)
        self.assertIn("eps", entities["financial_terms"])
    
    def test_expand_query(self):
        """Test expanding queries with financial context."""
        formatter = QueryFormatter()
        
        # Test expanding stock symbol
        query = "What is the current performance of AAPL?"
        expanded = formatter.expand_query(query)
        self.assertIn("AAPL", expanded)
        self.assertIn("Apple Inc.", expanded)
        
        # Test expanding financial term
        query = "Explain eps calculation"
        expanded = formatter.expand_query(query)
        self.assertIn("eps", expanded)
        self.assertIn("earnings per share", expanded)
    
    def test_optimize_query_function(self):
        """Test the optimize_query function."""
        # Test simple query
        simple_query = "AAPL performance"
        optimized = optimize_query(simple_query)
        self.assertIsInstance(optimized, str)
        
        # Test complex query
        complex_query = "How will the recent Fed interest rate decision impact technology stocks compared to financial sector stocks?"
        optimized = optimize_query(complex_query)
        
        # Should either return a list of decomposed queries or a single string
        if isinstance(optimized, list):
            self.assertGreater(len(optimized), 1)
        else:
            self.assertIsInstance(optimized, str)


class TestResponseGeneration(unittest.TestCase):
    """Tests for response generation components."""
    
    def test_context_builder(self):
        """Test building context from articles."""
        builder = ContextBuilder(max_context_length=500, relevance_threshold=0.5)
        
        # Test with empty articles
        context = builder.build_context([])
        self.assertEqual(context, "No relevant information found.")
        
        # Test with articles
        articles = [
            {
                "id": "id1",
                "text": "This is test article 1",
                "metadata": {
                    "title": "Test Article 1",
                    "source": "Source1",
                    "published": "2023-01-01T00:00:00"
                },
                "relevance": 0.9
            },
            {
                "id": "id2",
                "text": "This is test article 2",
                "metadata": {
                    "title": "Test Article 2",
                    "source": "Source2",
                    "published": "2023-01-02T00:00:00"
                },
                "relevance": 0.7
            }
        ]
        
        context = builder.build_context(articles)
        
        # Check that both articles are included in context
        self.assertIn("Test Article 1", context)
        self.assertIn("Test Article 2", context)
        
        # Check that articles are correctly formatted
        self.assertIn("[Article 1 - Source1", context)
        self.assertIn("[Article 2 - Source2", context)
    
    def test_template_manager(self):
        """Test template management."""
        manager = TemplateManager()
        
        # Check that all template types exist
        for template_type in ["stock_performance", "market_analysis", "general"]:
            template = manager.get_template(template_type)
            self.assertIsInstance(template, str)
        
        # Check that invalid type gets default template
        template = manager.get_template("non_existent_type")
        self.assertEqual(template, manager.templates["general"])


@pytest.mark.integration
class TestIntegration(unittest.TestCase):
    """Integration tests for the RAG system."""
    
    @patch("src.rag.news_collector.NewsCollector.search_articles")
    def test_response_generator_integration(self, mock_search):
        """Test the integration of response generator components."""
        # Mock search results
        mock_search.return_value = [
            {
                "id": "id1",
                "text": "Apple Inc. (AAPL) stock has shown positive performance. The stock price increased by 2.5% yesterday.",
                "metadata": {
                    "title": "Apple Stock Performance",
                    "source": "TestSource",
                    "published": "2023-01-01T00:00:00",
                    "link": "https://example.com"
                },
                "relevance": 0.9
            }
        ]
        
        # Create response generator
        generator = ResponseGenerator()
        
        # Generate response
        response = generator.generate_response("What is the performance of AAPL?")
        
        # Check response structure
        self.assertIn("query", response)
        self.assertIn("response", response)
        self.assertIn("query_type", response)
        self.assertIn("sources", response)
        
        # Check that response contains stock ticker
        self.assertIn("AAPL", response["response"])
        
        # Check that query type is correct
        self.assertEqual(response["query_type"], "stock_performance")


if __name__ == "__main__":
    unittest.main() 