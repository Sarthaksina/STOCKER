"""
RAG pipeline for STOCKER Pro.

This module provides a Retrieval-Augmented Generation (RAG) system that combines
financial data and news with LLM capabilities to provide context-aware financial insights.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import logging

from src.core.config import Config
from src.core.exceptions import RAGError
from src.core.logging import logger
from src.intelligence.vector_store import VectorStore

class RAGPipeline:
    """
    RAG pipeline for financial insights.
    
    Provides a complete pipeline for retrieval-augmented generation combining
    financial news, market data, and other information to enhance LLM outputs.
    """
    
    def __init__(self, config=None):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Optional configuration for the RAG pipeline
        """
        self.config = config or {}
        self.vector_store = VectorStore(config)
        self.llm_client = None  # Will be initialized on demand
        self.cache_dir = self.config.get('cache_dir', './data/rag_cache/')
        self.cache_expiry = self.config.get('cache_expiry', 86400)  # 24 hours
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def initialize_llm_client(self):
        """Initialize the LLM client if not already initialized."""
        if self.llm_client is None:
            from src.intelligence.llm import LLMClient
            self.llm_client = LLMClient(self.config)
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process documents for the RAG system.
        
        Args:
            documents: List of documents to process
        """
        # Process and chunk documents
        chunked_docs = self._chunk_documents(documents)
        
        # Create embeddings and store in vector database
        self.vector_store.add_documents(chunked_docs)
    
    def process_news(self, news_items: List[Dict[str, Any]]) -> None:
        """
        Process news items for the RAG system.
        
        Args:
            news_items: List of news items to process
        """
        # Convert news items to document format
        documents = []
        for item in news_items:
            documents.append({
                'id': f"news_{hash(item['url'])}",
                'text': f"{item['title']}\n\n{item['summary']}",
                'metadata': {
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'published_at': item.get('published_at', ''),
                    'type': 'news'
                }
            })
        
        # Process the documents
        self.process_documents(documents)
    
    def query(self, query_text: str, 
             filter_criteria: Optional[Dict[str, Any]] = None,
             context_items: Optional[List[Dict[str, Any]]] = None,
             use_cache: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query_text: Query text to process
            filter_criteria: Optional filter criteria for vector search
            context_items: Optional additional context items
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with query results
        """
        # Initialize LLM client if needed
        self.initialize_llm_client()
        
        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(query_text, filter_criteria)
            cache_result = self._check_cache(cache_key)
            if cache_result:
                return cache_result
        
        # Formulate the query
        enhanced_query = self._formulate_query(query_text)
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(
            enhanced_query,
            filter_criteria=filter_criteria,
            limit=10
        )
        
        # Generate context
        context = self._generate_context(retrieved_docs, context_items)
        
        # Generate response
        response = self._generate_response(query_text, context)
        
        # Cache result
        if use_cache:
            self._cache_result(cache_key, response)
        
        return response
    
    def _chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces for better retrieval.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        chunk_size = self.config.get('chunk_size', 500)
        chunk_overlap = self.config.get('chunk_overlap', 100)
        
        for doc in documents:
            text = doc['text']
            metadata = doc.get('metadata', {})
            
            # Simple text chunking by words
            words = text.split()
            
            if len(words) <= chunk_size:
                # Document is small enough, no chunking needed
                chunked_docs.append({
                    'id': doc.get('id', f"chunk_{len(chunked_docs)}"),
                    'text': text,
                    'metadata': metadata
                })
            else:
                # Create overlapping chunks
                for i in range(0, len(words), chunk_size - chunk_overlap):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) < chunk_size / 2 and i > 0:
                        # Skip very small chunks at the end
                        continue
                    
                    chunk_text = ' '.join(chunk_words)
                    chunk_id = f"{doc.get('id', 'doc')}_{i // (chunk_size - chunk_overlap)}"
                    
                    chunked_docs.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'metadata': {
                            **metadata,
                            'chunk_index': i // (chunk_size - chunk_overlap),
                            'total_chunks': len(words) // (chunk_size - chunk_overlap) + 1
                        }
                    })
        
        return chunked_docs
    
    def _formulate_query(self, query_text: str) -> str:
        """
        Formulate an enhanced query.
        
        Args:
            query_text: Original query text
            
        Returns:
            Enhanced query text
        """
        # Initialize LLM client if needed
        self.initialize_llm_client()
        
        # For simple queries, just return the original
        if len(query_text.split()) <= 5 or '?' not in query_text:
            return query_text
        
        try:
            # Use LLM to enhance the query
            prompt = f"""
            You are an AI assistant specializing in financial markets.
            Please rewrite the following query to make it more effective for semantic search.
            Focus on key concepts, entities, and financial terms while keeping it concise.
            
            Original query: {query_text}
            
            Enhanced query:
            """
            
            response = self.llm_client.complete(prompt, max_tokens=100)
            enhanced_query = response.strip()
            
            if enhanced_query:
                return enhanced_query
            else:
                return query_text
                
        except Exception as e:
            logger.warning(f"Error formulating enhanced query: {e}")
            return query_text
    
    def _generate_context(self, documents: List[Dict[str, Any]], 
                         additional_context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate context from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            additional_context: Optional additional context items
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add retrieved documents
        if documents:
            context_parts.append("## Retrieved Information")
            for i, doc in enumerate(documents, 1):
                source = doc.get('metadata', {}).get('source', 'Unknown')
                doc_type = doc.get('metadata', {}).get('type', 'document')
                date = doc.get('metadata', {}).get('published_at', '')
                
                context_parts.append(f"### Document {i}: {source} ({doc_type})")
                if date:
                    context_parts.append(f"Date: {date}")
                context_parts.append(doc['text'])
                context_parts.append("---")
        
        # Add additional context
        if additional_context:
            context_parts.append("## Additional Context")
            for i, ctx in enumerate(additional_context, 1):
                title = ctx.get('title', f'Context {i}')
                context_parts.append(f"### {title}")
                context_parts.append(ctx.get('text', ''))
                context_parts.append("---")
        
        return "\n\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate response using LLM.
        
        Args:
            query: User query
            context: Context information
            
        Returns:
            Response dictionary
        """
        # Initialize LLM client if needed
        self.initialize_llm_client()
        
        prompt = f"""
        You are an AI financial analyst assistant who provides accurate, helpful information about 
        financial markets, stocks, and investments based on the provided context.
        
        If the context doesn't contain relevant information to answer the query, acknowledge this
        and provide general information or best practices related to the query rather than making up facts.
        
        Always indicate the source of information when available.
        
        CONTEXT:
        {context}
        
        USER QUERY:
        {query}
        
        Please provide a comprehensive answer that addresses the user's query:
        """
        
        try:
            response_text = self.llm_client.complete(prompt, max_tokens=800)
            
            return {
                'query': query,
                'response': response_text,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'query': query,
                'response': f"I apologize, but I encountered an error while processing your query. Please try again or rephrase your question.",
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_cache_key(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate a cache key for the query."""
        import hashlib
        
        key_parts = [query]
        if filters:
            key_parts.append(json.dumps(filters, sort_keys=True))
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if result is in cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < self.cache_expiry:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache query result."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")
    
    def clear_cache(self) -> None:
        """Clear the RAG cache."""
        import glob
        
        try:
            files = glob.glob(os.path.join(self.cache_dir, "*.json"))
            for file in files:
                os.remove(file)
            
            logger.info(f"Cleared {len(files)} items from RAG cache")
            
        except Exception as e:
            logger.error(f"Error clearing RAG cache: {e}")
            raise RAGError(f"Failed to clear RAG cache: {e}")
    
    def create_investment_analysis(self, ticker: str, 
                                 include_news: bool = True,
                                 include_market_data: bool = True) -> Dict[str, Any]:
        """
        Create an investment analysis for a stock.
        
        Args:
            ticker: Stock ticker symbol
            include_news: Whether to include recent news
            include_market_data: Whether to include market data
            
        Returns:
            Investment analysis as a dictionary
        """
        self.initialize_llm_client()
        
        # Collect news if requested
        news_context = ""
        if include_news:
            from src.intelligence.news import NewsProcessor
            news_client = NewsProcessor(self.config)
            news_items = news_client.get_company_news(ticker, days=30)
            if news_items:
                news_texts = [f"Title: {item['title']}\nSource: {item['source']}\nDate: {item['published_at']}\n{item['summary']}" 
                           for item in news_items[:5]]
                news_context = "## Recent News\n\n" + "\n\n---\n\n".join(news_texts)
        
        # Collect market data if requested
        market_data_context = ""
        if include_market_data:
            try:
                from src.data.manager import DataManager
                data_manager = DataManager()
                stock_data = data_manager.get_stock_data(ticker, period='6mo')
                
                if not stock_data.empty:
                    recent_data = stock_data.tail(20)
                    current_price = recent_data['close'].iloc[-1]
                    prev_price = recent_data['close'].iloc[-2]
                    daily_change = (current_price - prev_price) / prev_price * 100
                    
                    month_ago = stock_data['close'].iloc[-21] if len(stock_data) >= 21 else stock_data['close'].iloc[0]
                    month_change = (current_price - month_ago) / month_ago * 100
                    
                    volume = recent_data['volume'].iloc[-1]
                    avg_volume = recent_data['volume'].mean()
                    
                    market_data_context = f"""## Market Data
Current Price: ${current_price:.2f}
Daily Change: {daily_change:.2f}%
30-Day Change: {month_change:.2f}%
Volume: {volume:,.0f} (Average: {avg_volume:,.0f})
"""
            except Exception as e:
                logger.warning(f"Error fetching market data for {ticker}: {e}")
        
        # Combine context
        context = f"# Analysis for {ticker}\n\n"
        if market_data_context:
            context += market_data_context + "\n\n"
        if news_context:
            context += news_context
        
        # Generate analysis
        prompt = f"""
        You are a professional financial analyst tasked with creating an investment analysis report
        for {ticker} based on the provided context. 
        
        Your analysis should cover:
        1. A brief overview of the company
        2. Recent performance summary
        3. Key news and developments
        4. Strengths and risks
        5. Outlook
        
        Use a professional, balanced tone and focus on factual information rather than speculation.
        
        CONTEXT:
        {context}
        
        Please provide your comprehensive investment analysis:
        """
        
        try:
            analysis_text = self.llm_client.complete(prompt, max_tokens=1200)
            
            return {
                'ticker': ticker,
                'analysis': analysis_text,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error generating investment analysis: {e}")
            return {
                'ticker': ticker,
                'analysis': f"I apologize, but I encountered an error while generating the investment analysis for {ticker}. Please try again later.",
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            } 