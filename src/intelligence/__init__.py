"""
Intelligence module for STOCKER Pro.

This module provides artificial intelligence components including LLMs,
news processing, and RAG pipelines for financial analysis.
"""

from src.intelligence.llm import (
    LLMClient,
    get_llm_client,
    generate_analysis,
    generate_summary
)

from src.intelligence.news import (
    NewsCollector,
    get_news_collector,
    process_news,
    analyze_sentiment
)

from src.intelligence.rag import (
    RAGPipeline,
    get_rag_pipeline,
    query_financial_knowledge,
    generate_market_insights
)

from src.intelligence.vector_store import (
    VectorStore,
    get_vector_store,
    add_documents,
    similarity_search
)

__all__ = [
    # LLM components
    'LLMClient',
    'get_llm_client',
    'generate_analysis',
    'generate_summary',
    
    # News components
    'NewsCollector',
    'get_news_collector',
    'process_news',
    'analyze_sentiment',
    
    # RAG components
    'RAGPipeline',
    'get_rag_pipeline',
    'query_financial_knowledge',
    'generate_market_insights',
    
    # Vector store components
    'VectorStore',
    'get_vector_store',
    'add_documents',
    'similarity_search'
]
