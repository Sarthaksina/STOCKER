"""
Run a complete RAG pipeline demonstration with financial data.
This script demonstrates the full end-to-end functionality of the financial RAG system.
"""
import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

from src.rag.chroma_db import get_chroma_db
from src.rag.news_collector import get_news_collector
from src.rag.chunking import process_and_embed_article
from src.rag.query_formulation import optimize_query
from src.rag.response_generator import generate_financial_insight
from src.configuration.config import settings

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_pipeline() -> Dict[str, Any]:
    """
    Set up and initialize all components of the RAG pipeline.
    
    Returns:
        Dictionary with initialized components
    """
    logger.info("Setting up RAG pipeline components...")
    
    # Initialize ChromaDB
    chroma_db = get_chroma_db(collection_name="financial_news")
    logger.info("ChromaDB initialized")
    
    # Initialize news collector
    news_collector = get_news_collector()
    logger.info("News collector initialized")
    
    # Return components
    components = {
        "chroma_db": chroma_db,
        "news_collector": news_collector
    }
    
    return components

def collect_news(news_collector) -> Dict[str, Any]:
    """
    Collect financial news articles and store in vector database.
    
    Args:
        news_collector: NewsCollector instance
    
    Returns:
        Dictionary with collection job statistics
    """
    logger.info("Starting news collection...")
    
    # Run collection job
    job_stats = news_collector.run_collection_job()
    
    logger.info(f"News collection complete. Collected {job_stats['articles_collected']} articles, stored {job_stats['articles_stored']}")
    
    return job_stats

def run_sample_queries(components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run sample financial queries through the RAG pipeline.
    
    Args:
        components: Dictionary with pipeline components
    
    Returns:
        List of query results
    """
    # Sample queries
    queries = [
        "What is the current performance of AAPL?",
        "How has the stock market been performing this week?",
        "What is the impact of recent Fed decisions on tech stocks?",
        "What are the growth prospects for Tesla in the electric vehicle market?",
        "How are inflation concerns affecting consumer goods companies?"
    ]
    
    results = []
    
    logger.info(f"Running {len(queries)} sample queries...")
    
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {query}")
        
        # Optimize query
        optimized_query = optimize_query(query)
        logger.info(f"Optimized query: {optimized_query}")
        
        # Generate insight
        start_time = time.time()
        insight = generate_financial_insight(query)
        processing_time = time.time() - start_time
        
        logger.info(f"Generated insight in {processing_time:.2f} seconds")
        
        # Store result
        result = {
            "query": query,
            "optimized_query": optimized_query,
            "insight": insight,
            "processing_time": processing_time
        }
        
        results.append(result)
        
        # Avoid hitting rate limits
        if i < len(queries) - 1:
            time.sleep(1)
    
    return results

def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary report of the RAG pipeline run.
    
    Args:
        results: List of query results
    
    Returns:
        Dictionary with report data
    """
    # Calculate statistics
    total_time = sum(result["processing_time"] for result in results)
    avg_time = total_time / len(results) if results else 0
    
    # Count sources
    sources = []
    source_counts = {}
    
    for result in results:
        for source in result["insight"]["sources"]:
            sources.append(source)
            source_name = source["source"]
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
    
    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "queries_processed": len(results),
        "total_processing_time": total_time,
        "average_processing_time": avg_time,
        "unique_sources_used": len(set(source["source"] for source in sources)),
        "source_distribution": source_counts,
        "query_types": {result["insight"]["query_type"]: result["insight"]["query_type"] for result in results}
    }
    
    return report

def save_results(results: List[Dict[str, Any]], report: Dict[str, Any], output_dir: str) -> None:
    """
    Save pipeline results and report to files.
    
    Args:
        results: List of query results
        report: Pipeline run report
        output_dir: Directory to save results in
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    results_path = os.path.join(output_dir, f"rag_results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Save report
    report_path = os.path.join(output_dir, f"rag_report_{timestamp}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Report saved to {report_path}")

def print_sample_result(result: Dict[str, Any]) -> None:
    """
    Print a sample result to the console.
    
    Args:
        result: Query result to print
    """
    print("\n" + "="*80)
    print(f"QUERY: {result['query']}")
    print(f"OPTIMIZED QUERY: {result['optimized_query']}")
    print("-"*80)
    print(result['insight']['response'])
    print("-"*80)
    print("SOURCES:")
    for source in result['insight']['sources']:
        print(f"- {source['title']} ({source['source']})")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    print("="*80)

def run_pipeline() -> None:
    """Run the complete RAG pipeline demonstration."""
    logger.info("Starting RAG pipeline demonstration...")
    
    try:
        # Setup components
        components = setup_pipeline()
        
        # Collect news
        collection_stats = collect_news(components["news_collector"])
        
        # Run sample queries
        results = run_sample_queries(components)
        
        # Generate report
        report = generate_report(results)
        
        # Save results
        save_results(results, report, os.path.join(settings.data_dir, "rag_output"))
        
        # Print a sample result
        if results:
            print_sample_result(results[0])
        
        logger.info("RAG pipeline demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error running RAG pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline() 