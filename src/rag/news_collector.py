"""
Financial news collector for the RAG system.
Fetches news from various sources and processes them for vector storage.
"""
import os
import time
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.configuration.config import settings
from src.rag.chroma_db import get_chroma_db
from src.exception.exception import NewsCollectionError

# Initialize logger
logger = logging.getLogger(__name__)

# Define paths for caching
CACHE_DIR = os.path.join(settings.data_dir, "news_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class NewsCollector:
    """
    Collects financial news from various sources, processes and stores them for the RAG system.
    Supports RSS feeds, web scraping, and API integrations.
    """
    def __init__(self, cache_expiry_hours: int = 24):
        """
        Initialize the news collector with caching settings.
        
        Args:
            cache_expiry_hours: Hours before cached news is considered expired
        """
        self.cache_expiry_hours = cache_expiry_hours
        self.chroma_db = get_chroma_db(collection_name="financial_news")
        
        # Define RSS feed sources for financial news
        self.rss_feeds = {
            "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
            "cnbc": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
            "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss"
        }
        
        # Define headers for web requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def collect_from_rss(self, feed_name: str) -> List[Dict[str, Any]]:
        """
        Collect news articles from an RSS feed.
        
        Args:
            feed_name: Name of the feed to collect from
        
        Returns:
            List of news article dictionaries
        """
        try:
            if feed_name not in self.rss_feeds:
                raise NewsCollectionError(f"Unknown RSS feed: {feed_name}")
                
            feed_url = self.rss_feeds[feed_name]
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries:
                # Create a unique ID based on title and link
                article_id = hashlib.md5((entry.title + entry.link).encode()).hexdigest()
                
                # Extract publication date
                published = entry.get('published', entry.get('pubDate', None))
                if published:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])
                    except (AttributeError, TypeError):
                        pub_date = datetime.now()
                else:
                    pub_date = datetime.now()
                
                # Extract and clean content
                content = entry.get('content', [{}])[0].get('value', '') if 'content' in entry else ''
                if not content and 'summary' in entry:
                    content = entry.summary
                    
                # Clean HTML content
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text()
                
                article = {
                    "id": article_id,
                    "title": entry.title,
                    "link": entry.link,
                    "content": content,
                    "published": pub_date.isoformat(),
                    "source": feed_name,
                    "type": "rss"
                }
                articles.append(article)
                
            logger.info(f"Collected {len(articles)} articles from {feed_name} RSS feed")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from RSS feed {feed_name}: {e}")
            raise NewsCollectionError(f"Failed to collect news from {feed_name}: {e}")
    
    def collect_all_feeds(self) -> List[Dict[str, Any]]:
        """
        Collect news from all configured RSS feeds.
        
        Returns:
            List of news article dictionaries
        """
        all_articles = []
        for feed_name in self.rss_feeds:
            try:
                articles = self.collect_from_rss(feed_name)
                all_articles.extend(articles)
                # Avoid hitting rate limits
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Skipping feed {feed_name} due to error: {e}")
        
        return all_articles
    
    def _clean_and_prepare_content(self, article: Dict[str, Any]) -> str:
        """
        Clean and prepare article content for embedding.
        
        Args:
            article: Article dictionary
        
        Returns:
            Cleaned and prepared text
        """
        # Combine title and content for better context
        combined_text = f"{article['title']}\n\n{article.get('content', '')}"
        
        # Remove excessive whitespace
        cleaned_text = ' '.join(combined_text.split())
        
        # Truncate if too long (most embedding models have token limits)
        if len(cleaned_text) > 10000:
            cleaned_text = cleaned_text[:10000] + "..."
            
        return cleaned_text
    
    def store_articles_in_chroma(self, articles: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Store articles in ChromaDB for vector search.
        
        Args:
            articles: List of article dictionaries
        
        Returns:
            Tuple of (number of articles stored, number of articles skipped)
        """
        if not articles:
            return 0, 0
            
        documents = []
        ids = []
        metadatas = []
        
        for article in articles:
            # Prepare document text for embedding
            document_text = self._clean_and_prepare_content(article)
            
            # Prepare metadata (exclude large content field)
            metadata = {
                "title": article["title"],
                "link": article["link"],
                "published": article["published"],
                "source": article["source"],
                "type": article["type"]
            }
            
            documents.append(document_text)
            ids.append(article["id"])
            metadatas.append(metadata)
        
        # Add to ChromaDB
        try:
            self.chroma_db.add_documents(documents=documents, ids=ids, metadatas=metadatas)
            logger.info(f"Stored {len(documents)} articles in ChromaDB")
            return len(documents), 0
        except Exception as e:
            logger.error(f"Error storing articles in ChromaDB: {e}")
            # If batch insert fails, try individually
            stored = 0
            skipped = 0
            for i, (doc, doc_id, metadata) in enumerate(zip(documents, ids, metadatas)):
                try:
                    self.chroma_db.add_documents([doc], [doc_id], [metadata])
                    stored += 1
                except Exception:
                    skipped += 1
            logger.info(f"Individual storage: {stored} stored, {skipped} skipped")
            return stored, skipped
    
    def search_articles(self, query: str, n_results: int = 5, 
                       filter_by_source: Optional[str] = None,
                       max_age_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for articles related to a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_by_source: Optional filter for specific news source
            max_age_days: Optional filter for maximum article age in days
        
        Returns:
            List of articles related to the query
        """
        # Build filter criteria
        filter_criteria = {}
        if filter_by_source:
            filter_criteria["source"] = filter_by_source
            
        if max_age_days:
            min_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            # This assumes ChromaDB can handle datetime comparison
            # If not, we'll need to filter results afterwards
            filter_criteria["published"] = {"$gte": min_date}
        
        # Search in ChromaDB
        results = self.chroma_db.search(
            query=query,
            n_results=n_results,
            filter_criteria=filter_criteria if filter_criteria else None
        )
        
        # Format results
        articles = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # Calculate relevance score (1 - normalized distance)
            relevance = 1 - (distance / max(results["distances"][0]) if results["distances"][0] else 0)
            
            article = {
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": metadata,
                "relevance": round(relevance, 3)
            }
            articles.append(article)
            
        return articles

    def run_collection_job(self) -> Dict[str, Any]:
        """
        Run a full collection job from all sources and store in ChromaDB.
        
        Returns:
            Dictionary with job statistics
        """
        start_time = time.time()
        stats = {
            "started_at": datetime.now().isoformat(),
            "feeds_processed": 0,
            "articles_collected": 0,
            "articles_stored": 0,
            "articles_skipped": 0,
            "errors": []
        }
        
        try:
            # Collect from RSS feeds
            all_articles = self.collect_all_feeds()
            stats["feeds_processed"] = len(self.rss_feeds)
            stats["articles_collected"] = len(all_articles)
            
            # Store in ChromaDB
            stored, skipped = self.store_articles_in_chroma(all_articles)
            stats["articles_stored"] = stored
            stats["articles_skipped"] = skipped
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in collection job: {error_msg}")
            stats["errors"].append(error_msg)
        
        # Add timing information
        stats["completed_at"] = datetime.now().isoformat()
        stats["execution_time_seconds"] = time.time() - start_time
        
        return stats


# Factory function to get NewsCollector
def get_news_collector() -> NewsCollector:
    """
    Get a NewsCollector instance.
    
    Returns:
        NewsCollector instance
    """
    return NewsCollector()


if __name__ == "__main__":
    # Set up basic logging for standalone usage
    logging.basicConfig(level=logging.INFO)
    
    # Run a collection job
    collector = get_news_collector()
    job_stats = collector.run_collection_job()
    
    # Print stats
    print(json.dumps(job_stats, indent=2))
    
    # Example search
    results = collector.search_articles("stock market trends", n_results=3)
    for i, article in enumerate(results):
        print(f"\n--- Result {i+1} (Relevance: {article['relevance']}) ---")
        print(f"Title: {article['metadata']['title']}")
        print(f"Source: {article['metadata']['source']}")
        print(f"Link: {article['metadata']['link']}")
        print(f"Published: {article['metadata']['published']}")
        print(f"Excerpt: {article['text'][:200]}...") 