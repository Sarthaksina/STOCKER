"""
News collection and processing for STOCKER Pro.

This module provides functionality to collect, process, and analyze financial news 
from various sources for market intelligence and sentiment analysis.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging

from src.core.config import Config
from src.core.exceptions import NewsCollectionError
from src.core.logging import logger

class NewsProcessor:
    """
    News processor for collecting and analyzing financial news.
    
    Provides methods to fetch news from various sources, parse content,
    and prepare it for further analysis or embeddings.
    """
    
    def __init__(self, config=None):
        """
        Initialize the news processor.
        
        Args:
            config: Optional configuration for news collection
        """
        self.config = config or {}
        self.api_keys = {
            'alpha_vantage': self.config.get('alpha_vantage_api_key', os.environ.get('ALPHA_VANTAGE_API_KEY', '')),
            'newsapi': self.config.get('newsapi_key', os.environ.get('NEWSAPI_KEY', '')),
            'financial_modeling_prep': self.config.get('fmp_api_key', os.environ.get('FMP_API_KEY', ''))
        }
        self.cache_dir = self.config.get('cache_dir', './data/news_cache/')
        self.cache_expiry = self.config.get('cache_expiry', 86400)  # 24 hours by default
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_company_news(self, ticker: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent news for a specific company.
        
        Args:
            ticker: Company ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles as dictionaries
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{ticker}_news.json")
            
            # Check cache
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < self.cache_expiry:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        logger.info(f"Using cached news for {ticker}")
                        return cached_data
            
            # Combine news from multiple sources
            news_items = []
            
            # Try Alpha Vantage
            if self.api_keys['alpha_vantage']:
                av_news = self._get_alpha_vantage_news(ticker)
                if av_news:
                    news_items.extend(av_news)
            
            # Try NewsAPI
            if self.api_keys['newsapi']:
                newsapi_news = self._get_newsapi_news(ticker, days)
                if newsapi_news:
                    news_items.extend(newsapi_news)
            
            # Try Financial Modeling Prep
            if self.api_keys['financial_modeling_prep']:
                fmp_news = self._get_fmp_news(ticker, days)
                if fmp_news:
                    news_items.extend(fmp_news)
            
            # Deduplicate news
            deduplicated = self._deduplicate_news(news_items)
            
            # Sort by published date (most recent first)
            deduplicated.sort(key=lambda x: x.get('published_at', ''), reverse=True)
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(deduplicated, f)
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            return []
    
    def get_market_news(self, days: int = 3) -> List[Dict[str, Any]]:
        """
        Get recent general market news.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of news articles as dictionaries
        """
        try:
            cache_file = os.path.join(self.cache_dir, "market_news.json")
            
            # Check cache
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < self.cache_expiry:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        logger.info("Using cached market news")
                        return cached_data
            
            # Define market news sources and topics
            market_keywords = [
                'stock market', 'wall street', 'nasdaq', 'dow jones', 'S&P 500',
                'federal reserve', 'interest rates', 'inflation', 'market outlook',
                'economic data', 'market analysis'
            ]
            
            # Combine news from multiple sources
            news_items = []
            
            # Try NewsAPI
            if self.api_keys['newsapi']:
                for keyword in market_keywords:
                    keyword_news = self._get_newsapi_news(keyword, days)
                    if keyword_news:
                        news_items.extend(keyword_news)
            
            # Try Financial Modeling Prep
            if self.api_keys['financial_modeling_prep']:
                fmp_news = self._get_fmp_market_news(days)
                if fmp_news:
                    news_items.extend(fmp_news)
            
            # Deduplicate news
            deduplicated = self._deduplicate_news(news_items)
            
            # Sort by published date (most recent first)
            deduplicated.sort(key=lambda x: x.get('published_at', ''), reverse=True)
            
            # Limit to a reasonable number
            deduplicated = deduplicated[:100]
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(deduplicated, f)
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error getting market news: {e}")
            return []
    
    def get_economic_calendar(self, days_ahead: int = 7, days_past: int = 0) -> List[Dict[str, Any]]:
        """
        Get economic calendar events.
        
        Args:
            days_ahead: Number of days to look ahead
            days_past: Number of days to look back
            
        Returns:
            List of economic events as dictionaries
        """
        try:
            cache_file = os.path.join(self.cache_dir, "economic_calendar.json")
            
            # Check cache
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 21600:  # 6 hour cache for economic calendar
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        logger.info("Using cached economic calendar")
                        return cached_data
            
            # Calculate date range
            today = datetime.now().date()
            start_date = (today - timedelta(days=days_past)).strftime('%Y-%m-%d')
            end_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            # Get data from Financial Modeling Prep
            if self.api_keys['financial_modeling_prep']:
                events = self._get_fmp_economic_calendar(start_date, end_date)
            else:
                events = []
            
            # Sort by date
            events.sort(key=lambda x: x.get('date', ''))
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(events, f)
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting economic calendar: {e}")
            return []
    
    def extract_article_content(self, url: str) -> str:
        """
        Extract the text content from a news article URL.
        
        Args:
            url: URL of the news article
            
        Returns:
            Extracted article text
        """
        try:
            # Check for cached version
            cache_key = self._url_to_cache_key(url)
            cache_file = os.path.join(self.cache_dir, f"article_{cache_key}.txt")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Request the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text and clean it up
            article_text = soup.get_text(separator=' ')
            lines = [line.strip() for line in article_text.splitlines()]
            chunks = [line for line in lines if line]
            article_text = ' '.join(chunks)
            
            # Basic cleanup - remove excessive whitespace
            article_text = ' '.join(article_text.split())
            
            # Cache the content
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(article_text)
            
            return article_text
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""
    
    def analyze_news_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of a collection of news articles.
        
        Args:
            news_items: List of news articles
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # This is a placeholder - in a real system we would use an NLP model
        # or sentiment analysis API to analyze the sentiment
        
        return {
            'average_sentiment': 0.0,
            'sentiment_distribution': {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            },
            'trending_topics': []
        }
    
    def _url_to_cache_key(self, url: str) -> str:
        """Convert a URL to a cache key."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    def _deduplicate_news(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate news items based on title similarity."""
        if not news_items:
            return []
        
        # Use a simple approach based on title similarity
        deduplicated = []
        titles = set()
        
        for item in news_items:
            title = item.get('title', '').lower()
            title_hash = hash(title)
            
            # Check for exact duplicates
            if title_hash in titles:
                continue
            
            # Check for similar titles (simple approach)
            duplicate = False
            for existing_title in titles:
                # If over 80% of words match, consider it a duplicate
                words1 = set(title.split())
                words2 = set(existing_title.split())
                if len(words1) > 3 and len(words2) > 3:
                    common_words = words1.intersection(words2)
                    if len(common_words) / max(len(words1), len(words2)) > 0.8:
                        duplicate = True
                        break
            
            if not duplicate:
                titles.add(title_hash)
                deduplicated.append(item)
        
        return deduplicated
    
    def _get_alpha_vantage_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Get news from Alpha Vantage API."""
        if not self.api_keys['alpha_vantage']:
            return []
        
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_keys['alpha_vantage']}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            news_items = []
            for item in data.get('feed', []):
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'published_at': item.get('time_published', ''),
                    'sentiment': item.get('overall_sentiment_score', 0)
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news for {ticker}: {e}")
            return []
    
    def _get_newsapi_news(self, query: str, days: int) -> List[Dict[str, Any]]:
        """Get news from NewsAPI."""
        if not self.api_keys['newsapi']:
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&language=en&from={from_date}&"
                f"sortBy=publishedAt&apiKey={self.api_keys['newsapi']}"
            )
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            news_items = []
            for item in data.get('articles', []):
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('description', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', {}).get('name', ''),
                    'published_at': item.get('publishedAt', ''),
                    'image_url': item.get('urlToImage', '')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news for {query}: {e}")
            return []
    
    def _get_fmp_news(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        """Get news from Financial Modeling Prep API."""
        if not self.api_keys['financial_modeling_prep']:
            return []
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=100&apikey={self.api_keys['financial_modeling_prep']}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            news_items = []
            for item in data:
                pub_date = datetime.strptime(item.get('publishedDate', ''), '%Y-%m-%d %H:%M:%S')
                if pub_date >= cutoff_date:
                    news_items.append({
                        'title': item.get('title', ''),
                        'summary': item.get('text', ''),
                        'url': item.get('url', ''),
                        'source': item.get('site', ''),
                        'published_at': item.get('publishedDate', ''),
                        'image_url': item.get('image', '')
                    })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching FMP news for {ticker}: {e}")
            return []
    
    def _get_fmp_market_news(self, days: int) -> List[Dict[str, Any]]:
        """Get general market news from Financial Modeling Prep API."""
        if not self.api_keys['financial_modeling_prep']:
            return []
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?limit=100&apikey={self.api_keys['financial_modeling_prep']}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            news_items = []
            for item in data:
                pub_date = datetime.strptime(item.get('publishedDate', ''), '%Y-%m-%d %H:%M:%S')
                if pub_date >= cutoff_date:
                    news_items.append({
                        'title': item.get('title', ''),
                        'summary': item.get('text', ''),
                        'url': item.get('url', ''),
                        'source': item.get('site', ''),
                        'published_at': item.get('publishedDate', ''),
                        'image_url': item.get('image', '')
                    })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching FMP market news: {e}")
            return []
    
    def _get_fmp_economic_calendar(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Get economic calendar from Financial Modeling Prep API."""
        if not self.api_keys['financial_modeling_prep']:
            return []
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={from_date}&to={to_date}&apikey={self.api_keys['financial_modeling_prep']}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            events = []
            for item in data:
                events.append({
                    'event': item.get('event', ''),
                    'date': item.get('date', ''),
                    'country': item.get('country', ''),
                    'impact': item.get('impact', ''),
                    'actual': item.get('actual', ''),
                    'previous': item.get('previous', ''),
                    'forecast': item.get('forecast', ''),
                    'unit': item.get('unit', '')
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return [] 