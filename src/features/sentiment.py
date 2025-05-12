"""Sentiment analysis module for STOCKER Pro.

This module provides functions for analyzing sentiment from news, social media,
and other text sources related to financial markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import re
import random  # For placeholder implementation

from src.core.logging import logger


def get_news_sentiment(symbol: str, days: int = 7, sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get sentiment analysis for news articles related to a symbol.
    
    Args:
        symbol: Stock symbol
        days: Number of days to look back
        sources: List of news sources to include (None for all)
        
    Returns:
        Dictionary with sentiment analysis results
    """
    try:
        # This is a placeholder implementation
        # In a real implementation, this would fetch news and analyze sentiment
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate placeholder news data
        news_count = min(days * 3, 20)  # Simulate 3 articles per day, max 20
        news_items = []
        
        for i in range(news_count):
            # Generate random date within range
            days_ago = random.randint(0, days-1)
            article_date = end_date - timedelta(days=days_ago)
            
            # Generate random sentiment score (-1 to 1)
            sentiment_score = random.uniform(-1, 1)
            
            # Determine sentiment category
            if sentiment_score > 0.3:
                sentiment = 'positive'
            elif sentiment_score < -0.3:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Generate placeholder title
            if sentiment == 'positive':
                title = f"{symbol} Exceeds Expectations in Latest Quarter"
            elif sentiment == 'negative':
                title = f"{symbol} Faces Challenges Amid Market Uncertainty"
            else:
                title = f"{symbol} Reports Mixed Results in Recent Performance"
            
            # Add random source
            default_sources = ['Bloomberg', 'CNBC', 'Reuters', 'WSJ', 'MarketWatch']
            source = random.choice(sources if sources else default_sources)
            
            news_items.append({
                'date': article_date.strftime('%Y-%m-%d'),
                'time': f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
                'title': title,
                'source': source,
                'url': f"https://example.com/{source.lower()}/{symbol.lower()}-news-{i}",
                'sentiment_score': sentiment_score,
                'sentiment': sentiment
            })
        
        # Sort by date (newest first)
        news_items.sort(key=lambda x: x['date'] + ' ' + x['time'], reverse=True)
        
        # Calculate overall sentiment metrics
        sentiment_scores = [item['sentiment_score'] for item in news_items]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        sentiment_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
        
        # Count by sentiment category
        positive_count = sum(1 for item in news_items if item['sentiment'] == 'positive')
        negative_count = sum(1 for item in news_items if item['sentiment'] == 'negative')
        neutral_count = sum(1 for item in news_items if item['sentiment'] == 'neutral')
        
        # Calculate sentiment trend (last 3 days vs previous period)
        recent_days = 3
        if days > recent_days and len(news_items) > recent_days:
            recent_items = [item for item in news_items 
                           if (end_date - datetime.strptime(item['date'], '%Y-%m-%d')).days <= recent_days]
            earlier_items = [item for item in news_items 
                            if (end_date - datetime.strptime(item['date'], '%Y-%m-%d')).days > recent_days]
            
            recent_sentiment = np.mean([item['sentiment_score'] for item in recent_items]) if recent_items else 0
            earlier_sentiment = np.mean([item['sentiment_score'] for item in earlier_items]) if earlier_items else 0
            
            sentiment_trend = recent_sentiment - earlier_sentiment
        else:
            sentiment_trend = 0
        
        return {
            'symbol': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'news_count': len(news_items),
            'average_sentiment': avg_sentiment,
            'sentiment_std': sentiment_std,
            'sentiment_trend': sentiment_trend,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'news_items': news_items
        }
        
    except Exception as e:
        logger.error(f"Error getting news sentiment: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'news_count': 0,
            'news_items': []
        }


def get_social_sentiment(symbol: str, days: int = 7, platforms: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get sentiment analysis from social media platforms for a symbol.
    
    Args:
        symbol: Stock symbol
        days: Number of days to look back
        platforms: List of social platforms to include (None for all)
        
    Returns:
        Dictionary with social sentiment analysis results
    """
    try:
        # This is a placeholder implementation
        # In a real implementation, this would fetch social media data and analyze sentiment
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Default platforms
        default_platforms = ['Twitter', 'Reddit', 'StockTwits']
        platforms_to_use = platforms if platforms else default_platforms
        
        # Generate placeholder data for each platform
        platform_data = {}
        overall_sentiment = 0
        total_mentions = 0
        
        for platform in platforms_to_use:
            # Generate random metrics
            mentions = random.randint(50, 500)
            sentiment_score = random.uniform(-0.8, 0.8)
            
            # Generate daily data
            daily_data = []
            for i in range(days):
                day_date = start_date + timedelta(days=i)
                day_mentions = random.randint(5, mentions // days)
                day_sentiment = random.uniform(sentiment_score - 0.2, sentiment_score + 0.2)
                
                daily_data.append({
                    'date': day_date.strftime('%Y-%m-%d'),
                    'mentions': day_mentions,
                    'sentiment_score': day_sentiment
                })
            
            # Add platform data
            platform_data[platform] = {
                'mentions': mentions,
                'sentiment_score': sentiment_score,
                'bullish_percentage': (sentiment_score + 1) * 50,  # Convert -1 to 1 scale to 0-100%
                'daily_data': daily_data
            }
            
            # Update overall metrics
            overall_sentiment += sentiment_score * mentions
            total_mentions += mentions
        
        # Calculate overall sentiment
        if total_mentions > 0:
            overall_sentiment = overall_sentiment / total_mentions
        else:
            overall_sentiment = 0
        
        return {
            'symbol': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_mentions': total_mentions,
            'overall_sentiment': overall_sentiment,
            'bullish_percentage': (overall_sentiment + 1) * 50,  # Convert -1 to 1 scale to 0-100%
            'platforms': platform_data
        }
        
    except Exception as e:
        logger.error(f"Error getting social sentiment: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'total_mentions': 0,
            'platforms': {}
        }


def analyze_sentiment_impact(symbol: str, price_data: pd.DataFrame, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the impact of sentiment on price movements.
    
    Args:
        symbol: Stock symbol
        price_data: DataFrame with price history
        sentiment_data: Dictionary with sentiment data
        
    Returns:
        Dictionary with sentiment impact analysis
    """
    try:
        if price_data.empty or 'close' not in price_data.columns:
            logger.error("Invalid price data provided")
            return {}
            
        if not sentiment_data or 'news_items' not in sentiment_data:
            logger.error("Invalid sentiment data provided")
            return {}
        
        # Extract price data
        prices = price_data['close']
        returns = prices.pct_change().dropna()
        
        # Extract sentiment data
        news_items = sentiment_data.get('news_items', [])
        
        # Group news by date
        news_by_date = {}
        for item in news_items:
            date = item['date']
            if date not in news_by_date:
                news_by_date[date] = []
            news_by_date[date].append(item)
        
        # Calculate average sentiment by date
        sentiment_by_date = {}
        for date, items in news_by_date.items():
            sentiment_scores = [item['sentiment_score'] for item in items]
            sentiment_by_date[date] = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Convert to Series for analysis
        sentiment_series = pd.Series(sentiment_by_date)
        sentiment_series.index = pd.to_datetime(sentiment_series.index)
        
        # Align with price data
        aligned_data = pd.DataFrame()
        aligned_data['returns'] = returns
        
        # Add sentiment with 1-day lag (to analyze impact on next day's returns)
        sentiment_series_shifted = sentiment_series.shift(1)
        aligned_data['sentiment'] = sentiment_series_shifted
        
        # Drop rows with missing data
        aligned_data = aligned_data.dropna()
        
        # Calculate correlation
        correlation = aligned_data['returns'].corr(aligned_data['sentiment']) \
            if not aligned_data.empty else 0
        
        # Analyze returns following positive vs negative sentiment
        positive_sentiment = aligned_data[aligned_data['sentiment'] > 0.3]
        negative_sentiment = aligned_data[aligned_data['sentiment'] < -0.3]
        neutral_sentiment = aligned_data[(aligned_data['sentiment'] >= -0.3) & 
                                        (aligned_data['sentiment'] <= 0.3)]
        
        avg_return_after_positive = positive_sentiment['returns'].mean() if not positive_sentiment.empty else 0
        avg_return_after_negative = negative_sentiment['returns'].mean() if not negative_sentiment.empty else 0
        avg_return_after_neutral = neutral_sentiment['returns'].mean() if not neutral_sentiment.empty else 0
        
        # Calculate sentiment effectiveness
        # Positive sentiment should lead to positive returns, negative to negative
        correct_predictions = len(positive_sentiment[positive_sentiment['returns'] > 0]) + \
                            len(negative_sentiment[negative_sentiment['returns'] < 0])
        total_predictions = len(positive_sentiment) + len(negative_sentiment)
        
        sentiment_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        return {
            'symbol': symbol,
            'correlation': correlation,
            'avg_return_after_positive': avg_return_after_positive,
            'avg_return_after_negative': avg_return_after_negative,
            'avg_return_after_neutral': avg_return_after_neutral,
            'sentiment_accuracy': sentiment_accuracy,
            'data_points': len(aligned_data)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment impact: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'correlation': 0,
            'data_points': 0
        }


def extract_sentiment_keywords(text: str, max_keywords: int = 10) -> List[Dict[str, Any]]:
    """
    Extract key sentiment-driving keywords from text.
    
    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords with sentiment scores
    """
    try:
        # This is a placeholder implementation
        # In a real implementation, this would use NLP to extract keywords
        
        # Define some financial sentiment keywords
        positive_keywords = [
            'growth', 'profit', 'increase', 'beat', 'exceed', 'outperform', 
            'strong', 'positive', 'upgrade', 'bullish', 'opportunity'
        ]
        
        negative_keywords = [
            'decline', 'loss', 'decrease', 'miss', 'below', 'underperform',
            'weak', 'negative', 'downgrade', 'bearish', 'risk'
        ]
        
        neutral_keywords = [
            'report', 'announce', 'statement', 'quarter', 'year', 'market',
            'stock', 'share', 'price', 'trading', 'investor'
        ]
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Find keyword matches
        keywords = []
        
        for word in positive_keywords:
            if word in text_lower:
                count = len(re.findall(r'\b' + word + r'\b', text_lower))
                if count > 0:
                    keywords.append({
                        'keyword': word,
                        'count': count,
                        'sentiment': 'positive',
                        'sentiment_score': random.uniform(0.3, 0.9)  # Random positive score
                    })
        
        for word in negative_keywords:
            if word in text_lower:
                count = len(re.findall(r'\b' + word + r'\b', text_lower))
                if count > 0:
                    keywords.append({
                        'keyword': word,
                        'count': count,
                        'sentiment': 'negative',
                        'sentiment_score': random.uniform(-0.9, -0.3)  # Random negative score
                    })
        
        for word in neutral_keywords:
            if word in text_lower:
                count = len(re.findall(r'\b' + word + r'\b', text_lower))
                if count > 0:
                    keywords.append({
                        'keyword': word,
                        'count': count,
                        'sentiment': 'neutral',
                        'sentiment_score': random.uniform(-0.2, 0.2)  # Random neutral score
                    })
        
        # Sort by count and limit to max_keywords
        keywords.sort(key=lambda x: x['count'], reverse=True)
        keywords = keywords[:max_keywords]
        
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting sentiment keywords: {e}")
        return []
