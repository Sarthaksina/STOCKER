"""Market data routes for STOCKER Pro API.

This module provides API endpoints for market data, including holdings, events, news, and sentiment.
"""
from fastapi import APIRouter, Query, Path
from typing import Dict, List, Optional, Any

from src.features.holdings import analyze_holdings
from src.features.events import get_corporate_events
from src.features.sentiment import get_news_sentiment
from src.components.news_agent import search_news
from src.core.config import Config

router = APIRouter(prefix="/market", tags=["market_data"])
config = Config()

@router.get("/holdings/{symbol}")
def api_holdings(symbol: str):
    """Get holdings data for a stock symbol."""
    return analyze_holdings(symbol, config)

@router.get("/events/{symbol}")
def api_events(symbol: str):
    """Fetch corporate events (dividends, splits) for a symbol."""
    return get_corporate_events(symbol)

@router.get("/sentiment/{symbol}")
def api_sentiment(symbol: str):
    """Fetch recent news sentiment for a symbol."""
    return get_news_sentiment(symbol, config)

@router.get("/news/{symbol}")
def api_news(
    symbol: str,
    max_articles: Optional[int] = Query(None, ge=1),
    summarize: bool = Query(True),
    sentiment: bool = Query(True)
):
    """Fetch recent news articles with optional summarization and sentiment analysis."""
    limit = max_articles or config.max_news_articles
    articles = search_news(symbol, limit, summarize, sentiment)
    return {"symbol": symbol, "articles": articles}
