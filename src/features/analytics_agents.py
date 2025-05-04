import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import logging

from src.configuration.config import settings
from src.features.portfolio import recommend_portfolio, self_assess_portfolio, advanced_rebalance_portfolio, suggest_high_quality_stocks, peer_compare, top_n_recommender, chart_performance
from src.features.vector_search import add_vector_similarity
import feedparser
from src.constant.constants import GOOGLE_NEWS_RSS
from src.llm_utils import analyze_sentiment
from src.db import get_collection

logger = logging.getLogger(__name__)

# --- Core Analytics (inlined from analysis.py)
def performance_analysis(prices: List[float]) -> Dict[str, float]:
    """
    Returns total return, annualized return, and volatility from price series.
    """
    returns = np.diff(prices) / prices[:-1]
    total_return = (prices[-1] / prices[0]) - 1
    ann_return = (1 + total_return) ** (12 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = np.std(returns) * np.sqrt(12)
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "volatility": float(volatility)
    }

def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.04) -> float:
    """
    Returns the Sharpe ratio of returns series.
    """
    excess = np.array(returns) - risk_free_rate / 12
    return float(np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(12))

def valuation_metrics(stock_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns PE, PB, dividend yield, and market cap from fundamentals.
    """
    return {
        "pe": stock_data.get("pe", np.nan),
        "pb": stock_data.get("pb", np.nan),
        "div_yield": stock_data.get("div_yield", np.nan),
        "market_cap": stock_data.get("market_cap", np.nan)
    }

def alpha_beta(prices: List[float], benchmark: List[float]) -> Dict[str, float]:
    """
    Computes alpha and beta relative to benchmark.
    """
    returns = np.diff(prices) / prices[:-1]
    bench_ret = np.diff(benchmark) / benchmark[:-1]
    cov = np.cov(returns, bench_ret)[0][1]
    # Handle zero variance in benchmark returns: perfect correlation
    denom = np.var(bench_ret)
    if denom < 1e-8:
        beta = 1.0
        alpha = 0.0
    else:
        beta = cov / denom
        alpha = np.mean(returns) - beta * np.mean(bench_ret)
    return {"alpha": float(alpha), "beta": float(beta)}

def attribution_analysis(portfolio: Dict[str, float], returns: Dict[str, float]) -> Dict[str, float]:
    """
    Decomposes portfolio return into asset contributions.
    """
    contrib = {k: portfolio[k] * returns.get(k, 0) for k in portfolio}
    total = sum(contrib.values())
    return {k: v / total if total else 0 for k, v in contrib.items()}

def exposure_analysis(portfolio: Dict[str, float], sector_map: Dict[str, str], asset_class_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Checks sector and asset class exposures.
    """
    sector_exp, asset_exp = {}, {}
    for stock, weight in portfolio.items():
        sector = sector_map.get(stock, "Other")
        asset = asset_class_map.get(stock, "Other")
        sector_exp[sector] = sector_exp.get(sector, 0) + weight
        asset_exp[asset] = asset_exp.get(asset, 0) + weight
    over_exp = {k: v for k, v in sector_exp.items() if v > 0.4}
    under_exp = {k: v for k, v in sector_exp.items() if v < 0.1}
    return {"sector_exposure": sector_exp, "asset_exposure": asset_exp, "over_exposed": over_exp, "under_exposed": under_exp}

def momentum_analysis(prices: List[float]) -> float:
    """
    Simple momentum: last price / 6-month average.
    """
    if len(prices) < 6:
        return np.nan
    return float(prices[-1] / np.mean(prices[-6:]))

class AnalyticsAgents:
    """
    Orchestrates analytics and portfolio features.
    """
    def __init__(self):
        self.tasks: Dict[str, Any] = {
            'portfolio': self.portfolio,
            'self_assess': self.self_assess,
            'rebalance': self.rebalance,
            'suggest_hq': self.suggest_hq,
            'peer': self.peer,
            'top_peers': self.top_peers,
            'charts': self.charts,
            'risk_var': self.risk_var,
            'risk_drawdown': self.risk_drawdown,
            'risk_sharpe': self.risk_sharpe,
            'sentiment_agg': self.sentiment_agg,
            'holdings': self.holdings,
            'analysis_performance': self.analysis_performance,
            'analysis_sharpe': self.analysis_sharpe,
            'analysis_valuation': self.analysis_valuation,
            'analysis_alpha_beta': self.analysis_alpha_beta,
            'analysis_attribution': self.analysis_attribution,
            'analysis_momentum': self.analysis_momentum,
            'vector_search': self.vector_search
        }

    def portfolio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return recommend_portfolio(params, settings)

    def self_assess(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self_assess_portfolio(
            user_portfolio=params.get('user_portfolio'),
            stock_data_map=params.get('stock_data_map', {}),
            sector_map=params.get('sector_map', {}),
            asset_class_map=params.get('asset_class_map', {}),
            price_history_map=params.get('price_history_map', {})
        )

    def rebalance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return advanced_rebalance_portfolio(
            user_portfolio=params.get('user_portfolio'),
            stock_data_map=params.get('stock_data_map', {}),
            sector_map=params.get('sector_map', {}),
            asset_class_map=params.get('asset_class_map', {}),
            price_history_map=params.get('price_history_map', {}),
            benchmark_prices=params.get('benchmark_prices', []),
            config=settings
        )

    def suggest_hq(self, params: Dict[str, Any]) -> List[Any]:
        return suggest_high_quality_stocks(
            config=settings,
            market_data=params.get('market_data', {}),
            filters=params.get('filters', {})
        )

    def peer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return peer_compare(
            price_history_map=params.get('price_history_map', {}),
            target=params.get('target', ''),
            n=params.get('n', 5)
        )

    def top_peers(self, params: Dict[str, Any]) -> List[str]:
        df = params.get('df')
        if df is None:
            return []
        return top_n_recommender(df, params.get('score_col', 'score'), params.get('n', 5))

    def charts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return chart_performance(
            dates=params.get('dates', []),
            price_history_map=params.get('price_history_map', {})
        )

    def risk_var(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return var_historical(
            returns=params.get('returns', []),
            confidence=params.get('confidence', 0.05)
        )

    def risk_drawdown(self, params: Dict[str, Any]) -> Any:
        return max_drawdown(prices=params.get('prices', []))

    def risk_sharpe(self, params: Dict[str, Any]) -> Any:
        return rolling_sharpe(
            returns=params.get('returns', []),
            window=params.get('window', 12),
            risk_free_rate=params.get('risk_free_rate', 0.04)
        )

    def sentiment_agg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return get_news_sentiment(
            symbol=params.get('symbol', ''),
            config=settings
        )

    def holdings(self, params: Dict[str, Any]) -> Any:
        return analyze_holdings(
            symbol=params.get('symbol', ''),
            config=settings
        )

    def analysis_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return performance_analysis(params.get('prices', []))

    def analysis_sharpe(self, params: Dict[str, Any]) -> float:
        return sharpe_ratio(params.get('returns', []), params.get('risk_free_rate', 0.04))

    def analysis_valuation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return valuation_metrics(params.get('stock_data', {}))

    def analysis_alpha_beta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return alpha_beta(
            prices=params.get('prices', []),
            benchmark=params.get('benchmark', [])
        )

    def analysis_attribution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return attribution_analysis(
            portfolio=params.get('portfolio', {}),
            returns=params.get('returns', {})
        )

    def analysis_momentum(self, params: Dict[str, Any]) -> float:
        return momentum_analysis(params.get('prices', []))

    def vector_search(self, params: Dict[str, Any]) -> Any:
        return add_vector_similarity(
            df=params.get('df'),
            vector_col=params.get('vector_col', 'embedding'),
            query_vec=params.get('query_vec'),
            out_col=params.get('out_col', 'similarity')
        )

# Inline risk, sentiment, and holdings utilities
def var_historical(returns: List[float], confidence: float = 0.05) -> Dict[str, float]:
    arr = np.array(returns)
    if arr.size == 0:
        return {"var": None}
    # Historical VaR: worst loss (no interpolation)
    return {"var": float(-np.min(arr))}

def max_drawdown(prices: List[float]) -> Dict[str, float]:
    arr = np.array(prices)
    if arr.size == 0:
        return {"max_drawdown": None}
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return {"max_drawdown": float(drawdown.min())}

def rolling_sharpe(returns: List[float], window: int = 12, risk_free_rate: float = 0.04) -> Dict[str, Any]:
    arr = np.array(returns)
    result: List[Optional[float]] = []
    rf_per_period = risk_free_rate / window
    for i in range(len(arr)):
        if i + 1 < window:
            result.append(None)
        else:
            window_ret = arr[i + 1 - window: i + 1]
            excess = window_ret - rf_per_period
            mean_ex = np.mean(excess)
            std_ex = np.std(excess) + 1e-8
            result.append(float(mean_ex / std_ex * np.sqrt(window)))
    return {"rolling_sharpe": result}

def get_news_sentiment(symbol: str, config) -> Dict[str, Any]:
    url = GOOGLE_NEWS_RSS.format(query=symbol)
    feed = feedparser.parse(url)
    entries = feed.entries[: config.max_news_articles]
    results: List[Dict[str, Any]] = []
    for entry in entries:
        title = entry.get("title", "")
        label = analyze_sentiment(title)
        results.append({"title": title, "sentiment": label})
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for item in results:
        counts[item["sentiment"]] = counts.get(item["sentiment"], 0) + 1
    return {"symbol": symbol, "counts": counts, "details": results}

def analyze_holdings(symbol: str, config) -> Any:
    collection = get_collection("shareholding_patterns", config.mongodb_db_name)
    docs = list(collection.find({"symbol": symbol}, {"_id": 0}))
    return docs or {"message": f"No holdings data found for {symbol}"}
