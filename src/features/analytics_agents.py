import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.configuration.config import settings
# Import from refactored portfolio module
# Import from the portfolio facade module which maintains backward compatibility
from src.features.portfolio import (
    PortfolioManager, # Import the manager class
    get_portfolio_manager, # Keep if used directly, otherwise remove
    recommend_portfolio, self_assess_portfolio, advanced_rebalance_portfolio,
    get_portfolio_manager
)
# Import all metrics functions from the consolidated metrics file
from src.features.portfolio.portfolio_metrics_consolidated import (
    peer_compare, chart_performance, performance_analysis as portfolio_performance_analysis, 
    sharpe_ratio as portfolio_sharpe_ratio, valuation_metrics as portfolio_valuation_metrics, 
    alpha_beta as portfolio_alpha_beta, attribution_analysis as portfolio_attribution_analysis, 
    momentum_analysis as portfolio_momentum_analysis, sentiment_agg
)
# Import risk functions from portfolio_risk
from src.features.portfolio.portfolio_risk import (
    calculate_var, calculate_cvar, calculate_drawdown
)
from src.utils.helpers import top_n_recommender
from src.features.vector_search import add_vector_similarity

logger = logging.getLogger(__name__)

# --- Core Analytics Functions (Moved to portfolio modules) ---

# Functions sharpe_ratio, valuation_metrics, alpha_beta, attribution_analysis, momentum_analysis moved to portfolio_metrics.py
# Function exposure_analysis moved to portfolio_core.py (PortfolioManager class)

class AnalyticsAgents:
    """
    Orchestrates analytics and portfolio features.
    """
    def __init__(self):
        # Get portfolio manager instance
        self.pm = get_portfolio_manager()
        
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
        # Use PortfolioManager's method
        return self.pm.suggest_stocks(
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

    def risk_var(self, params: Dict[str, Any]) -> float:
        # Use imported calculate_var function
        returns_series = pd.Series(params.get('returns', []))
        return calculate_var(
            returns=returns_series,
            confidence_level=params.get('confidence', 0.95), # Default to 95%
            method=params.get('method', 'historical') # Allow specifying method
        )

    def risk_drawdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Use imported calculate_drawdown function
        returns_series = pd.Series(params.get('returns', [])) # Needs returns, not prices
        if 'prices' in params and not returns_series.any(): # Calculate returns if only prices given
             prices = params.get('prices', [])
             if len(prices) > 1:
                 returns_series = pd.Series(np.diff(prices) / prices[:-1])
        
        drawdown_info = calculate_drawdown(returns=returns_series)
        return {
            'max_drawdown': drawdown_info[1], 
            'average_drawdown': drawdown_info[2]
            # drawdown_series (drawdown_info[0]) is likely too large for direct return
        }

    def risk_sharpe(self, params: Dict[str, Any]) -> float:
        # Use the moved sharpe_ratio function
        # Note: rolling_sharpe is not defined here, using the basic one
        return portfolio_sharpe_ratio(
            returns=params.get('returns', []),
            risk_free_rate=params.get('risk_free_rate', 0.04)
        )
        # If rolling sharpe is needed, it should be implemented in portfolio_metrics or portfolio_risk

    def sentiment_agg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Use the imported sentiment_agg function
        return sentiment_agg(params)

    def holdings(self, params: Dict[str, Any]) -> Any:
        return analyze_holdings(
            symbol=params.get('symbol', ''),
            config=settings
        )

    def analysis_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Use the moved performance_analysis function
        return portfolio_performance_analysis(params.get('prices', []))

    def analysis_sharpe(self, params: Dict[str, Any]) -> float:
        # Use the moved sharpe_ratio function
        return portfolio_sharpe_ratio(params.get('returns', []), params.get('risk_free_rate', 0.04))

    def analysis_valuation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Use the moved valuation_metrics function
        return portfolio_valuation_metrics(params.get('stock_data', {}))

    def analysis_alpha_beta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Use the moved alpha_beta function
        return portfolio_alpha_beta(
            prices=params.get('prices', []),
            benchmark=params.get('benchmark', [])
        )

    def analysis_attribution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Use the moved attribution_analysis function
        return portfolio_attribution_analysis(
            portfolio=params.get('portfolio', {}),
            returns=params.get('returns', {})
        )

    def analysis_momentum(self, params: Dict[str, Any]) -> float:
        # Use the moved momentum_analysis function
        return portfolio_momentum_analysis(
            prices=params.get('prices', []),
            window=params.get('window', 6) # Allow specifying window
        )

    def vector_search(self, params: Dict[str, Any]) -> Any:
        return add_vector_similarity(
            df=params.get('df'),
            vector_col=params.get('vector_col', 'embedding'),
            query_vec=params.get('query_vec'),
            out_col=params.get('out_col', 'similarity')
        )

# Inline risk and holdings utilities (Sentiment moved)
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

# get_news_sentiment moved to financial_analysis.py as part of sentiment_agg

def analyze_holdings(symbol: str, config) -> Any:
    collection = get_collection("shareholding_patterns", config.mongodb_db_name)
    docs = list(collection.find({"symbol": symbol}, {"_id": 0}))
    return docs or {"message": f"No holdings data found for {symbol}"}
