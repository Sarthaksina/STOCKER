"""
Portfolio optimization and exposure analysis for STOCKER.
- Mean-variance optimization, exposure checks, etc.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def mean_variance_portfolio(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """
    Computes mean-variance optimal weights (no constraints).
    """
    cov = returns.cov()
    mean = returns.mean()
    inv_cov = np.linalg.pinv(cov)
    weights = inv_cov @ mean
    weights = weights / np.sum(weights)
    weights_list = weights.tolist()
    return {"weights": weights_list, "assets": list(returns.columns)}

def exposure_analysis(portfolio: Dict[str, float], sector_map: Dict[str, str], asset_class_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Checks for over/under-exposure to sectors and asset classes.
    """
    sector_exp = {}
    asset_exp = {}
    for stock, weight in portfolio.items():
        sector = sector_map.get(stock, "Other")
        asset = asset_class_map.get(stock, "Other")
        sector_exp[sector] = sector_exp.get(sector, 0) + weight
        asset_exp[asset] = asset_exp.get(asset, 0) + weight
    over_exp = {k: v for k, v in sector_exp.items() if v > 0.4}
    under_exp = {k: v for k, v in sector_exp.items() if v < 0.1}
    return {"sector_exposure": sector_exp, "asset_exposure": asset_exp, "over_exposed": over_exp, "under_exposed": under_exp}

# Add missing portfolio functions required by API
def recommend_portfolio(user_info: Dict[str, Any], config) -> Dict[str, Any]:
    """
    Recommends an optimal portfolio using mean-variance optimization.
    Expects 'price_history_map' in user_info: {symbol: [prices,...]}.
    """
    price_map = user_info.get("price_history_map")
    if not price_map:
        return {"error": "price_history_map is required for portfolio recommendation"}
    df_prices = pd.DataFrame(price_map)
    df_prices = df_prices.dropna(axis=0)
    if df_prices.empty:
        return {"error": "No valid price data provided"}
    # Compute returns and perform mean-variance optimization
    returns = df_prices.pct_change().dropna()
    optimization = mean_variance_portfolio(returns)
    return optimization

def self_assess_portfolio(user_portfolio: Dict[str, float],
                          stock_data_map: Dict[str, Any],
                          sector_map: Dict[str, str],
                          asset_class_map: Dict[str, str],
                          price_history_map: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for portfolio self-assessment."""
    return {"message": "self_assess_portfolio not yet implemented"}

def advanced_rebalance_portfolio(user_portfolio: Dict[str, float],
                                 stock_data_map: Dict[str, Any],
                                 sector_map: Dict[str, str],
                                 asset_class_map: Dict[str, str],
                                 price_history_map: Dict[str, Any],
                                 benchmark_prices: List[Any],
                                 config) -> Dict[str, Any]:
    """Placeholder for advanced portfolio rebalancing."""
    return {"message": "advanced_rebalance_portfolio not yet implemented"}

def suggest_high_quality_stocks(config, market_data: Dict[str, Any], filters: Dict[str, Any]) -> List[Any]:
    """Placeholder for high-quality stock suggestions."""
    return []

# --- Peer Comparison (merged from peer_comparison.py)
def top_n_recommender(df: pd.DataFrame, score_col: str = "score", n: int = 5) -> List[str]:
    """
    Returns top-N items by score.
    """
    if score_col in df.columns:
        return df.sort_values(score_col, ascending=False).head(n).index.tolist()
    return []

def peer_compare(price_history_map: Dict[str, List[float]], target: str, n: int = 5) -> Dict[str, Any]:
    """
    Compare the target symbol to peers by return correlation.
    Returns top-n peers with correlation values.
    """
    df = pd.DataFrame(price_history_map).dropna()
    if target not in df.columns:
        return {"error": f"Target '{target}' not in price history map"}
    returns = df.pct_change().dropna()
    target_ret = returns[target]
    corrs = returns.corrwith(target_ret).drop(target)
    top = corrs.nlargest(n)
    return {
        "target": target,
        "peers": [{"symbol": sym, "correlation": float(corr)} for sym, corr in top.items()]
    }

# --- Charting (merged from charts.py)
def chart_performance(dates: List[str], price_history_map: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Compute quarterly and yearly returns for given price series.
    - dates: list of date strings (ISO format).
    - price_history_map: symbol -> list of prices matching dates.
    Returns dict with 'quarterly' and 'yearly' mappings.
    """
    df = pd.DataFrame(price_history_map)
    df.index = pd.to_datetime(dates)
    df = df.sort_index()
    # Quarterly returns
    first_q = df.resample('Q').first()
    last_q = df.resample('Q').last()
    q_ret = (last_q / first_q - 1).to_dict('index')
    # Yearly returns
    first_y = df.resample('A').first()
    last_y = df.resample('A').last()
    y_ret = (last_y / first_y - 1).to_dict('index')
    # Serialize dates
    quarterly = {date.strftime('%Y-%m-%d'): returns for date, returns in q_ret.items()}
    yearly = {date.strftime('%Y-%m-%d'): returns for date, returns in y_ret.items()}
    return {'quarterly': quarterly, 'yearly': yearly}
