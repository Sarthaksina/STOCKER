"""Analysis routes for STOCKER Pro API.

This module provides API endpoints for financial analysis, risk assessment, and peer comparison.
"""
from fastapi import APIRouter, Query, Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from src.features.analysis import (
    performance_analysis, 
    sharpe_ratio, 
    valuation_metrics, 
    alpha_beta, 
    attribution_analysis, 
    momentum_analysis
)
from src.features.risk import var_historical, max_drawdown, rolling_sharpe
from src.features.peer_comparison import peer_compare
from src.features.charts import chart_performance

router = APIRouter(prefix="/analysis", tags=["analysis"])

# --- Analysis Models ---
class PriceSeries(BaseModel):
    prices: List[float]

class ReturnSeries(BaseModel):
    returns: List[float]
    risk_free_rate: Optional[float] = 0.04

class ValuationRequest(BaseModel):
    stock_data: Dict[str, float]

class AlphaBetaRequest(BaseModel):
    prices: List[float]
    benchmark: List[float]

class AttributionRequest(BaseModel):
    portfolio: Dict[str, float]
    returns: Dict[str, float]

class MomentumRequest(BaseModel):
    prices: List[float]

# --- Risk Models ---
class RiskVarRequest(BaseModel):
    returns: List[float]
    confidence: Optional[float] = 0.05

class DrawdownRequest(BaseModel):
    prices: List[float]

class RollingSharpeRequest(BaseModel):
    returns: List[float]
    window: Optional[int] = 12
    risk_free_rate: Optional[float] = 0.04

# --- Peer Comparison Models ---
class PeerCompareRequest(BaseModel):
    price_history_map: Dict[str, List[float]]
    target: str
    n: Optional[int] = 5

# --- Charts Models ---
class ChartRequest(BaseModel):
    dates: List[str]
    price_history_map: Dict[str, List[float]]

# --- Analysis Endpoints ---
@router.post("/performance")
def api_analysis_performance(req: PriceSeries):
    """Analyze performance of a price series."""
    return performance_analysis(req.prices)

@router.post("/sharpe")
def api_analysis_sharpe(req: ReturnSeries):
    """Calculate Sharpe ratio for a return series."""
    return {"sharpe_ratio": sharpe_ratio(req.returns, req.risk_free_rate)}

@router.post("/valuation")
def api_analysis_valuation(req: ValuationRequest):
    """Calculate valuation metrics for a stock."""
    return valuation_metrics(req.stock_data)

@router.post("/alpha-beta")
def api_analysis_alpha_beta(req: AlphaBetaRequest):
    """Calculate alpha and beta for a stock relative to a benchmark."""
    return alpha_beta(req.prices, req.benchmark)

@router.post("/attribution")
def api_analysis_attribution(req: AttributionRequest):
    """Perform attribution analysis on a portfolio."""
    return attribution_analysis(req.portfolio, req.returns)

@router.post("/momentum")
def api_analysis_momentum(req: MomentumRequest):
    """Calculate momentum for a price series."""
    return {"momentum": momentum_analysis(req.prices)}

# --- Peer Comparison Endpoint ---
@router.post("/peer/compare")
def api_peer_compare(req: PeerCompareRequest):
    """Compare a target symbol to peers by return correlation."""
    return peer_compare(req.price_history_map, req.target, req.n)

# --- Charts Endpoints ---
@router.post("/charts/performance")
def api_charts_performance(req: ChartRequest):
    """Compute quarterly and yearly returns for price history."""
    return chart_performance(req.dates, req.price_history_map)

# --- Risk Endpoints ---
@router.post("/risk/var")
def api_risk_var(req: RiskVarRequest):
    """Calculate Value at Risk (VaR) for a return series."""
    return var_historical(req.returns, req.confidence)

@router.post("/risk/drawdown")
def api_risk_drawdown(req: DrawdownRequest):
    """Calculate maximum drawdown for a price series."""
    return max_drawdown(req.prices)

@router.post("/risk/rolling-sharpe")
def api_risk_rolling_sharpe(req: RollingSharpeRequest):
    """Calculate rolling Sharpe ratio for a return series."""
    return rolling_sharpe(req.returns, req.window, req.risk_free_rate)
