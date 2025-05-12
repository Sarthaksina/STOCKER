"""Portfolio routes for STOCKER Pro API.

This module provides API endpoints for portfolio management, analysis, and optimization.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from src.features.portfolio import (
    recommend_portfolio, 
    self_assess_portfolio, 
    advanced_rebalance_portfolio, 
    suggest_high_quality_stocks, 
    get_portfolio_manager
)
from src.core.config import Config

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
config = Config()

class UserInfo(BaseModel):
    age: Optional[int] = None
    risk_appetite: Optional[str] = None
    sip_amount: Optional[float] = None
    lumpsum: Optional[float] = None
    years: Optional[int] = None
    user_portfolio: Optional[Dict[str, float]] = None
    stock_data_map: Optional[Dict[str, dict]] = None
    sector_map: Optional[Dict[str, str]] = None
    asset_class_map: Optional[Dict[str, str]] = None
    price_history_map: Optional[Dict[str, list]] = None
    benchmark_prices: Optional[List[float]] = None

@router.post("/recommend")
def api_recommend_portfolio(user_info: UserInfo):
    """Recommend a portfolio based on user information."""
    return recommend_portfolio(user_info.dict(), config)

@router.post("/rebalance")
def api_rebalance_portfolio(user_info: UserInfo):
    """Rebalance an existing portfolio."""
    if not user_info.user_portfolio:
        return {"error": "user_portfolio required"}
    return advanced_rebalance_portfolio(
        user_info.user_portfolio,
        user_info.stock_data_map or {},
        user_info.sector_map or {},
        user_info.asset_class_map or {},
        user_info.price_history_map or {},
        user_info.benchmark_prices or [],
        config
    )

@router.post("/assess")
def api_self_assess_portfolio(user_info: UserInfo):
    """Assess an existing portfolio."""
    if not user_info.user_portfolio:
        return {"error": "user_portfolio required"}
    return self_assess_portfolio(
        user_info.user_portfolio,
        user_info.stock_data_map or {},
        user_info.sector_map or {},
        user_info.asset_class_map or {},
        user_info.price_history_map or {}
    )

@router.get("/high-quality-stocks")
def api_high_quality_stocks():
    """Get a list of high-quality stocks."""
    # For demo, expects config.symbols, dummy data
    return suggest_high_quality_stocks(config, {}, {})
