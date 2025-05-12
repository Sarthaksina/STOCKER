"""
API routes module for STOCKER Pro.

This module contains API route definitions for different endpoints.
"""

from fastapi import APIRouter

from src.api.routes.auth import router as auth_router
from src.api.routes.stocks import router as stocks_router
from src.api.routes.portfolio import router as portfolio_router
from src.api.routes.analysis import router as analysis_router
from src.api.routes.market_data import router as market_data_router
from src.api.routes.agent import router as agent_router

router = APIRouter()

router.include_router(auth_router)
router.include_router(stocks_router)
router.include_router(portfolio_router)
router.include_router(analysis_router)
router.include_router(market_data_router)
router.include_router(agent_router)

__all__ = [
    'auth_router',
    'stocks_router',
    'portfolio_router',
    'analysis_router',
    'market_data_router',
    'agent_router',
    'router'
]
