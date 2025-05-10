"""
API routes module for STOCKER Pro.

This module contains API route definitions for different endpoints.
"""

from src.api.routes.auth import router as auth_router
from src.api.routes.stocks import router as stocks_router

routers = [
    auth_router,
    stocks_router
]

__all__ = [
    'auth_router',
    'stocks_router',
    'routers'
]
