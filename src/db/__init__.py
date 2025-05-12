"""
Database models and session management for STOCKER Pro.

This package provides database models and utilities for session management.
"""

from src.db.session import (
    get_db,
    get_db_session,
    get_mongodb_client,
    get_mongodb_db,
    get_collection,
    setup_database,
    close_db_connections
)

from src.db.models import (
    StockPrice,
    StockDataResponse,
    CompanyInfo,
    Portfolio,
    PredictionModel,
    PredictionRequest,
    PredictionResponse,
    TimeFrame,
    AssetType,
    ModelType
)
