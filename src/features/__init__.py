"""
Feature engineering and analysis for STOCKER Pro.

This package provides components for feature engineering, technical indicators,
analytics and portfolio optimization.
"""

# Feature engineering
from src.features.engineering import (
    FeatureEngineer,
    generate_features,
    create_timeframe_features,
    create_lag_features,
    create_rolling_features
)

# Technical indicators
from src.features.indicators import (
    calculate_technical_indicators,
    calculate_macd,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_volatility,
    calculate_trend_indicators
)

# Analytics
from src.features.analytics import (
    analyze_seasonality,
    analyze_trend,
    detect_outliers,
    detect_anomalies,
    calculate_correlation_matrix
)

# Portfolio
from src.features.portfolio import (
    PortfolioFacade,
    PortfolioManager,
    PortfolioOptimizer,
    EfficientFrontier,
    calculate_portfolio_statistics,
    optimize_portfolio
)
