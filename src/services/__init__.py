"""
Services module for STOCKER Pro.

This module provides business logic services that coordinate between
data access, models, and other components.
"""

from src.services.auth import (
    AuthService,
    get_auth_service,
    verify_password,
    create_access_token
)

from src.services.portfolio import (
    PortfolioService,
    get_portfolio_service,
    create_portfolio,
    update_portfolio
)

from src.services.prediction import (
    PredictionService,
    get_prediction_service,
    predict_stock_price,
    predict_portfolio_performance
)

from src.services.training import (
    TrainingService,
    get_training_service,
    train_model,
    evaluate_model
)

__all__ = [
    # Auth services
    'AuthService',
    'get_auth_service',
    'verify_password',
    'create_access_token',
    
    # Portfolio services
    'PortfolioService',
    'get_portfolio_service',
    'create_portfolio',
    'update_portfolio',
    
    # Prediction services
    'PredictionService',
    'get_prediction_service',
    'predict_stock_price',
    'predict_portfolio_performance',
    
    # Training services
    'TrainingService',
    'get_training_service',
    'train_model',
    'evaluate_model'
]
