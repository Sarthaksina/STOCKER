"""
Core functionality module for STOCKER Pro.

This module contains essential core functionality like configuration,
logging, error handling, and general utilities.
"""

from src.core.config import StockerConfig, get_config
from src.core.exceptions import (
    StockerError,
    ConfigurationError,
    DataValidationError,
    ModelTrainingError,
    ModelInferenceError,
    ModelLoadingError,
    FeatureEngineeringError,
    StockerPredictionError,
    PortfolioError
)
from src.core.logging import get_logger, configure_logging
from src.core.utils import (
    get_project_root,
    timer,
    retry,
    validate_dataframe,
    sanitize_input,
    create_directory_if_not_exists
)

__all__ = [
    # Configuration
    'StockerConfig',
    'get_config',
    
    # Exceptions
    'StockerError',
    'ConfigurationError',
    'DataValidationError',
    'ModelTrainingError',
    'ModelInferenceError',
    'ModelLoadingError',
    'FeatureEngineeringError',
    'StockerPredictionError',
    'PortfolioError',
    
    # Logging
    'get_logger',
    'configure_logging',
    
    # Utilities
    'get_project_root',
    'timer',
    'retry',
    'validate_dataframe',
    'sanitize_input',
    'create_directory_if_not_exists'
]
