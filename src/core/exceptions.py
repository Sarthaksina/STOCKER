"""
Centralized exception handling for STOCKER Pro.

This module defines all custom exceptions used throughout the application.
"""
from typing import Any, Dict, Optional
import traceback


class StockerError(Exception):
    """Base exception class for all STOCKER Pro errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize StockerError.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        self.message = message
        self.code = code or "STOCKER_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary.
        
        Returns:
            Dictionary representation of the exception
        """
        result = {
            "error": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.code}: {self.message} - {self.details}"
        return f"{self.code}: {self.message}"


class ConfigurationError(StockerError):
    """Error raised when there's a configuration issue."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize ConfigurationError."""
        super().__init__(message, "CONFIGURATION_ERROR", details)


class DataSourceError(StockerError):
    """Error raised when there's an issue with a data source."""
    
    def __init__(self, message: str, source: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize DataSourceError.
        
        Args:
            message: Error message
            source: Data source name
            details: Additional error details
        """
        details = details or {}
        details["source"] = source
        super().__init__(message, "DATA_SOURCE_ERROR", details)


class DatabaseError(StockerError):
    """Error raised when there's a database issue."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize DatabaseError."""
        super().__init__(message, "DATABASE_ERROR", details)


class DataValidationError(StockerError):
    """Error raised when data validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize DataValidationError."""
        super().__init__(message, "DATA_VALIDATION_ERROR", details)


class ModelTrainingError(StockerError):
    """Error raised when model training fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize ModelTrainingError."""
        super().__init__(message, "MODEL_TRAINING_ERROR", details)


class ModelInferenceError(StockerError):
    """Error raised when model inference fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize ModelInferenceError."""
        super().__init__(message, "MODEL_INFERENCE_ERROR", details)


class ModelLoadingError(StockerError):
    """Error raised when model loading fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize ModelLoadingError."""
        super().__init__(message, "MODEL_LOADING_ERROR", details)


class FeatureEngineeringError(StockerError):
    """Error raised when feature engineering fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize FeatureEngineeringError."""
        super().__init__(message, "FEATURE_ENGINEERING_ERROR", details)


class StockerPredictionError(StockerError):
    """Error raised when prediction pipeline fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize StockerPredictionError."""
        super().__init__(message, "PREDICTION_ERROR", details)


class APIError(StockerError):
    """Error raised when there's an API issue."""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        """
        Initialize APIError.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        details = details or {}
        details["status_code"] = status_code
        super().__init__(message, "API_ERROR", details)


class AuthenticationError(StockerError):
    """Error raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        """Initialize AuthenticationError."""
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(StockerError):
    """Error raised when authorization fails."""
    
    def __init__(self, message: str = "Not authorized", details: Optional[Dict[str, Any]] = None):
        """Initialize AuthorizationError."""
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class PortfolioError(StockerError):
    """Error raised when there's an issue with portfolio operations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize PortfolioError."""
        super().__init__(message, "PORTFOLIO_ERROR", details)


class PortfolioAnalysisError(StockerError):
    """Error raised when there's an issue with portfolio analysis."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize PortfolioAnalysisError."""
        super().__init__(message, "PORTFOLIO_ANALYSIS_ERROR", details)


class PortfolioOptimizationError(StockerError):
    """Error raised when there's an issue with portfolio optimization."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize PortfolioOptimizationError."""
        super().__init__(message, "PORTFOLIO_OPTIMIZATION_ERROR", details)


class DataAccessError(StockerError):
    """Error raised when there's an issue accessing data."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize DataAccessError."""
        super().__init__(message, "DATA_ACCESS_ERROR", details)


class DataNotFoundError(StockerError):
    """Error raised when requested data is not found."""
    
    def __init__(self, message: str, entity_id: Optional[str] = None, source: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize DataNotFoundError.
        
        Args:
            message: Error message
            entity_id: ID of the entity that was not found (e.g., symbol, model_id)
            source: Source where the entity was not found (e.g., 'mongodb', 'alpha_vantage')
            details: Additional error details
        """
        details = details or {}
        if entity_id:
            details["entity_id"] = entity_id
        if source:
            details["source"] = source
        super().__init__(message, "DATA_NOT_FOUND_ERROR", details)


class DataSourceRateLimitError(StockerError):
    """Error raised when a data source rate limit is exceeded."""
    
    def __init__(self, message: str, source: str, details: Optional[Dict[str, Any]] = None):
        """Initialize DataSourceRateLimitError."""
        details = details or {}
        details["source"] = source
        super().__init__(message, "DATA_SOURCE_RATE_LIMIT_ERROR", details)


class InvalidDataError(StockerError):
    """Error raised when data is invalid or malformed."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize InvalidDataError."""
        super().__init__(message, "INVALID_DATA_ERROR", details)


class DatabaseConnectionError(StockerError):
    """Error raised when there's an issue connecting to a database."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize DatabaseConnectionError."""
        super().__init__(message, "DATABASE_CONNECTION_ERROR", details)


class DataIngestionError(StockerError):
    """Error raised when there's an issue during data ingestion."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize DataIngestionError."""
        super().__init__(message, "DATA_INGESTION_ERROR", details)


def format_exception(exc: Exception) -> Dict[str, Any]:
    """
    Format an exception for logging or API response.
    
    Args:
        exc: Exception to format
        
    Returns:
        Dictionary representation of the exception
    """
    if isinstance(exc, StockerError):
        return exc.to_dict()
    
    return {
        "error": exc.__class__.__name__,
        "message": str(exc),
        "details": {
            "traceback": traceback.format_exc()
        }
    }


# Alias for backward compatibility
StockerBaseException = StockerError