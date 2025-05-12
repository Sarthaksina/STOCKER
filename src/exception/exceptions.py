# src/utils/exceptions.py
"""
Centralized custom exceptions and error utilities for the STOCKER project.
Provides rich error messages with file name, line number, and traceback info for debugging and logging.
"""
import sys
import logging
import traceback
from typing import Optional, List, Dict, Any

# === Error Message Utility ===
def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.
    :param error: The exception that occurred.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    exc_tb = error_detail.exc_info()[2]
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"
    line_number = exc_tb.tb_lineno if exc_tb else -1
    error_message = (
        f"Error occurred in python script: [{file_name}] at line number [{line_number}]: {str(error)}"
    )
    logging.error(error_message)
    return error_message

# === Custom Exceptions ===
class StockerBaseException(Exception):
    """Base exception for all custom STOCKER errors."""
    pass

class DatabaseConnectionError(StockerBaseException):
    """Custom exception for database connection errors."""
    pass

class DataIngestionError(StockerBaseException):
    """Custom exception for data ingestion errors."""
    pass

class PipelineExecutionError(StockerBaseException):
    """Custom exception for pipeline execution errors."""
    pass

class ModelInferenceError(StockerBaseException):
    """Custom exception for ML/DL model inference errors."""
    pass

class AlphaVantageAPIException(StockerBaseException):
    """
    Exception raised for Alpha Vantage API errors.
    
    This could be due to:
    - API rate limiting
    - Invalid API parameters
    - Authentication issues
    - Service unavailability
    - Parsing errors in API responses
    """
    pass

# === Data Access Exceptions ===
class DataAccessError(StockerBaseException):
    """Base exception for all data access errors."""
    
    def __init__(self, message: str, source: Optional[str] = None):
        self.source = source
        self.message = message
        super().__init__(f"{message} (Source: {source})" if source else message)


class DataNotFoundError(DataAccessError):
    """Exception raised when requested data is not found."""
    
    def __init__(self, message: str, symbol: Optional[str] = None, source: Optional[str] = None):
        self.symbol = symbol
        symbol_info = f" for symbol: {symbol}" if symbol else ""
        super().__init__(f"{message}{symbol_info}", source)


class DataSourceConnectionError(DataAccessError):
    """Exception raised when connection to a data source fails."""
    
    def __init__(self, source: str, details: Optional[str] = None):
        message = f"Failed to connect to {source}"
        if details:
            message += f": {details}"
        super().__init__(message, source)


class DataSourceAuthError(DataAccessError):
    """Exception raised when authentication with a data source fails."""
    
    def __init__(self, source: str, details: Optional[str] = None):
        message = f"Authentication failed for {source}"
        if details:
            message += f": {details}"
        super().__init__(message, source)


class DataSourceRateLimitError(DataAccessError):
    """Exception raised when a data source rate limit is exceeded."""
    
    def __init__(self, source: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {source}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, source)


class InvalidDataError(DataAccessError):
    """Exception raised when received data is invalid or corrupted."""
    
    def __init__(self, message: str, source: Optional[str] = None, data_sample: Optional[Any] = None):
        self.data_sample = data_sample
        super().__init__(message, source)

# Add more custom exceptions as your project grows.
