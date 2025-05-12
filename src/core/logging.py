"""
Advanced logging configuration for STOCKER Pro.

This module provides enhanced logging functionality with file and console output,
colored formatting, and contextual information.
"""
import logging
import os
import sys
from typing import Optional, Dict, Any, Union
from datetime import datetime
import traceback
from functools import wraps
import json


class ColoredFormatter(logging.Formatter):
    """Logging formatter with colored output for console."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[38;5;39m',    # Light blue
        'INFO': '\033[38;5;35m',     # Green
        'WARNING': '\033[38;5;214m', # Orange
        'ERROR': '\033[38;5;196m',   # Red
        'CRITICAL': '\033[48;5;196m\033[38;5;231m', # White on red background
        'RESET': '\033[0m'           # Reset to default
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors for console output."""
        # Save original levelname to restore it later
        original_levelname = record.levelname
        levelname = record.levelname
        
        # Apply color if available for this level
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return result


class JSONFormatter(logging.Formatter):
    """Logging formatter that outputs JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra attributes
        for key, value in record.__dict__.items():
            if key.startswith('_') and not key.startswith('__'):
                try:
                    # Try to serialize the value to ensure it's JSON-compatible
                    json.dumps({key[1:]: value})
                    log_data[key[1:]] = value
                except (TypeError, OverflowError):
                    # If it's not serializable, convert to string
                    log_data[key[1:]] = str(value)
        
        return json.dumps(log_data)


def get_advanced_logger(
    name: str,
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = False,
    log_dir: str = "logs",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    json_format: bool = False,
    capture_warnings: bool = True
) -> logging.Logger:
    """
    Create an advanced logger with console and/or file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        log_format: Log format string
        json_format: Whether to use JSON format for file logging
        capture_warnings: Whether to capture warnings
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Configure logging to console
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for console
        formatter = ColoredFormatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Configure logging to file
    if log_to_file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file path
        log_file = os.path.join(log_dir, f"{name}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Use JSON formatter if requested
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(log_format)
            
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Capture warnings
    if capture_warnings:
        logging.captureWarnings(True)
    
    return logger


def log_exceptions(logger: Optional[logging.Logger] = None):
    """
    Decorator to log exceptions raised by functions.
    
    Args:
        logger: Logger instance to use (or will create a default one)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger (passed or from function's module)
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {e}",
                    exc_info=True,
                    extra={
                        "_function": func.__name__,
                        "_args": str(args),
                        "_kwargs": str(kwargs),
                    }
                )
                raise
        return wrapper
    return decorator


def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger instance to use (or will create a default one)
        level: Logging level
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger (passed or from function's module)
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # Log function call
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            logger.log(level, f"Calling {func.__name__}({signature})")
            
            # Execute function and log result
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {e.__class__.__name__}: {e}")
                raise
        return wrapper
    return decorator


def add_context(logger: logging.Logger, **context):
    """
    Add context to logger for structured logging.
    
    Args:
        logger: Logger instance
        **context: Context key-value pairs
    """
    # Create filter to add context to log records
    class ContextFilter(logging.Filter):
        def filter(self, record):
            for key, value in context.items():
                setattr(record, f"_{key}", value)
            return True
    
    # Add filter to logger
    logger.addFilter(ContextFilter())
    
    return logger


def get_context_logger(name: str, **context):
    """
    Get a logger with context.
    
    Args:
        name: Logger name
        **context: Context key-value pairs
        
    Returns:
        Logger with context
    """
    logger = get_advanced_logger(name)
    return add_context(logger, **context)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    This is a simplified wrapper around get_advanced_logger for backward compatibility.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    return get_advanced_logger(name, level=level)


def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure global logging settings.
    
    Args:
        log_level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, or None to disable file logging
    """
    # Convert string level to logging level
    level = getattr(logging, log_level.upper())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]: 
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Capture warnings
    logging.captureWarnings(True)
    
    # Log configuration
    root_logger.info(f"Logging configured with level {log_level}")
    if log_file:
        root_logger.info(f"Logging to file: {log_file}")
    
    return root_logger


# Alias for backward compatibility
setup_logging = configure_logging

# Default logger instance for the module
logger = get_logger("stocker")