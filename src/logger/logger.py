"""
Centralized logging configuration for STOCKER Pro.
This module provides unified logging setup and utility functions.
"""
import os
import logging
import logging.config
import yaml
from typing import Optional, Dict, Any
from pathlib import Path
import datetime
import traceback
import sys

from src.configuration.config import DEFAULT_LOGGING_CONFIG, DEFAULT_LOG_FILE

def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> None:
    """
    Set up logging configuration from a YAML file.
    
    Args:
        config_path: Path to logging configuration file
        default_level: Default logging level if config file is not found
        log_dir: Directory to store log files
    """
    config_path = config_path or DEFAULT_LOGGING_CONFIG
    
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                
                # Update log file paths if log_dir is provided
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    for handler in config.get('handlers', {}).values():
                        if 'filename' in handler:
                            handler['filename'] = os.path.join(log_dir, os.path.basename(handler['filename']))
                
                logging.config.dictConfig(config)
                logging.info(f"Logging configured from {config_path}")
            except Exception as e:
                print(f"Error loading logging configuration: {e}")
                logging.basicConfig(level=default_level)
                logging.error(f"Error loading logging configuration: {e}")
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config file not found at {config_path}. Using basic configuration.")

def get_advanced_logger(
    name: str,
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = False,
    log_dir: Optional[str] = None,
    log_format: Optional[str] = None,
    propagate: bool = False
) -> logging.Logger:
    """
    Get a configured logger with advanced options.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory to store log files
        log_format: Custom log format
        propagate: Whether to propagate to parent loggers
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = propagate
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")
        else:
            log_dir = os.path.dirname(DEFAULT_LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_exception(logger: logging.Logger, e: Exception, message: str = "An error occurred") -> None:
    """
    Log an exception with traceback.
    
    Args:
        logger: Logger to use
        e: Exception to log
        message: Custom message to include
    """
    logger.error(f"{message}: {str(e)}")
    logger.debug(f"Exception details: {traceback.format_exc()}")

def get_performance_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Get a logger specifically configured for performance monitoring.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        
    Returns:
        Configured performance logger
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"performance_{name}.log")
    else:
        log_dir = os.path.dirname(DEFAULT_LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"performance_{name}.log")
    
    logger = logging.getLogger(f"performance.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_performance(logger: logging.Logger, operation: str, start_time: float, end_time: float) -> None:
    """
    Log performance metrics for an operation.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        start_time: Start time (from time.time())
        end_time: End time (from time.time())
    """
    duration = end_time - start_time
    logger.info(f"Performance: {operation} completed in {duration:.4f} seconds")

def log_dataframe_info(logger: logging.Logger, df, name: str) -> None:
    """
    Log information about a pandas DataFrame.
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name or description of the DataFrame
    """
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            logger.info(f"DataFrame {name}: shape={df.shape}, columns={df.columns.tolist()}")
            logger.debug(f"DataFrame {name} dtypes:\n{df.dtypes}")
            logger.debug(f"DataFrame {name} head:\n{df.head()}")
            logger.debug(f"DataFrame {name} null counts:\n{df.isnull().sum()}")
        else:
            logger.warning(f"{name} is not a pandas DataFrame")
    except ImportError:
        logger.warning("pandas not available for logging DataFrame info")

def log_model_metrics(logger: logging.Logger, metrics: Dict[str, Any], model_name: str) -> None:
    """
    Log model evaluation metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    logger.info(f"Metrics for {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")

# Usage: from logger import setup_logging; setup_logging(); logger = logging.getLogger(__name__)
