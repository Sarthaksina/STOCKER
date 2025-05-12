"""STOCKER Pro Utilities Package

This package provides access to the utility functions defined in the core/utils.py module.
It imports and re-exports the functions to maintain backward compatibility.
"""

# Import all functions from the core utils module
from src.core.utils import (
    # File and Path Utilities
    get_project_root,
    create_directory_if_not_exists,
    load_json,
    save_json,
    load_pickle,
    save_pickle,
    
    # Decorators
    timer,
    retry,
    log_exceptions,
    cache_to_file,
    
    # Data Validation
    validate_dataframe,
    sanitize_input,
    
    # Formatting
    format_number,
    format_percentage,
    
    # Time Series
    get_rolling_windows,
    calculate_returns,
    
    # Caching
    save_to_cache,
    load_from_cache,
    clear_cache,
    get_cache_size,
    
    # Error Handling
    error_message_detail
)

# Also import from core.logging for backward compatibility
from src.core.logging import get_logger, setup_logging
