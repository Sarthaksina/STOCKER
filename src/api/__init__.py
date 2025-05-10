"""
API module for STOCKER Pro.

This module provides the REST API interface for the application.
"""

from src.api.server import app, get_app
from src.api.dependencies import get_db, get_current_user, get_data_manager

__all__ = [
    # Application
    'app',
    'get_app',
    
    # Dependencies
    'get_db',
    'get_current_user',
    'get_data_manager'
]
