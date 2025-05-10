"""
Command-line interface module for STOCKER Pro.

This module provides a command-line interface for interacting with the application.
"""

from src.cli.commands import (
    cli_app,
    train_command,
    predict_command,
    data_command,
    evaluate_command,
    portfolio_command
)

__all__ = [
    'cli_app',
    'train_command',
    'predict_command',
    'data_command',
    'evaluate_command',
    'portfolio_command'
]
