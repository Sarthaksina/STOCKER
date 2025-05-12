"""
Command-line interface module for STOCKER Pro.

This module provides a command-line interface for interacting with the application.
"""

from src.cli.commands import (
    cli_app,
    main,
    parse_args,
    handle_data_get,
    handle_data_company,
    handle_predict_stock,
    handle_predict_batch,
    handle_predict_models,
    handle_portfolio_optimize,
    handle_train_model,
    handle_train_list,
    run_interactive_cli
)

__all__ = [
    'cli_app',
    'main',
    'parse_args',
    'handle_data_get',
    'handle_data_company',
    'handle_predict_stock',
    'handle_predict_batch',
    'handle_predict_models',
    'handle_portfolio_optimize',
    'handle_train_model',
    'handle_train_list',
    'run_interactive_cli'
]
