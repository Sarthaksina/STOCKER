"""
STOCKER Pro - Main Application

This is the main entry point for the STOCKER Pro application.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from src.core.config import get_config, StockerConfig
from src.core.logging import configure_logging, get_logger
from src.api.server import get_app
from src.ui.dashboard import create_dashboard, run_dashboard
from src.cli.commands import cli_app

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="STOCKER Pro Application")
    
    parser.add_argument(
        "--mode",
        choices=["api", "ui", "cli"],
        default="api",
        help="Application mode (api, ui, or cli)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address for API or UI server"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API or UI server"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config_path = args.config if args.config else os.environ.get("STOCKER_CONFIG")
    config = get_config(config_path)
    
    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    configure_logging(level=log_level)
    
    logger.info(f"Starting STOCKER Pro in {args.mode} mode")
    
    # Run application in the selected mode
    if args.mode == "api":
        # Run FastAPI application
        app = get_app(config)
        
        # Import here to avoid dependency unless needed
        import uvicorn
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=log_level.lower()
        )
        
    elif args.mode == "ui":
        # Run Dashboard UI
        app = create_dashboard(
            app_title="STOCKER Pro Dashboard",
            debug=args.debug
        )
        
        run_dashboard(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    elif args.mode == "cli":
        # Run CLI application
        cli_app()

if __name__ == "__main__":
    main()
