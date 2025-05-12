"""
STOCKER Pro - Main Application

This is the main entry point for the STOCKER Pro application.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

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
    # Load environment variables from .env file
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    # Check if we're running in CLI mode
    # If the first argument is not an option (doesn't start with -), it's a CLI command
    import sys
    cli_mode = len(sys.argv) > 1 and not sys.argv[1].startswith('-')
    
    if cli_mode and cli_app is not None:
        # Run the Typer CLI directly
        configure_logging(log_level="INFO")
        logger.info("Starting STOCKER Pro in CLI mode")
        cli_app()
        return
    
    # Parse command line arguments for non-CLI modes
    args = parse_args()
    
    # Load configuration
    config_path = args.config if args.config else os.environ.get("STOCKER_CONFIG")
    config = get_config(config_path)
    
    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    configure_logging(log_level=log_level)
    
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
        try:
            logger.info(f"Creating dashboard with sample data")
            app = create_dashboard(
                app_title="STOCKER Pro Dashboard",
                theme="light",
                debug=args.debug
            )
            
            # Initialize dashboard with automatic updates
            from src.ui.dashboard import update_dashboard
            update_dashboard(app, interval_seconds=30)
            
            logger.info(f"Running dashboard on {args.host}:{args.port}")
            run_dashboard(
                app,
                host=args.host,
                port=args.port,
                debug=args.debug
            )
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.mode == "cli":
        # Run CLI application
        if cli_app is not None:
            # Use Typer CLI if available
            cli_app()
        else:
            # Fall back to argparse CLI
            from src.cli.commands import main as cli_main
            cli_main()

if __name__ == "__main__":
    main()
