"""
Command-line interface for STOCKER Pro.

This module provides a CLI for interacting with STOCKER Pro.
"""
import argparse
import sys
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import pandas as pd

from src.core.config import config
from src.core.logging import setup_logging, logger
from src.data.manager import DataManager
from src.services.portfolio import PortfolioService
from src.services.prediction import PredictionService
from src.services.training import TrainingService
from src.db.models import ModelType, PredictionHorizon, OptimizationMethod
from src.db.session import setup_database, close_db_connections


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="STOCKER Pro - Financial Market Intelligence CLI")
    
    # Add interactive mode option
    parser.add_argument("--interactive", "-i", action="store_true", 
                      help="Run in interactive mode")
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 'data' command
    data_parser = subparsers.add_parser("data", help="Data operations")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data command")
    
    # 'data get' command
    data_get_parser = data_subparsers.add_parser("get", help="Get stock data")
    data_get_parser.add_argument("symbol", help="Stock symbol")
    data_get_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    data_get_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    data_get_parser.add_argument("--interval", choices=["daily", "weekly", "monthly"], 
                                default="daily", help="Data interval")
    data_get_parser.add_argument("--output", help="Output file path (CSV)")
    
    # 'data company' command
    data_company_parser = data_subparsers.add_parser("company", help="Get company info")
    data_company_parser.add_argument("symbol", help="Stock symbol")
    
    # 'predict' command
    predict_parser = subparsers.add_parser("predict", help="Prediction operations")
    predict_subparsers = predict_parser.add_subparsers(dest="predict_command", help="Prediction command")
    
    # 'predict stock' command
    predict_stock_parser = predict_subparsers.add_parser("stock", help="Predict stock price")
    predict_stock_parser.add_argument("symbol", help="Stock symbol")
    predict_stock_parser.add_argument("--model-type", choices=[m.value for m in ModelType], 
                                    help="Model type")
    predict_stock_parser.add_argument("--model-id", help="Specific model ID")
    predict_stock_parser.add_argument("--horizon", choices=[h.value for h in PredictionHorizon], 
                                    default=PredictionHorizon.DAY_5.value, help="Prediction horizon")
    predict_stock_parser.add_argument("--confidence", action="store_true", 
                                    help="Include confidence intervals")
    
    # 'predict batch' command
    predict_batch_parser = predict_subparsers.add_parser("batch", help="Batch predict")
    predict_batch_parser.add_argument("symbols", nargs="+", help="Stock symbols")
    predict_batch_parser.add_argument("--model-type", choices=[m.value for m in ModelType], 
                                    help="Model type")
    predict_batch_parser.add_argument("--horizon", choices=[h.value for h in PredictionHorizon], 
                                    default=PredictionHorizon.DAY_5.value, help="Prediction horizon")
    predict_batch_parser.add_argument("--confidence", action="store_true", 
                                    help="Include confidence intervals")
    
    # 'predict models' command
    predict_models_parser = predict_subparsers.add_parser("models", help="List models")
    predict_models_parser.add_argument("--symbol", help="Filter by symbol")
    
    # 'portfolio' command
    portfolio_parser = subparsers.add_parser("portfolio", help="Portfolio operations")
    portfolio_subparsers = portfolio_parser.add_subparsers(dest="portfolio_command", help="Portfolio command")
    
    # 'portfolio optimize' command
    portfolio_optimize_parser = portfolio_subparsers.add_parser("optimize", help="Optimize portfolio")
    portfolio_optimize_parser.add_argument("symbols", nargs="+", help="Stock symbols")
    portfolio_optimize_parser.add_argument("--method", choices=[m.value for m in OptimizationMethod], 
                                        default=OptimizationMethod.EFFICIENT_FRONTIER.value, 
                                        help="Optimization method")
    portfolio_optimize_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    portfolio_optimize_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    portfolio_optimize_parser.add_argument("--risk-free-rate", type=float, default=0.02, 
                                        help="Risk-free rate")
    
    # 'train' command
    train_parser = subparsers.add_parser("train", help="Training operations")
    train_subparsers = train_parser.add_subparsers(dest="train_command", help="Training command")
    
    # 'train model' command
    train_model_parser = train_subparsers.add_parser("model", help="Train a model")
    train_model_parser.add_argument("symbol", help="Stock symbol")
    train_model_parser.add_argument("--model-type", choices=[m.value for m in ModelType], 
                                  default=ModelType.LSTM.value, help="Model type")
    train_model_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    train_model_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    train_model_parser.add_argument("--horizon", choices=[h.value for h in PredictionHorizon], 
                                  default=PredictionHorizon.DAY_5.value, help="Prediction horizon")
    
    # 'train list' command
    train_list_parser = train_subparsers.add_parser("list", help="List trained models")
    train_list_parser.add_argument("--symbol", help="Filter by symbol")
    train_list_parser.add_argument("--model-type", choices=[m.value for m in ModelType], 
                                 help="Filter by model type")
    
    return parser.parse_args()


def format_output(data, pretty=True):
    """Format data for output."""
    if pretty:
        return json.dumps(data, indent=2, default=str)
    else:
        return json.dumps(data, default=str)


def save_to_csv(data, path):
    """Save data to CSV file."""
    if isinstance(data, pd.DataFrame):
        data.to_csv(path)
    elif isinstance(data, List) and all(isinstance(item, Dict) for item in data):
        pd.DataFrame(data).to_csv(path, index=False)
    elif isinstance(data, Dict):
        pd.Series(data).to_csv(path, header=False)
    else:
        raise ValueError("Unsupported data type for CSV export")


def handle_data_get(args):
    """Handle 'data get' command."""
    data_manager = DataManager()
    
    try:
        stock_data = data_manager.get_stock_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval
        )
        
        if args.output:
            # Save to CSV
            if args.output.endswith(".csv"):
                stock_data.to_csv(args.output)
                print(f"Data saved to {args.output}")
            else:
                print("Output file must have .csv extension")
        else:
            # Print to stdout
            prices = [{
                "date": index.isoformat(),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume")
            } for index, row in stock_data.iterrows()]
            
            response = {
                "symbol": args.symbol,
                "interval": args.interval,
                "count": len(prices),
                "first_date": prices[0]["date"] if prices else None,
                "last_date": prices[-1]["date"] if prices else None,
                "prices": prices[:5] + ["..."] + prices[-5:] if len(prices) > 10 else prices
            }
            
            print(format_output(response))
            
    except Exception as e:
        logger.error(f"Error retrieving stock data: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def handle_data_company(args):
    """Handle 'data company' command."""
    data_manager = DataManager()
    
    try:
        company_info = data_manager.get_company_info(args.symbol)
        print(format_output(company_info))
        
    except Exception as e:
        logger.error(f"Error retrieving company info: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def handle_predict_stock(args):
    """Handle 'predict stock' command."""
    prediction_service = PredictionService()
    
    try:
        # Convert string args to enums
        model_type = ModelType(args.model_type) if args.model_type else None
        horizon = PredictionHorizon(args.horizon)
        
        from src.db.models import PredictionRequest
        
        request = PredictionRequest(
            symbol=args.symbol,
            model_type=model_type,
            model_id=args.model_id,
            prediction_horizon=horizon,
            include_confidence_intervals=args.confidence
        )
        
        result = prediction_service.predict(request)
        print(format_output(result))
        
    except Exception as e:
        logger.error(f"Error predicting stock: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def handle_predict_batch(args):
    """Handle 'predict batch' command."""
    prediction_service = PredictionService()
    
    try:
        # Convert string args to enums
        model_type = ModelType(args.model_type) if args.model_type else None
        horizon = PredictionHorizon(args.horizon)
        
        results = prediction_service.batch_predict(
            symbols=args.symbols,
            model_type=model_type,
            horizon=horizon,
            include_confidence=args.confidence
        )
        
        print(format_output(results))
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def handle_predict_models(args):
    """Handle 'predict models' command."""
    prediction_service = PredictionService()
    
    try:
        models = prediction_service.get_available_models(args.symbol)
        print(format_output(models))
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def handle_portfolio_optimize(args):
    """Handle 'portfolio optimize' command."""
    portfolio_service = PortfolioService()
    
    try:
        # Convert string args to enums
        optimization_method = OptimizationMethod(args.method)
        
        result = portfolio_service.optimize_portfolio(
            symbols=args.symbols,
            optimization_method=optimization_method,
            start_date=args.start_date,
            end_date=args.end_date,
            risk_free_rate=args.risk_free_rate
        )
        
        print(format_output(result))
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def handle_train_model(args):
    """Handle 'train model' command."""
    training_service = TrainingService()
    
    try:
        # Convert string args to enums
        model_type = ModelType(args.model_type)
        horizon = PredictionHorizon(args.horizon)
        
        result = training_service.train_model(
            symbol=args.symbol,
            model_type=model_type,
            start_date=args.start_date,
            end_date=args.end_date,
            prediction_horizon=horizon
        )
        
        print(format_output(result))
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def handle_train_list(args):
    """Handle 'train list' command."""
    training_service = TrainingService()
    
    try:
        # Convert string args to enums
        model_type = ModelType(args.model_type) if args.model_type else None
        
        models = training_service.list_models(args.symbol, model_type)
        print(format_output(models))
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def print_interactive_menu():
    """Print the interactive CLI menu."""
    print("\n==== STOCKER CLI ====")
    print("1. Portfolio Recommendation")
    print("2. Holdings Analysis")
    print("3. News & Sentiment")
    print("4. Earnings Call Summary")
    print("5. Peer Comparison")
    print("6. Event & Macro Event Detection")
    print("7. Exit")


def get_user_info():
    """Get user information for portfolio planning."""
    print("\nEnter your details for personalized planning:")
    age = int(input("Age: "))
    risk = input("Risk Appetite (conservative/moderate/aggressive): ").strip().lower()
    sip = float(input("Monthly SIP amount (0 if none): "))
    lumpsum = float(input("Lumpsum investment (0 if none): "))
    years = int(input("Investment horizon (years): "))
    return {
        "age": age,
        "risk_appetite": risk,
        "sip_amount": sip if sip > 0 else None,
        "lumpsum": lumpsum if lumpsum > 0 else None,
        "years": years
    }


def run_interactive_cli():
    """Run the interactive CLI."""
    from src.entity.config_entity import StockerConfig
    from src.features.agent import StockerAgent
    import pprint
    
    config = StockerConfig()
    agent = StockerAgent(config)
    
    while True:
        print_interactive_menu()
        choice = input("Choose an option: ").strip()
        
        if choice == "1":
            user_info = get_user_info()
            result = agent.answer("portfolio", user_info)
            pprint.pprint(result)
        elif choice == "2":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"holdings {symbol}")
            pprint.pprint(result)
        elif choice == "3":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"news {symbol}")
            pprint.pprint(result)
        elif choice == "4":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"concall {symbol}")
            pprint.pprint(result)
        elif choice == "5":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"peer {symbol}")
            pprint.pprint(result)
        elif choice == "6":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"event {symbol}")
            pprint.pprint(result)
        elif choice == "7":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Try again.")


# Create a Typer app for CLI commands
cli_app = None
try:
    import typer
    cli_app = typer.Typer(name="stocker", help="STOCKER Pro - Financial Market Intelligence CLI")
    
    # Define CLI commands using Typer
    @cli_app.command("data")
    def typer_data_command(
        symbol: str = typer.Argument(..., help="Stock symbol"),
        start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
        end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
        interval: str = typer.Option("daily", help="Data interval (daily, weekly, monthly)")
    ):
        """Get stock data."""
        class Args:
            pass
        args = Args()
        args.symbol = symbol
        args.start_date = start_date
        args.end_date = end_date
        args.interval = interval
        args.output = None
        handle_data_get(args)
    
    @cli_app.command("predict")
    def typer_predict_command(
        symbol: str = typer.Argument(..., help="Stock symbol"),
        model_type: str = typer.Option(None, help="Model type"),
        horizon: str = typer.Option("5d", help="Prediction horizon")
    ):
        """Predict stock price."""
        class Args:
            pass
        args = Args()
        args.symbol = symbol
        args.model_type = model_type
        args.horizon = horizon
        args.confidence = False
        args.model_id = None
        handle_predict_stock(args)
    
    @cli_app.command("interactive")
    def typer_interactive_command():
        """Run interactive CLI."""
        run_interactive_cli()
        
except ImportError:
    # Typer is not installed, we'll use argparse instead
    cli_app = None
    logger.warning("Typer is not installed. Using argparse for CLI instead.")

def main():
    """Main entry point for CLI."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Set up database
    setup_database()
    
    try:
        # Check for interactive mode
        if hasattr(args, 'interactive') and args.interactive:
            run_interactive_cli()
            return
        
        # Handle commands
        if args.command == "data":
            if args.data_command == "get":
                handle_data_get(args)
            elif args.data_command == "company":
                handle_data_company(args)
            else:
                print("Unknown data command")
                sys.exit(1)
                
        elif args.command == "predict":
            if args.predict_command == "stock":
                handle_predict_stock(args)
            elif args.predict_command == "batch":
                handle_predict_batch(args)
            elif args.predict_command == "models":
                handle_predict_models(args)
            else:
                print("Unknown predict command")
                sys.exit(1)
                
        elif args.command == "portfolio":
            if args.portfolio_command == "optimize":
                handle_portfolio_optimize(args)
            else:
                print("Unknown portfolio command")
                sys.exit(1)
                
        elif args.command == "train":
            if args.train_command == "model":
                handle_train_model(args)
            elif args.train_command == "list":
                handle_train_list(args)
            else:
                print("Unknown train command")
                sys.exit(1)
                
        else:
            print("Unknown command")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
        
    finally:
        # Clean up
        close_db_connections()


if __name__ == "__main__":
    main()