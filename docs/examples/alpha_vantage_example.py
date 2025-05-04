"""
Example script demonstrating the Alpha Vantage API client integration in STOCKER Pro.

This example shows how to:
1. Initialize the Alpha Vantage client
2. Fetch company overview data
3. Fetch time series data
4. Fetch financial statements
5. Handle errors and rate limiting
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configuration import Config
from src.data_access.alpha_vantage_client import AlphaVantageClient
from src.exception.exceptions import AlphaVantageAPIException
from src.utils.cache_utils import clear_cache, get_cache_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def display_company_overview(client, symbol):
    """Display company overview data."""
    try:
        logger.info(f"Fetching company overview for {symbol}...")
        overview = client.get_company_overview(symbol)
        
        # Print basic company info
        print(f"\n===== {overview.get('Name', symbol)} ({symbol}) =====")
        print(f"Sector: {overview.get('Sector', 'N/A')}")
        print(f"Industry: {overview.get('Industry', 'N/A')}")
        print(f"Exchange: {overview.get('Exchange', 'N/A')}")
        print(f"Country: {overview.get('Country', 'N/A')}")
        print(f"Market Cap: ${int(float(overview.get('MarketCapitalization', 0)) / 1_000_000):.2f}M")
        
        # Print key financial metrics
        print("\n----- Key Metrics -----")
        print(f"P/E Ratio: {overview.get('PERatio', 'N/A')}")
        print(f"EPS (TTM): {overview.get('EPS', 'N/A')}")
        print(f"Dividend Yield: {overview.get('DividendYield', 'N/A')}")
        print(f"52-Week High: {overview.get('52WeekHigh', 'N/A')}")
        print(f"52-Week Low: {overview.get('52WeekLow', 'N/A')}")
        print(f"50-Day MA: {overview.get('50DayMovingAverage', 'N/A')}")
        print(f"200-Day MA: {overview.get('200DayMovingAverage', 'N/A')}")
        
        return overview
    except AlphaVantageAPIException as e:
        logger.error(f"Failed to fetch company overview: {e}")
        return None

def display_income_statement(client, symbol):
    """Display income statement data."""
    try:
        logger.info(f"Fetching income statement for {symbol}...")
        income_data = client.get_income_statement(symbol)
        
        if 'annualReports' in income_data and len(income_data['annualReports']) > 0:
            latest_report = income_data['annualReports'][0]
            print("\n----- Latest Annual Income Statement -----")
            print(f"Fiscal Year: {latest_report.get('fiscalDateEnding', 'N/A')}")
            print(f"Total Revenue: ${int(float(latest_report.get('totalRevenue', 0)) / 1_000_000):.2f}M")
            print(f"Cost of Revenue: ${int(float(latest_report.get('costOfRevenue', 0)) / 1_000_000):.2f}M")
            print(f"Gross Profit: ${int(float(latest_report.get('grossProfit', 0)) / 1_000_000):.2f}M")
            print(f"Operating Income: ${int(float(latest_report.get('operatingIncome', 0)) / 1_000_000):.2f}M")
            print(f"Net Income: ${int(float(latest_report.get('netIncome', 0)) / 1_000_000):.2f}M")
            print(f"EPS: ${latest_report.get('reportedEPS', 'N/A')}")
        
        return income_data
    except AlphaVantageAPIException as e:
        logger.error(f"Failed to fetch income statement: {e}")
        return None

def visualize_stock_data(client, symbol):
    """Visualize stock price data."""
    try:
        logger.info(f"Fetching daily stock data for {symbol}...")
        stock_data = client.get_time_series_daily(symbol, outputsize="compact")
        
        if not stock_data.empty:
            # Create a plot of closing prices
            plt.figure(figsize=(12, 6))
            plt.plot(stock_data.index, stock_data['close'])
            plt.title(f"{symbol} Stock Price (Last {len(stock_data)} Trading Days)")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.grid(True)
            plt.tight_layout()
            
            # Check if 'plots' directory exists, if not create it
            plots_dir = Path(__file__).parent / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Save the plot
            plot_path = plots_dir / f"{symbol}_stock_price.png"
            plt.savefig(plot_path)
            logger.info(f"Plot saved to {plot_path}")
            
            # Calculate some statistics
            print("\n----- Stock Price Statistics -----")
            print(f"Current Price: ${stock_data['close'].iloc[-1]:.2f}")
            print(f"100-Day Change: {(stock_data['close'].iloc[-1] / stock_data['close'].iloc[0] - 1) * 100:.2f}%")
            print(f"Average Volume: {stock_data['volume'].mean():.0f}")
            print(f"Average Daily Range: ${(stock_data['high'] - stock_data['low']).mean():.2f}")
        
        return stock_data
    except AlphaVantageAPIException as e:
        logger.error(f"Failed to fetch stock data: {e}")
        return None

def main():
    """Main function demonstrating Alpha Vantage client usage."""
    # Initialize configuration and client
    config = Config()
    api_key = config.get_alpha_vantage_api_key()
    
    if not api_key:
        logger.error("No Alpha Vantage API key found. Please set ALPHA_VANTAGE_API_KEY in environment or .env file")
        print("\nTo get an API key, visit: https://www.alphavantage.co/support/#api-key")
        return
    
    # Initialize client with cache directory
    cache_dir = Path(__file__).parent.parent.parent / "cache" / "alpha_vantage"
    client = AlphaVantageClient(config, cache_dir=str(cache_dir))
    
    # Display cache information
    cache_size = get_cache_size(cache_dir)
    logger.info(f"Cache size: {cache_size / 1024:.2f} KB")
    
    # Get symbols from command line or use defaults
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["MSFT", "AAPL", "GOOGL"]
    
    for symbol in symbols:
        try:
            # Display company overview
            overview = display_company_overview(client, symbol)
            
            # Display income statement
            income_data = display_income_statement(client, symbol)
            
            # Visualize stock data
            stock_data = visualize_stock_data(client, symbol)
            
            print("\n" + "=" * 60 + "\n")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Show cache information after processing
    new_cache_size = get_cache_size(cache_dir)
    logger.info(f"New cache size: {new_cache_size / 1024:.2f} KB")
    logger.info(f"Added to cache: {(new_cache_size - cache_size) / 1024:.2f} KB")

if __name__ == "__main__":
    main() 