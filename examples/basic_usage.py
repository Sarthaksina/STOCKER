"""
Basic usage example for STOCKER Pro.

This script demonstrates the core functionality of STOCKER Pro, including
data retrieval, feature engineering, and visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.core.config import config
from src.core.logging import setup_logging, logger
from src.data.manager import DataManager
from src.features.engineering import FeatureEngineering

# Set up logging
setup_logging()
logger.info("Starting STOCKER Pro basic example")

# Initialize data manager
data_manager = DataManager()

# Define parameters
symbol = "AAPL"
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

try:
    # Fetch stock data
    stock_data = data_manager.get_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="daily"
    )
    
    logger.info(f"Retrieved {len(stock_data)} data points for {symbol}")
    
    # Display basic information
    print(f"\nStock data for {symbol}:")
    print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
    print(f"Number of records: {len(stock_data)}")
    print("\nLast 5 records:")
    print(stock_data.tail())
    
    # Generate features
    feature_eng = FeatureEngineering(stock_data)
    features_df = feature_eng.add_price_features()
    features_df = feature_eng.add_moving_averages(windows=[20, 50, 200])
    features_df = feature_eng.add_technical_indicators()
    
    logger.info(f"Generated {len(features_df.columns)} features")
    
    # Display generated features
    print(f"\nGenerated features ({len(features_df.columns)} total):")
    print(f"Features: {', '.join(features_df.columns[:10])}...")
    
    # Plot stock data with indicators
    plt.figure(figsize=(12, 8))
    
    # Price and moving averages
    plt.subplot(2, 1, 1)
    plt.plot(features_df.index, features_df['close'], label='Close Price')
    plt.plot(features_df.index, features_df['sma_20'], label='SMA 20')
    plt.plot(features_df.index, features_df['sma_50'], label='SMA 50')
    plt.plot(features_df.index, features_df['sma_200'], label='SMA 200')
    plt.title(f'{symbol} Price and Moving Averages')
    plt.legend()
    plt.grid(True)
    
    # RSI indicator
    plt.subplot(2, 1, 2)
    plt.plot(features_df.index, features_df['rsi'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.title('Relative Strength Index (RSI)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_analysis.png")
    logger.info(f"Saved analysis chart to {symbol}_analysis.png")
    
    # Get company information
    company_info = data_manager.get_company_info(symbol)
    
    print("\nCompany Information:")
    for key, value in company_info.items():
        if key not in ['symbol', 'type', 'source', 'ingestion_date']:
            print(f"{key}: {value}")

except Exception as e:
    logger.error(f"Error in example script: {e}")
    raise

logger.info("STOCKER Pro basic example completed successfully") 