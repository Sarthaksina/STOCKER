"""
Example script demonstrating the ensemble model for stock price prediction.

This script shows how to:
1. Load stock price data
2. Configure and train LSTM, XGBoost, and LightGBM models
3. Create an ensemble model combining all three
4. Make predictions and evaluate performance
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import yfinance as yf
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom modules
from src.ml import LSTMModel, XGBoostModel, LightGBMModel, EnsembleModel
from src.ml.evaluation import evaluate_financial_model, plot_financial_evaluation

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with stock data
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Add some basic features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Rolling_Vol_10'] = data['Returns'].rolling(window=10).std()
    data['Rolling_Vol_30'] = data['Returns'].rolling(window=30).std()
    
    # Drop rows with NaN values
    data = data.dropna()
    
    logger.info(f"Fetched {len(data)} data points")
    return data

def prepare_data(data, sequence_length=10, target_column='Close', train_ratio=0.8):
    """
    Prepare data for training and testing.
    
    Args:
        data: DataFrame with stock data
        sequence_length: Length of input sequences
        target_column: Target column to predict
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler)
    """
    # Get target values
    target_values = data[target_column].values
    
    # Calculate train/test split point
    split_idx = int(len(data) * train_ratio)
    
    # Split into training and test sets
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    logger.info(f"Training data: {len(train_data)} points, Test data: {len(test_data)} points")
    
    return train_data, test_data

def main():
    # Set parameters
    ticker = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)  # Get ~3 years of data
    sequence_length = 20
    prediction_horizon = 1  # Predict 1 day ahead
    
    # Create output directory
    output_dir = os.path.join('models', 'ensemble_example')
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch and prepare data
    data = fetch_stock_data(ticker, start_date, end_date)
    train_data, test_data = prepare_data(data, sequence_length=sequence_length)
    
    # Extract close prices for training
    train_prices = train_data['Close'].values
    test_prices = test_data['Close'].values
    
    # Configure models
    lstm_config = {
        "input_dim": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "sequence_length": sequence_length,
        "epochs": 50,
        "learning_rate": 0.001
    }
    
    xgb_config = {
        "objective": "reg:squarederror",
        "learning_rate": 0.01,
        "max_depth": 6,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "sequence_length": sequence_length,
        "prediction_length": prediction_horizon
    }
    
    lgb_config = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 300,
        "sequence_length": sequence_length,
        "prediction_length": prediction_horizon
    }
    
    ensemble_config = {
        "ensemble_method": "weighted_avg",
        "weights": {"lstm_model": 0.4, "xgb_model": 0.3, "lgb_model": 0.3}
    }
    
    # Initialize models
    lstm_model = LSTMModel(name="lstm_model", config=lstm_config)
    xgb_model = XGBoostModel(name="xgb_model", config=xgb_config)
    lgb_model = LightGBMModel(name="lgb_model", config=lgb_config)
    
    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_history = lstm_model.fit(train_prices, train_prices)
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    xgb_history = xgb_model.fit(train_prices, train_prices)
    
    # Train LightGBM model
    logger.info("Training LightGBM model...")
    lgb_history = lgb_model.fit(train_prices, train_prices)
    
    # Create and train ensemble model
    logger.info("Creating ensemble model...")
    ensemble_model = EnsembleModel(name="stock_ensemble", config=ensemble_config)
    ensemble_model.add_model(lstm_model)
    ensemble_model.add_model(xgb_model)
    ensemble_model.add_model(lgb_model)
    
    # Make predictions on test data
    logger.info("Making predictions...")
    lstm_preds = lstm_model.predict(test_prices)
    xgb_preds = xgb_model.predict(test_prices)
    lgb_preds = lgb_model.predict(test_prices)
    ensemble_preds = ensemble_model.predict(test_prices)
    
    # Evaluate models
    logger.info("Evaluating models...")
    lstm_metrics = evaluate_financial_model(test_prices, lstm_preds)
    xgb_metrics = evaluate_financial_model(test_prices, xgb_preds)
    lgb_metrics = evaluate_financial_model(test_prices, lgb_preds)
    ensemble_metrics = evaluate_financial_model(test_prices, ensemble_preds)
    
    # Print key metrics
    print("\nModel Evaluation Metrics:")
    print("-" * 50)
    print(f"LSTM - RMSE: {lstm_metrics['rmse']:.2f}, Dir Acc: {lstm_metrics['directional_accuracy']:.2f}")
    print(f"XGBoost - RMSE: {xgb_metrics['rmse']:.2f}, Dir Acc: {xgb_metrics['directional_accuracy']:.2f}")
    print(f"LightGBM - RMSE: {lgb_metrics['rmse']:.2f}, Dir Acc: {lgb_metrics['directional_accuracy']:.2f}")
    print(f"Ensemble - RMSE: {ensemble_metrics['rmse']:.2f}, Dir Acc: {ensemble_metrics['directional_accuracy']:.2f}")
    
    # Plot results
    logger.info("Plotting results...")
    
    # Convert index to datetime for plotting
    test_dates = test_data.index
    
    # Plot each model's predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, test_prices, label='Actual', linewidth=2)
    plt.plot(test_dates, lstm_preds, label='LSTM', linestyle='--')
    plt.plot(test_dates, xgb_preds, label='XGBoost', linestyle='--')
    plt.plot(test_dates, lgb_preds, label='LightGBM', linestyle='--')
    plt.plot(test_dates, ensemble_preds, label='Ensemble', linewidth=2)
    plt.title(f'{ticker} Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_predictions.png'))
    
    # Plot detailed evaluation of ensemble model
    fig, axes = plot_financial_evaluation(
        test_prices, 
        ensemble_preds, 
        dates=test_dates,
        title=f"Ensemble Model: {ticker}"
    )
    plt.savefig(os.path.join(output_dir, 'ensemble_evaluation.png'))
    
    # Save models
    logger.info("Saving models...")
    lstm_model.save(os.path.join(output_dir, 'lstm'))
    xgb_model.save(os.path.join(output_dir, 'xgboost'))
    lgb_model.save(os.path.join(output_dir, 'lightgbm'))
    ensemble_model.save(os.path.join(output_dir, 'ensemble'))
    
    logger.info("Example completed successfully!")
    
if __name__ == "__main__":
    main() 