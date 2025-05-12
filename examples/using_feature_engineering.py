"""Example script demonstrating how to use the consolidated feature engineering module.

This script shows how to use the FeatureEngineer class from the consolidated
feature engineering module to create various features for financial time series data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import FeatureEngineer, remove_outliers, normalize_data
from src.core.exceptions import FeatureEngineeringError


def load_sample_data():
    """Load sample stock data or generate synthetic data if not available."""
    try:
        # Try to load sample data from CSV if available
        data_path = Path(__file__).parent / 'data' / 'sample_stock_data.csv'
        if data_path.exists():
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Could not load sample data: {e}")
    
    # Generate synthetic data if sample data is not available
    print("Generating synthetic stock data...")
    dates = pd.date_range(start='2022-01-01', periods=252, freq='B')  # Business days for a year
    
    # Create a trending market with some volatility
    close = 100 + np.cumsum(np.random.normal(0.05, 1, 252))  # Slight upward drift
    high = close + np.random.uniform(0, 2, 252)
    low = close - np.random.uniform(0, 2, 252)
    open_price = close - np.random.uniform(-1, 1, 252)
    volume = np.random.uniform(1000000, 5000000, 252)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def demonstrate_lag_features(fe, data):
    """Demonstrate creating lag features."""
    print("\n=== Creating Lag Features ===")
    
    # Create lag features for close prices
    result = fe.create_lag_features(['close'], [1, 5, 10])
    
    # Show the first few rows with lag features
    print("\nFirst few rows with lag features:")
    lag_cols = [col for col in result.columns if 'lag' in col]
    print(result[['close'] + lag_cols].head(12))
    
    # Explain the lag features
    print("\nLag features explanation:")
    print("- close_lag_1: The close price from the previous day")
    print("- close_lag_5: The close price from 5 days ago")
    print("- close_lag_10: The close price from 10 days ago")
    
    return result


def demonstrate_rolling_features(fe, data):
    """Demonstrate creating rolling window features."""
    print("\n=== Creating Rolling Window Features ===")
    
    # Create rolling features for close prices
    result = fe.create_rolling_features(['close'], [5, 20], ['mean', 'std'])
    
    # Show the first few rows with rolling features
    print("\nFirst few rows with rolling features:")
    rolling_cols = [col for col in result.columns if 'rolling' in col]
    print(result[['close'] + rolling_cols].head(22))
    
    # Explain the rolling features
    print("\nRolling features explanation:")
    print("- close_rolling_mean_5: 5-day moving average of close prices")
    print("- close_rolling_std_5: 5-day standard deviation of close prices")
    print("- close_rolling_mean_20: 20-day moving average of close prices")
    print("- close_rolling_std_20: 20-day standard deviation of close prices")
    
    return result


def demonstrate_return_features(fe, data):
    """Demonstrate creating return features."""
    print("\n=== Creating Return Features ===")
    
    # Create return features for close prices
    result = fe.create_return_features(['close'], [1, 5, 20])
    
    # Show the first few rows with return features
    print("\nFirst few rows with return features:")
    return_cols = [col for col in result.columns if 'return' in col]
    print(result[['close'] + return_cols].head(22))
    
    # Explain the return features
    print("\nReturn features explanation:")
    print("- close_return_1: 1-day percentage return")
    print("- close_return_5: 5-day percentage return")
    print("- close_return_20: 20-day percentage return")
    
    return result


def demonstrate_date_features(fe, data):
    """Demonstrate creating date-based features."""
    print("\n=== Creating Date Features ===")
    
    # Create date features
    result = fe.create_date_features()
    
    # Show the first few rows with date features
    print("\nFirst few rows with date features:")
    date_cols = ['day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'is_month_start', 'is_month_end']
    print(result[date_cols].head())
    
    # Explain the date features
    print("\nDate features explanation:")
    print("- day_of_week: Day of the week (0=Monday, 6=Sunday)")
    print("- day_of_month: Day of the month")
    print("- month: Month of the year")
    print("- quarter: Quarter of the year")
    print("- year: Year")
    print("- is_month_start: Whether the date is the start of a month")
    print("- is_month_end: Whether the date is the end of a month")
    
    return result


def demonstrate_outlier_removal(data):
    """Demonstrate outlier removal."""
    print("\n=== Removing Outliers ===")
    
    # Create a DataFrame with outliers
    df_with_outliers = data.copy()
    df_with_outliers.loc[df_with_outliers.index[50], 'close'] = data['close'].mean() + data['close'].std() * 5  # Add an outlier
    df_with_outliers.loc[df_with_outliers.index[100], 'close'] = data['close'].mean() - data['close'].std() * 5  # Add another outlier
    
    # Show the data with outliers
    print("\nData with outliers:")
    print(f"Close price range: {df_with_outliers['close'].min():.2f} to {df_with_outliers['close'].max():.2f}")
    print(f"Close price mean: {df_with_outliers['close'].mean():.2f}")
    print(f"Close price std: {df_with_outliers['close'].std():.2f}")
    
    # Remove outliers using the Z-score method
    result = remove_outliers(df_with_outliers, method='zscore', threshold=3.0)
    
    # Show the data after outlier removal
    print("\nData after outlier removal:")
    print(f"Close price range: {result['close'].min():.2f} to {result['close'].max():.2f}")
    print(f"Close price mean: {result['close'].mean():.2f}")
    print(f"Close price std: {result['close'].std():.2f}")
    
    return result


def demonstrate_data_normalization(data):
    """Demonstrate data normalization."""
    print("\n=== Normalizing Data ===")
    
    # Show the original data statistics
    print("\nOriginal data statistics:")
    print(data[['open', 'high', 'low', 'close', 'volume']].describe())
    
    # Normalize the data
    result = normalize_data(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Show the normalized data statistics
    print("\nNormalized data statistics:")
    print(result[['open', 'high', 'low', 'close', 'volume']].describe())
    
    return result


def demonstrate_all_features(fe, data):
    """Demonstrate creating all features at once."""
    print("\n=== Creating All Features ===")
    
    # Create all features
    result = fe.create_all_features()
    
    # Show the number of features created
    original_cols = set(data.columns)
    feature_cols = set(result.columns) - original_cols
    print(f"\nCreated {len(feature_cols)} new features")
    
    # Group features by type
    lag_cols = [col for col in feature_cols if 'lag' in col]
    rolling_cols = [col for col in feature_cols if 'rolling' in col]
    return_cols = [col for col in feature_cols if 'return' in col]
    date_cols = ['day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'is_month_start', 'is_month_end']
    date_cols = [col for col in date_cols if col in feature_cols]
    
    print(f"Lag features: {len(lag_cols)}")
    print(f"Rolling features: {len(rolling_cols)}")
    print(f"Return features: {len(return_cols)}")
    print(f"Date features: {len(date_cols)}")
    
    return result


def plot_features(data, features_data):
    """Plot some of the created features for visualization."""
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Close price and moving averages
    axs[0].plot(data.index, data['close'], label='Close Price')
    if 'close_rolling_mean_5' in features_data.columns and 'close_rolling_mean_20' in features_data.columns:
        axs[0].plot(features_data.index, features_data['close_rolling_mean_5'], 'r--', label='5-day MA')
        axs[0].plot(features_data.index, features_data['close_rolling_mean_20'], 'g--', label='20-day MA')
    axs[0].set_title('Close Price and Moving Averages')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Returns
    if 'close_return_1' in features_data.columns and 'close_return_5' in features_data.columns:
        axs[1].plot(features_data.index, features_data['close_return_1'], label='1-day Return')
        axs[1].plot(features_data.index, features_data['close_return_5'], 'r--', label='5-day Return')
        axs[1].set_title('Returns')
        axs[1].legend()
        axs[1].grid(True)
    
    # Plot 3: Volatility (standard deviation)
    if 'close_rolling_std_20' in features_data.columns:
        axs[2].plot(features_data.index, features_data['close_rolling_std_20'], label='20-day Volatility')
        axs[2].set_title('Volatility (20-day Standard Deviation)')
        axs[2].legend()
        axs[2].grid(True)
    
    plt.tight_layout()
    
    # Create directory for saving plots if it doesn't exist
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Save the plot
    plt.savefig(output_dir / 'feature_engineering.png')
    print(f"\nPlot saved to {output_dir / 'feature_engineering.png'}")
    
    # Show the plot if running interactively
    plt.show()


def main():
    """Main function to demonstrate the feature engineering module."""
    print("STOCKER Feature Engineering Example")
    print("===================================\n")
    
    # Load sample data
    data = load_sample_data()
    print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Columns: {list(data.columns)}")
    
    # Create a FeatureEngineer instance
    fe = FeatureEngineer(data)
    
    # Demonstrate lag features
    lag_data = demonstrate_lag_features(fe, data)
    
    # Demonstrate rolling features
    rolling_data = demonstrate_rolling_features(fe, data)
    
    # Demonstrate return features
    return_data = demonstrate_return_features(fe, data)
    
    # Demonstrate date features
    date_data = demonstrate_date_features(fe, data)
    
    # Demonstrate outlier removal
    outlier_data = demonstrate_outlier_removal(data)
    
    # Demonstrate data normalization
    normalized_data = demonstrate_data_normalization(data)
    
    # Demonstrate creating all features
    all_features_data = demonstrate_all_features(fe, data)
    
    # Plot the features
    print("\nPlotting features...")
    plot_features(data, all_features_data)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
