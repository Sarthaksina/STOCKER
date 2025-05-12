"""Example script demonstrating how to use the consolidated technical indicators module.

This script shows how to use both the class-based and function-based interfaces
of the consolidated technical indicators module to analyze stock data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.indicators_final_new import (
    TechnicalIndicators,
    calculate_macd,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_stochastic_oscillator,
    calculate_volatility,
    calculate_trend_indicators,
    calculate_technical_indicators
)


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


def demonstrate_class_interface(data):
    """Demonstrate using the class-based interface for technical indicators."""
    print("\n=== Using Class-Based Interface ===")
    
    # Create an instance of TechnicalIndicators
    ti = TechnicalIndicators()
    
    # Calculate MACD
    print("\nCalculating MACD...")
    macd_result = ti.calculate_macd(data)
    print(f"MACD columns added: {[col for col in macd_result.columns if 'macd' in col]}")
    
    # Calculate RSI
    print("\nCalculating RSI...")
    rsi_result = ti.calculate_rsi(data)
    print(f"RSI value range: {rsi_result['rsi'].min():.2f} to {rsi_result['rsi'].max():.2f}")
    
    # Calculate Bollinger Bands
    print("\nCalculating Bollinger Bands...")
    bb_result = ti.calculate_bollinger_bands(data)
    print(f"Bollinger Bands columns added: {[col for col in bb_result.columns if 'bb_' in col]}")
    
    # Calculate Stochastic Oscillator
    print("\nCalculating Stochastic Oscillator...")
    stoch_result = ti.calculate_stochastic_oscillator(data)
    print(f"Stochastic Oscillator columns added: {[col for col in stoch_result.columns if 'stoch_' in col]}")
    
    # Calculate ADX
    print("\nCalculating Average Directional Index...")
    adx_result = ti.calculate_average_directional_index(data)
    print(f"ADX columns added: {[col for col in adx_result.columns if 'adx' in col or 'di' in col]}")
    
    return adx_result  # Return the result with all indicators added


def demonstrate_function_interface(data):
    """Demonstrate using the function-based interface for technical indicators."""
    print("\n=== Using Function-Based Interface ===")
    
    # Calculate MACD
    print("\nCalculating MACD...")
    macd_result = calculate_macd(data)
    print(f"MACD columns added: {[col for col in macd_result.columns if 'macd' in col]}")
    
    # Calculate RSI
    print("\nCalculating RSI...")
    rsi_result = calculate_rsi(data)
    print(f"RSI value range: {rsi_result['rsi'].min():.2f} to {rsi_result['rsi'].max():.2f}")
    
    # Calculate Bollinger Bands
    print("\nCalculating Bollinger Bands...")
    bb_result = calculate_bollinger_bands(data)
    print(f"Bollinger Bands columns added: {[col for col in bb_result.columns if 'bb_' in col]}")
    
    # Calculate Volatility
    print("\nCalculating Volatility...")
    vol_result = calculate_volatility(data)
    print(f"Volatility columns added: {[col for col in vol_result.columns if 'volatility_' in col]}")
    
    # Calculate Trend Indicators
    print("\nCalculating Trend Indicators...")
    trend_result = calculate_trend_indicators(data)
    print(f"Trend columns added: {[col for col in trend_result.columns if 'sma_' in col or 'ema_' in col][:5]}...")
    
    return trend_result  # Return the result with trend indicators added


def demonstrate_comprehensive_analysis(data):
    """Demonstrate using the comprehensive technical indicators calculation."""
    print("\n=== Using Comprehensive Technical Analysis ===")
    
    # Calculate all technical indicators at once
    print("\nCalculating all technical indicators...")
    result = calculate_technical_indicators(data)
    
    # Show the number of indicators added
    original_cols = set(data.columns)
    indicator_cols = set(result.columns) - original_cols
    print(f"Added {len(indicator_cols)} technical indicators to the data")
    
    # Show some of the indicator categories
    trend_cols = [col for col in indicator_cols if 'sma_' in col or 'ema_' in col]
    momentum_cols = [col for col in indicator_cols if 'rsi' in col or 'macd' in col or 'stoch_' in col]
    volatility_cols = [col for col in indicator_cols if 'bb_' in col or 'atr' in col or 'volatility_' in col]
    
    print(f"\nTrend indicators: {len(trend_cols)} columns")
    print(f"Momentum indicators: {len(momentum_cols)} columns")
    print(f"Volatility indicators: {len(volatility_cols)} columns")
    
    return result


def plot_indicators(data):
    """Plot some of the calculated indicators for visualization."""
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot 1: Price and Bollinger Bands
    axs[0].plot(data.index, data['close'], label='Close Price')
    if 'bb_upper' in data.columns:
        axs[0].plot(data.index, data['bb_upper'], 'r--', label='Upper BB')
        axs[0].plot(data.index, data['bb_middle'], 'g--', label='Middle BB')
        axs[0].plot(data.index, data['bb_lower'], 'r--', label='Lower BB')
    axs[0].set_title('Price and Bollinger Bands')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: MACD
    if 'macd_line' in data.columns and 'macd_signal' in data.columns:
        axs[1].plot(data.index, data['macd_line'], label='MACD Line')
        axs[1].plot(data.index, data['macd_signal'], 'r--', label='Signal Line')
        axs[1].bar(data.index, data['macd_histogram'], alpha=0.3, label='Histogram')
        axs[1].set_title('MACD')
        axs[1].legend()
        axs[1].grid(True)
    
    # Plot 3: RSI
    if 'rsi' in data.columns:
        axs[2].plot(data.index, data['rsi'], label='RSI')
        axs[2].axhline(y=70, color='r', linestyle='--', alpha=0.3)
        axs[2].axhline(y=30, color='g', linestyle='--', alpha=0.3)
        axs[2].set_title('RSI')
        axs[2].set_ylim(0, 100)
        axs[2].legend()
        axs[2].grid(True)
    
    # Plot 4: Stochastic Oscillator
    if 'stoch_k' in data.columns and 'stoch_d' in data.columns:
        axs[3].plot(data.index, data['stoch_k'], label='%K')
        axs[3].plot(data.index, data['stoch_d'], 'r--', label='%D')
        axs[3].axhline(y=80, color='r', linestyle='--', alpha=0.3)
        axs[3].axhline(y=20, color='g', linestyle='--', alpha=0.3)
        axs[3].set_title('Stochastic Oscillator')
        axs[3].set_ylim(0, 100)
        axs[3].legend()
        axs[3].grid(True)
    
    plt.tight_layout()
    
    # Create directory for saving plots if it doesn't exist
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Save the plot
    plt.savefig(output_dir / 'technical_indicators.png')
    print(f"\nPlot saved to {output_dir / 'technical_indicators.png'}")
    
    # Show the plot if running interactively
    plt.show()


def main():
    """Main function to demonstrate the technical indicators module."""
    print("STOCKER Technical Indicators Example")
    print("====================================\n")
    
    # Load sample data
    data = load_sample_data()
    print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Columns: {list(data.columns)}")
    
    # Demonstrate class-based interface
    class_result = demonstrate_class_interface(data)
    
    # Demonstrate function-based interface
    function_result = demonstrate_function_interface(data)
    
    # Demonstrate comprehensive analysis
    comprehensive_result = demonstrate_comprehensive_analysis(data)
    
    # Plot the indicators
    print("\nPlotting technical indicators...")
    plot_indicators(comprehensive_result)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
