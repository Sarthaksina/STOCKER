"""
Example script demonstrating the technical indicators module in STOCKER Pro.

This example shows how to:
1. Fetch historical stock data using Alpha Vantage
2. Calculate various technical indicators
3. Visualize the indicators on interactive charts
4. Analyze signal combinations for potential trade setups
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configuration import Config
from src.data_access.alpha_vantage_client import AlphaVantageClient
from src.features.technical_indicators import TechnicalIndicators
from src.exception.exceptions import AlphaVantageAPIException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol, outputsize="full"):
    """
    Fetch historical stock data for a symbol.
    
    Args:
        symbol: Stock symbol
        outputsize: 'compact' for recent data, 'full' for all available data
        
    Returns:
        DataFrame with stock data
    """
    config = Config()
    api_key = config.get_alpha_vantage_api_key()
    
    if not api_key:
        logger.error("No Alpha Vantage API key found. Please set ALPHA_VANTAGE_API_KEY in environment or .env file")
        print("\nTo get an API key, visit: https://www.alphavantage.co/support/#api-key")
        return None
    
    # Initialize client with cache directory
    cache_dir = Path(__file__).parent.parent.parent / "cache" / "alpha_vantage"
    client = AlphaVantageClient(config, cache_dir=str(cache_dir))
    
    try:
        # Fetch daily stock data
        logger.info(f"Fetching stock data for {symbol}...")
        df = client.get_time_series_daily(symbol, outputsize=outputsize)
        
        # Log basic info
        logger.info(f"Retrieved {len(df)} days of data for {symbol}")
        logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        return df
        
    except AlphaVantageAPIException as e:
        logger.error(f"Failed to fetch stock data: {e}")
        return None

def create_trend_analysis_chart(df, indicators_df, title, output_path=None):
    """
    Create a chart with price and trend indicators.
    
    Args:
        df: Original stock price DataFrame
        indicators_df: DataFrame with technical indicators
        title: Chart title
        output_path: Path to save the chart image (optional)
    """
    # Create figure and subplot grid
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
    
    # Price and Moving Averages plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Close Price', color='black', alpha=0.7)
    ax1.plot(df.index, indicators_df['sma_20'], label='SMA 20', color='blue', alpha=0.8)
    ax1.plot(df.index, indicators_df['sma_50'], label='SMA 50', color='orange', alpha=0.8)
    ax1.plot(df.index, indicators_df['sma_200'], label='SMA 200', color='red', alpha=0.8)
    
    # Bollinger Bands
    ax1.plot(df.index, indicators_df['bb_upper'], label='Upper BB', color='gray', linestyle='--', alpha=0.5)
    ax1.plot(df.index, indicators_df['bb_lower'], label='Lower BB', color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(df.index, indicators_df['bb_upper'], indicators_df['bb_lower'], color='lightgray', alpha=0.2)
    
    ax1.set_title(f"{title} - Price and Trend Indicators")
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # MACD plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, indicators_df['macd_line'], label='MACD Line', color='blue')
    ax2.plot(df.index, indicators_df['macd_signal'], label='Signal Line', color='red')
    ax2.bar(df.index, indicators_df['macd_histogram'], label='Histogram', color='green', alpha=0.5, width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # RSI plot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, indicators_df['rsi_14'], label='RSI 14', color='purple')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)  # Overbought line
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)  # Oversold line
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Chart saved to {output_path}")
        
    return fig

def create_oscillator_chart(df, indicators_df, title, output_path=None):
    """
    Create a chart with oscillator indicators.
    
    Args:
        df: Original stock price DataFrame
        indicators_df: DataFrame with technical indicators
        title: Chart title
        output_path: Path to save the chart image (optional)
    """
    # Create figure and subplot grid
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # Price plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Close Price', color='black')
    
    # Add volume as bars on the same plot with separate y-axis
    ax1v = ax1.twinx()
    ax1v.bar(df.index, df['volume'], label='Volume', color='gray', alpha=0.3, width=0.8)
    ax1v.set_ylabel('Volume')
    
    ax1.set_title(f"{title} - Price and Oscillator Indicators")
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1v.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Stochastic plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, indicators_df['stoch_k'], label='%K', color='blue')
    ax2.plot(df.index, indicators_df['stoch_d'], label='%D', color='red')
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5)  # Overbought line
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.5)  # Oversold line
    ax2.set_ylabel('Stochastic')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # ADX plot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, indicators_df['adx'], label='ADX', color='black')
    ax3.plot(df.index, indicators_df['pos_di'], label='+DI', color='green')
    ax3.plot(df.index, indicators_df['neg_di'], label='-DI', color='red')
    ax3.axhline(y=25, color='gray', linestyle='--', alpha=0.5)  # Strong trend threshold
    ax3.set_ylabel('ADX')
    ax3.set_ylim(0, 60)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # OBV plot
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, indicators_df['obv'] / 1e6, label='OBV (millions)', color='purple')
    ax4.set_ylabel('OBV')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Chart saved to {output_path}")
        
    return fig

def identify_trading_signals(df, indicators_df):
    """
    Identify potential trading signals based on technical indicators.
    
    Args:
        df: Original stock price DataFrame
        indicators_df: DataFrame with technical indicators
        
    Returns:
        DataFrame with buy and sell signals
    """
    # Create a new DataFrame for signals
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['close']
    
    # 1. Golden Cross (SMA 50 crosses above SMA 200)
    signals['golden_cross'] = (
        (indicators_df['sma_50'] > indicators_df['sma_200']) & 
        (indicators_df['sma_50'].shift(1) <= indicators_df['sma_200'].shift(1))
    )
    
    # 2. Death Cross (SMA 50 crosses below SMA 200)
    signals['death_cross'] = (
        (indicators_df['sma_50'] < indicators_df['sma_200']) & 
        (indicators_df['sma_50'].shift(1) >= indicators_df['sma_200'].shift(1))
    )
    
    # 3. MACD Bullish Crossover
    signals['macd_bullish_cross'] = (
        (indicators_df['macd_line'] > indicators_df['macd_signal']) & 
        (indicators_df['macd_line'].shift(1) <= indicators_df['macd_signal'].shift(1))
    )
    
    # 4. MACD Bearish Crossover
    signals['macd_bearish_cross'] = (
        (indicators_df['macd_line'] < indicators_df['macd_signal']) & 
        (indicators_df['macd_line'].shift(1) >= indicators_df['macd_signal'].shift(1))
    )
    
    # 5. RSI Oversold Condition (RSI crosses above 30 from below)
    signals['rsi_oversold'] = (
        (indicators_df['rsi_14'] > 30) & 
        (indicators_df['rsi_14'].shift(1) <= 30)
    )
    
    # 6. RSI Overbought Condition (RSI crosses below 70 from above)
    signals['rsi_overbought'] = (
        (indicators_df['rsi_14'] < 70) & 
        (indicators_df['rsi_14'].shift(1) >= 70)
    )
    
    # 7. Bollinger Band Breakout (Price crosses above upper band)
    signals['bb_upper_breakout'] = (
        (df['close'] > indicators_df['bb_upper']) & 
        (df['close'].shift(1) <= indicators_df['bb_upper'].shift(1))
    )
    
    # 8. Bollinger Band Breakdown (Price crosses below lower band)
    signals['bb_lower_breakdown'] = (
        (df['close'] < indicators_df['bb_lower']) & 
        (df['close'].shift(1) >= indicators_df['bb_lower'].shift(1))
    )
    
    # 9. Stochastic Bullish Crossover
    signals['stoch_bullish_cross'] = (
        (indicators_df['stoch_k'] > indicators_df['stoch_d']) & 
        (indicators_df['stoch_k'].shift(1) <= indicators_df['stoch_d'].shift(1)) & 
        (indicators_df['stoch_k'] < 50)  # Below 50 for stronger signal
    )
    
    # 10. Stochastic Bearish Crossover
    signals['stoch_bearish_cross'] = (
        (indicators_df['stoch_k'] < indicators_df['stoch_d']) & 
        (indicators_df['stoch_k'].shift(1) >= indicators_df['stoch_d'].shift(1)) & 
        (indicators_df['stoch_k'] > 50)  # Above 50 for stronger signal
    )
    
    # 11. ADX Strong Trend (ADX rises above 25)
    signals['adx_strong_trend'] = (
        (indicators_df['adx'] > 25) & 
        (indicators_df['adx'].shift(1) <= 25)
    )
    
    # 12. Volume Spike (Volume > 2x 20-day average)
    volume_sma = df['volume'].rolling(window=20).mean()
    signals['volume_spike'] = df['volume'] > (volume_sma * 2)
    
    # 13. Combined Buy Signal
    signals['buy_signal'] = (
        signals['macd_bullish_cross'] | 
        signals['rsi_oversold'] | 
        signals['stoch_bullish_cross']
    )
    
    # 14. Combined Sell Signal
    signals['sell_signal'] = (
        signals['macd_bearish_cross'] | 
        signals['rsi_overbought'] | 
        signals['stoch_bearish_cross']
    )
    
    # 15. Strong Buy Signal (at least 2 buy indicators)
    buy_columns = ['macd_bullish_cross', 'rsi_oversold', 'stoch_bullish_cross', 'golden_cross', 'bb_lower_breakdown']
    signals['buy_count'] = signals[buy_columns].sum(axis=1)
    signals['strong_buy'] = signals['buy_count'] >= 2
    
    # 16. Strong Sell Signal (at least 2 sell indicators)
    sell_columns = ['macd_bearish_cross', 'rsi_overbought', 'stoch_bearish_cross', 'death_cross', 'bb_upper_breakout']
    signals['sell_count'] = signals[sell_columns].sum(axis=1)
    signals['strong_sell'] = signals['sell_count'] >= 2
    
    # Replace NaN with False for boolean columns
    bool_columns = signals.columns[signals.columns != 'price']
    for col in bool_columns:
        if signals[col].dtype == bool:
            signals[col] = signals[col].fillna(False)
    
    return signals

def analyze_signals_effectiveness(df, signals, lookforward_days=20):
    """
    Analyze the effectiveness of trading signals by calculating
    future returns after each signal.
    
    Args:
        df: Original stock price DataFrame
        signals: DataFrame with trading signals
        lookforward_days: Number of days to look forward for return calculation
        
    Returns:
        Analysis results
    """
    # Calculate forward returns
    forward_returns = {}
    
    # Function to calculate future return
    def calc_return(start_idx, days):
        if start_idx + days >= len(df):
            end_idx = len(df) - 1
        else:
            end_idx = start_idx + days
        
        start_price = df['close'].iloc[start_idx]
        end_price = df['close'].iloc[end_idx]
        return (end_price / start_price - 1) * 100
    
    # Signal types to analyze
    signal_types = [
        'golden_cross', 'death_cross', 'macd_bullish_cross', 'macd_bearish_cross',
        'rsi_oversold', 'rsi_overbought', 'bb_upper_breakout', 'bb_lower_breakdown',
        'stoch_bullish_cross', 'stoch_bearish_cross', 'strong_buy', 'strong_sell'
    ]
    
    # Analyze each signal type
    results = {}
    for signal_type in signal_types:
        signal_dates = signals.index[signals[signal_type]].tolist()
        
        if not signal_dates:
            results[signal_type] = {
                'count': 0,
                'avg_return': np.nan,
                'win_rate': np.nan,
                'max_return': np.nan,
                'min_return': np.nan
            }
            continue
        
        # Calculate returns for each signal
        returns = []
        for date in signal_dates:
            idx = df.index.get_loc(date)
            returns.append(calc_return(idx, lookforward_days))
        
        # Skip if no valid returns
        if not returns:
            continue
            
        # Calculate statistics
        avg_return = np.mean(returns)
        win_rate = sum(1 for r in returns if (r > 0 and 'bull' in signal_type or 'buy' in signal_type) or 
                                         (r < 0 and 'bear' in signal_type or 'sell' in signal_type)) / len(returns)
        max_return = np.max(returns)
        min_return = np.min(returns)
        
        results[signal_type] = {
            'count': len(returns),
            'avg_return': avg_return,
            'win_rate': win_rate * 100,  # as percentage
            'max_return': max_return,
            'min_return': min_return
        }
    
    return results

def print_signal_analysis(analysis_results):
    """
    Print signal analysis results in a formatted table.
    
    Args:
        analysis_results: Signal analysis results
    """
    # Define signal types in a more readable format
    signal_names = {
        'golden_cross': 'Golden Cross (SMA50 > SMA200)',
        'death_cross': 'Death Cross (SMA50 < SMA200)',
        'macd_bullish_cross': 'MACD Bullish Crossover',
        'macd_bearish_cross': 'MACD Bearish Crossover',
        'rsi_oversold': 'RSI Oversold (>30 from below)',
        'rsi_overbought': 'RSI Overbought (<70 from above)',
        'bb_upper_breakout': 'Bollinger Upper Band Breakout',
        'bb_lower_breakdown': 'Bollinger Lower Band Breakdown',
        'stoch_bullish_cross': 'Stochastic Bullish Crossover',
        'stoch_bearish_cross': 'Stochastic Bearish Crossover',
        'strong_buy': 'Strong Buy Signal (2+ indicators)',
        'strong_sell': 'Strong Sell Signal (2+ indicators)'
    }
    
    # Print header
    print("\n=== SIGNAL ANALYSIS RESULTS ===")
    print(f"{'Signal Type':<40} {'Count':>6} {'Avg Return %':>12} {'Win Rate %':>12} {'Max Return %':>12} {'Min Return %':>12}")
    print("-" * 100)
    
    # Print each signal's results
    for signal_type, metrics in analysis_results.items():
        signal_name = signal_names.get(signal_type, signal_type)
        count = metrics['count']
        
        if count > 0:
            avg_return = f"{metrics['avg_return']:.2f}"
            win_rate = f"{metrics['win_rate']:.1f}"
            max_return = f"{metrics['max_return']:.2f}"
            min_return = f"{metrics['min_return']:.2f}"
        else:
            avg_return = win_rate = max_return = min_return = "N/A"
        
        print(f"{signal_name:<40} {count:>6} {avg_return:>12} {win_rate:>12} {max_return:>12} {min_return:>12}")

def main():
    """Main function demonstrating technical indicators."""
    # Fetch stock data for an example symbol
    symbol = "MSFT"  # Default symbol
    
    # Check if symbol is provided as command line argument
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    
    # Fetch stock data
    df = fetch_stock_data(symbol, outputsize="full")
    
    if df is None or df.empty:
        logger.error("Failed to fetch stock data. Exiting.")
        return
    
    # Trim data to last 2 years for better visualization
    two_years_ago = df.index.max() - pd.DateOffset(years=2)
    df_recent = df[df.index >= two_years_ago]
    
    # Calculate technical indicators
    logger.info("Calculating technical indicators...")
    indicators_df = TechnicalIndicators.add_all_indicators(df_recent)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create trend analysis chart
    trend_chart_path = plots_dir / f"{symbol}_trend_analysis.png"
    create_trend_analysis_chart(df_recent, indicators_df, f"{symbol} Trend Analysis", trend_chart_path)
    
    # Create oscillator chart
    oscillator_chart_path = plots_dir / f"{symbol}_oscillators.png"
    create_oscillator_chart(df_recent, indicators_df, f"{symbol} Oscillator Analysis", oscillator_chart_path)
    
    # Identify trading signals
    logger.info("Identifying trading signals...")
    signals = identify_trading_signals(df_recent, indicators_df)
    
    # Count and print signals
    signal_count = {col: signals[col].sum() for col in signals.columns if signals[col].dtype == bool}
    
    print(f"\n=== {symbol} TRADING SIGNALS ===")
    print(f"{'Signal Type':<25} {'Count':>5}")
    print("-" * 32)
    for signal_type, count in signal_count.items():
        if signal_type not in ['price', 'buy_count', 'sell_count']:
            print(f"{signal_type:<25} {count:>5}")
    
    # Calculate and print signal effectiveness
    logger.info("Analyzing signal effectiveness...")
    analysis_results = analyze_signals_effectiveness(df_recent, signals)
    print_signal_analysis(analysis_results)
    
    # Print recent signals (last 5 trading days)
    recent_days = 5
    last_days = signals.tail(recent_days).copy()
    
    print(f"\n=== RECENT SIGNALS (LAST {recent_days} TRADING DAYS) ===")
    for idx, row in last_days.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        price = row['price']
        
        active_signals = [signal for signal in signal_count.keys() 
                        if signal not in ['price', 'buy_count', 'sell_count'] and row[signal]]
        
        if not active_signals:
            print(f"{date_str} - Price: ${price:.2f} - No active signals")
        else:
            signal_str = ', '.join(active_signals)
            print(f"{date_str} - Price: ${price:.2f} - Signals: {signal_str}")
    
    logger.info("Technical indicator analysis complete!")

if __name__ == "__main__":
    main() 