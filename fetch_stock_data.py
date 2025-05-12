#!/usr/bin/env python
"""
Simplified stock data fetcher using Alpha Vantage API

This script fetches stock data directly from Alpha Vantage without requiring MongoDB.
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import argparse

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def fetch_stock_data(symbol: str, interval: str = 'daily', outputsize: str = 'full') -> pd.DataFrame:
    """
    Fetch stock data directly from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol
        interval: Data interval ('daily', 'weekly', 'monthly')
        outputsize: 'compact' (100 data points) or 'full' (all available data)
        
    Returns:
        DataFrame with stock price data
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("Alpha Vantage API key is required. Set the ALPHA_VANTAGE_API_KEY environment variable.")
    
    # Map interval to Alpha Vantage function (using free tier endpoints)
    function_map = {
        'daily': 'TIME_SERIES_DAILY',
        'weekly': 'TIME_SERIES_WEEKLY',
        'monthly': 'TIME_SERIES_MONTHLY'
    }
    
    if interval not in function_map:
        raise ValueError(f"Invalid interval: {interval}. Must be one of {list(function_map.keys())}")
    
    function = function_map[interval]
    
    # Build request parameters
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    # Make the request
    print(f"Fetching {interval} data for {symbol}...")
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    data = response.json()
    
    # Print the full response for debugging
    print("API Response:")
    print(json.dumps(data, indent=2))
    
    # Check for error messages
    if 'Error Message' in data:
        raise Exception(f"API returned error: {data['Error Message']}")
    
    # Determine the time series key based on the function
    time_series_keys = {
        'TIME_SERIES_DAILY': 'Time Series (Daily)',
        'TIME_SERIES_WEEKLY': 'Weekly Time Series',
        'TIME_SERIES_MONTHLY': 'Monthly Time Series'
    }
    
    time_series_key = time_series_keys[function]
    
    if time_series_key not in data:
        raise Exception(f"No time series data found in API response. Keys: {list(data.keys())}")
    
    # Extract time series data
    time_series = data[time_series_key]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Rename columns to remove the numbering prefix
    df.columns = [col.split('. ')[1] for col in df.columns]
    
    # Convert columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Set index to datetime
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    return df

def format_output(symbol: str, df: pd.DataFrame, interval: str) -> Dict[str, Any]:
    """
    Format the DataFrame into a JSON-friendly dictionary.
    
    Args:
        symbol: Stock symbol
        df: DataFrame with stock data
        interval: Data interval
        
    Returns:
        Dictionary with formatted data
    """
    # Get first and last dates
    first_date = df.index.min().isoformat()
    last_date = df.index.max().isoformat()
    
    # Format prices as list of dictionaries
    prices = []
    for date, row in df.iterrows():
        price = {
            "date": date.isoformat(),
            "open": row.get("1. open", row.get("open", None)),
            "high": row.get("2. high", row.get("high", None)),
            "low": row.get("3. low", row.get("low", None)),
            "close": row.get("4. close", row.get("close", None)),
            "volume": row.get("5. volume", row.get("volume", None))
        }
        prices.append(price)
    
    # Create output dictionary
    output = {
        "symbol": symbol,
        "interval": interval,
        "count": len(prices),
        "first_date": first_date,
        "last_date": last_date,
        "prices": prices
    }
    
    return output

def main():
    """
    Main function to parse arguments and fetch stock data.
    """
    parser = argparse.ArgumentParser(description="Fetch stock data from Alpha Vantage")
    parser.add_argument("symbol", help="Stock symbol")
    parser.add_argument("--interval", choices=["daily", "weekly", "monthly"], default="daily",
                        help="Data interval (daily, weekly, monthly)")
    parser.add_argument("--outputsize", choices=["compact", "full"], default="full",
                        help="Output size (compact=100 data points, full=all available data)")
    parser.add_argument("--output", help="Output file path (JSON format)")
    
    args = parser.parse_args()
    
    try:
        # Fetch data
        df = fetch_stock_data(args.symbol, args.interval, args.outputsize)
        
        # Format output
        output = format_output(args.symbol, df, args.interval)
        
        # Print or save output
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Data saved to {args.output}")
        else:
            print(json.dumps(output, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
