import pandas as pd
from yahooquery import Ticker

tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS']

print("Running per-ticker history debug")
for sym in tickers:
    try:
        hist = Ticker(sym).history(period='6mo', interval='1d')
        print(f"{sym} -> type: {type(hist)}")
        if isinstance(hist, pd.DataFrame):
            print(f"{sym} columns: {list(hist.columns)}")
        elif isinstance(hist, dict):
            print(f"{sym} dict keys: {list(hist.keys())}")
        else:
            print(f"{sym} returned: {hist}")
    except Exception as e:
        print(f"{sym} error: {e}")
