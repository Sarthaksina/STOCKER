import os
print("STOCKER FastAPI server started")
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import plotly.graph_objs as go
from yahooquery import Ticker

def run_holdings_analytics():
    tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS']
    tkr = Ticker(tickers)
    price_data = tkr.price
    prices = {}
    for sym in tickers:
        info = price_data.get(sym, {})
        price = info.get('regularMarketPrice')
        if price is None:
            print(f"[yahooquery] No price for {sym}: {info}")
        else:
            prices[sym] = price
    if not prices:
        return {'result': [], 'explanation': 'YahooQuery data unavailable for all tickers.'}
    total = sum(prices.values())
    result = [{'ticker': sym, 'price': prices[sym], 'weight': prices[sym]/total} for sym in tickers if sym in prices]
    return {'result': result, 'explanation': 'Live weights from YahooQuery'}

def run_portfolio_optimization():
    tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS']
    returns_dict = {}
    for sym in tickers:
        print(f"DEBUG [{sym}] fetching history")
        try:
            hist = Ticker(sym).history(period='6mo', interval='1d')
            print(f"DEBUG [{sym}] hist type: {type(hist)}")
            if isinstance(hist, pd.DataFrame):
                print(f"DEBUG [{sym}] hist columns: {list(hist.columns)}")
                if 'adjclose' not in hist.columns:
                    print(f"DEBUG [{sym}] 'adjclose' not in columns")
                    continue
                series = hist['adjclose']
            elif isinstance(hist, dict):
                df_sym = hist.get(sym)
                print(f"DEBUG [{sym}] hist dict item type: {type(df_sym)}")
                if isinstance(df_sym, pd.DataFrame) and 'adjclose' in df_sym.columns:
                    series = df_sym['adjclose']
                else:
                    print(f"DEBUG [{sym}] 'adjclose' missing in DataFrame")
                    continue
            else:
                print(f"DEBUG [{sym}] unexpected history format: {type(hist)}")
                continue
            print(f"DEBUG [{sym}] series length: {len(series)}")
            ret = series.pct_change().mean()
            print(f"DEBUG [{sym}] raw return: {ret}")
            if not np.isfinite(ret):
                print(f"DEBUG [{sym}] return not finite")
                continue
            returns_dict[sym] = float(ret)
        except Exception as e:
            print(f"ERROR [{sym}] {e}")
    print("DEBUG returns_dict:", returns_dict)
    if not returns_dict:
        return {'result': {'weights': [], 'assets': []}, 'explanation': 'No valid return data.'}
    total_ret = sum(returns_dict.values())
    assets = list(returns_dict.keys())
    weights = [returns_dict[s] / total_ret for s in assets]
    print("DEBUG final weights:", weights)
    return {'result': {'weights': weights, 'assets': assets}, 'explanation': 'Mean return weights from YahooQuery'}

def run_news_agent():
    # Placeholder for real news API
    return [{'title': 'TCS hits new high!', 'sentiment': 'bullish'}]

def run_price_chart(ticker: str):
    import logging
    import yfinance as yf
    df = yf.download(ticker, period="6mo")
    print("[Chart] DataFrame columns:", df.columns)
    print("[Chart] DataFrame shape:", df.shape)
    if 'Adj Close' not in df:
        print("[Chart] DataFrame (head):\n", df.head())
        return go.Figure().to_json()
    adj_close = df['Adj Close']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=adj_close.index, y=adj_close.values, mode='lines', name=ticker))
    fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price (INR)")
    return fig.to_json()

# ...add more as needed

app = FastAPI(title="STOCKER Analytics Dashboard", docs_url="/docs")

app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

class HoldingsResponse(BaseModel):
    result: List[dict]
    explanation: str

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/holdings", response_model=HoldingsResponse)
def holdings():
    print("HOLDINGS ENDPOINT HIT")
    return run_holdings_analytics()

@app.get("/api/portfolio-optimization")
def portfolio_optimization():
    print("DEBUG: PORTFOLIO ENDPOINT HIT")
    try:
        result = run_portfolio_optimization()
        print("DEBUG: run_portfolio_optimization returned:\n", result)
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("ERROR in portfolio_optimization:\n", tb)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})

@app.get("/api/news")
def news():
    return run_news_agent()

@app.get("/api/price-chart/{ticker}")
def price_chart(ticker: str):
    return JSONResponse(content={"plotly_json": run_price_chart(ticker)})

# Add more endpoints for each analytics feature

if __name__ == "__main__":
    uvicorn.run("src.web.main:app", host="0.0.0.0", port=8000, reload=True)
