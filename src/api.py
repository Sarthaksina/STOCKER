"""
FastAPI backend for STOCKER: exposes endpoints for portfolio analytics, rebalancing, holdings, news, events, reporting, and more.
"""
from fastapi import FastAPI, UploadFile, File, Form, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from src.entity.config_entity import StockerConfig
from src.features.agent import Agent as StockerAgent
from src.features.portfolio import recommend_portfolio, self_assess_portfolio, advanced_rebalance_portfolio, suggest_high_quality_stocks
from src.features.holdings import analyze_holdings
from src.features.events import get_corporate_events
from src.features.sentiment import get_news_sentiment
from src.components.news_agent import search_news
from src.features.risk import var_historical, max_drawdown, rolling_sharpe
from src.features.analysis import performance_analysis, sharpe_ratio, valuation_metrics, alpha_beta, attribution_analysis, momentum_analysis
from src.features.peer_comparison import peer_compare
from src.features.charts import chart_performance
from src.features.mega_agent import MegaAgent
import uvicorn

app = FastAPI()
config = StockerConfig()
agent = StockerAgent(config)
mega_agent = MegaAgent()

@app.get("/")
def root():
    """Health check route."""
    return {"message": "Stocker API is up and running"}

class UserInfo(BaseModel):
    age: Optional[int]
    risk_appetite: Optional[str]
    sip_amount: Optional[float]
    lumpsum: Optional[float]
    years: Optional[int]
    user_portfolio: Optional[Dict[str, float]]
    stock_data_map: Optional[Dict[str, dict]]
    sector_map: Optional[Dict[str, str]]
    asset_class_map: Optional[Dict[str, str]]
    price_history_map: Optional[Dict[str, list]]
    benchmark_prices: Optional[List[float]]

@app.post("/recommend_portfolio")
def api_recommend_portfolio(user_info: UserInfo):
    return recommend_portfolio(user_info.dict(), config)

@app.post("/rebalance_portfolio")
def api_rebalance_portfolio(user_info: UserInfo):
    if not user_info.user_portfolio:
        return {"error": "user_portfolio required"}
    return advanced_rebalance_portfolio(
        user_info.user_portfolio,
        user_info.stock_data_map or {},
        user_info.sector_map or {},
        user_info.asset_class_map or {},
        user_info.price_history_map or {},
        user_info.benchmark_prices or [],
        config
    )

@app.post("/self_assess_portfolio")
def api_self_assess_portfolio(user_info: UserInfo):
    if not user_info.user_portfolio:
        return {"error": "user_portfolio required"}
    return self_assess_portfolio(
        user_info.user_portfolio,
        user_info.stock_data_map or {},
        user_info.sector_map or {},
        user_info.asset_class_map or {},
        user_info.price_history_map or {}
    )

@app.get("/high_quality_stocks")
def api_high_quality_stocks():
    # For demo, expects config.symbols, dummy data
    return suggest_high_quality_stocks(config, {}, {})

@app.get("/holdings/{symbol}")
def api_holdings(symbol: str):
    return analyze_holdings(symbol, config)

@app.get("/events/{symbol}")
def api_events(symbol: str):
    """Fetch corporate events (dividends, splits) for a symbol."""
    return get_corporate_events(symbol)

@app.get("/sentiment/{symbol}")
def api_sentiment(symbol: str):
    """Fetch recent news sentiment for a symbol."""
    return get_news_sentiment(symbol, config)

@app.get("/news/{symbol}")
def api_news(
    symbol: str,
    max_articles: Optional[int] = Query(None, ge=1),
    summarize: bool = Query(True),
    sentiment: bool = Query(True)
):
    """Fetch recent news articles with optional summarization and sentiment analysis."""
    limit = max_articles or config.max_news_articles
    articles = search_news(symbol, limit, summarize, sentiment)
    return {"symbol": symbol, "articles": articles}

@app.post("/agent")
def api_agent(task: str, params: Dict[str, Any]):
    """Execute any agent task by name using MegaAgent."""
    return mega_agent.execute(task, params)

# --- Analysis Models ---
class PriceSeries(BaseModel):
    prices: List[float]

class ReturnSeries(BaseModel):
    returns: List[float]
    risk_free_rate: Optional[float] = 0.04

class ValuationRequest(BaseModel):
    stock_data: Dict[str, float]

class AlphaBetaRequest(BaseModel):
    prices: List[float]
    benchmark: List[float]

class AttributionRequest(BaseModel):
    portfolio: Dict[str, float]
    returns: Dict[str, float]

class MomentumRequest(BaseModel):
    prices: List[float]

# --- Risk Models ---
class RiskVarRequest(BaseModel):
    returns: List[float]
    confidence: Optional[float] = 0.05

class DrawdownRequest(BaseModel):
    prices: List[float]

class RollingSharpeRequest(BaseModel):
    returns: List[float]
    window: Optional[int] = 12
    risk_free_rate: Optional[float] = 0.04

# --- Peer Comparison Models ---
class PeerCompareRequest(BaseModel):
    price_history_map: Dict[str, List[float]]
    target: str
    n: Optional[int] = 5

# --- Charts Models ---
class ChartRequest(BaseModel):
    dates: List[str]
    price_history_map: Dict[str, List[float]]

# --- Analysis Endpoints ---
@app.post("/analysis/performance")
def api_analysis_performance(req: PriceSeries):
    return performance_analysis(req.prices)

@app.post("/analysis/sharpe")
def api_analysis_sharpe(req: ReturnSeries):
    return {"sharpe_ratio": sharpe_ratio(req.returns, req.risk_free_rate)}

@app.post("/analysis/valuation")
def api_analysis_valuation(req: ValuationRequest):
    return valuation_metrics(req.stock_data)

@app.post("/analysis/alpha_beta")
def api_analysis_alpha_beta(req: AlphaBetaRequest):
    return alpha_beta(req.prices, req.benchmark)

@app.post("/analysis/attribution")
def api_analysis_attribution(req: AttributionRequest):
    return attribution_analysis(req.portfolio, req.returns)

@app.post("/analysis/momentum")
def api_analysis_momentum(req: MomentumRequest):
    return {"momentum": momentum_analysis(req.prices)}

# --- Peer Comparison Endpoint ---
@app.post("/peer/compare")
def api_peer_compare(req: PeerCompareRequest):
    """Compare a target symbol to peers by return correlation."""
    return peer_compare(req.price_history_map, req.target, req.n)

# --- Charts Endpoints ---
@app.post("/charts/performance")
def api_charts_performance(req: ChartRequest):
    """Compute quarterly and yearly returns for price history."""
    return chart_performance(req.dates, req.price_history_map)

# --- Risk Endpoints ---
@app.post("/risk/var")
def api_risk_var(req: RiskVarRequest):
    return var_historical(req.returns, req.confidence)

@app.post("/risk/drawdown")
def api_risk_drawdown(req: DrawdownRequest):
    return max_drawdown(req.prices)

@app.post("/risk/rolling_sharpe")
def api_risk_rolling_sharpe(req: RollingSharpeRequest):
    return rolling_sharpe(req.returns, req.window, req.risk_free_rate)

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
