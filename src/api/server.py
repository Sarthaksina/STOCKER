"""
FastAPI server for STOCKER Pro.

This module provides the main FastAPI application for STOCKER Pro API.
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
import time

from src.core.config import config
from src.core.logging import logger
from src.api.dependencies import get_auth_service, get_data_manager, get_portfolio_service, get_prediction_service
from src.services.auth import AuthService, AuthError
from src.services.portfolio import PortfolioService, PortfolioError
from src.services.prediction import PredictionService, PredictionError
from src.db.models import (
    PredictionRequest, PredictionResponse, 
    StockDataFilter, StockDataResponse, 
    Portfolio, OptimizationRequest, OptimizationResult,
    BacktestRequest, BacktestResult,
    CompanyInfo
)

# Create FastAPI app
app = FastAPI(
    title="STOCKER Pro API",
    description="Financial Market Intelligence Platform API",
    version="1.0.0"
)

# Import route modules
from src.api.routes import portfolio, analysis, market_data, agent

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from route modules
app.include_router(portfolio.router)
app.include_router(analysis.router)
app.include_router(market_data.router)
app.include_router(agent.router)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.debug(f"Request: {request.method} {request.url.path} - Completed in {process_time:.4f}s")
    
    return response

# Error handling
@app.exception_handler(AuthError)
async def auth_error_handler(request, exc):
    return JSONResponse(
        status_code=401,
        content={"message": str(exc)}
    )

@app.exception_handler(PortfolioError)
async def portfolio_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)}
    )

@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "STOCKER Pro API",
        "version": "1.0.0",
        "description": "Financial Market Intelligence Platform"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Authentication endpoints
@app.post("/auth/register")
async def register_user(
    username: str, email: str, password: str,
    first_name: Optional[str] = None, last_name: Optional[str] = None,
    auth_service: AuthService = Depends(get_auth_service)
):
    user = auth_service.register_user(username, email, password, first_name, last_name)
    return user

@app.post("/auth/login")
async def login(
    username_or_email: str, password: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    user = auth_service.authenticate_user(username_or_email, password)
    token = auth_service.generate_token(user)
    return {"access_token": token, "token_type": "bearer", "user": user}

# Stock data endpoints
@app.get("/data/stock/{symbol}")
async def get_stock_data(
    symbol: str = Path(..., description="Stock symbol"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("daily", description="Data interval (daily, weekly, monthly)"),
    data_manager = Depends(get_data_manager)
):
    try:
        stock_data = data_manager.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        # Convert to response model
        response = StockDataResponse(
            symbol=symbol,
            time_frame=interval,
            prices=[{
                "date": index.isoformat(),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
                "adjusted_close": row.get("adjusted_close")
            } for index, row in stock_data.iterrows()],
            metadata={
                "start_date": stock_data.index.min().isoformat() if not stock_data.empty else None,
                "end_date": stock_data.index.max().isoformat() if not stock_data.empty else None,
                "data_points": len(stock_data)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving stock data for {symbol}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to retrieve stock data: {str(e)}")

@app.get("/data/company/{symbol}")
async def get_company_info(
    symbol: str = Path(..., description="Stock symbol"),
    data_manager = Depends(get_data_manager)
):
    try:
        company_info = data_manager.get_company_info(symbol)
        return CompanyInfo(**company_info)
        
    except Exception as e:
        logger.error(f"Error retrieving company info for {symbol}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to retrieve company info: {str(e)}")

# Prediction endpoints
@app.post("/predict/stock")
async def predict_stock(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    try:
        result = prediction_service.predict(request)
        return result
        
    except PredictionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/models")
async def list_models(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    try:
        models = prediction_service.get_available_models(symbol)
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

# Portfolio endpoints
@app.post("/portfolio")
async def create_portfolio(
    portfolio: Portfolio,
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        result = portfolio_service.create_portfolio(
            name=portfolio.name,
            user_id=portfolio.user_id,
            assets=portfolio.assets,
            description=portfolio.description,
            risk_profile=portfolio.risk_profile,
            optimization_method=portfolio.optimization_method
        )
        return result
        
    except PortfolioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create portfolio: {str(e)}")

@app.get("/portfolio/{portfolio_id}")
async def get_portfolio(
    portfolio_id: str = Path(..., description="Portfolio ID"),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        portfolio = portfolio_service.get_portfolio(portfolio_id)
        return portfolio
        
    except PortfolioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve portfolio: {str(e)}")

@app.get("/portfolio/user/{user_id}")
async def get_user_portfolios(
    user_id: str = Path(..., description="User ID"),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        portfolios = portfolio_service.get_user_portfolios(user_id)
        return portfolios
        
    except Exception as e:
        logger.error(f"Error retrieving user portfolios: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user portfolios: {str(e)}")

@app.put("/portfolio/{portfolio_id}")
async def update_portfolio(
    portfolio_id: str = Path(..., description="Portfolio ID"),
    updates: Dict[str, Any] = None,
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        updated_portfolio = portfolio_service.update_portfolio(portfolio_id, updates)
        return updated_portfolio
        
    except PortfolioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update portfolio: {str(e)}")

@app.delete("/portfolio/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: str = Path(..., description="Portfolio ID"),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        success = portfolio_service.delete_portfolio(portfolio_id)
        return {"success": success}
        
    except PortfolioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete portfolio: {str(e)}")

@app.post("/portfolio/optimize")
async def optimize_portfolio(
    request: OptimizationRequest,
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        result = portfolio_service.optimize_portfolio(
            symbols=request.symbols,
            optimization_method=request.optimization_method,
            start_date=request.start_date.isoformat() if request.start_date else None,
            end_date=request.end_date.isoformat() if request.end_date else None,
            risk_free_rate=request.risk_free_rate,
            target_return=request.target_return,
            target_risk=request.target_risk,
            constraints=request.constraints
        )
        return OptimizationResult(**result)
        
    except PortfolioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize portfolio: {str(e)}")

@app.post("/portfolio/backtest")
async def backtest_portfolio(
    request: BacktestRequest,
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        result = portfolio_service.backtest_portfolio(
            portfolio_id=request.portfolio_id,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            initial_capital=request.initial_capital
        )
        return BacktestResult(**result)
        
    except PortfolioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error backtesting portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to backtest portfolio: {str(e)}")

@app.get("/portfolio/{portfolio_id}/risk")
async def analyze_portfolio_risk(
    portfolio_id: str = Path(..., description="Portfolio ID"),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    try:
        risk_analysis = portfolio_service.analyze_portfolio_risk(portfolio_id)
        return risk_analysis
        
    except PortfolioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze portfolio risk: {str(e)}")

# Additional endpoints can be added here

def get_app() -> FastAPI:
    """
    Get the FastAPI application instance.
    
    Returns:
        FastAPI: The FastAPI application instance.
    """
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)