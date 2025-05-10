"""
Stock data and prediction routes for STOCKER Pro API.

This module provides routes for stock data retrieval and price predictions.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime, date

from src.api.dependencies import get_data_manager, get_prediction_service
from src.data.manager import DataManager
from src.services.prediction import PredictionService, PredictionError
from src.db.models import (
    PredictionRequest, PredictionResponse, 
    StockDataFilter, StockDataResponse,
    CompanyInfo, ModelType, PredictionHorizon
)


# Create router
router = APIRouter(
    prefix="/stocks",
    tags=["stocks"],
    responses={
        404: {"description": "Stock not found"},
        400: {"description": "Bad request"},
        500: {"description": "Internal server error"}
    }
)


@router.get("/{symbol}", response_model=StockDataResponse)
async def get_stock_data(
    symbol: str = Path(..., description="Stock symbol"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("daily", description="Data interval (daily, weekly, monthly)"),
    data_manager: DataManager = Depends(get_data_manager)
):
    """
    Get historical stock price data.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (daily, weekly, monthly)
        data_manager: Data manager
        
    Returns:
        Stock price data
        
    Raises:
        HTTPException: If data retrieval fails
    """
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
        raise HTTPException(status_code=400, detail=f"Failed to retrieve stock data: {str(e)}")


@router.get("/{symbol}/company", response_model=CompanyInfo)
async def get_company_info(
    symbol: str = Path(..., description="Stock symbol"),
    data_manager: DataManager = Depends(get_data_manager)
):
    """
    Get company information.
    
    Args:
        symbol: Stock symbol
        data_manager: Data manager
        
    Returns:
        Company information
        
    Raises:
        HTTPException: If data retrieval fails
    """
    try:
        company_info = data_manager.get_company_info(symbol)
        return CompanyInfo(**company_info)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve company info: {str(e)}")


@router.post("/{symbol}/predict", response_model=Dict[str, Any])
async def predict_stock(
    symbol: str = Path(..., description="Stock symbol"),
    model_type: Optional[ModelType] = Query(None, description="Model type to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use"),
    horizon: PredictionHorizon = Query(PredictionHorizon.DAY_5, description="Prediction horizon"),
    include_confidence: bool = Query(False, description="Include confidence intervals"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Generate stock price prediction.
    
    Args:
        symbol: Stock symbol
        model_type: Type of model to use
        model_id: Specific model ID to use
        horizon: Prediction horizon
        include_confidence: Whether to include confidence intervals
        prediction_service: Prediction service
        
    Returns:
        Prediction result
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        request = PredictionRequest(
            symbol=symbol,
            model_type=model_type,
            model_id=model_id,
            prediction_horizon=horizon,
            include_confidence_intervals=include_confidence
        )
        
        result = prediction_service.predict(request)
        return result
        
    except PredictionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch_predict", response_model=Dict[str, Dict[str, Any]])
async def batch_predict(
    symbols: List[str],
    model_type: Optional[ModelType] = Query(None, description="Model type to use"),
    horizon: PredictionHorizon = Query(PredictionHorizon.DAY_5, description="Prediction horizon"),
    include_confidence: bool = Query(False, description="Include confidence intervals"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Generate predictions for multiple stocks.
    
    Args:
        symbols: List of stock symbols
        model_type: Type of model to use
        horizon: Prediction horizon
        include_confidence: Whether to include confidence intervals
        prediction_service: Prediction service
        
    Returns:
        Dictionary mapping symbols to prediction results
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        results = prediction_service.batch_predict(
            symbols=symbols,
            model_type=model_type,
            horizon=horizon,
            include_confidence=include_confidence
        )
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    List available prediction models.
    
    Args:
        symbol: Filter by symbol
        prediction_service: Prediction service
        
    Returns:
        List of model metadata
        
    Raises:
        HTTPException: If listing fails
    """
    try:
        models = prediction_service.get_available_models(symbol)
        return models
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model_info(
    model_id: str = Path(..., description="Model ID"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Get model information.
    
    Args:
        model_id: Model ID
        prediction_service: Prediction service
        
    Returns:
        Model metadata
        
    Raises:
        HTTPException: If model not found
    """
    try:
        # Use the method from the training service instead, but it's not injected here
        from src.services.training import TrainingService
        training_service = TrainingService()
        
        model_info = training_service.get_model_info(model_id)
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}") 