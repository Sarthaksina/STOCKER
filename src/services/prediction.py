"""
Prediction service for STOCKER Pro.

This module provides services for generating price predictions and
market forecasts using various ML models.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

from src.core.config import config
from src.core.logging import logger
from src.core.exceptions import StockerBaseException, ModelInferenceError
from src.db.models import PredictionModel, PredictionRequest, PredictionResponse, ModelType, PredictionHorizon
from src.data.manager import DataManager
from src.features.engineering import FeatureEngineering


class PredictionError(StockerBaseException):
    """Exception raised for prediction-related errors."""
    pass


class PredictionService:
    """
    Price prediction and forecasting service.
    
    Provides methods for generating predictions using various models,
    managing prediction requests, and storing prediction results.
    """
    
    def __init__(self):
        """Initialize the prediction service."""
        from src.db.session import get_mongodb_db
        
        try:
            # Initialize database connection
            db = get_mongodb_db()
            self.models_collection = db[config.database.models_collection]
            
            # Initialize data manager
            self.data_manager = DataManager()
            
            # Try to import ML models
            try:
                from src.ml.models import XGBoostModel, LightGBMModel
                self.XGBoostModel = XGBoostModel
                self.LightGBMModel = LightGBMModel
                self.gradient_boosting_available = True
            except ImportError:
                self.gradient_boosting_available = False
                logger.warning("Gradient boosting models not available.")
                
            try:
                from src.ml.models import LSTMModel
                self.LSTMModel = LSTMModel
                self.lstm_available = True
            except ImportError:
                self.lstm_available = False
                logger.warning("LSTM models not available.")
                
            try:
                from src.ml.ensemble import EnsembleModel
                self.EnsembleModel = EnsembleModel
                self.ensemble_available = True
            except ImportError:
                self.ensemble_available = False
                logger.warning("Ensemble models not available.")
            
            logger.info("Prediction service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction service: {e}")
            raise
    
    def get_model_instance(self, model_type: ModelType, model_id: Optional[str] = None) -> Any:
        """
        Get a model instance based on model type.
        
        Args:
            model_type: Type of model
            model_id: Optional model ID
            
        Returns:
            Model instance
            
        Raises:
            PredictionError: If model type is not supported or available
        """
        try:
            if model_type == ModelType.XGBOOST:
                if not self.gradient_boosting_available:
                    raise PredictionError("XGBoost model is not available")
                return self.XGBoostModel(model_id)
                
            elif model_type == ModelType.LIGHTGBM:
                if not self.gradient_boosting_available:
                    raise PredictionError("LightGBM model is not available")
                return self.LightGBMModel(model_id)
                
            elif model_type == ModelType.LSTM:
                if not self.lstm_available:
                    raise PredictionError("LSTM model is not available")
                return self.LSTMModel(model_id)
                
            elif model_type == ModelType.ENSEMBLE:
                if not self.ensemble_available:
                    raise PredictionError("Ensemble model is not available")
                return self.EnsembleModel(model_id)
                
            else:
                raise PredictionError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to get model instance: {e}")
            raise PredictionError(f"Failed to get model instance: {e}")
    
    def get_model_by_id(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Tuple of (model instance, model metadata)
            
        Raises:
            PredictionError: If model not found
        """
        try:
            # Find model in database
            model_data = self.models_collection.find_one({"model_id": model_id})
            
            if not model_data:
                logger.warning(f"Model not found: {model_id}")
                raise PredictionError(f"Model not found: {model_id}")
            
            # Create model instance
            model_type = model_data.get("model_type")
            model_instance = self.get_model_instance(model_type, model_id)
            
            # Load model from storage
            model_path = model_data.get("file_path")
            if not model_path:
                raise PredictionError(f"Model file path not found for model {model_id}")
                
            loaded_model = model_instance.load(model_path)
            
            return loaded_model, model_data
            
        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            raise PredictionError(f"Failed to get model: {e}")
    
    def predict(self, request: Union[PredictionRequest, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a prediction.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Convert to PredictionRequest if dict
            if isinstance(request, dict):
                request = PredictionRequest(**request)
            
            # Get symbol and prediction horizon
            symbol = request.symbol
            horizon = request.prediction_horizon
            
            # Get model
            model_instance = None
            model_type = None
            
            if request.model_id:
                # Use specific model
                model_instance, model_data = self.get_model_by_id(request.model_id)
                model_type = model_data.get("model_type")
            elif request.model_type:
                # Use model type
                model_type = request.model_type
                
                # Find the latest model of this type for this symbol
                model_data = self.models_collection.find_one(
                    {"model_type": model_type, "target_symbol": symbol},
                    sort=[("created_at", -1)]
                )
                
                if model_data:
                    model_id = model_data.get("model_id")
                    model_instance, _ = self.get_model_by_id(model_id)
                else:
                    # No model found, use default
                    model_instance = self.get_model_instance(model_type)
                    logger.warning(f"No existing model found for {symbol} using {model_type}. Using default model.")
            else:
                # Use default model type
                model_type = config.model.default_model_type
                model_instance = self.get_model_instance(model_type)
                logger.info(f"Using default model type: {model_type}")
            
            # Get data and generate features
            end_date = datetime.now().strftime("%Y-%m-%d")
            lookback_days = 365  # Get one year of data for features
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            stock_data = self.data_manager.get_stock_data(symbol, start_date, end_date)
            
            # Generate features
            feature_eng = FeatureEngineering(stock_data)
            features_df = feature_eng.generate_all_features(include_targets=False)
            
            # Handle missing values
            features_df = feature_eng.handle_missing_values(features_df)
            
            # Make prediction
            try:
                if request.include_confidence_intervals:
                    predictions, confidence = model_instance.predict_with_confidence(features_df.tail(1))
                    
                    # Format prediction result
                    prediction_date = datetime.now().strftime("%Y-%m-%d")
                    current_price = float(stock_data.iloc[-1]["close"])
                    predicted_price = float(predictions[0])
                    lower_bound = float(predicted_price - confidence[0][0])
                    upper_bound = float(predicted_price + confidence[0][1])
                    
                    prediction_data = {
                        "date": prediction_date,
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "confidence_lower": lower_bound,
                        "confidence_upper": upper_bound,
                        "horizon": horizon
                    }
                    
                    confidence_data = [
                        {"date": prediction_date, "lower": lower_bound, "upper": upper_bound}
                    ]
                else:
                    predictions = model_instance.predict(features_df.tail(1))
                    
                    # Format prediction result
                    prediction_date = datetime.now().strftime("%Y-%m-%d")
                    current_price = float(stock_data.iloc[-1]["close"])
                    predicted_price = float(predictions[0])
                    
                    prediction_data = {
                        "date": prediction_date,
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "horizon": horizon
                    }
                    
                    confidence_data = None
            
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise ModelInferenceError(f"Failed to generate prediction: {e}")
            
            # Create response
            response = PredictionResponse(
                symbol=symbol,
                model_id=model_instance.model_id,
                model_type=model_type,
                prediction_horizon=horizon,
                predictions=[prediction_data],
                confidence_intervals=confidence_data,
                predicted_at=datetime.now().isoformat(),
                metadata={
                    "features_used": list(features_df.columns)[:10],  # First 10 features for brevity
                    "feature_count": len(features_df.columns),
                    "data_points": len(features_df)
                }
            )
            
            logger.info(f"Prediction generated successfully for {symbol}")
            return response.dict()
            
        except (PredictionError, ModelInferenceError):
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Failed to generate prediction: {e}")
    
    def batch_predict(self, symbols: List[str], model_type: Optional[ModelType] = None,
                    horizon: PredictionHorizon = PredictionHorizon.DAY_5,
                    include_confidence: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for multiple symbols.
        
        Args:
            symbols: List of symbols
            model_type: Type of model to use (optional)
            horizon: Prediction horizon
            include_confidence: Whether to include confidence intervals
            
        Returns:
            Dictionary mapping symbols to prediction responses
        """
        results = {}
        
        for symbol in symbols:
            try:
                request = PredictionRequest(
                    symbol=symbol,
                    model_type=model_type,
                    prediction_horizon=horizon,
                    include_confidence_intervals=include_confidence
                )
                
                results[symbol] = self.predict(request)
                
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return results
    
    def get_available_models(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available prediction models.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of model metadata
        """
        try:
            query = {}
            if symbol:
                query["target_symbol"] = symbol
                
            models = list(self.models_collection.find(query))
            
            # Remove MongoDB ObjectIds
            for model in models:
                model.pop("_id", None)
                
            return models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            raise PredictionError(f"Failed to get available models: {e}") 


# Utility functions
def predict_stock_price(symbol: str, model_type: Optional[ModelType] = None,
                    horizon: PredictionHorizon = PredictionHorizon.DAY_5,
                    include_confidence: bool = False) -> Dict[str, Any]:
    """
    Generate a prediction for a stock price.
    
    This is a convenience wrapper around PredictionService.predict.
    
    Args:
        symbol: Stock symbol
        model_type: Type of model to use (optional)
        horizon: Prediction horizon
        include_confidence: Whether to include confidence intervals
        
    Returns:
        Prediction response
    """
    prediction_service = get_prediction_service()
    request = PredictionRequest(
        symbol=symbol,
        model_type=model_type,
        prediction_horizon=horizon,
        include_confidence_intervals=include_confidence
    )
    return prediction_service.predict(request)


def predict_portfolio_performance(symbols: List[str], weights: List[float],
                               horizon: PredictionHorizon = PredictionHorizon.DAY_5) -> Dict[str, Any]:
    """
    Predict the performance of a portfolio.
    
    Args:
        symbols: List of stock symbols
        weights: List of weights corresponding to each symbol
        horizon: Prediction horizon
        
    Returns:
        Portfolio performance prediction
    """
    if len(symbols) != len(weights):
        raise ValueError("Number of symbols must match number of weights")
        
    prediction_service = get_prediction_service()
    predictions = prediction_service.batch_predict(symbols, horizon=horizon)
    
    # Calculate weighted average of predictions
    total_return = 0.0
    current_value = 0.0
    predicted_value = 0.0
    
    for i, symbol in enumerate(symbols):
        if symbol in predictions and "error" not in predictions[symbol]:
            pred = predictions[symbol]
            if "predictions" in pred and len(pred["predictions"]) > 0:
                p = pred["predictions"][0]
                weight = weights[i]
                
                current_price = p["current_price"]
                predicted_price = p["predicted_price"]
                
                current_value += current_price * weight
                predicted_value += predicted_price * weight
    
    if current_value > 0:
        total_return = (predicted_value - current_value) / current_value
    
    return {
        "symbols": symbols,
        "weights": weights,
        "current_value": current_value,
        "predicted_value": predicted_value,
        "predicted_return": total_return,
        "horizon": horizon.value,
        "prediction_date": datetime.now().isoformat()
    }


# Singleton instance
_prediction_service_instance = None


def get_prediction_service() -> PredictionService:
    """
    Get the singleton instance of the PredictionService.
    
    Returns:
        PredictionService instance
    """
    global _prediction_service_instance
    
    if _prediction_service_instance is None:
        _prediction_service_instance = PredictionService()
        
    return _prediction_service_instance