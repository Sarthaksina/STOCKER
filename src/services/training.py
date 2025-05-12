"""
Model training service for STOCKER Pro.

This module provides services for training, evaluating, and managing ML models
for stock price prediction and other financial forecasting tasks.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import os
import uuid
import json

from src.core.config import config
from src.core.logging import logger
from src.core.exceptions import StockerBaseException
from src.db.models import PredictionModel, ModelType, PredictionHorizon, TrainingJob
from src.data.manager import DataManager
from src.features.engineering import FeatureEngineering


class TrainingError(StockerBaseException):
    """Exception raised for model training-related errors."""
    pass


class TrainingService:
    """
    Model training and management service.
    
    Provides methods for training ML models, managing training jobs,
    and evaluating model performance.
    """
    
    def __init__(self):
        """Initialize the training service."""
        from src.db.session import get_mongodb_db
        
        try:
            # Initialize database connection
            db = get_mongodb_db()
            self.models_collection = db[config.database.models_collection]
            self.training_jobs_collection = db["training_jobs"]  # For tracking training jobs
            
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
            
            logger.info("Training service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize training service: {e}")
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
            TrainingError: If model type is not supported or available
        """
        try:
            if model_type == ModelType.XGBOOST:
                if not self.gradient_boosting_available:
                    raise TrainingError("XGBoost model is not available")
                return self.XGBoostModel(model_id)
                
            elif model_type == ModelType.LIGHTGBM:
                if not self.gradient_boosting_available:
                    raise TrainingError("LightGBM model is not available")
                return self.LightGBMModel(model_id)
                
            elif model_type == ModelType.LSTM:
                if not self.lstm_available:
                    raise TrainingError("LSTM model is not available")
                return self.LSTMModel(model_id)
                
            elif model_type == ModelType.ENSEMBLE:
                if not self.ensemble_available:
                    raise TrainingError("Ensemble model is not available")
                return self.EnsembleModel(model_id)
                
            else:
                raise TrainingError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to get model instance: {e}")
            raise TrainingError(f"Failed to get model instance: {e}")
    
    def prepare_training_data(self, symbol: str, start_date: Optional[str] = None,
                             end_date: Optional[str] = None, target_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            target_horizon: Prediction horizon in days
            
        Returns:
            Tuple of (features_df, target_series)
            
        Raises:
            TrainingError: If data preparation fails
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            if not start_date:
                # Default to 5 years of data
                start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
            
            # Get stock data
            stock_data = self.data_manager.get_stock_data(symbol, start_date, end_date)
            
            # Check if we have enough data
            if len(stock_data) < 100:
                raise TrainingError(f"Not enough data for {symbol}. Need at least 100 data points.")
            
            # Generate features
            feature_eng = FeatureEngineering(stock_data)
            features_df = feature_eng.generate_all_features(include_targets=True)
            
            # Get target variable
            target_col = f'target_{target_horizon}d_return'
            
            if target_col not in features_df.columns:
                raise TrainingError(f"Target column '{target_col}' not found in generated features")
                
            # Extract features and target
            y = features_df[target_col]
            X = features_df.drop(columns=[col for col in features_df.columns if col.startswith('target_')])
            
            # Handle missing values
            X = feature_eng.handle_missing_values(X)
            
            # Drop rows with NaN in target
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Prepared training data for {symbol}: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise TrainingError(f"Failed to prepare training data: {e}")
    
    def train_model(self, symbol: str, model_type: ModelType, start_date: Optional[str] = None,
                   end_date: Optional[str] = None, prediction_horizon: PredictionHorizon = PredictionHorizon.DAY_5,
                   hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a model.
        
        Args:
            symbol: Stock symbol
            model_type: Type of model
            start_date: Start date for training data
            end_date: End date for training data
            prediction_horizon: Prediction horizon
            hyperparameters: Model hyperparameters
            
        Returns:
            Dictionary with training results and model info
            
        Raises:
            TrainingError: If training fails
        """
        # Create a training job
        job_id = str(uuid.uuid4())
        training_job = TrainingJob(
            id=job_id,
            model_id=f"{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status="started",
            parameters={
                "symbol": symbol,
                "model_type": model_type,
                "start_date": start_date,
                "end_date": end_date,
                "prediction_horizon": prediction_horizon,
                "hyperparameters": hyperparameters or {}
            },
            start_time=datetime.now().isoformat()
        )
        
        # Insert job into database
        self.training_jobs_collection.insert_one(training_job.dict())
        
        try:
            # Get horizon in days
            horizon_days = int(prediction_horizon.value[:-1])
            
            # Prepare data
            X, y = self.prepare_training_data(symbol, start_date, end_date, horizon_days)
            
            # Split data
            test_size = config.model.training_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # Create model instance
            model = self.get_model_instance(model_type)
            
            # Update hyperparameters if provided
            if hyperparameters:
                model.hyperparameters.update(hyperparameters)
            
            # Train model
            logger.info(f"Training {model_type} model for {symbol}")
            
            # Create validation data tuple
            validation_data = (X_test, y_test)
            
            # Fit model
            training_history = model.fit(X_train, y_train, validation_data=validation_data)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Save model
            model_dir = os.path.join(config.model.model_save_dir, symbol)
            os.makedirs(model_dir, exist_ok=True)
            model_path = model.save(model_dir)
            
            # Extract feature importance
            feature_importance = model.get_feature_importance(X.columns.tolist())
            
            # Create model metadata
            model_metadata = PredictionModel(
                id=model.model_id,
                name=f"{symbol} {model_type} {prediction_horizon}",
                description=f"{model_type} model for {symbol} with {prediction_horizon} prediction horizon",
                model_type=model_type,
                target_symbol=symbol,
                features=list(X.columns),
                prediction_horizon=prediction_horizon,
                training_start_date=start_date,
                training_end_date=end_date,
                metrics=metrics,
                parameters=model.hyperparameters,
                file_path=model_path,
                status="trained"
            )
            
            # Insert model metadata into database
            self.models_collection.insert_one(model_metadata.dict())
            
            # Update training job
            self.training_jobs_collection.update_one(
                {"id": job_id},
                {
                    "$set": {
                        "status": "completed",
                        "end_time": datetime.now().isoformat(),
                        "result_metrics": metrics
                    }
                }
            )
            
            logger.info(f"Model {model.model_id} trained and saved successfully")
            
            # Return results
            return {
                "model_id": model.model_id,
                "model_type": model_type,
                "target_symbol": symbol,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "model_path": model_path,
                "status": "completed"
            }
            
        except Exception as e:
            # Update training job with error
            self.training_jobs_collection.update_one(
                {"id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "end_time": datetime.now().isoformat(),
                        "logs": [f"Error: {str(e)}"]
                    }
                }
            )
            
            logger.error(f"Model training failed: {e}")
            raise TrainingError(f"Failed to train model: {e}")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata
            
        Raises:
            TrainingError: If model not found
        """
        try:
            model_data = self.models_collection.find_one({"id": model_id})
            
            if not model_data:
                logger.warning(f"Model not found: {model_id}")
                raise TrainingError(f"Model not found: {model_id}")
            
            # Remove MongoDB ObjectId
            model_data.pop("_id", None)
            
            return model_data
            
        except TrainingError:
            raise
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise TrainingError(f"Failed to get model info: {e}")
    
    def get_training_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get training job information.
        
        Args:
            job_id: Job ID
            
        Returns:
            Training job metadata
            
        Raises:
            TrainingError: If job not found
        """
        try:
            job_data = self.training_jobs_collection.find_one({"id": job_id})
            
            if not job_data:
                logger.warning(f"Training job not found: {job_id}")
                raise TrainingError(f"Training job not found: {job_id}")
            
            # Remove MongoDB ObjectId
            job_data.pop("_id", None)
            
            return job_data
            
        except TrainingError:
            raise
        except Exception as e:
            logger.error(f"Failed to get training job: {e}")
            raise TrainingError(f"Failed to get training job: {e}")
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful
            
        Raises:
            TrainingError: If deletion fails
        """
        try:
            # Get model info
            model_data = self.get_model_info(model_id)
            
            # Delete model file
            model_path = model_data.get("file_path")
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
                
                # Delete metadata JSON if it exists
                meta_path = os.path.splitext(model_path)[0] + "_metadata.json"
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            
            # Delete from database
            result = self.models_collection.delete_one({"id": model_id})
            
            if result.deleted_count == 0:
                logger.warning(f"Model deletion failed: Model not found in database ({model_id})")
                raise TrainingError(f"Model not found in database: {model_id}")
            
            logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except TrainingError:
            raise
        except Exception as e:
            logger.error(f"Model deletion failed: {e}")
            raise TrainingError(f"Failed to delete model: {e}")
    
    def retrain_model(self, model_id: str, use_latest_data: bool = True) -> Dict[str, Any]:
        """
        Retrain an existing model.
        
        Args:
            model_id: Model ID
            use_latest_data: Whether to use the latest data
            
        Returns:
            Dictionary with training results and model info
            
        Raises:
            TrainingError: If retraining fails
        """
        try:
            # Get original model info
            model_data = self.get_model_info(model_id)
            
            # Extract parameters
            symbol = model_data.get("target_symbol")
            model_type = model_data.get("model_type")
            prediction_horizon = model_data.get("prediction_horizon")
            hyperparameters = model_data.get("parameters")
            
            # Set dates
            start_date = None
            end_date = None
            
            if not use_latest_data:
                # Use original training date range
                start_date = model_data.get("training_start_date")
                end_date = model_data.get("training_end_date")
            
            # Train new model
            result = self.train_model(
                symbol=symbol,
                model_type=model_type,
                start_date=start_date,
                end_date=end_date,
                prediction_horizon=prediction_horizon,
                hyperparameters=hyperparameters
            )
            
            logger.info(f"Model {model_id} retrained successfully as {result['model_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            raise TrainingError(f"Failed to retrain model: {e}")
    
    def list_models(self, symbol: Optional[str] = None, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Args:
            symbol: Filter by symbol
            model_type: Filter by model type
            
        Returns:
            List of model metadata
        """
        try:
            query = {}
            
            if symbol:
                query["target_symbol"] = symbol
                
            if model_type:
                query["model_type"] = model_type
                
            models = list(self.models_collection.find(query))
            
            # Remove MongoDB ObjectIds
            for model in models:
                model.pop("_id", None)
                
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise TrainingError(f"Failed to list models: {e}")
    
    def train_ensemble(self, symbol: str, base_models: List[str], start_date: Optional[str] = None,
                      end_date: Optional[str] = None, prediction_horizon: PredictionHorizon = PredictionHorizon.DAY_5) -> Dict[str, Any]:
        """
        Train an ensemble model using existing base models.
        
        Args:
            symbol: Stock symbol
            base_models: List of base model IDs
            start_date: Start date for training data
            end_date: End date for training data
            prediction_horizon: Prediction horizon
            
        Returns:
            Dictionary with training results and model info
            
        Raises:
            TrainingError: If training fails
        """
        if not self.ensemble_available:
            raise TrainingError("Ensemble models are not available")
            
        try:
            # Get base models
            models = []
            for model_id in base_models:
                model_data = self.get_model_info(model_id)
                
                # Check if model is for the same symbol
                if model_data.get("target_symbol") != symbol:
                    raise TrainingError(f"Model {model_id} is for symbol {model_data.get('target_symbol')}, not {symbol}")
                
                # Load model
                model_path = model_data.get("file_path")
                model_type = model_data.get("model_type")
                
                model_instance = self.get_model_instance(model_type)
                loaded_model = model_instance.load(model_path)
                
                models.append(loaded_model)
            
            # Create ensemble model
            ensemble = self.EnsembleModel()
            
            # Prepare data
            horizon_days = int(prediction_horizon.value[:-1])
            X, y = self.prepare_training_data(symbol, start_date, end_date, horizon_days)
            
            # Split data
            test_size = config.model.training_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # Train ensemble with base models
            ensemble.set_base_models(models)
            training_history = ensemble.fit(X_train, y_train, validation_data=(X_test, y_test))
            
            # Evaluate ensemble
            metrics = ensemble.evaluate(X_test, y_test)
            
            # Save ensemble model
            model_dir = os.path.join(config.model.model_save_dir, symbol)
            os.makedirs(model_dir, exist_ok=True)
            model_path = ensemble.save(model_dir)
            
            # Create model metadata
            model_metadata = PredictionModel(
                id=ensemble.model_id,
                name=f"{symbol} Ensemble {prediction_horizon}",
                description=f"Ensemble model for {symbol} with {prediction_horizon} prediction horizon",
                model_type=ModelType.ENSEMBLE,
                target_symbol=symbol,
                features=list(X.columns),
                prediction_horizon=prediction_horizon,
                training_start_date=start_date,
                training_end_date=end_date,
                metrics=metrics,
                parameters={"base_models": base_models},
                file_path=model_path,
                status="trained"
            )
            
            # Insert model metadata into database
            self.models_collection.insert_one(model_metadata.dict())
            
            logger.info(f"Ensemble model {ensemble.model_id} trained and saved successfully")
            
            # Return results
            return {
                "model_id": ensemble.model_id,
                "model_type": ModelType.ENSEMBLE,
                "target_symbol": symbol,
                "metrics": metrics,
                "base_models": base_models,
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "model_path": model_path,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            raise TrainingError(f"Failed to train ensemble model: {e}") 


# Utility functions
def train_model(symbol: str, model_type: ModelType = ModelType.XGBOOST,
             start_date: Optional[str] = None, end_date: Optional[str] = None,
             prediction_horizon: PredictionHorizon = PredictionHorizon.DAY_5,
             hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Train a model for stock price prediction.
    
    This is a convenience wrapper around TrainingService.train_model.
    
    Args:
        symbol: Stock symbol
        model_type: Type of model
        start_date: Start date for training data
        end_date: End date for training data
        prediction_horizon: Prediction horizon
        hyperparameters: Model hyperparameters
        
    Returns:
        Dictionary with training results and model info
    """
    training_service = get_training_service()
    return training_service.train_model(
        symbol, model_type, start_date, end_date, prediction_horizon, hyperparameters
    )


def evaluate_model(model_id: str) -> Dict[str, Any]:
    """
    Get model information including evaluation metrics.
    
    This is a convenience wrapper around TrainingService.get_model_info.
    
    Args:
        model_id: Model ID
        
    Returns:
        Model metadata including evaluation metrics
    """
    training_service = get_training_service()
    return training_service.get_model_info(model_id)


# Singleton instance
_training_service_instance = None


def get_training_service() -> TrainingService:
    """
    Get the singleton instance of the TrainingService.
    
    Returns:
        TrainingService instance
    """
    global _training_service_instance
    
    if _training_service_instance is None:
        _training_service_instance = TrainingService()
        
    return _training_service_instance