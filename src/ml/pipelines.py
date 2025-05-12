"""
Machine learning pipelines for STOCKER Pro.

This module provides standardized pipelines for training and evaluating 
machine learning models for financial prediction tasks.
"""
import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.core.config import StockerConfig
from src.ml.base import BaseModel
from src.ml.models import LSTMModel, XGBoostModel, LightGBMModel, EnsembleModel
from src.ml.evaluation import ModelEvaluator
from src.core.logging import get_advanced_logger
from src.core.exceptions import ModelLoadingError, ModelTrainingError

logger = logging.getLogger(__name__)

class ModelPipeline:
    """Base class for ML pipelines with common functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, run_dir: Optional[str] = None):
        """
        Initialize the model pipeline.
        
        Args:
            config: Configuration parameters
            run_dir: Directory for storing run artifacts
        """
        self.config = config or {}
        self.run_dir = run_dir or self._create_run_dir()
        self.logger = get_advanced_logger(
            f"{self.__class__.__name__}", 
            log_to_file=True, 
            log_dir=os.path.join(self.run_dir, "logs")
        )
        self.artifacts = {}
        self.start_time = time.time()
        
    def _create_run_dir(self, base_dir: str = "runs") -> str:
        """Create a timestamped run directory for artifacts."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
        
    def _save_metadata(self, metadata: Dict[str, Any]) -> str:
        """Save run metadata to the run directory."""
        metadata_path = os.path.join(self.run_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata_path
    
    def run(self) -> Dict[str, Any]:
        """
        Run the pipeline (to be implemented by subclasses).
        
        Returns:
            Dictionary of artifacts and results
        """
        raise NotImplementedError("Subclasses must implement this method")


class TrainingPipeline(ModelPipeline):
    """Pipeline for training machine learning models."""
    
    def __init__(self, config: Dict[str, Any], run_dir: Optional[str] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration parameters
            run_dir: Directory for storing run artifacts
        """
        super().__init__(config, run_dir)
        
        # Set up MLflow tracking
        self.mlflow_experiment = config.get('mlflow_experiment', 'STOCKER_TRAINING')
        mlflow.set_experiment(self.mlflow_experiment)
        self.mlflow_client = MlflowClient()
        
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            features_df: DataFrame with features and target
            
        Returns:
            Tuple of (X, y) for model training
        """
        target_col = self.config.get('target_col', 'target')
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in features DataFrame")
            
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]
        
        return X, y
        
    def create_model(self) -> BaseModel:
        """
        Create a model instance based on the configuration.
        
        Returns:
            Model instance
        """
        model_type = self.config.get('model_type', 'lstm')
        model_name = self.config.get('model_name', f"{model_type}_{datetime.now().strftime('%Y%m%d')}")
        model_params = self.config.get('model_params', {})
        
        model_classes = {
            'lstm': LSTMModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'ensemble': EnsembleModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        ModelClass = model_classes[model_type]
        model = ModelClass(name=model_name, model_type=model_type, config=model_params)
        
        return model
        
    def train_model(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Train a model on the given data.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        model = self.create_model()
        
        # Log parameters with MLflow
        with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(self.config)
            
            # Train the model
            try:
                training_history = model.fit(X, y)
                
                # Log metrics
                for metric_name, metric_value in training_history.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                        
                # Save the model
                model_path = os.path.join(self.run_dir, "model")
                os.makedirs(model_path, exist_ok=True)
                model.save(model_path)
                mlflow.log_artifacts(model_path, "model")
                
                self.logger.info(f"Model training completed and saved to {model_path}")
                
                return model, training_history
                
            except Exception as e:
                self.logger.error(f"Model training failed: {str(e)}")
                raise ModelTrainingError(f"Failed to train model: {str(e)}")
    
    def evaluate_model(self, model: BaseModel, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X: Test features
            y: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        evaluator = ModelEvaluator(self.config)
        metrics = evaluator.evaluate_model(
            model_name=model.name,
            y_true=y.values if isinstance(y, pd.Series) else y,
            y_pred=model.predict(X)
        )
        
        # Log metrics with MLflow
        with mlflow.start_run(run_name=f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
        
        return metrics
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary of artifacts and results
        """
        self.logger.info("Starting training pipeline")
        
        try:
            # Prepare data
            features_df = self.config.get('features_df')
            if features_df is None:
                raise ValueError("Features DataFrame not provided in configuration")
                
            X, y = self.prepare_data(features_df)
            self.artifacts['X'] = X
            self.artifacts['y'] = y
            
            # Train model
            model, training_history = self.train_model(X, y)
            self.artifacts['model'] = model
            self.artifacts['training_history'] = training_history
            
            # Evaluate model
            eval_metrics = self.evaluate_model(model, X, y)
            self.artifacts['evaluation_metrics'] = eval_metrics
            
            # Save metadata
            metadata = {
                "pipeline": "training",
                "model_type": self.config.get('model_type', 'lstm'),
                "model_name": model.name,
                "metrics": eval_metrics,
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.start_time
            }
            metadata_path = self._save_metadata(metadata)
            self.artifacts['metadata_path'] = metadata_path
            
            self.logger.info(f"Training pipeline completed in {time.time() - self.start_time:.2f} seconds")
            
            return self.artifacts
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise


class PredictionPipeline(ModelPipeline):
    """Pipeline for generating predictions with trained models."""
    
    def __init__(self, config: Dict[str, Any], run_dir: Optional[str] = None):
        """
        Initialize the prediction pipeline.
        
        Args:
            config: Configuration parameters
            run_dir: Directory for storing run artifacts
        """
        super().__init__(config, run_dir)
    
    def load_model(self) -> BaseModel:
        """
        Load a trained model for prediction.
        
        Returns:
            Loaded model
        """
        model_path = self.config.get('model_path')
        if not model_path:
            raise ValueError("Model path not specified in configuration")
        
        model_type = self.config.get('model_type', 'lstm')
        model_name = self.config.get('model_name', f"{model_type}_loaded")
        
        model_classes = {
            'lstm': LSTMModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'ensemble': EnsembleModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        ModelClass = model_classes[model_type]
        
        try:
            # Create empty model instance and load from disk
            model = ModelClass(name=model_name, model_type=model_type, config={})
            model.load(model_path)
            
            self.logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise ModelLoadingError(f"Failed to load model: {str(e)}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete prediction pipeline.
        
        Returns:
            Dictionary of artifacts and results
        """
        self.logger.info("Starting prediction pipeline")
        
        try:
            # Load model
            model = self.load_model()
            self.artifacts['model'] = model
            
            # Prepare features
            features = self.config.get('features')
            if features is None:
                raise ValueError("Features not provided in configuration")
                
            # Make predictions
            predictions = model.predict(features)
            self.artifacts['predictions'] = predictions
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'prediction': predictions
            })
            predictions_path = os.path.join(self.run_dir, "predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            self.artifacts['predictions_path'] = predictions_path
            
            # Save metadata
            metadata = {
                "pipeline": "prediction",
                "model_type": self.config.get('model_type', 'lstm'),
                "model_name": model.name,
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.start_time
            }
            metadata_path = self._save_metadata(metadata)
            self.artifacts['metadata_path'] = metadata_path
            
            self.logger.info(f"Prediction pipeline completed in {time.time() - self.start_time:.2f} seconds")
            
            return self.artifacts
            
        except Exception as e:
            self.logger.error(f"Prediction pipeline failed: {str(e)}")
            raise


def train_model(
    features: pd.DataFrame, 
    config: Dict[str, Any], 
    run_dir: Optional[str] = None
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Train a model using the TrainingPipeline.
    
    Args:
        features: DataFrame with features and target
        config: Configuration parameters
        run_dir: Directory for storing run artifacts
        
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    # Add features to config
    config = config.copy()
    config['features_df'] = features
    
    # Run training pipeline
    pipeline = TrainingPipeline(config, run_dir)
    artifacts = pipeline.run()
    
    return artifacts['model'], artifacts['evaluation_metrics']


def predict(
    model: BaseModel, 
    features: Union[np.ndarray, pd.DataFrame], 
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Generate predictions using a trained model.
    
    Args:
        model: Trained model
        features: Features to predict on
        config: Optional configuration parameters
        
    Returns:
        Array of predictions
    """
    return model.predict(features) 