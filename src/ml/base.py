"""
Base model interface for all financial prediction models in STOCKER Pro.
This module defines the common interface that all models must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, Tuple, List
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all financial prediction models.
    Defines the interface that must be implemented by all models.
    """
    
    def __init__(self, name: str, model_type: str, config: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            name: A unique name for this model instance
            model_type: The type of model (e.g., 'lstm', 'xgboost', 'lightgbm')
            config: Model configuration parameters
        """
        self.name = name
        self.model_type = model_type
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "model_type": model_type,
            "name": name,
        }

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        Args:
            X: Training features
            y: Target values
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Dict containing training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probabilistic predictions with the model (for classification models).
        For regression models, could return prediction intervals.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probabilities or confidence intervals
        """
        pass
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate the model on test data with financial metrics.
        
        Args:
            X: Test features
            y: True target values
        
        Returns:
            Dict of evaluation metrics
        """
        predictions = self.predict(X)
        
        # Convert to numpy arrays if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        # Basic metrics
        metrics = {
            "mse": np.mean((y - predictions)**2),
            "mae": np.mean(np.abs(y - predictions)),
            "rmse": np.sqrt(np.mean((y - predictions)**2)),
        }
        
        # Add financial metrics if possible
        try:
            # Directional accuracy (for financial predictions)
            if len(y) > 1 and len(predictions) > 1:
                y_direction = np.sign(y[1:] - y[:-1])
                pred_direction = np.sign(predictions[1:] - predictions[:-1])
                dir_accuracy = np.mean(y_direction == pred_direction)
                metrics["directional_accuracy"] = float(dir_accuracy)
        except Exception as e:
            logger.warning(f"Could not calculate directional accuracy: {e}")
            
        return metrics
    
    def save(self, path: str) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Directory path for saving the model
            
        Returns:
            Path to the saved model file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save a model that has not been fitted")
        
        os.makedirs(path, exist_ok=True)
        model_file = os.path.join(path, f"{self.name}_{self.model_type}_model.joblib")
        
        # Save model-specific artifacts
        self._save_model_artifacts(path)
        
        # Save metadata
        metadata_file = os.path.join(path, f"{self.name}_{self.model_type}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f)
            
        return model_file
    
    def load(self, path: str) -> 'BaseModel':
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model directory
            
        Returns:
            Loaded model instance
        """
        # Load model-specific artifacts
        self._load_model_artifacts(path)
        
        # Load metadata
        metadata_file = os.path.join(path, f"{self.name}_{self.model_type}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        self.is_fitted = True
        return self
    
    @abstractmethod
    def _save_model_artifacts(self, path: str) -> None:
        """
        Save model-specific artifacts.
        
        Args:
            path: Directory path for saving the model artifacts
        """
        pass
    
    @abstractmethod
    def _load_model_artifacts(self, path: str) -> None:
        """
        Load model-specific artifacts.
        
        Args:
            path: Directory path for loading the model artifacts
        """
        pass
    
    @abstractmethod
    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if supported by the model.
        
        Returns:
            Dictionary mapping feature names to importance scores or None if not supported
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} ({self.model_type})" 