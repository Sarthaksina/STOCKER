"""
Ensemble model implementation for STOCKER Pro.
This module combines multiple models for improved prediction accuracy.
"""
from typing import Dict, Any, Union, Optional, Tuple, List
import numpy as np
import pandas as pd
import os
import joblib
import logging
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.ml.base_model import BaseModel
from src.ml.lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple base models.
    Supports weighted averaging, voting, and stacking approaches.
    """
    
    def __init__(self, name: str = "stock_ensemble_model", 
                 config: Optional[Dict[str, Any]] = None,
                 models: Optional[List[BaseModel]] = None):
        """
        Initialize the ensemble model.
        
        Args:
            name: Model name
            config: Model configuration containing parameters like:
                   - ensemble_method: Method for combining predictions ('weighted_avg', 'voting', 'stacking')
                   - weights: Optional dictionary mapping model names to weights
                   - directional_voting: For voting method, whether to vote on direction (up/down)
                   - meta_model_type: For stacking, the type of meta-model
                   - meta_model_config: For stacking, configuration for meta-model
                   - cv_folds: For stacking, number of cross-validation folds
                   - stack_with_orig_features: For stacking, whether to include original features
            models: List of base models to include in the ensemble
        """
        default_config = {
            "ensemble_method": "weighted_avg",  # 'weighted_avg', 'voting', or 'stacking'
            "directional_voting": False,        # For voting, whether to vote on direction
            "meta_model_type": "lightgbm",      # For stacking, type of meta-model
            "meta_model_config": None,          # For stacking, meta-model config
            "cv_folds": 5,                      # For stacking, number of CV folds
            "stack_with_orig_features": False,  # For stacking, whether to include original features
        }
        
        # Override defaults with provided config
        if config:
            default_config.update(config)
            
        super().__init__(name=name, model_type="ensemble", config=default_config)
        
        # Initialize models list and weights
        self.models = []
        self.model_names = []
        self.weights = {}
        self.meta_model = None
        
        # Add models if provided
        if models:
            for model in models:
                self.add_model(model)
                
    def add_model(self, model: BaseModel, weight: Optional[float] = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Model to add
            weight: Optional weight for the model (will be normalized with others)
        """
        if model.name in self.model_names:
            raise ValueError(f"Model with name '{model.name}' already exists in ensemble")
        
        self.models.append(model)
        self.model_names.append(model.name)
        
        # Update weights
        if weight is None:
            # Equal weighting if no weight specified
            for name in self.model_names:
                self.weights[name] = 1.0 / len(self.model_names)
        else:
            # Add new weight and renormalize
            self.weights[model.name] = weight
            total_weight = sum(self.weights.values())
            for name in self.weights:
                self.weights[name] /= total_weight
                
        logger.info(f"Added model {model.name} to ensemble {self.name} with weight {self.weights[model.name]:.3f}")
        
    def remove_model(self, model_name: str) -> None:
        """
        Remove a model from the ensemble.
        
        Args:
            model_name: Name of the model to remove
        """
        if model_name not in self.model_names:
            raise ValueError(f"Model '{model_name}' not found in ensemble")
        
        # Find model index
        idx = self.model_names.index(model_name)
        
        # Remove model
        self.models.pop(idx)
        self.model_names.remove(model_name)
        del self.weights[model_name]
        
        # Renormalize weights
        if self.model_names:
            total_weight = sum(self.weights.values())
            for name in self.weights:
                self.weights[name] /= total_weight
        
        logger.info(f"Removed model {model_name} from ensemble {self.name}")
        
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for the ensemble models.
        
        Args:
            weights: Dictionary mapping model names to weights
        """
        # Validate weights
        for name in weights:
            if name not in self.model_names:
                raise ValueError(f"Model '{name}' not found in ensemble")
        
        # Set weights
        for name in weights:
            self.weights[name] = weights[name]
            
        # Normalize weights
        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight
            
        logger.info(f"Updated ensemble weights: {self.weights}")
        
    def _init_meta_model(self) -> BaseModel:
        """
        Initialize meta-model for stacking.
        
        Returns:
            Initialized meta-model
        """
        # Currently only supporting LightGBM as meta-model
        if self.config["meta_model_type"].lower() == "lightgbm":
            # Default meta-model config
            meta_config = {
                "objective": "regression",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "n_estimators": 100
            }
            
            # Override with provided config
            if self.config["meta_model_config"]:
                meta_config.update(self.config["meta_model_config"])
                
            return LightGBMModel(name=f"{self.name}_meta", config=meta_config)
        else:
            raise ValueError(f"Meta-model type '{self.config['meta_model_type']}' not supported")
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        For weighted average and voting, this trains the base models.
        For stacking, this also trains the meta-model.
        
        Args:
            X: Training features
            y: Target values
            validation_data: Optional validation data
            
        Returns:
            Training history
        """
        ensemble_method = self.config["ensemble_method"]
        
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            
        # Check if we need to train base models
        need_train_base = any(not model.is_fitted for model in self.models)
        
        # Training history
        history = {
            "base_models": {},
            "meta_model": None
        }
        
        # Train base models if needed
        if need_train_base:
            logger.info(f"Training {len(self.models)} base models for ensemble {self.name}")
            for model in self.models:
                if not model.is_fitted:
                    model_history = model.fit(X, y, validation_data)
                    history["base_models"][model.name] = model_history
        
        # For stacking, train meta-model
        if ensemble_method == "stacking":
            logger.info(f"Training stacking ensemble with {self.config['meta_model_type']} meta-model")
            
            # Initialize meta-model if not already initialized
            if self.meta_model is None:
                self.meta_model = self._init_meta_model()
            
            # Generate meta-features using cross-validation
            meta_features = self._generate_meta_features(X, y)
            
            # Train meta-model on meta-features
            meta_history = self.meta_model.fit(meta_features, y)
            history["meta_model"] = meta_history
            
        self.is_fitted = True
        logger.info(f"Ensemble model {self.name} training completed")
        
        return history
    
    def _generate_meta_features(self, X: Union[np.ndarray, pd.DataFrame], 
                               y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Generate meta-features for stacking using cross-validation.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Meta-features array
        """
        n_samples = len(X)
        n_models = len(self.models)
        
        # Initialize meta-features array
        meta_features = np.zeros((n_samples, n_models))
        
        # Set up cross-validation
        cv = KFold(n_splits=self.config["cv_folds"], shuffle=True, random_state=42)
        
        # Generate predictions for each fold
        for train_idx, val_idx in cv.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Train each model on this fold and predict
            for i, model in enumerate(self.models):
                # Create a clone of the model to avoid data leakage
                fold_model = model.__class__(name=f"{model.name}_fold", config=model.config)
                
                # Train on this fold
                fold_model.fit(X_train, y_train)
                
                # Predict on validation data
                val_preds = fold_model.predict(X_val)
                
                # Store predictions as meta-features
                meta_features[val_idx, i] = val_preds
        
        # If including original features
        if self.config["stack_with_orig_features"]:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
                
            # Combine meta-features with original features
            meta_features = np.hstack([meta_features, X_array])
            
        return meta_features
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the ensemble model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        # Get predictions from all base models
        predictions = []
        for model in self.models:
            model_preds = model.predict(X)
            predictions.append(model_preds)
            
        # Convert to numpy arrays and ensure consistent shape
        predictions = [np.asarray(p).reshape(-1) for p in predictions]
        
        ensemble_method = self.config["ensemble_method"]
        
        if ensemble_method == "weighted_avg":
            # Weighted average of predictions
            ensemble_pred = np.zeros_like(predictions[0])
            for i, model in enumerate(self.models):
                weight = self.weights[model.name]
                ensemble_pred += weight * predictions[i]
                
        elif ensemble_method == "voting":
            if self.config["directional_voting"]:
                # Directional voting (predict direction: up/down)
                votes = np.zeros_like(predictions[0])
                
                for i, model in enumerate(self.models):
                    weight = self.weights[model.name]
                    
                    # Get directions: 1 for up, -1 for down, 0 for no change
                    if len(predictions[i]) > 1:
                        dirs = np.sign(np.diff(predictions[i]))
                        # Pad with 0 to maintain original length
                        dirs = np.append(0, dirs)
                    else:
                        dirs = np.zeros_like(predictions[i])
                        
                    votes += weight * dirs
                    
                # Final vote: sign of weighted votes
                ensemble_pred = np.sign(votes)
                
            else:
                # Simple voting (binary classification)
                votes = np.zeros_like(predictions[0])
                
                for i, model in enumerate(self.models):
                    weight = self.weights[model.name]
                    # Threshold at 0.5 for binary classification
                    binary_pred = (predictions[i] >= 0.5).astype(int)
                    votes += weight * binary_pred
                    
                # Final vote: 1 if weighted sum > 0.5, else 0
                ensemble_pred = (votes > 0.5).astype(int)
                
        elif ensemble_method == "stacking":
            # For stacking, we need to generate meta-features first
            meta_features = np.column_stack(predictions)
            
            # If including original features
            if self.config["stack_with_orig_features"]:
                if isinstance(X, pd.DataFrame):
                    X_array = X.values
                else:
                    X_array = X
                    
                # Combine meta-features with original features
                meta_features = np.hstack([meta_features, X_array])
                
            # Make predictions with meta-model
            ensemble_pred = self.meta_model.predict(meta_features)
            
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
            
        return ensemble_pred
    
    def add_deep_learning_model(self, model: BaseModel, weight: Optional[float] = None) -> None:
        """
        Special method for adding DL models with additional validation
        """
        if not hasattr(model, 'fit_generator') and not hasattr(model, 'fit'):
            raise ValueError("Model must be a deep learning model with fit/fit_generator method")
        
        self.add_model(model, weight)
        logger.info(f"Added deep learning model {model.name} to ensemble")
    
    def predict_with_uncertainty(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get predictions with uncertainty estimates
        """
        preds = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                preds.append(model.predict_proba(X))
            else:
                preds.append(model.predict(X))
        
        return {
            "mean_prediction": np.mean(preds, axis=0),
            "std_deviation": np.std(preds, axis=0),
            "min_prediction": np.min(preds, axis=0),
            "max_prediction": np.max(preds, axis=0)
        }
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probabilistic predictions with the ensemble model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probabilistic predictions or confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        ensemble_method = self.config["ensemble_method"]
        
        # For regression tasks with weighted_avg, return basic confidence interval
        if ensemble_method == "weighted_avg":
            # Get base predictions
            point_pred = self.predict(X)
            
            # Get probabilistic predictions from base models if available
            base_intervals = []
            for model in self.models:
                try:
                    model_interval = model.predict_proba(X)
                    base_intervals.append(model_interval)
                except:
                    # If a model doesn't support intervals, use point prediction
                    # with a small buffer (±5%)
                    pred = model.predict(X)
                    lower = pred * 0.95
                    upper = pred * 1.05
                    mock_interval = np.column_stack([pred, lower, upper])
                    base_intervals.append(mock_interval)
            
            # Combine intervals with weighted averaging
            combined_lower = np.zeros_like(point_pred)
            combined_upper = np.zeros_like(point_pred)
            
            for i, model in enumerate(self.models):
                weight = self.weights[model.name]
                combined_lower += weight * base_intervals[i][:, 1]  # Lower bound
                combined_upper += weight * base_intervals[i][:, 2]  # Upper bound
                
            # Return [prediction, lower_bound, upper_bound]
            return np.column_stack([point_pred, combined_lower, combined_upper])
            
        elif ensemble_method == "stacking" and self.meta_model:
            # For stacking, use meta-model's probabilistic predictions
            # Generate meta-features first
            base_preds = [model.predict(X) for model in self.models]
            meta_features = np.column_stack(base_preds)
            
            # If including original features
            if self.config["stack_with_orig_features"]:
                if isinstance(X, pd.DataFrame):
                    X_array = X.values
                else:
                    X_array = X
                meta_features = np.hstack([meta_features, X_array])
                
            # Use meta-model's predict_proba
            return self.meta_model.predict_proba(meta_features)
            
        else:
            # For voting or other methods, convert to basic confidence interval
            point_pred = self.predict(X)
            lower_bound = point_pred * 0.95  # 5% below prediction
            upper_bound = point_pred * 1.05  # 5% above prediction
            
            return np.column_stack([point_pred, lower_bound, upper_bound])
    
    def _save_model_artifacts(self, path: str) -> None:
        """
        Save ensemble model artifacts.
        
        Args:
            path: Directory path for saving
        """
        # Save ensemble configuration
        config_file = os.path.join(path, f"{self.name}_config.json")
        with open(config_file, 'w') as f:
            # Create a copy of config to avoid modifying the original
            config_copy = self.config.copy()
            # Remove meta_model_config if present to avoid JSON serialization issues
            if 'meta_model_config' in config_copy and config_copy['meta_model_config'] is not None:
                config_copy['meta_model_config'] = dict(config_copy['meta_model_config'])
            json.dump(config_copy, f)
        
        # Save weights
        weights_file = os.path.join(path, f"{self.name}_weights.joblib")
        joblib.dump(self.weights, weights_file)
        
        # Save model names
        names_file = os.path.join(path, f"{self.name}_model_names.joblib")
        joblib.dump(self.model_names, names_file)
        
        # Save meta-model if it exists
        if self.meta_model is not None:
            meta_model_dir = os.path.join(path, "meta_model")
            os.makedirs(meta_model_dir, exist_ok=True)
            self.meta_model.save(meta_model_dir)
    
    def _load_model_artifacts(self, path: str) -> None:
        """
        Load ensemble model artifacts.
        
        Args:
            path: Directory path for loading
        """
        # Load ensemble configuration
        config_file = os.path.join(path, f"{self.name}_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        # Load weights
        weights_file = os.path.join(path, f"{self.name}_weights.joblib")
        if os.path.exists(weights_file):
            self.weights = joblib.load(weights_file)
        
        # Load model names
        names_file = os.path.join(path, f"{self.name}_model_names.joblib")
        if os.path.exists(names_file):
            self.model_names = joblib.load(names_file)
        
        # Load meta-model if it exists
        meta_model_dir = os.path.join(path, "meta_model")
        if os.path.exists(meta_model_dir):
            self.meta_model = self._init_meta_model()
            self.meta_model.load(meta_model_dir)
            
        # Note: The actual base models need to be loaded and added separately
        
    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance by combining importance from base models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Initialize combined importance
        combined_importance = {}
        
        # Collect feature importance from all models
        for model in self.models:
            # Get importance if available
            model_importance = model.feature_importance()
            
            if model_importance is not None:
                # Get model weight
                weight = self.weights[model.name]
                
                # Add weighted importance to combined importance
                for feature, importance in model_importance.items():
                    if feature in combined_importance:
                        combined_importance[feature] += importance * weight
                    else:
                        combined_importance[feature] = importance * weight
        
        # If no models provided feature importance
        if not combined_importance:
            logger.warning("No base models provided feature importance")
            return None
            
        # Normalize to sum to 1
        total_importance = sum(combined_importance.values())
        if total_importance > 0:
            for feature in combined_importance:
                combined_importance[feature] /= total_importance
                
        return combined_importance