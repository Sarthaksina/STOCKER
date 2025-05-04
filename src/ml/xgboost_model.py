"""
XGBoost model for stock price prediction implementing the BaseModel interface.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Union, Optional, Tuple, List
import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler
import json

from src.ml.base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost model implementation for stock price prediction.
    Inherits from BaseModel interface.
    
    This model uses XGBoost's gradient boosting capabilities for financial time series prediction.
    """
    
    def __init__(self, name: str = "xgboost_stock_predictor", 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the XGBoost model with configuration.
        
        Args:
            name: Model name
            config: Model configuration including XGBoost parameters:
                   - objective: Objective function (reg:squarederror for regression)
                   - learning_rate: Step size shrinkage to prevent overfitting
                   - max_depth: Maximum depth of a tree
                   - n_estimators: Number of boosting rounds
                   - subsample: Subsample ratio of training instances
                   - colsample_bytree: Subsample ratio of columns for each tree
                   - early_stopping_rounds: Stop if performance doesn't improve
                   - sequence_length: Length of historical sequence for features
                   - prediction_length: Number of steps to predict ahead
        """
        default_config = {
            "objective": "reg:squarederror",    # Default regression objective
            "learning_rate": 0.01,              # Default learning rate
            "max_depth": 6,                     # Default tree depth
            "n_estimators": 1000,               # Default number of estimators
            "subsample": 0.8,                   # Default subsample ratio
            "colsample_bytree": 0.8,            # Default column sample ratio
            "early_stopping_rounds": 50,        # Early stopping to prevent overfitting
            "eval_metric": "rmse",              # Default evaluation metric
            "sequence_length": 10,              # Length of time sequence for feature engineering
            "prediction_length": 1,             # Default prediction horizon
            "verbosity": 1,                     # Verbosity of XGBoost
            "use_gpu": False                    # Enable GPU acceleration if available
        }
        
        # Override defaults with provided config
        if config:
            default_config.update(config)
            
        super().__init__(name=name, model_type="xgboost", config=default_config)
        
        # Initialize model components
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        # Create XGBoost parameters dictionary
        self.xgb_params = {k: v for k, v in self.config.items() 
                          if k not in ["sequence_length", "prediction_length", "use_gpu"]}
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features for XGBoost from the input data.
        
        Args:
            data: Input time series data
            
        Returns:
            DataFrame with engineered features
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                # Convert 1D array to DataFrame with single column
                data = pd.DataFrame(data, columns=["price"])
            else:
                # Convert 2D array to DataFrame with generic column names
                data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # If we're starting with just price data, create a price column if not present
        if len(df.columns) == 1 and 'price' not in df.columns:
            df.columns = ['price']
        
        seq_length = self.config["sequence_length"]
        
        # Create lagged features for all columns
        for col in df.columns:
            for lag in range(1, seq_length + 1):
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        # Create rolling statistics (if we have enough data)
        if len(df) >= seq_length:
            for col in df.columns[:len(data.columns)]:  # Only use original columns
                # Rolling mean
                df[f"{col}_rolling_mean_{seq_length}"] = df[col].rolling(window=seq_length).mean()
                # Rolling standard deviation
                df[f"{col}_rolling_std_{seq_length}"] = df[col].rolling(window=seq_length).std()
                # Rolling min/max
                df[f"{col}_rolling_min_{seq_length}"] = df[col].rolling(window=seq_length).min()
                df[f"{col}_rolling_max_{seq_length}"] = df[col].rolling(window=seq_length).max()
        
        # Create percent changes
        for col in df.columns[:len(data.columns)]:  # Only use original columns
            df[f"{col}_pct_change"] = df[col].pct_change()
            df[f"{col}_pct_change_abs"] = np.abs(df[col].pct_change())
        
        # Create targets (future values) based on prediction_length
        pred_length = self.config["prediction_length"]
        for col in data.columns:
            df[f"{col}_target_{pred_length}"] = df[col].shift(-pred_length)
        
        # Drop rows with NaN values (due to lagging/shifting)
        df = df.dropna()
        
        return df
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Optional[Union[np.ndarray, pd.Series]] = None,
                      is_training: bool = True) -> Tuple:
        """
        Prepare data for XGBoost model (create features, scale).
        
        Args:
            X: Input features
            y: Target values (optional)
            is_training: Whether this is for training (fit scaler) or inference
            
        Returns:
            Prepared data
        """
        # If y is provided separately, we'll use it instead of creating targets
        if y is not None:
            # Generate features from X
            X_featured = self._create_features(X)
            
            # Remove target columns (they'll be replaced by y)
            cols_to_drop = [col for col in X_featured.columns if '_target_' in col]
            X_featured = X_featured.drop(columns=cols_to_drop)
            
            # Convert y to DataFrame if it's a Series
            if isinstance(y, pd.Series):
                y_df = pd.DataFrame(y)
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:
                    y_df = pd.DataFrame(y, columns=['target'])
                else:
                    y_df = pd.DataFrame(y, columns=[f'target_{i}' for i in range(y.shape[1])])
            else:
                y_df = y
                
            # Store feature names before scaling
            self.feature_names = X_featured.columns.tolist()
            
            # Scale features if we have a DataFrame
            if isinstance(X_featured, pd.DataFrame):
                if is_training:
                    X_scaled = pd.DataFrame(
                        self.scaler.fit_transform(X_featured),
                        columns=X_featured.columns
                    )
                else:
                    X_scaled = pd.DataFrame(
                        self.scaler.transform(X_featured),
                        columns=X_featured.columns
                    )
            else:
                if is_training:
                    X_scaled = self.scaler.fit_transform(X_featured)
                else:
                    X_scaled = self.scaler.transform(X_featured)
            
            return X_scaled, y_df
        
        else:
            # For prediction or if targets are included in X
            # Create features
            data_featured = self._create_features(X)
            
            # Separate features and target
            target_cols = [col for col in data_featured.columns if '_target_' in col]
            
            if not target_cols and not is_training:
                # For prediction (no targets needed)
                # Store feature names before scaling
                self.feature_names = data_featured.columns.tolist()
                
                # Scale features
                if isinstance(data_featured, pd.DataFrame):
                    X_scaled = pd.DataFrame(
                        self.scaler.transform(data_featured),
                        columns=data_featured.columns
                    )
                else:
                    X_scaled = self.scaler.transform(data_featured)
                    
                return X_scaled, None
            
            if not target_cols:
                raise ValueError("No target columns found. For training, X must contain data with sufficient "
                                "points to create target shifts or y must be provided separately.")
            
            # Select just the last target column
            target_col = target_cols[-1]  # Use the last target (furthest in future)
            
            # Split into features and target
            y_df = data_featured[target_col]
            X_df = data_featured.drop(columns=target_cols)
            
            # Store feature names before scaling
            self.feature_names = X_df.columns.tolist()
            
            # Scale features
            if is_training:
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X_df),
                    columns=X_df.columns
                )
            else:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X_df),
                    columns=X_df.columns
                )
            
            return X_scaled, y_df
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[Union[np.ndarray, pd.Series]] = None,
            validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X: Training data
            y: Target values (optional if X contains sufficient historical data)
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Training history
        """
        # Prepare training data
        X_train, y_train = self._prepare_data(X, y, is_training=True)
        
        # Handle validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, is_training=False)
            eval_set = [(X_val_prep, y_val_prep)]
        else:
            eval_set = None
        
        # Convert to DMatrix for faster processing
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if eval_set:
            deval = xgb.DMatrix(X_val_prep, label=y_val_prep)
            watchlist = [(dtrain, 'train'), (deval, 'eval')]
        else:
            watchlist = [(dtrain, 'train')]
        
        # Configure GPU if requested and available
        if self.config.get("use_gpu", False):
            self.xgb_params["tree_method"] = "gpu_hist"
            self.xgb_params["gpu_id"] = 0
        
        # Train the model
        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.config["n_estimators"],
            evals=watchlist,
            early_stopping_rounds=self.config.get("early_stopping_rounds"),
            verbose_eval=self.config.get("verbosity", 1)
        )
        
        # Update metadata
        self.is_fitted = True
        best_iteration = getattr(self.model, "best_iteration", self.config["n_estimators"])
        self.metadata.update({
            "best_iteration": best_iteration,
            "feature_importance": {k: v for k, v in zip(
                self.feature_names, 
                self.model.get_score(importance_type='gain').values()
            )}
        })
        
        # Create a history dict similar to Keras models
        history = {
            "loss": [],
            "val_loss": [] if validation_data is not None else None
        }
        
        # Extract metrics from XGBoost model
        if hasattr(self.model, "eval_result"):
            for i in range(len(self.model.eval_result.get("train", []))):
                history["loss"].append(self.model.eval_result["train"][i])
                if validation_data is not None:
                    history["val_loss"].append(self.model.eval_result["eval"][i])
        
        return history
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the XGBoost model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Prepare data
        X_prep, _ = self._prepare_data(X, is_training=False)
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X_prep)
        
        # Make predictions
        predictions = self.model.predict(dtest)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        For XGBoost regression, returns the predictions with prediction intervals.
        For classification, returns class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Array with predictions and uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Prepare data
        X_prep, _ = self._prepare_data(X, is_training=False)
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X_prep)
        
        # Get predictions
        predictions = self.model.predict(dtest)
        
        # Check if this is a classification model (has num_class > 0)
        if hasattr(self.model, 'attr') and 'num_class' in self.model.attr:
            # For classification, return raw probabilities
            return predictions
        else:
            # For regression, use XGBoost's built-in quantile regression capability
            # We'd need to train 3 separate models for this to be accurate
            # Here we'll approximate with a simple percentile approach
            
            # Get leaf indices for each sample
            leaf_indices = self.model.predict(dtest, pred_leaf=True)
            
            # Group predictions by leaf indices
            leaf_to_preds = {}
            for i, leaves in enumerate(leaf_indices):
                leaves_tuple = tuple(leaves.tolist())
                if leaves_tuple not in leaf_to_preds:
                    leaf_to_preds[leaves_tuple] = []
                leaf_to_preds[leaves_tuple].append(predictions[i])
                
            # Calculate intervals for each sample
            intervals = np.zeros((len(predictions), 3))
            for i, leaves in enumerate(leaf_indices):
                leaves_tuple = tuple(leaves.tolist())
                leaf_preds = leaf_to_preds[leaves_tuple]
                if len(leaf_preds) >= 5:  # Only use percentiles with enough samples
                    intervals[i, 0] = predictions[i]  # Mean prediction
                    intervals[i, 1] = np.percentile(leaf_preds, 5)  # Lower bound
                    intervals[i, 2] = np.percentile(leaf_preds, 95)  # Upper bound
                else:
                    # Not enough samples for this leaf, use simple percentage
                    intervals[i, 0] = predictions[i]
                    intervals[i, 1] = predictions[i] * 0.9  # Lower bound
                    intervals[i, 2] = predictions[i] * 1.1  # Upper bound
                    
            return intervals
    
    def _save_model_artifacts(self, path: str) -> None:
        """
        Save XGBoost model artifacts.
        
        Args:
            path: Directory path for saving
        """
        # Save XGBoost model
        model_file = os.path.join(path, f"{self.name}_xgb_model.json")
        self.model.save_model(model_file)
        
        # Save scaler
        scaler_file = os.path.join(path, f"{self.name}_scaler.joblib")
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature names
        if self.feature_names:
            features_file = os.path.join(path, f"{self.name}_features.json")
            with open(features_file, 'w') as f:
                json.dump(self.feature_names, f)
    
    def _load_model_artifacts(self, path: str) -> None:
        """
        Load XGBoost model artifacts.
        
        Args:
            path: Directory path for loading
        """
        # Load XGBoost model
        model_file = os.path.join(path, f"{self.name}_xgb_model.json")
        self.model = xgb.Booster()
        self.model.load_model(model_file)
        
        # Load scaler
        scaler_file = os.path.join(path, f"{self.name}_scaler.joblib")
        self.scaler = joblib.load(scaler_file)
        
        # Load feature names
        features_file = os.path.join(path, f"{self.name}_features.json")
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        # Get feature importance scores
        importance_gain = self.model.get_score(importance_type='gain')
        
        # Map scores to feature names
        feature_scores = {}
        for feature, score in importance_gain.items():
            feature_id = int(feature.replace('f', '')) if feature.startswith('f') else feature
            if isinstance(feature_id, int):
                # XGBoost sometimes uses f0, f1, etc. for feature indices
                if feature_id < len(self.feature_names):
                    feature_name = self.feature_names[feature_id]
                    feature_scores[feature_name] = score
            else:
                # XGBoost is using the provided feature names
                feature_scores[feature] = score
        
        return feature_scores 