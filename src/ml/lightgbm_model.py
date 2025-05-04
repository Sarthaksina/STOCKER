"""
LightGBM model for stock price prediction implementing the BaseModel interface.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Union, Optional, Tuple, List
import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler
import json

from src.ml.base_model import BaseModel

logger = logging.getLogger(__name__)

class LightGBMModel(BaseModel):
    """
    LightGBM model implementation for stock price prediction.
    Inherits from BaseModel interface.
    
    LightGBM offers advantages for financial data with its leaf-wise tree growth and
    efficient handling of large feature spaces common in time series features.
    """
    
    def __init__(self, name: str = "lightgbm_stock_predictor", 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LightGBM model with configuration.
        
        Args:
            name: Model name
            config: Model configuration including LightGBM parameters:
                   - objective: Objective function (regression or binary)
                   - boosting_type: Boosting type (gbdt, dart, goss)
                   - learning_rate: Learning rate for boosting
                   - num_leaves: Max number of leaves in one tree
                   - max_depth: Max tree depth
                   - n_estimators: Number of boosting iterations
                   - subsample: Subsample ratio of training instances
                   - subsample_freq: Frequency for bagging
                   - colsample_bytree: Feature fraction
                   - min_child_samples: Min data in one leaf
                   - early_stopping_rounds: Early stopping rounds
                   - sequence_length: Historical time steps for feature creation
                   - prediction_length: Steps ahead to predict
                   - categorical_features: List of categorical feature names/indices
        """
        default_config = {
            "objective": "regression",         # Default objective for stock price prediction
            "boosting_type": "gbdt",           # Default boosting type
            "learning_rate": 0.01,             # Default learning rate
            "num_leaves": 31,                  # Default number of leaves
            "max_depth": -1,                   # Unlimited depth by default
            "n_estimators": 1000,              # Default number of estimators
            "subsample": 0.8,                  # Subsample ratio
            "subsample_freq": 1,               # Frequency for bagging
            "colsample_bytree": 0.8,           # Feature fraction
            "min_child_samples": 20,           # Min data in leaf
            "reg_alpha": 0.0,                  # L1 regularization
            "reg_lambda": 0.0,                 # L2 regularization
            "early_stopping_rounds": 50,       # Early stopping rounds
            "metric": "rmse",                  # Default evaluation metric
            "sequence_length": 10,             # Historical window size
            "prediction_length": 1,            # Steps ahead to predict
            "categorical_features": None,      # Categorical features
            "use_gpu": False,                  # GPU acceleration
            "verbose": 1                       # Verbosity
        }
        
        # Override defaults with provided config
        if config:
            default_config.update(config)
            
        super().__init__(name=name, model_type="lightgbm", config=default_config)
        
        # Initialize model components
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        # Create LightGBM parameters dictionary (excluding sequence_length and prediction_length)
        self.lgb_params = {k: v for k, v in self.config.items() 
                           if k not in ["sequence_length", "prediction_length", "categorical_features"]}
        
        # Set GPU if requested
        if self.config.get("use_gpu", False):
            self.lgb_params["device"] = "gpu"
            self.lgb_params["gpu_platform_id"] = 0
            self.lgb_params["gpu_device_id"] = 0
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features for LightGBM from the input data.
        This creates lagged features, rolling statistics, and technical indicators.
        
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
        
        # Create lagged features for all numeric columns
        for col in df.select_dtypes(include=np.number).columns:
            for lag in range(1, seq_length + 1):
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        # Create rolling statistics
        if len(df) >= seq_length:
            for col in df.select_dtypes(include=np.number).columns[:len(data.columns)]:
                # Rolling mean
                df[f"{col}_rolling_mean_{seq_length}"] = df[col].rolling(window=seq_length).mean()
                # Rolling standard deviation
                df[f"{col}_rolling_std_{seq_length}"] = df[col].rolling(window=seq_length).std()
                # Rolling min/max
                df[f"{col}_rolling_min_{seq_length}"] = df[col].rolling(window=seq_length).min()
                df[f"{col}_rolling_max_{seq_length}"] = df[col].rolling(window=seq_length).max()
                # Rolling median (more robust to outliers)
                df[f"{col}_rolling_median_{seq_length}"] = df[col].rolling(window=seq_length).median()
                
                # Exponential moving average (faster response to recent changes)
                df[f"{col}_ema_{seq_length}"] = df[col].ewm(span=seq_length).mean()
                
                # Rate of change (momentum indicator)
                df[f"{col}_roc_{seq_length}"] = (df[col] / df[col].shift(seq_length) - 1) * 100
        
        # Create percent changes for all numeric columns
        for col in df.select_dtypes(include=np.number).columns[:len(data.columns)]:
            df[f"{col}_pct_change_1"] = df[col].pct_change(1)
            df[f"{col}_pct_change_5"] = df[col].pct_change(5) if len(df) > 5 else np.nan
            
            # Acceleration (change of change)
            df[f"{col}_acceleration"] = df[col].pct_change(1).pct_change(1)
            
            # Absolute changes (volatility indicators)
            df[f"{col}_abs_change"] = df[col].diff().abs()
        
        # Add day of week, month features if the index is a datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['day_of_month'] = df.index.day
            df['quarter'] = df.index.quarter
        
        # Create target column (future value) based on prediction_length
        pred_length = self.config["prediction_length"]
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_target_{pred_length}"] = df[col].shift(-pred_length)
        
        # Drop rows with NaN values (due to lagging/shifting)
        df = df.dropna()
        
        return df
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Optional[Union[np.ndarray, pd.Series]] = None,
                      is_training: bool = True) -> Tuple:
        """
        Prepare data for LightGBM model (create features, scale).
        
        Args:
            X: Input features
            y: Target values (optional)
            is_training: Whether this is for training (fit scaler) or inference
            
        Returns:
            Prepared data
        """
        # Handle case where y is provided separately
        if y is not None:
            # Generate features from X
            X_featured = self._create_features(X)
            
            # Remove target columns (they'll be replaced by y)
            cols_to_drop = [col for col in X_featured.columns if '_target_' in col]
            X_featured = X_featured.drop(columns=cols_to_drop)
            
            # Convert y to DataFrame if it's a Series or array
            if isinstance(y, pd.Series):
                y_df = pd.DataFrame(y)
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:
                    y_df = pd.DataFrame(y, columns=['target'])
                else:
                    y_df = pd.DataFrame(y, columns=[f'target_{i}' for i in range(y.shape[1])])
            else:
                y_df = y
                
            # Store feature names for later use
            self.feature_names = X_featured.columns.tolist()
            
            # Get categorical features if specified
            cat_features = self.config.get("categorical_features")
            if cat_features is not None:
                # If categorical features are specified by name, convert to indices
                if isinstance(cat_features[0], str):
                    cat_indices = [X_featured.columns.get_loc(col) for col in cat_features if col in X_featured.columns]
                    self.categorical_feature_indices = cat_indices
                else:
                    self.categorical_feature_indices = cat_features
                    
                # Exclude categorical features from scaling
                numeric_features = [col for i, col in enumerate(X_featured.columns) 
                                   if i not in self.categorical_feature_indices]
                X_numeric = X_featured[numeric_features].copy()
                X_categorical = X_featured[[col for i, col in enumerate(X_featured.columns) 
                                         if i in self.categorical_feature_indices]].copy()
                
                # Scale only numeric features
                if is_training:
                    X_numeric_scaled = pd.DataFrame(
                        self.scaler.fit_transform(X_numeric),
                        columns=numeric_features
                    )
                else:
                    X_numeric_scaled = pd.DataFrame(
                        self.scaler.transform(X_numeric),
                        columns=numeric_features
                    )
                
                # Combine back with categorical features
                X_scaled = pd.concat([X_numeric_scaled, X_categorical], axis=1)
                # Reorder columns to match original
                X_scaled = X_scaled[X_featured.columns]
            else:
                # Scale all features if no categorical features
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
            
            return X_scaled, y_df
        
        else:
            # For prediction or if targets are included in X
            # Create features
            data_featured = self._create_features(X)
            
            # Separate features and target
            target_cols = [col for col in data_featured.columns if '_target_' in col]
            
            if not target_cols and not is_training:
                # For prediction (no targets needed)
                # Store feature names for later use
                self.feature_names = data_featured.columns.tolist()
                
                # Handle categorical features
                cat_features = self.config.get("categorical_features")
                if cat_features is not None:
                    # If categorical features are specified by name, convert to indices
                    if isinstance(cat_features[0], str):
                        cat_indices = [data_featured.columns.get_loc(col) for col in cat_features if col in data_featured.columns]
                        self.categorical_feature_indices = cat_indices
                    else:
                        self.categorical_feature_indices = cat_features
                        
                    # Exclude categorical features from scaling
                    numeric_features = [col for i, col in enumerate(data_featured.columns) 
                                       if i not in self.categorical_feature_indices]
                    X_numeric = data_featured[numeric_features].copy()
                    X_categorical = data_featured[[col for i, col in enumerate(data_featured.columns) 
                                             if i in self.categorical_feature_indices]].copy()
                    
                    # Scale only numeric features
                    X_numeric_scaled = pd.DataFrame(
                        self.scaler.transform(X_numeric),
                        columns=numeric_features
                    )
                    
                    # Combine back with categorical features
                    X_scaled = pd.concat([X_numeric_scaled, X_categorical], axis=1)
                    # Reorder columns to match original
                    X_scaled = X_scaled[data_featured.columns]
                else:
                    # Scale all features if no categorical features
                    X_scaled = pd.DataFrame(
                        self.scaler.transform(data_featured),
                        columns=data_featured.columns
                    )
                    
                return X_scaled, None
            
            if not target_cols:
                raise ValueError("No target columns found. For training, X must contain data with sufficient "
                                "points to create target shifts or y must be provided separately.")
            
            # Select the last target column for the latest prediction
            target_col = target_cols[-1]  # Use the last target (furthest in future)
            
            # Split into features and target
            y_df = data_featured[target_col]
            X_df = data_featured.drop(columns=target_cols)
            
            # Store feature names for later use
            self.feature_names = X_df.columns.tolist()
            
            # Handle categorical features
            cat_features = self.config.get("categorical_features")
            if cat_features is not None:
                # If categorical features are specified by name, convert to indices
                if isinstance(cat_features[0], str):
                    cat_indices = [X_df.columns.get_loc(col) for col in cat_features if col in X_df.columns]
                    self.categorical_feature_indices = cat_indices
                else:
                    self.categorical_feature_indices = cat_features
                    
                # Exclude categorical features from scaling
                numeric_features = [col for i, col in enumerate(X_df.columns) 
                                   if i not in self.categorical_feature_indices]
                X_numeric = X_df[numeric_features].copy()
                X_categorical = X_df[[col for i, col in enumerate(X_df.columns) 
                                     if i in self.categorical_feature_indices]].copy()
                
                # Scale only numeric features
                if is_training:
                    X_numeric_scaled = pd.DataFrame(
                        self.scaler.fit_transform(X_numeric),
                        columns=numeric_features
                    )
                else:
                    X_numeric_scaled = pd.DataFrame(
                        self.scaler.transform(X_numeric),
                        columns=numeric_features
                    )
                
                # Combine back with categorical features
                X_scaled = pd.concat([X_numeric_scaled, X_categorical], axis=1)
                # Reorder columns to match original
                X_scaled = X_scaled[X_df.columns]
            else:
                # Scale all features if no categorical features
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
        Train the LightGBM model.
        
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
        
        # Create LightGBM dataset
        categorical_feature = getattr(self, 'categorical_feature_indices', None)
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            feature_name=self.feature_names,
            categorical_feature=categorical_feature
        )
        
        # Create validation set if provided
        if eval_set:
            valid_data = lgb.Dataset(
                X_val_prep, 
                label=y_val_prep,
                feature_name=self.feature_names,
                categorical_feature=categorical_feature,
                reference=train_data
            )
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # Train model
        callbacks = []
        if self.config["verbose"] > 0:
            callbacks.append(lgb.log_evaluation(self.config["verbose"]))
        
        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_data,
            num_boost_round=self.config["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            early_stopping_rounds=self.config.get("early_stopping_rounds"),
            categorical_feature=categorical_feature
        )
        
        # Update metadata
        self.is_fitted = True
        best_iteration = getattr(self.model, "best_iteration", self.config["n_estimators"])
        
        # Get feature importances
        importances = {}
        if self.feature_names:
            importances = dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain')
            ))
        
        self.metadata.update({
            "best_iteration": best_iteration,
            "feature_importance": importances
        })
        
        # Create a history dict similar to Keras models
        history = {
            "loss": self.model.best_score.get('train', {}).get(self.config["metric"], []),
            "val_loss": self.model.best_score.get('valid', {}).get(self.config["metric"], []) if validation_data else None
        }
        
        return history
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the LightGBM model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Prepare data
        X_prep, _ = self._prepare_data(X, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X_prep)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        For LightGBM regression, returns the predictions with uncertainty estimates.
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
        
        # Check if this is a classification model
        if self.config.get("objective", "").startswith("binary") or self.config.get("objective", "").startswith("multi"):
            # For classification, return raw probabilities
            return self.model.predict(X_prep, raw_score=False)
        else:
            # For regression, use prediction intervals
            # Get predictions
            predictions = self.model.predict(X_prep)
            
            # Get prediction standard deviation based on tree variance
            pred_stddev = np.zeros_like(predictions)
            n_trees = self.model.num_trees()
            
            # Collect predictions from each tree for variance calculation
            tree_preds = np.zeros((X_prep.shape[0], n_trees))
            for i in range(n_trees):
                tree_preds[:, i] = self.model.predict(X_prep, start_iteration=i, num_iteration=1)
            
            # Calculate standard deviation across trees
            pred_stddev = np.std(tree_preds, axis=1)
            
            # Create prediction intervals (mean ± 2 std deviations for 95% interval)
            intervals = np.zeros((len(predictions), 3))
            intervals[:, 0] = predictions  # Mean prediction
            intervals[:, 1] = predictions - 2 * pred_stddev  # Lower bound
            intervals[:, 2] = predictions + 2 * pred_stddev  # Upper bound
            
            return intervals
    
    def _save_model_artifacts(self, path: str) -> None:
        """
        Save LightGBM model artifacts.
        
        Args:
            path: Directory path for saving
        """
        # Save LightGBM model
        model_file = os.path.join(path, f"{self.name}_lgbm_model.txt")
        self.model.save_model(model_file)
        
        # Save scaler
        scaler_file = os.path.join(path, f"{self.name}_scaler.joblib")
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature names
        if self.feature_names:
            features_file = os.path.join(path, f"{self.name}_features.json")
            with open(features_file, 'w') as f:
                json.dump(self.feature_names, f)
        
        # Save categorical features if any
        if hasattr(self, 'categorical_feature_indices') and self.categorical_feature_indices:
            cat_file = os.path.join(path, f"{self.name}_categorical_features.json")
            with open(cat_file, 'w') as f:
                json.dump(self.categorical_feature_indices, f)
    
    def _load_model_artifacts(self, path: str) -> None:
        """
        Load LightGBM model artifacts.
        
        Args:
            path: Directory path for loading
        """
        # Load LightGBM model
        model_file = os.path.join(path, f"{self.name}_lgbm_model.txt")
        self.model = lgb.Booster(model_file=model_file)
        
        # Load scaler
        scaler_file = os.path.join(path, f"{self.name}_scaler.joblib")
        self.scaler = joblib.load(scaler_file)
        
        # Load feature names
        features_file = os.path.join(path, f"{self.name}_features.json")
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)
                
        # Load categorical features if any
        cat_file = os.path.join(path, f"{self.name}_categorical_features.json")
        if os.path.exists(cat_file):
            with open(cat_file, 'r') as f:
                self.categorical_feature_indices = json.load(f)
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        # Get raw feature importance scores
        importance = self.model.feature_importance(importance_type='gain')
        
        # Map to feature names
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)} 