"""
LSTM model for stock price prediction implementing the BaseModel interface.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Union, Optional, Tuple, List
import os
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler

from src.ml.base_model import BaseModel

logger = logging.getLogger(__name__)

class LSTMNetwork(nn.Module):
    """
    LSTM neural network architecture for time series prediction.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(BaseModel):
    """
    LSTM model implementation for stock price prediction.
    Inherits from BaseModel interface.
    """
    
    def __init__(self, name: str = "lstm_stock_predictor", 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LSTM model with configuration.
        
        Args:
            name: Model name
            config: Model configuration containing parameters like:
                   - input_dim: Input feature dimension
                   - hidden_dim: Hidden layer dimension
                   - num_layers: Number of LSTM layers
                   - output_dim: Output dimension (usually 1 for price prediction)
                   - sequence_length: Length of input sequences
                   - learning_rate: Learning rate for optimizer
                   - epochs: Number of training epochs
                   - batch_size: Batch size for training
        """
        default_config = {
            "input_dim": 1,            # Default for single feature (price)
            "hidden_dim": 64,          # Size of hidden state
            "num_layers": 2,           # Number of LSTM layers
            "output_dim": 1,           # Predict a single value (price)
            "sequence_length": 10,     # Number of time steps to look back
            "learning_rate": 0.001,    # Adam optimizer learning rate
            "epochs": 50,              # Training epochs
            "batch_size": 32,          # Batch size for training
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        # Override defaults with provided config
        if config:
            default_config.update(config)
            
        super().__init__(name=name, model_type="lstm", config=default_config)
        
        # Initialize model components (but don't build network yet)
        self.scaler = MinMaxScaler()
        self.network = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = self.config["device"]
        
    def _build_network(self):
        """
        Build the LSTM neural network.
        """
        self.network = LSTMNetwork(
            input_dim=self.config["input_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            output_dim=self.config["output_dim"]
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.config["learning_rate"]
        )
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        seq_length = self.config["sequence_length"]
        xs, ys = [], []
        
        for i in range(len(data) - seq_length):
            xs.append(data[i:i+seq_length])
            ys.append(data[i+seq_length])
            
        return np.array(xs), np.array(ys)
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Optional[Union[np.ndarray, pd.Series]] = None,
                     is_training: bool = True) -> Tuple:
        """
        Prepare data for LSTM model (scale, create sequences, convert to tensors).
        
        Args:
            X: Input features
            y: Target values (optional)
            is_training: Whether this is for training (fit scaler) or inference
            
        Returns:
            Prepared data as tensors
        """
        # If we receive a DataFrame or Series, convert to numpy
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        elif isinstance(X, pd.Series):
            X_values = X.values.reshape(-1, 1)
        else:
            X_values = X
            
        # For univariate time series, reshape single column
        if len(X_values.shape) == 1:
            X_values = X_values.reshape(-1, 1)
            
        # Scale data
        if is_training:
            X_scaled = self.scaler.fit_transform(X_values)
        else:
            X_scaled = self.scaler.transform(X_values)
            
        # For prediction (when y is None), return the last sequence
        if y is None:
            seq_length = self.config["sequence_length"]
            if len(X_scaled) < seq_length:
                raise ValueError(f"Not enough data points. Need at least {seq_length} points.")
                
            # Get last sequence for prediction
            X_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            X_tensor = torch.tensor(X_sequence, dtype=torch.float32).to(self.device)
            return (X_tensor,)
            
        # For training, create sequences
        X_seq, y_seq = self._create_sequences(X_scaled)
        
        # Reshape for LSTM [batch, seq_len, features]
        X_seq_reshaped = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], -1)
        
        # Convert to pytorch tensors
        X_tensor = torch.tensor(X_seq_reshaped, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)
        
        if self.config["output_dim"] > 1:
            # For multivariate output
            y_tensor = y_tensor.reshape(-1, self.config["output_dim"])
        else:
            # For univariate output
            y_tensor = y_tensor.reshape(-1, 1)
            
        return X_tensor, y_tensor
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series], 
            validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Training time series data
            y: Target values (usually not needed for time series as X contains all data)
            validation_data: Optional validation data
            
        Returns:
            Training history
        """
        # Build network if not already built
        if self.network is None:
            self._build_network()
            
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y, is_training=True)
        
        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val, is_training=False)
            
        # Training history
        history = {
            "loss": [],
            "val_loss": [] if validation_data is not None else None
        }
        
        # Set to training mode
        self.network.train()
        
        # Train for specified epochs
        for epoch in range(self.config["epochs"]):
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.network(X_tensor)
            
            # Calculate loss
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Record training loss
            history["loss"].append(loss.item())
            
            # Validation if provided
            if validation_data is not None:
                with torch.no_grad():
                    self.network.eval()
                    val_outputs = self.network(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor)
                    history["val_loss"].append(val_loss.item())
                    self.network.train()
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config['epochs']}, Loss: {loss.item():.4f}")
                if validation_data is not None:
                    logger.info(f"Validation Loss: {val_loss.item():.4f}")
        
        self.is_fitted = True
        self.metadata["trained_epochs"] = self.config["epochs"]
        self.metadata["final_loss"] = history["loss"][-1]
        
        if validation_data is not None:
            self.metadata["final_val_loss"] = history["val_loss"][-1]
        
        return history
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the LSTM model.
        
        Args:
            X: Input features or time series data
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        # Set to evaluation mode
        self.network.eval()
        
        # Prepare data
        X_tensor = self._prepare_data(X, is_training=False)[0]
        
        # Make predictions
        with torch.no_grad():
            predictions_scaled = self.network(X_tensor).cpu().numpy()
        
        # Inverse transform to get original scale
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten() if predictions.shape[1] == 1 else predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        For LSTM regression model, returns the predictions with a mocked confidence interval.
        
        Args:
            X: Input features
            
        Returns:
            Array where each row contains [prediction, lower_bound, upper_bound]
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        # Get predictions
        predictions = self.predict(X)
        
        # For simplicity, create a mock confidence interval of ±5%
        # In a real implementation, this would use proper prediction intervals
        lower_bound = predictions * 0.95
        upper_bound = predictions * 1.05
        
        # Combine into array
        if predictions.ndim == 1:
            return np.vstack([predictions, lower_bound, upper_bound]).T
        else:
            # For multivariate output
            intervals = np.zeros((predictions.shape[0], predictions.shape[1], 3))
            intervals[:, :, 0] = predictions
            intervals[:, :, 1] = lower_bound
            intervals[:, :, 2] = upper_bound
            return intervals
    
    def _save_model_artifacts(self, path: str) -> None:
        """
        Save LSTM model artifacts.
        
        Args:
            path: Directory path for saving
        """
        # Save PyTorch model
        model_file = os.path.join(path, f"{self.name}_network.pt")
        torch.save(self.network.state_dict(), model_file)
        
        # Save scaler
        scaler_file = os.path.join(path, f"{self.name}_scaler.joblib")
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature names if available
        if self.feature_names:
            features_file = os.path.join(path, f"{self.name}_features.joblib")
            joblib.dump(self.feature_names, features_file)
    
    def _load_model_artifacts(self, path: str) -> None:
        """
        Load LSTM model artifacts.
        
        Args:
            path: Directory path for loading
        """
        # Build network if not already built
        if self.network is None:
            self._build_network()
            
        # Load PyTorch model
        model_file = os.path.join(path, f"{self.name}_network.pt")
        self.network.load_state_dict(torch.load(model_file, map_location=self.device))
        
        # Load scaler
        scaler_file = os.path.join(path, f"{self.name}_scaler.joblib")
        self.scaler = joblib.load(scaler_file)
        
        # Load feature names if available
        features_file = os.path.join(path, f"{self.name}_features.joblib")
        if os.path.exists(features_file):
            self.feature_names = joblib.load(features_file)
    
    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance - not directly available for LSTM.
        
        Returns:
            None as LSTM does not provide direct feature importance
        """
        logger.warning("Feature importance is not directly available for LSTM models")
        return None 