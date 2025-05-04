"""
Tests for the BaseModel abstract class implementation.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from typing import Dict, Any
from datetime import datetime

from src.ml.base_model import BaseModel


# Create a concrete implementation of BaseModel for testing
class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing purposes."""
    
    def __init__(self, name: str = "test_model", config: Dict[str, Any] = None):
        super().__init__(name=name, model_type="test", config=config or {})
        self.trained = False
    
    def fit(self, X, y, validation_data=None):
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.model = "dummy_model"
        self.is_fitted = True
        self.trained = True
        return {"loss": 0.1, "val_loss": 0.2}
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return np.ones(len(X))
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return np.ones((len(X), 2)) * 0.5
    
    def _save_model_artifacts(self, path: str):
        with open(os.path.join(path, f"{self.name}_model.txt"), 'w') as f:
            f.write("model_data")
    
    def _load_model_artifacts(self, path: str):
        with open(os.path.join(path, f"{self.name}_model.txt"), 'r') as f:
            content = f.read()
            if content == "model_data":
                self.model = "dummy_model"
                self.is_fitted = True
    
    def feature_importance(self):
        if not self.is_fitted:
            return None
        return {f"feature_{i}": 1.0 / len(self.feature_names) for i in range(len(self.feature_names))}


class TestBaseModel:
    """Tests for the BaseModel functionality."""
    
    @pytest.fixture
    def model(self):
        """Return a concrete model implementation for testing."""
        return ConcreteModel(name="test_model", config={"param1": 10, "param2": "value"})
    
    @pytest.fixture
    def test_data(self):
        """Generate test data for model training and prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        return X, y
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.name == "test_model"
        assert model.model_type == "test"
        assert model.config["param1"] == 10
        assert model.config["param2"] == "value"
        assert model.is_fitted is False
        assert model.model is None
        assert isinstance(model.metadata, dict)
        assert "created_at" in model.metadata
        assert model.metadata["model_type"] == "test"
        assert model.metadata["name"] == "test_model"
    
    def test_fit_predict(self, model, test_data):
        """Test model fit and predict methods."""
        X, y = test_data
        
        # Test fit method
        history = model.fit(X, y)
        assert model.is_fitted is True
        assert model.model is not None
        assert history["loss"] == 0.1
        assert history["val_loss"] == 0.2
        
        # Test predict method
        predictions = model.predict(X)
        assert predictions.shape == (100,)
        assert np.all(predictions == 1.0)
        
        # Test predict_proba method
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.all(proba == 0.5)
    
    def test_predict_before_fit(self, model, test_data):
        """Test predicting before fitting raises error."""
        X, _ = test_data
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)
    
    def test_evaluate(self, model, test_data):
        """Test model evaluation."""
        X, y = test_data
        
        # First fit the model
        model.fit(X, y)
        
        # Test evaluate method
        metrics = model.evaluate(X, y)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        
        # Since we're predicting all ones and y is random, metrics won't be good
        # but we can verify they are calculated
        assert metrics["mse"] > 0
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
    
    def test_feature_importance(self, model, test_data):
        """Test feature importance calculation."""
        X, y = test_data
        
        # Without fitting, should return None
        assert model.feature_importance() is None
        
        # After fitting
        model.fit(X, y)
        importance = model.feature_importance()
        assert importance is not None
        assert len(importance) == X.shape[1]
        assert sum(importance.values()) == pytest.approx(1.0)
    
    def test_save_load(self, model, test_data, temp_model_dir):
        """Test model saving and loading."""
        X, y = test_data
        
        # Fit the model first
        model.fit(X, y)
        
        # Save the model
        model_path = model.save(temp_model_dir)
        assert os.path.exists(os.path.join(temp_model_dir, f"test_model_test_metadata.json"))
        assert os.path.exists(os.path.join(temp_model_dir, f"test_model_model.txt"))
        
        # Create a new model instance
        new_model = ConcreteModel(name="test_model")
        assert new_model.is_fitted is False
        
        # Load the model
        new_model.load(temp_model_dir)
        assert new_model.is_fitted is True
        assert new_model.metadata["name"] == "test_model"
        assert new_model.metadata["model_type"] == "test"
        
        # Check predictions match
        X_test = np.random.randn(10, 5)
        original_preds = model.predict(X_test)
        loaded_preds = new_model.predict(X_test)
        np.testing.assert_array_equal(original_preds, loaded_preds)
    
    def test_save_without_fitting(self, model, temp_model_dir):
        """Test that saving a model without fitting raises an error."""
        with pytest.raises(ValueError, match="Cannot save a model that has not been fitted"):
            model.save(temp_model_dir)
    
    def test_string_representation(self, model):
        """Test the string representation of the model."""
        assert str(model) == "test_model (test)" 