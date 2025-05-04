"""
Tests for the ensemble model implementation.
"""
import pytest
import numpy as np
import pandas as pd
import os
import shutil
import tempfile
from typing import List, Dict, Tuple

from src.ml import LSTMModel, XGBoostModel, LightGBMModel, EnsembleModel

# Test data generation functions
def generate_synthetic_price_data(
    n_samples: int = 200, 
    freq: str = '1D', 
    trend: float = 0.1, 
    noise: float = 1.0,
    start_price: float = 100.0
) -> pd.DataFrame:
    """
    Generate synthetic price data for testing.
    
    Args:
        n_samples: Number of samples
        freq: Time frequency
        trend: Trend factor
        noise: Noise factor
        start_price: Starting price
        
    Returns:
        DataFrame with synthetic price data
    """
    # Generate date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq=freq)
    
    # Generate price data
    prices = np.zeros(n_samples)
    prices[0] = start_price
    
    # Add trend and noise
    for i in range(1, n_samples):
        # Previous price + trend + random noise
        prices[i] = prices[i-1] * (1 + trend/100) + np.random.normal(0, noise)
        
        # Ensure prices are positive
        if prices[i] <= 0:
            prices[i] = prices[i-1] * 0.9  # 10% drop if would go negative
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    return df

def create_test_models(config_override: Dict = None) -> Tuple[List, Dict]:
    """
    Create test models for ensemble.
    
    Args:
        config_override: Override default configs
        
    Returns:
        Tuple of (list of models, model configs)
    """
    # Default configs
    lstm_config = {
        "input_dim": 1,
        "hidden_dim": 32,
        "num_layers": 1,
        "sequence_length": 5,
        "epochs": 5,
        "learning_rate": 0.01
    }
    
    xgb_config = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 50,
        "sequence_length": 5,
        "prediction_length": 1
    }
    
    lgb_config = {
        "objective": "regression",
        "learning_rate": 0.1,
        "num_leaves": 7,
        "max_depth": 3,
        "n_estimators": 50,
        "sequence_length": 5,
        "prediction_length": 1
    }
    
    # Override with provided configs if any
    if config_override:
        if 'lstm' in config_override:
            lstm_config.update(config_override['lstm'])
        if 'xgboost' in config_override:
            xgb_config.update(config_override['xgboost'])
        if 'lightgbm' in config_override:
            lgb_config.update(config_override['lightgbm'])
    
    # Create models
    lstm_model = LSTMModel(name="test_lstm", config=lstm_config)
    xgb_model = XGBoostModel(name="test_xgb", config=xgb_config)
    lgb_model = LightGBMModel(name="test_lgb", config=lgb_config)
    
    models = [lstm_model, xgb_model, lgb_model]
    configs = {
        'lstm': lstm_config,
        'xgboost': xgb_config,
        'lightgbm': lgb_config
    }
    
    return models, configs

class TestEnsembleModel:
    """Tests for the EnsembleModel class."""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data."""
        data = generate_synthetic_price_data(n_samples=100, noise=2.0)
        
        # Split into train and test
        train_data = data.iloc[:80]
        test_data = data.iloc[80:]
        
        return {
            'train': train_data['price'].values,
            'test': test_data['price'].values
        }
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_ensemble_initialization(self):
        """Test initialization of ensemble model."""
        # Create ensemble without models
        ensemble = EnsembleModel(name="test_ensemble")
        assert ensemble.name == "test_ensemble"
        assert ensemble.model_type == "ensemble"
        assert len(ensemble.models) == 0
        
        # Create models
        models, _ = create_test_models()
        
        # Create ensemble with models
        ensemble = EnsembleModel(name="test_ensemble", models=models)
        assert len(ensemble.models) == 3
        assert len(ensemble.model_names) == 3
        assert len(ensemble.weights) == 3
        
        # Check equal weights
        for weight in ensemble.weights.values():
            assert weight == pytest.approx(1/3)
    
    def test_add_remove_models(self):
        """Test adding and removing models."""
        # Create ensemble
        ensemble = EnsembleModel(name="test_ensemble")
        assert len(ensemble.models) == 0
        
        # Create models
        models, _ = create_test_models()
        
        # Add models one by one
        ensemble.add_model(models[0])
        assert len(ensemble.models) == 1
        assert ensemble.model_names[0] == "test_lstm"
        assert ensemble.weights["test_lstm"] == 1.0
        
        ensemble.add_model(models[1])
        assert len(ensemble.models) == 2
        assert ensemble.weights["test_lstm"] == pytest.approx(0.5)
        assert ensemble.weights["test_xgb"] == pytest.approx(0.5)
        
        ensemble.add_model(models[2], weight=2.0)
        assert len(ensemble.models) == 3
        # Weights should be normalized
        total_weight = ensemble.weights["test_lstm"] + ensemble.weights["test_xgb"] + ensemble.weights["test_lgb"]
        assert total_weight == pytest.approx(1.0)
        assert ensemble.weights["test_lgb"] > ensemble.weights["test_lstm"]
        
        # Remove a model
        ensemble.remove_model("test_xgb")
        assert len(ensemble.models) == 2
        assert "test_xgb" not in ensemble.model_names
        assert "test_xgb" not in ensemble.weights
        
        # Check weights are renormalized
        total_weight = ensemble.weights["test_lstm"] + ensemble.weights["test_lgb"]
        assert total_weight == pytest.approx(1.0)
    
    def test_set_weights(self):
        """Test setting custom weights."""
        # Create ensemble with models
        models, _ = create_test_models()
        ensemble = EnsembleModel(name="test_ensemble", models=models)
        
        # Set custom weights
        ensemble.set_weights({
            "test_lstm": 0.5,
            "test_xgb": 0.3,
            "test_lgb": 0.2
        })
        
        assert ensemble.weights["test_lstm"] == 0.5
        assert ensemble.weights["test_xgb"] == 0.3
        assert ensemble.weights["test_lgb"] == 0.2
        
        # Test normalization
        ensemble.set_weights({
            "test_lstm": 5,
            "test_xgb": 3,
            "test_lgb": 2
        })
        
        assert ensemble.weights["test_lstm"] == 0.5
        assert ensemble.weights["test_xgb"] == 0.3
        assert ensemble.weights["test_lgb"] == 0.2
        
        # Test invalid weights
        with pytest.raises(ValueError):
            ensemble.set_weights({
                "test_lstm": 1,
                "invalid_model": 1
            })
    
    def test_weighted_avg_ensemble(self, test_data):
        """Test weighted average ensemble."""
        # Create models
        models, _ = create_test_models()
        
        # Train individual models
        for model in models:
            model.fit(test_data['train'], test_data['train'])
        
        # Create ensemble
        ensemble_config = {
            "ensemble_method": "weighted_avg",
            "weights": {
                "test_lstm": 0.5,
                "test_xgb": 0.3,
                "test_lgb": 0.2
            }
        }
        ensemble = EnsembleModel(name="test_ensemble", models=models, config=ensemble_config)
        
        # Make predictions
        preds_lstm = models[0].predict(test_data['test'])
        preds_xgb = models[1].predict(test_data['test'])
        preds_lgb = models[2].predict(test_data['test'])
        
        preds_ensemble = ensemble.predict(test_data['test'])
        
        # Calculate expected weighted average
        expected_preds = 0.5 * preds_lstm + 0.3 * preds_xgb + 0.2 * preds_lgb
        
        # Check if predictions match
        assert np.allclose(preds_ensemble, expected_preds, rtol=1e-5)
    
    def test_voting_ensemble(self, test_data):
        """Test voting ensemble."""
        # Create models
        models, _ = create_test_models()
        
        # Train individual models
        for model in models:
            model.fit(test_data['train'], test_data['train'])
        
        # Create directional voting ensemble
        ensemble_config = {
            "ensemble_method": "voting",
            "directional_voting": True,
            "weights": {
                "test_lstm": 0.5,
                "test_xgb": 0.3,
                "test_lgb": 0.2
            }
        }
        ensemble = EnsembleModel(name="test_ensemble", models=models, config=ensemble_config)
        
        # Make predictions
        predictions = ensemble.predict(test_data['test'])
        
        # For directional voting, results should be -1, 0, or 1
        assert set(np.unique(predictions)).issubset({-1, 0, 1})
    
    def test_stacking_ensemble(self, test_data):
        """Test stacking ensemble."""
        # Create models with smaller configs for faster tests
        config_override = {
            'lstm': {'epochs': 2},
            'xgboost': {'n_estimators': 10},
            'lightgbm': {'n_estimators': 10}
        }
        models, _ = create_test_models(config_override)
        
        # Create stacking ensemble
        ensemble_config = {
            "ensemble_method": "stacking",
            "meta_model_type": "lightgbm",
            "meta_model_config": {
                "n_estimators": 10,
                "learning_rate": 0.1
            },
            "cv_folds": 2
        }
        ensemble = EnsembleModel(name="test_ensemble", models=models, config=ensemble_config)
        
        # Train ensemble
        ensemble.fit(test_data['train'], test_data['train'])
        
        # Check if meta-model was created
        assert ensemble.meta_model is not None
        
        # Make predictions
        predictions = ensemble.predict(test_data['test'])
        
        # Check if predictions have the right shape
        assert predictions.shape == test_data['test'].shape
    
    def test_save_load(self, test_data, temp_model_dir):
        """Test saving and loading ensemble model."""
        # Create and train models
        models, _ = create_test_models()
        for model in models:
            model.fit(test_data['train'], test_data['train'])
        
        # Create and train ensemble
        ensemble = EnsembleModel(name="test_ensemble", models=models)
        
        # Save models and ensemble
        model_dir = os.path.join(temp_model_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        for model in models:
            model.save(os.path.join(model_dir, model.name))
        
        ensemble.save(os.path.join(model_dir, "ensemble"))
        
        # Make predictions before loading
        preds_before = ensemble.predict(test_data['test'])
        
        # Create a new ensemble
        new_ensemble = EnsembleModel(name="test_ensemble")
        
        # Load ensemble configuration
        new_ensemble.load(os.path.join(model_dir, "ensemble"))
        
        # Load models
        lstm_model = LSTMModel(name="test_lstm")
        lstm_model.load(os.path.join(model_dir, "test_lstm"))
        
        xgb_model = XGBoostModel(name="test_xgb")
        xgb_model.load(os.path.join(model_dir, "test_xgb"))
        
        lgb_model = LightGBMModel(name="test_lgb")
        lgb_model.load(os.path.join(model_dir, "test_lgb"))
        
        # Add models to ensemble
        new_ensemble.add_model(lstm_model)
        new_ensemble.add_model(xgb_model)
        new_ensemble.add_model(lgb_model)
        
        # Make predictions after loading
        preds_after = new_ensemble.predict(test_data['test'])
        
        # Check if predictions match
        assert np.allclose(preds_before, preds_after, rtol=1e-5)
    
    def test_feature_importance(self, test_data):
        """Test feature importance."""
        # Create models
        models, _ = create_test_models()
        
        # Train individual models
        for model in models:
            model.fit(test_data['train'], test_data['train'])
        
        # Get individual feature importance
        # LSTM models don't have feature importance
        xgb_importance = models[1].feature_importance()
        lgb_importance = models[2].feature_importance()
        
        # Check if feature importance is not None for tree-based models
        assert xgb_importance is not None
        assert lgb_importance is not None
        
        # Create ensemble
        ensemble_config = {
            "ensemble_method": "weighted_avg",
            "weights": {
                "test_lstm": 0.2,
                "test_xgb": 0.4,
                "test_lgb": 0.4
            }
        }
        ensemble = EnsembleModel(name="test_ensemble", models=models, config=ensemble_config)
        
        # Get ensemble feature importance
        ensemble_importance = ensemble.feature_importance()
        
        # Check if ensemble feature importance is not None
        assert ensemble_importance is not None
        
        # Since LSTM doesn't have feature importance, ensemble importance
        # should mainly come from XGBoost and LightGBM
        for feature in xgb_importance:
            if feature in ensemble_importance:
                assert ensemble_importance[feature] > 0 