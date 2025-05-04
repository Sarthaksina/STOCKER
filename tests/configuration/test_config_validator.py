"""
Tests for the configuration validator module.
"""
import pytest
from typing import Dict, Any

from src.configuration.config_validator import (
    validate_config,
    ConfigValidationError,
    positive_number,
    non_negative_number,
    in_range,
    one_of,
    validate_lstm_config,
    validate_xgboost_config,
    validate_lightgbm_config,
    validate_alpha_vantage_config,
)

class TestConfigValidators:
    """Tests for individual config validators."""
    
    def test_positive_number(self):
        """Test positive number validator."""
        assert positive_number(1) is True
        assert positive_number(0.1) is True
        assert positive_number(0) is False
        assert positive_number(-1) is False
    
    def test_non_negative_number(self):
        """Test non-negative number validator."""
        assert non_negative_number(1) is True
        assert non_negative_number(0.1) is True
        assert non_negative_number(0) is True
        assert non_negative_number(-0.1) is False
        assert non_negative_number(-1) is False
    
    def test_in_range(self):
        """Test in_range validator."""
        validator = in_range(1, 10)
        assert validator(1) is True
        assert validator(5) is True
        assert validator(10) is True
        assert validator(0) is False
        assert validator(11) is False
        
        # Test with floats
        validator = in_range(0.5, 1.5)
        assert validator(0.5) is True
        assert validator(1.0) is True
        assert validator(1.5) is True
        assert validator(0.4) is False
        assert validator(1.6) is False
    
    def test_one_of(self):
        """Test one_of validator."""
        validator = one_of(["a", "b", "c"])
        assert validator("a") is True
        assert validator("b") is True
        assert validator("c") is True
        assert validator("d") is False
        assert validator(1) is False
        
        # Test with mixed types
        validator = one_of([1, "a", True])
        assert validator(1) is True
        assert validator("a") is True
        assert validator(True) is True
        assert validator(2) is False
        assert validator("b") is False
        assert validator(False) is False

class TestConfigValidation:
    """Tests for the validate_config function."""
    
    def test_basic_validation(self):
        """Test basic config validation."""
        schema = {
            "param1": {
                "type": int,
                "required": True
            },
            "param2": {
                "type": str,
                "required": False,
                "default": "default_value"
            }
        }
        
        # Valid config
        config = {"param1": 10}
        validated = validate_config(config, schema)
        assert validated["param1"] == 10
        assert validated["param2"] == "default_value"
        
        # Valid config with both params
        config = {"param1": 10, "param2": "custom_value"}
        validated = validate_config(config, schema)
        assert validated["param1"] == 10
        assert validated["param2"] == "custom_value"
        
        # Invalid config (missing required param)
        config = {}
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config, schema)
        assert "Required parameter 'param1' is missing" in str(exc.value)
        
        # Invalid config (wrong type)
        config = {"param1": "not_an_int"}
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config, schema)
        assert "Parameter 'param1' has incorrect type" in str(exc.value)
    
    def test_validation_with_custom_validators(self):
        """Test validation with custom validators."""
        schema = {
            "age": {
                "type": int,
                "required": True,
                "validator": in_range(0, 120)
            },
            "score": {
                "type": float,
                "required": False,
                "validator": in_range(0.0, 1.0),
                "default": 0.5
            }
        }
        
        # Valid config
        config = {"age": 25, "score": 0.7}
        validated = validate_config(config, schema)
        assert validated["age"] == 25
        assert validated["score"] == 0.7
        
        # Invalid age (out of range)
        config = {"age": 150}
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config, schema)
        assert "Parameter 'age' failed validation" in str(exc.value)
        
        # Invalid score (out of range)
        config = {"age": 25, "score": 1.5}
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config, schema)
        assert "Parameter 'score' failed validation" in str(exc.value)
    
    def test_extra_parameters(self):
        """Test handling of extra parameters not in schema."""
        schema = {
            "param1": {
                "type": int,
                "required": True
            }
        }
        
        # Config with extra parameter
        config = {"param1": 10, "extra_param": "value"}
        validated = validate_config(config, schema)
        assert validated["param1"] == 10
        assert validated["extra_param"] == "value"

    def test_type_conversion(self):
        """Test numeric type conversion."""
        schema = {
            "int_param": {
                "type": int,
                "required": True
            },
            "float_param": {
                "type": float,
                "required": True
            }
        }
        
        # Convert float to int
        config = {"int_param": 10.5, "float_param": 20}
        validated = validate_config(config, schema)
        assert validated["int_param"] == 10
        assert isinstance(validated["int_param"], int)
        assert validated["float_param"] == 20.0
        assert isinstance(validated["float_param"], float)

class TestModelConfigValidation:
    """Tests for model-specific config validation."""
    
    def test_lstm_config_validation(self):
        """Test LSTM config validation."""
        # Valid minimal config
        config = {
            "learning_rate": 0.01,
            "sequence_length": 10,
            "hidden_dim": 64,
            "num_layers": 2,
            "epochs": 100
        }
        validated = validate_lstm_config(config)
        assert validated["learning_rate"] == 0.01
        assert validated["sequence_length"] == 10
        assert validated["hidden_dim"] == 64
        assert validated["num_layers"] == 2
        assert validated["epochs"] == 100
        assert validated["batch_size"] == 32  # default value
        
        # Invalid learning rate
        config = {
            "learning_rate": 2.0,  # out of range
            "sequence_length": 10,
            "hidden_dim": 64,
            "num_layers": 2,
            "epochs": 100
        }
        with pytest.raises(ConfigValidationError) as exc:
            validate_lstm_config(config)
        assert "Parameter 'learning_rate' failed validation" in str(exc.value)
    
    def test_xgboost_config_validation(self):
        """Test XGBoost config validation."""
        # Valid minimal config
        config = {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100
        }
        validated = validate_xgboost_config(config)
        assert validated["learning_rate"] == 0.1
        assert validated["max_depth"] == 6
        assert validated["n_estimators"] == 100
        assert validated["objective"] == "reg:squarederror"  # default value
        
        # Invalid max_depth
        config = {
            "learning_rate": 0.1,
            "max_depth": 0,  # out of range
            "n_estimators": 100
        }
        with pytest.raises(ConfigValidationError) as exc:
            validate_xgboost_config(config)
        assert "Parameter 'max_depth' failed validation" in str(exc.value)
    
    def test_lightgbm_config_validation(self):
        """Test LightGBM config validation."""
        # Valid minimal config
        config = {
            "learning_rate": 0.1,
            "num_leaves": 31,
            "n_estimators": 100
        }
        validated = validate_lightgbm_config(config)
        assert validated["learning_rate"] == 0.1
        assert validated["num_leaves"] == 31
        assert validated["n_estimators"] == 100
        assert validated["objective"] == "regression"  # default value
        
        # Invalid num_leaves
        config = {
            "learning_rate": 0.1,
            "num_leaves": 1,  # out of range
            "n_estimators": 100
        }
        with pytest.raises(ConfigValidationError) as exc:
            validate_lightgbm_config(config)
        assert "Parameter 'num_leaves' failed validation" in str(exc.value)
    
    def test_alpha_vantage_config_validation(self):
        """Test Alpha Vantage client config validation."""
        # Valid minimal config
        config = {
            "api_key": "test_api_key"
        }
        validated = validate_alpha_vantage_config(config)
        assert validated["api_key"] == "test_api_key"
        assert validated["cache_dir"] == "./cache/alpha_vantage"  # default value
        assert validated["cache_expiry"] == 24  # default value
        
        # Missing required api_key
        config = {}
        with pytest.raises(ConfigValidationError) as exc:
            validate_alpha_vantage_config(config)
        assert "Required parameter 'api_key' is missing" in str(exc.value) 