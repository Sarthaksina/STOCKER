"""Tests for the unified configuration module.

This module tests the functionality of the unified configuration system,
including loading from files, environment variables, and code.
"""

import os
import tempfile
import pathlib
import pytest
from src.unified_config import (
    StockerConfig, Environment, get_config, load_config_from_file,
    load_config_from_env, ENV_PREFIX
)


def test_default_config():
    """Test that default configuration is created correctly"""
    config = StockerConfig()
    
    # Check default values
    assert config.environment == Environment.DEVELOPMENT
    assert config.data_dir == "data"
    assert config.mongodb_uri == "mongodb://localhost:27017"
    assert config.db_name == "stocker_db"
    
    # Check nested configs
    assert config.portfolio_config.risk_free_rate == 0.04
    assert config.portfolio_config.benchmark_symbol == "SPY"
    assert config.data_source_config.yahoo_finance_enabled is True
    assert config.model_config.default_model == "ensemble"
    assert config.api_config.host == "127.0.0.1"


def test_to_dict():
    """Test conversion to dictionary"""
    config = StockerConfig()
    config_dict = config.to_dict()
    
    # Check main values
    assert config_dict["environment"] == "development"  # Enum converted to string
    assert config_dict["data_dir"] == "data"
    
    # Check nested values
    assert "portfolio_config" in config_dict
    assert config_dict["portfolio_config"]["risk_free_rate"] == 0.04
    assert "data_source_config" in config_dict
    assert "model_config" in config_dict
    assert "api_config" in config_dict


def test_from_dict():
    """Test updating from dictionary"""
    config = StockerConfig()
    
    # Update with new values
    update_dict = {
        "environment": "production",  # Will be converted to Enum
        "data_dir": "/custom/data",
        "portfolio_config": {
            "risk_free_rate": 0.05,
            "benchmark_symbol": "QQQ"
        },
        "model_config": {
            "lstm_units": 100
        }
    }
    
    config.from_dict(update_dict)
    
    # Check updated values
    assert config.environment == "production"  # String not converted to Enum in from_dict
    assert config.data_dir == "/custom/data"
    assert config.portfolio_config.risk_free_rate == 0.05
    assert config.portfolio_config.benchmark_symbol == "QQQ"
    assert config.model_config.lstm_units == 100
    
    # Check that other values remain default
    assert config.mongodb_uri == "mongodb://localhost:27017"
    assert config.model_config.lstm_dropout == 0.2


def test_save_and_load_yaml():
    """Test saving and loading configuration from YAML file"""
    config = StockerConfig()
    config.environment = Environment.PRODUCTION
    config.data_dir = "/custom/data"
    config.portfolio_config.risk_free_rate = 0.05
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "config.yaml"
        config.save_to_file(file_path)
        
        # Load from file
        loaded_config = load_config_from_file(file_path)
        
        # Check loaded values
        assert loaded_config["environment"] == "production"
        assert loaded_config["data_dir"] == "/custom/data"
        assert loaded_config["portfolio_config"]["risk_free_rate"] == 0.05


def test_save_and_load_json():
    """Test saving and loading configuration from JSON file"""
    config = StockerConfig()
    config.environment = Environment.TESTING
    config.api_config.port = 9000
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "config.json"
        config.save_to_file(file_path)
        
        # Load from file
        loaded_config = load_config_from_file(file_path)
        
        # Check loaded values
        assert loaded_config["environment"] == "testing"
        assert loaded_config["api_config"]["port"] == 9000


def test_load_config_from_env():
    """Test loading configuration from environment variables"""
    # Set environment variables
    os.environ[f"{ENV_PREFIX}ENVIRONMENT"] = "production"
    os.environ[f"{ENV_PREFIX}DATA_DIR"] = "/env/data"
    os.environ[f"{ENV_PREFIX}PORTFOLIO_CONFIG__RISK_FREE_RATE"] = "0.06"
    os.environ[f"{ENV_PREFIX}MODEL_CONFIG__LSTM_UNITS"] = "200"
    os.environ[f"{ENV_PREFIX}API_CONFIG__DEBUG"] = "true"
    
    # Load from environment
    env_config = load_config_from_env()
    
    # Check loaded values
    assert env_config["environment"] == "production"
    assert env_config["data_dir"] == "/env/data"
    assert env_config["portfolio_config"]["risk_free_rate"] == 0.06  # Converted to float
    assert env_config["model_config"]["lstm_units"] == 200  # Converted to int
    assert env_config["api_config"]["debug"] is True  # Converted to bool
    
    # Clean up environment
    for key in list(os.environ.keys()):
        if key.startswith(ENV_PREFIX):
            del os.environ[key]


def test_get_config_with_file(monkeypatch):
    """Test get_config with file path"""
    # Create a temporary config file
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "config.yaml"
        
        # Create config content
        config = StockerConfig()
        config.environment = Environment.PRODUCTION
        config.data_dir = "/custom/data"
        config.save_to_file(file_path)
        
        # Reset global instance
        monkeypatch.setattr("src.unified_config._config_instance", None)
        
        # Get config with file path
        loaded_config = get_config(file_path)
        
        # Check loaded values
        assert loaded_config.environment == Environment.PRODUCTION
        assert loaded_config.data_dir == "/custom/data"


def test_get_config_with_env(monkeypatch):
    """Test get_config with environment variables"""
    # Set environment variables
    os.environ[f"{ENV_PREFIX}DATA_DIR"] = "/env/data"
    os.environ[f"{ENV_PREFIX}PORTFOLIO_CONFIG__RISK_FREE_RATE"] = "0.07"
    
    # Reset global instance
    monkeypatch.setattr("src.unified_config._config_instance", None)
    
    # Get config
    config = get_config()
    
    # Check that environment variables override defaults
    assert config.data_dir == "/env/data"
    assert config.portfolio_config.risk_free_rate == 0.07
    
    # Clean up environment
    for key in list(os.environ.keys()):
        if key.startswith(ENV_PREFIX):
            del os.environ[key]


def test_config_example():
    """Example of how to use the configuration system"""
    # Create custom configuration
    config = StockerConfig()
    config.environment = Environment.PRODUCTION
    config.data_dir = "/app/data"
    config.mongodb_uri = "mongodb://user:password@mongodb:27017"
    
    # Configure data sources
    config.data_source_config.alpha_vantage_enabled = True
    config.data_source_config.alpha_vantage_api_key = "your_api_key"
    
    # Configure models
    config.model_config.default_model = "xgboost"
    config.model_config.xgboost_learning_rate = 0.005
    
    # Configure portfolio
    config.portfolio_config.risk_free_rate = 0.03
    config.portfolio_config.optimization_method = "min_variance"
    
    # Convert to dictionary (for saving to file or API)
    config_dict = config.to_dict()
    
    # Verify values
    assert config_dict["environment"] == "production"
    assert config_dict["data_source_config"]["alpha_vantage_enabled"] is True
    assert config_dict["model_config"]["default_model"] == "xgboost"
    assert config_dict["portfolio_config"]["optimization_method"] == "min_variance"