"""
Configuration management for STOCKER Pro.

This module provides centralized configuration management for the entire application.
"""
import os
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pydantic import BaseModel, Field, validator

# Default configuration
DEFAULT_CONFIG = {
    'app': {
        'name': 'STOCKER Pro',
        'version': '0.1.0',
        'environment': 'development',
        'debug': True,
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/stocker.log',
        'console': True,
    },
    'data': {
        'sources': {
            'alpha_vantage': {
                'api_key': None,
                'output_size': 'full',
                'data_type': 'json',
                'throttle_ms': 1000,
            },
            'mongodb': {
                'uri': 'mongodb://localhost:27017/',
                'database': 'stocker',
                'collection_prefix': '',
            },
        },
        'cache': {
            'enabled': True,
            'ttl': 3600,  # seconds
            'dir': 'cache',
        },
    },
    'model': {
        'default_type': 'ensemble',
        'model_save_dir': 'models',
        'hyperparameter_tuning': {
            'max_trials': 20,
            'early_stopping': True,
        },
    },
    'features': {
        'technical_indicators': {
            'ma_windows': [5, 10, 20, 50, 200],
            'rsi_periods': [14],
            'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
            'bb_params': {'window': 20, 'num_std': 2},
            'atr_periods': [14],
            'stoch_params': {'k_period': 14, 'd_period': 3},
        },
        'missing_value_method': 'ffill',
    },
    'api': {
        'host': '0.0.0.0',
        'port': 8000,
        'workers': 4,
        'cors_origins': ['*'],
        'rate_limit': {
            'max_requests': 100,
            'window_seconds': 60,
        },
    },
    'ui': {
        'theme': 'dark',
        'default_chart_type': 'candlestick',
        'default_timeframe': '1d',
    },
    'intelligence': {
        'vector_store': {
            'provider': 'chroma',
            'collection': 'stocker_docs',
            'embedding_model': 'all-MiniLM-L6-v2',
        },
        'llm': {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 500,
        },
        'news': {
            'sources': ['bloomberg', 'cnbc', 'reuters', 'wsj'],
            'max_age_days': 30,
        },
    },
}


@dataclass
class StockerConfig:
    """
    Unified configuration class for STOCKER Pro.
    
    This class can be used both as a dataclass for static configuration
    and as a dynamic dictionary-like object for runtime configuration.
    """
    # Application settings
    mode: str = "default"
    app_name: str = "STOCKER Pro"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True
    
    # Data settings
    symbols: List[str] = field(default_factory=list)
    start_date: str = None
    end_date: str = None
    data_sources: List[str] = field(default_factory=lambda: ["alpha_vantage"])
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_dir: str = "cache"
    
    # Model settings
    model_type: str = "ensemble"
    model_name: str = None
    model_path: str = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_tuning: bool = False
    max_trials: int = 20
    early_stopping: bool = True
    
    # Feature settings
    target_col: str = "close"
    forecast_horizon: int = 1
    train_test_split: float = 0.8
    feature_selection_method: str = "importance"
    
    # Technical indicators
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [14])
    macd_params: Dict[str, int] = field(default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9})
    bb_params: Dict[str, Any] = field(default_factory=lambda: {"window": 20, "num_std": 2})
    atr_periods: List[int] = field(default_factory=lambda: [14])
    stoch_params: Dict[str, int] = field(default_factory=lambda: {"k_period": 14, "d_period": 3})
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # UI settings
    ui_theme: str = "dark"
    default_chart_type: str = "candlestick"
    default_timeframe: str = "1d"
    
    # Intelligence settings
    vector_store_provider: str = "chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.1
    
    # Extra dynamic fields
    __extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived values after initialization."""
        # Set model name if not provided
        if self.model_name is None:
            self.model_name = f"{self.model_type}_{datetime.now().strftime('%Y%m%d')}"
    
    def __getattr__(self, name):
        """Allow dynamic access to configuration attributes."""
        if name in self.__extra_fields:
            return self.__extra_fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Allow setting both static and dynamic attributes."""
        if name.startswith("_") or hasattr(self.__class__, name):
            super().__setattr__(name, value)
        else:
            self.__extra_fields[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to dictionary."""
        config_dict = asdict(self)
        config_dict.update(self.__extra_fields)
        del config_dict["__extra_fields"]
        return config_dict
    
    def save(self, path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = self.to_dict()
        
        # Remove non-serializable objects
        clean_dict = {}
        for k, v in config_dict.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                clean_dict[k] = v
        
        with open(path, "w") as f:
            yaml.dump(clean_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'StockerConfig':
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Extract base fields
        base_fields = {k: v for k, v in config_dict.items() 
                      if k in cls.__annotations__}
        
        # Create instance with base fields
        instance = cls(**base_fields)
        
        # Add extra fields
        extra_fields = {k: v for k, v in config_dict.items() 
                       if k not in cls.__annotations__}
        instance.__extra_fields = extra_fields
        
        return instance
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StockerConfig':
        """Create a configuration instance from a dictionary."""
        # Extract base fields
        base_fields = {k: v for k, v in config_dict.items() 
                      if k in cls.__annotations__}
        
        # Create instance with base fields
        instance = cls(**base_fields)
        
        # Add extra fields
        extra_fields = {k: v for k, v in config_dict.items() 
                       if k not in cls.__annotations__}
        instance.__extra_fields = extra_fields
        
        return instance


class PydanticStockerConfig(BaseModel):
    """
    Pydantic model for STOCKER Pro configuration validation.
    Use this for API endpoints and data validation.
    """
    # Application settings
    mode: str = Field("default", description="Operation mode (train, predict, etc.)")
    app_name: str = Field("STOCKER Pro", description="Application name")
    app_version: str = Field("0.1.0", description="Application version")
    environment: str = Field("development", description="Environment (development, production)")
    debug: bool = Field(True, description="Debug mode")
    
    # Data settings
    symbols: List[str] = Field([], description="Stock symbols to analyze")
    start_date: Optional[str] = Field(None, description="Start date for data")
    end_date: Optional[str] = Field(None, description="End date for data")
    data_sources: List[str] = Field(["alpha_vantage"], description="Data sources")
    
    # Model settings
    model_type: str = Field("ensemble", description="Model type")
    model_name: Optional[str] = Field(None, description="Model name")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    
    # Feature settings
    target_col: str = Field("close", description="Target column for prediction")
    forecast_horizon: int = Field(1, description="Forecast horizon in days")
    
    @validator('start_date', 'end_date')
    def validate_date(cls, v):
        """Validate date format."""
        if v is None:
            return v
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate symbols list."""
        if len(v) == 0:
            return v
        for symbol in v:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Symbols must be non-empty strings")
        return v
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields


def load_config(path: Optional[str] = None) -> StockerConfig:
    """
    Load configuration from a file or environment variables.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration object
    """
    # Start with default configuration
    config_data = DEFAULT_CONFIG
    
    # Load from file if provided
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Recursively update configuration
                deep_update(config_data, file_config)
        except Exception as e:
            logging.warning(f"Failed to load config from {path}: {e}")
    
    # Override from environment variables
    env_prefix = "STOCKER_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            config_path = key[len(env_prefix):].lower().split("__")
            set_nested_key(config_data, config_path, parse_env_value(value))
    
    # Create a StockerConfig instance
    config = StockerConfig.from_dict(config_data)
    
    return config


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    Recursively update a dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for key, value in update_dict.items():
        if (
            key in base_dict and 
            isinstance(base_dict[key], dict) and 
            isinstance(value, dict)
        ):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def set_nested_key(d: Dict, keys: List[str], value: Any) -> None:
    """
    Set a value in a nested dictionary.
    
    Args:
        d: Dictionary to update
        keys: List of keys forming a path
        value: Value to set
    """
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def parse_env_value(value: str) -> Any:
    """
    Parse environment variable value.
    
    Args:
        value: String value from environment
        
    Returns:
        Parsed value (bool, int, float, or string)
    """
    # Handle boolean
    if value.lower() in ('true', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'no', 'n', '0'):
        return False
    
    # Handle numeric
    try:
        # Try as int
        return int(value)
    except ValueError:
        try:
            # Try as float
            return float(value)
        except ValueError:
            # Fall back to string
            return value
