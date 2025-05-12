"""STOCKER Pro Configuration Module

This module provides centralized configuration handling for the STOCKER Pro application.
It consolidates functionality for configuration management into a single, well-organized module.

Classes:
    Environment: Enum for different environment types
    ConfigValidationError: Exception for configuration validation failures
    BaseConfig: Base configuration class with common methods
    DataSourceConfig: Configuration for data sources
    ModelConfig: Configuration for prediction models
    APIConfig: Configuration for API settings
    PortfolioConfig: Configuration for portfolio analytics
    DatabaseConfig: Configuration for database connections
    AWSConfig: Configuration for AWS services
    StockerConfig: Main configuration class with all application settings
    PydanticStockerConfig: Pydantic model for configuration validation

Functions:
    validate_config: Validate configuration against a schema
    load_config: Load configuration from file or environment
    deep_update: Recursively update a dictionary
    set_nested_key: Set a value in a nested dictionary
    parse_env_value: Parse environment variable value

Constants:
    Various application constants and default values
"""

import os
import json
import logging
import dataclasses

# Handle case when PyYAML is not installed
try:
    import yaml
except ImportError:
    yaml = None
    print("WARNING: PyYAML is not installed. YAML configuration files will not be supported.")
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict, MISSING
from pathlib import Path
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator

# Environment variable prefix
ENV_PREFIX = "STOCKER_"

# Default configuration values
DEFAULT_DATA_DIR = "data"
DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_DB_NAME = "stocker_db"
STOCK_COLLECTION = "stocks"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2023-01-01"
APP_NAME = "STOCKER Pro"
VERSION = "0.1.0"
DEFAULT_LOG_FILE = "logs/stocker.log"
DEFAULT_LOGGING_CONFIG = "src/core/logging_config.yaml"
SUPPORTED_EXCHANGES = ["NSE", "BSE", "NYSE", "NASDAQ"]
DEFAULT_MODEL = "ensemble"
ENSEMBLE_WEIGHTS = {"lstm": 0.4, "xgboost": 0.3, "lightgbm": 0.3}
MAX_NEWS_ARTICLES = 50

class Environment(str, Enum):
    """Environment types for configuration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass

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
class BaseConfig:
    """Base configuration class with common methods"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class DataSourceConfig(BaseConfig):
    """Configuration for data sources"""
    # Yahoo Finance settings
    yahoo_finance_enabled: bool = True
    yahoo_cache_days: int = 7
    
    # Alpha Vantage settings
    alpha_vantage_enabled: bool = False
    alpha_vantage_api_key: str = ""
    alpha_vantage_requests_per_minute: int = 5
    
    # General data settings
    default_start_date: str = "2010-01-01"
    default_end_date: str = ""  # Empty means current date
    data_cache_dir: str = "cache"

@dataclass
class ModelConfig(BaseConfig):
    """Configuration for prediction models"""
    # General model settings
    default_model: str = "ensemble"
    model_save_dir: str = "models"
    
    # LSTM settings
    lstm_units: int = 50
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    # XGBoost settings
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.01
    xgboost_n_estimators: int = 1000
    xgboost_objective: str = "reg:squarederror"
    
    # LightGBM settings
    lightgbm_max_depth: int = 6
    lightgbm_learning_rate: float = 0.01
    lightgbm_n_estimators: int = 1000
    lightgbm_objective: str = "regression"
    
    # Ensemble settings
    ensemble_models: List[str] = field(default_factory=lambda: ["lstm", "xgboost", "lightgbm"])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])

@dataclass
class APIConfig(BaseConfig):
    """Configuration for API settings"""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    enable_docs: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_period_seconds: int = 3600
    jwt_secret_key: str = "stocker_secret"
    alpha_vantage_api_key: str = field(default_factory=lambda: os.environ.get("ALPHA_VANTAGE_API_KEY", ""))
    alpha_vantage_requests_per_minute: int = 5

@dataclass
class PortfolioConfig(BaseConfig):
    """Configuration for portfolio analytics"""
    risk_free_rate: float = 0.04
    benchmark_symbol: str = "SPY"
    rebalance_frequency: str = "monthly"
    optimization_method: str = "efficient_frontier"
    risk_aversion: float = 2.0
    max_position_size: float = 0.25
    min_position_size: float = 0.01
    target_volatility: float = 0.15
    presets_dir: str = "presets"
    cache_dir: str = "cache"

@dataclass
class DatabaseConfig(BaseConfig):
    """Configuration for database connections"""
    mongodb_connection_string: str = DEFAULT_MONGODB_URI
    mongodb_database_name: str = DEFAULT_DB_NAME
    stock_data_collection: str = "stocks"
    models_collection: str = "models"
    portfolio_collection: str = "portfolios"
    user_collection: str = "users"
    news_collection: str = "news"
    username: Optional[str] = None
    password: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AWSConfig(BaseConfig):
    """Configuration for AWS services"""
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"
    bucket: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataConfig(BaseConfig):
    """Configuration for data sources and settings"""
    default_start_date: str = DEFAULT_START_DATE
    default_end_date: str = ""
    cache_dir: str = "cache"
    sources: List[str] = field(default_factory=lambda: ["alpha_vantage"])
    missing_value_method: str = "ffill"
    technical_indicators: Dict[str, Any] = field(default_factory=lambda: {
        "ma_windows": [5, 10, 20, 50, 200],
        "rsi_periods": [14],
        "macd_params": {"fast": 12, "slow": 26, "signal": 9},
        "bb_params": {"window": 20, "num_std": 2},
        "atr_periods": [14],
        "stoch_params": {"k_period": 14, "d_period": 3}
    })

@dataclass
class StockerConfig(BaseConfig):
    """
    Main configuration class for STOCKER Pro.
    
    This class combines the functionality from both configuration implementations
    and provides a comprehensive configuration solution.
    """
    # Environment and project paths
    environment: Environment = Environment.DEVELOPMENT
    project_root: str = field(default_factory=lambda: os.getcwd())
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), DEFAULT_DATA_DIR))
    logs_dir: str = "logs"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    
    # Application settings
    mode: str = "default"
    app_name: str = APP_NAME
    app_version: str = VERSION
    debug: bool = True
    
    # Data settings
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    default_start_date: str = DEFAULT_START_DATE
    default_end_date: str = DEFAULT_END_DATE
    training_start_date: str = DEFAULT_START_DATE
    training_end_date: str = DEFAULT_END_DATE
    data_sources: List[str] = field(default_factory=lambda: ["alpha_vantage"])
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_dir: str = "cache"
    
    # API keys
    alpha_vantage_api_key: Optional[str] = None
    thunder_compute_api_key: Optional[str] = None
    
    # Model settings
    model_type: str = "ensemble"
    model_name: Optional[str] = None
    model_path: Optional[str] = None
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
    
    # Configuration sections
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aws: Optional[AWSConfig] = None
    
    # Extra dynamic fields
    __extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize directories and load environment variables"""
        # Ensure __extra_fields is initialized first to avoid recursion
        object.__setattr__(self, '_StockerConfig__extra_fields', {})
        
        # Ensure all dataclass fields are properly initialized
        # This is needed because our custom __getattr__ might interfere with dataclass initialization
        fields = self.__class__.__dataclass_fields__
        for field_name in fields:
            if not hasattr(self, field_name):
                # Get the default value from the field
                field_obj = fields[field_name]
                if field_obj.default is not dataclasses.MISSING:
                    setattr(self, field_name, field_obj.default)
                elif field_obj.default_factory is not dataclasses.MISSING:
                    setattr(self, field_name, field_obj.default_factory())
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.artifacts_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set model name if not provided
        if self.model_name is None:
            self.model_name = f"{self.model_type}_{datetime.now().strftime('%Y%m%d')}"
        
        # Load environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API keys
        if os.environ.get("ALPHA_VANTAGE_API_KEY"):
            self.alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
            self.data_source.alpha_vantage_enabled = True
            self.data_source.alpha_vantage_api_key = self.alpha_vantage_api_key
        
        if os.environ.get("THUNDER_COMPUTE_API_KEY"):
            self.thunder_compute_api_key = os.environ.get("THUNDER_COMPUTE_API_KEY")
        
        # Database configuration
        if os.environ.get("MONGODB_URI"):
            self.database.mongodb_connection_string = os.environ.get("MONGODB_URI")
        
        if os.environ.get("MONGODB_USERNAME"):
            self.database.username = os.environ.get("MONGODB_USERNAME")
        
        if os.environ.get("MONGODB_PASSWORD"):
            self.database.password = os.environ.get("MONGODB_PASSWORD")
        
        if os.environ.get("MONGODB_DB_NAME"):
            self.database.mongodb_database_name = os.environ.get("MONGODB_DB_NAME")
        
        # AWS configuration
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION")
        aws_bucket = os.environ.get("AWS_S3_BUCKET")
        
        if all([aws_access_key, aws_secret_key, aws_region, aws_bucket]):
            self.aws = AWSConfig(
                access_key=aws_access_key,
                secret_key=aws_secret_key,
                region=aws_region,
                bucket=aws_bucket
            )
    
    def __getattr__(self, name):
        """Allow dynamic access to configuration attributes."""
        # Use object.__getattribute__ to safely access __extra_fields
        try:
            extra_fields = object.__getattribute__(self, '_StockerConfig__extra_fields')
            if name in extra_fields:
                return extra_fields[name]
        except AttributeError:
            pass
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Allow setting both static and dynamic attributes."""
        if name.startswith("_") or hasattr(self.__class__, name):
            super().__setattr__(name, value)
        else:
            try:
                # Use object.__getattribute__ to safely access __extra_fields
                extra_fields = object.__getattribute__(self, '_StockerConfig__extra_fields')
                extra_fields[name] = value
            except AttributeError:
                # If __extra_fields doesn't exist yet, initialize it
                object.__setattr__(self, '_StockerConfig__extra_fields', {name: value})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == "__extra_fields":
                continue
                
            if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
                config_dict[key] = value.to_dict()
            elif isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
                
        # Add extra fields
        config_dict.update(self.__extra_fields)
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
            
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file"""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if file_path.suffix.lower() == ".json":
            with open(file_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:  # Default to YAML
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'StockerConfig':
        """Load configuration from a YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix.lower() == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        else:  # Default to YAML
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StockerConfig':
        """Create a configuration from a dictionary."""
        # Create a new instance with default values
        config = cls()
        
        # Update with values from dictionary
        for k, v in config_dict.items():
            if k == "data_source" and isinstance(v, dict):
                config.data_source.from_dict(v)
            elif k == "model" and isinstance(v, dict):
                config.model.from_dict(v)
            elif k == "api" and isinstance(v, dict):
                config.api.from_dict(v)
            elif k == "portfolio" and isinstance(v, dict):
                config.portfolio.from_dict(v)
            elif k == "database" and isinstance(v, dict):
                config.database.from_dict(v)
            elif k == "aws" and isinstance(v, dict):
                if config.aws is None:
                    config.aws = AWSConfig()
                config.aws.from_dict(v)
            elif hasattr(config, k):
                setattr(config, k, v)
            else:
                config.__extra_fields[k] = v
        
        return config


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


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d


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


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with configuration values
    """
    if yaml is None:
        logging.error(f"Cannot load YAML configuration from {file_path}: PyYAML is not installed")
        return {}
        
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        logging.error(f"Error loading YAML configuration from {file_path}: {e}")
        return {}


def load_config(config_path: Optional[str] = None, env_prefix: str = ENV_PREFIX) -> StockerConfig:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_path: Path to configuration file. If None, only environment variables are used.
        env_prefix: Prefix for environment variables.
        
    Returns:
        StockerConfig: Configuration object.
    """
    # Start with default configuration
    config = StockerConfig()
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        file_config = load_yaml_config(config_path)
        # Merge configurations
        if isinstance(file_config, dict):
            config = StockerConfig.from_dict(deep_update(config.to_dict(), file_config))
        elif hasattr(file_config, 'to_dict'):
            config = StockerConfig.from_dict(deep_update(config.to_dict(), file_config.to_dict()))
    
    # Override with environment variables
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(env_prefix):].lower()
            
            # Handle nested keys (e.g., STOCKER_API_HOST -> api.host)
            if "_" in config_key:
                parts = config_key.split("_", 1)
                if hasattr(config, parts[0]):
                    # If the first part is a known section, treat it as nested
                    section = getattr(config, parts[0])
                    if isinstance(section, dict):
                        section[parts[1]] = parse_env_value(value)
                    else:
                        # Try to set attribute on the section
                        try:
                            setattr(section, parts[1], parse_env_value(value))
                        except AttributeError:
                            # Fall back to setting it as a top-level attribute
                            setattr(config, config_key, parse_env_value(value))
                else:
                    # Set as a top-level attribute
                    setattr(config, config_key, parse_env_value(value))
            else:
                # Set as a top-level attribute
                setattr(config, config_key, parse_env_value(value))
    
    return config


def validate_config(config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a configuration dictionary against a schema"""
    validated_config = {}
    errors = []
    
    for param_name, param_schema in schema.items():
        is_required = param_schema.get('required', False)
        default_value = param_schema.get('default', None)
        param_type = param_schema.get('type', None)
        validator = param_schema.get('validator', None)
        
        if param_name in config:
            value = config[param_name]
            
            if param_type and not isinstance(value, param_type):
                if param_type in (int, float) and isinstance(value, (int, float)):
                    value = param_type(value)
                else:
                    errors.append(f"Parameter '{param_name}' has incorrect type. "
                                 f"Expected {param_type.__name__}, got {type(value).__name__}")
                    continue
            
            if validator and not validator(value):
                errors.append(f"Parameter '{param_name}' failed validation: {validator.__doc__ or 'invalid value'}")
                continue
                
            validated_config[param_name] = value
        elif is_required:
            errors.append(f"Required parameter '{param_name}' is missing")
        else:
            validated_config[param_name] = default_value
    
    for param_name, value in config.items():
        if param_name not in schema:
            validated_config[param_name] = value
    
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        logging.error(error_message)
        raise ConfigValidationError(error_message)
    
    return validated_config


# Create a global configuration instance
settings = load_config()

# For backward compatibility, also provide 'config' as an alias
config = settings

# Function to get the config instance
def get_config(config_path: Optional[str] = None) -> StockerConfig:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        StockerConfig: The global configuration instance.
    """
    if config_path:
        # Load configuration from the specified file
        return load_config(config_path)
    return settings


class Config:
    """
    Simple configuration class for backward compatibility.
    
    This class provides a simpler interface to access configuration values
    and is used by some modules that haven't been updated to use the new
    StockerConfig class.
    """
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the Config object.
        
        Args:
            config_dict: Optional dictionary with configuration values
        """
        self._config = config_dict or {}
        
        # Add common attributes from the global config
        self.symbols = getattr(settings, 'symbols', [])
        self.api = settings.api if hasattr(settings, 'api') else APIConfig()
        self.database = settings.database if hasattr(settings, 'database') else DatabaseConfig()
        self.model = settings.model if hasattr(settings, 'model') else ModelConfig()
        self.portfolio = settings.portfolio if hasattr(settings, 'portfolio') else PortfolioConfig()
        
    def __getattr__(self, name):
        """Get attribute from config dictionary if not found in object"""
        if name in self._config:
            return self._config[name]
        
        # Check if attribute exists in global config
        if hasattr(settings, name):
            return getattr(settings, name)
            
        # Default values for common attributes
        if name == 'debug':
            return getattr(settings, 'debug', False)
        elif name == 'environment':
            return getattr(settings, 'environment', 'development')
        elif name == 'jwt_secret_key':
            return getattr(settings.api, 'jwt_secret_key', 'stocker_secret') if hasattr(settings, 'api') else 'stocker_secret'
            
        # Raise attribute error if not found
        raise AttributeError(f"'Config' object has no attribute '{name}'")
