"""STOCKER Pro Configuration Module

This module provides centralized configuration handling for the STOCKER Pro application.
It consolidates functionality previously spread across multiple files into a single, well-organized module.

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

Functions:
    validate_config: Validate configuration against a schema
    load_config: Load configuration from file or environment

Constants:
    Various application constants and default values
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

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
ALPHA_VANTAGE_API_KEY = "your_api_key_here"
THUNDER_COMPUTE_API_KEY = "your_thunder_compute_key"
DEFAULT_LOG_FILE = "logs/stocker.log"
DEFAULT_LOGGING_CONFIG = "src/utils/logging_config.yaml"
SUPPORTED_EXCHANGES = ["NSE", "BSE", "NYSE", "NASDAQ"]
DEFAULT_MODEL = "ensemble"
ENSEMBLE_WEIGHTS = {"lstm": 0.4, "xgboost": 0.3, "lightgbm": 0.3}
NSE_SYMBOLS_URL = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
MAX_NEWS_ARTICLES = 50
DEFAULT_SUMMARIZATION_MODEL = "t5-small"
DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

class Environment(str, Enum):
    """Environment types for configuration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass

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
    uri: str = DEFAULT_MONGODB_URI
    db_name: str = DEFAULT_DB_NAME
    username: Optional[str] = None
    password: Optional[str] = None
    extra: Optional[Dict] = field(default_factory=dict)

@dataclass
class AWSConfig(BaseConfig):
    """Configuration for AWS services"""
    access_key: str
    secret_key: str
    region: str
    bucket: str
    extra: Optional[Dict] = field(default_factory=dict)

@dataclass
class StockerConfig(BaseConfig):
    """Main configuration class for STOCKER Pro"""
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Project paths
    project_root: str = field(default_factory=lambda: os.getcwd())
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), DEFAULT_DATA_DIR))
    logs_dir: str = "logs"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    
    # Data settings
    default_start_date: str = DEFAULT_START_DATE
    default_end_date: str = DEFAULT_END_DATE
    training_start_date: str = DEFAULT_START_DATE
    training_end_date: str = DEFAULT_END_DATE
    training_symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    
    # API keys
    alpha_vantage_api_key: Optional[str] = None
    thunder_compute_api_key: Optional[str] = None
    
    # Configuration sections
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    aws: Optional[AWSConfig] = None
    
    def __post_init__(self):
        """Initialize directories and load environment variables"""
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.artifacts_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
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
            self.database.uri = os.environ.get("MONGODB_URI")
        
        if os.environ.get("MONGODB_USERNAME"):
            self.database.username = os.environ.get("MONGODB_USERNAME")
        
        if os.environ.get("MONGODB_PASSWORD"):
            self.database.password = os.environ.get("MONGODB_PASSWORD")
        
        if os.environ.get("MONGODB_DB_NAME"):
            self.database.db_name = os.environ.get("MONGODB_DB_NAME")
        
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
                config_dict[key] = value.to_dict()
            elif isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
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
        logger.error(error_message)
        raise ConfigValidationError(error_message)
    
    return validated_config

def load_config(config_path: Optional[str] = None) -> StockerConfig:
    """Load configuration from file or create default"""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config_dict = json.load(f)
                else:  # Default to YAML
                    config_dict = yaml.safe_load(f) or {}
            
            config = StockerConfig()
            config.from_dict(config_dict)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    # Try default locations
    default_locations = [
        os.path.join(os.getcwd(), "config.json"),
        os.path.join(os.getcwd(), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),
    ]
    
    for loc in default_locations:
        if os.path.exists(loc):
            logger.info(f"Found config at {loc}")
            return load_config(loc)
    
    logger.info("No config file found, using defaults")
    return StockerConfig()