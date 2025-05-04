"""
Configuration management for STOCKER Pro.
This module provides centralized configuration handling with validation and environment variable support.
"""
import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Moved from constants.py
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

# Moved from config_entity.py
@dataclass
class DatabaseConfig:
    uri: str = DEFAULT_MONGODB_URI
    db_name: str = DEFAULT_DB_NAME
    username: Optional[str] = None
    password: Optional[str] = None
    extra: Optional[Dict] = field(default_factory=dict)

@dataclass
class AWSConfig:
    access_key: str
    secret_key: str
    region: str
    bucket: str
    extra: Optional[Dict] = field(default_factory=dict)

# Moved from config_validator.py
class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass

def validate_config(config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a configuration dictionary against a schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema defining required parameters and their types
        
    Returns:
        The validated configuration dictionary (may include defaults)
        
    Raises:
        ConfigValidationError: If validation fails
    """
    validated_config = {}
    errors = []
    
    # Check for required parameters
    for param_name, param_schema in schema.items():
        is_required = param_schema.get('required', False)
        default_value = param_schema.get('default', None)
        param_type = param_schema.get('type', None)
        validator = param_schema.get('validator', None)
        
        # Check if parameter exists
        if param_name in config:
            value = config[param_name]
            
            # Type checking
            if param_type and not isinstance(value, param_type):
                # Special case for numeric types
                if param_type in (int, float) and isinstance(value, (int, float)):
                    # Allow int to float conversion
                    value = param_type(value)
                else:
                    errors.append(f"Parameter '{param_name}' has incorrect type. "
                                 f"Expected {param_type.__name__}, got {type(value).__name__}")
                    continue
            
            # Custom validation
            if validator and not validator(value):
                errors.append(f"Parameter '{param_name}' failed validation: {validator.__doc__ or 'invalid value'}")
                continue
            
            validated_config[param_name] = value
        elif is_required:
            errors.append(f"Required parameter '{param_name}' is missing")
        elif default_value is not None:
            validated_config[param_name] = default_value
    
    # Allow additional parameters not in schema (we don't want to be too strict)
    for param_name, value in config.items():
        if param_name not in schema:
            validated_config[param_name] = value
    
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        logger.error(error_message)
        raise ConfigValidationError(error_message)
    
    return validated_config

# Common validators
def positive_number(value: Union[int, float]) -> bool:
    """Value must be a positive number."""
    return value > 0

def non_negative_number(value: Union[int, float]) -> bool:
    """Value must be a non-negative number."""
    return value >= 0

def in_range(min_val: Union[int, float], max_val: Union[int, float]):
    """Value must be in the specified range (inclusive)."""
    def validator(value: Union[int, float]) -> bool:
        return min_val <= value <= max_val
    validator.__doc__ = f"Value must be between {min_val} and {max_val} (inclusive)"
    return validator

def one_of(valid_values: List[Any]):
    """Value must be one of the specified values."""
    def validator(value: Any) -> bool:
        return value in valid_values
    validator.__doc__ = f"Value must be one of: {', '.join(str(v) for v in valid_values)}"
    return validator

# Pre-defined schemas for common components
ML_BASE_SCHEMA = {
    "learning_rate": {
        "type": float,
        "required": True,
        "validator": in_range(0.0001, 1.0),
        "default": 0.01
    },
    "random_seed": {
        "type": int,
        "required": False,
        "default": 42
    },
    "validation_split": {
        "type": float,
        "required": False,
        "validator": in_range(0.0, 0.5),
        "default": 0.2
    }
}

LSTM_MODEL_SCHEMA = {
    **ML_BASE_SCHEMA,
    "sequence_length": {
        "type": int,
        "required": True,
        "validator": positive_number,
        "default": 10
    },
    "hidden_dim": {
        "type": int,
        "required": True,
        "validator": positive_number,
        "default": 64
    },
    "num_layers": {
        "type": int,
        "required": True,
        "validator": positive_number,
        "default": 2
    },
    "dropout": {
        "type": float,
        "required": False,
        "validator": in_range(0.0, 0.9),
        "default": 0.2
    },
    "epochs": {
        "type": int,
        "required": True,
        "validator": positive_number,
        "default": 100
    },
    "batch_size": {
        "type": int,
        "required": False,
        "validator": positive_number,
        "default": 32
    },
}

XGBOOST_MODEL_SCHEMA = {
    **ML_BASE_SCHEMA,
    "max_depth": {
        "type": int,
        "required": True,
        "validator": in_range(1, 20),
        "default": 6
    },
    "n_estimators": {
        "type": int,
        "required": True,
        "validator": positive_number,
        "default": 100
    },
    "subsample": {
        "type": float,
        "required": False,
        "validator": in_range(0.1, 1.0),
        "default": 0.8
    },
    "colsample_bytree": {
        "type": float,
        "required": False,
        "validator": in_range(0.1, 1.0),
        "default": 0.8
    },
    "objective": {
        "type": str,
        "required": False,
        "validator": one_of(["reg:squarederror", "reg:logistic", "binary:logistic"]),
        "default": "reg:squarederror"
    }
}

LIGHTGBM_MODEL_SCHEMA = {
    **ML_BASE_SCHEMA,
    "num_leaves": {
        "type": int,
        "required": True,
        "validator": in_range(2, 256),
        "default": 31
    },
    "max_depth": {
        "type": int,
        "required": False,
        "validator": in_range(-1, 20),
        "default": -1
    },
    "n_estimators": {
        "type": int,
        "required": True,
        "validator": positive_number,
        "default": 100
    },
    "subsample": {
        "type": float,
        "required": False,
        "validator": in_range(0.1, 1.0),
        "default": 0.8
    },
    "objective": {
        "type": str,
        "required": False,
        "default": "regression"
    }
}

@dataclass
class StockerConfig:
    """
    Main configuration class for STOCKER Pro.
    Contains all configuration parameters with sensible defaults.
    """
    # Project paths
    project_root: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "data"))
    models_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "models"))
    logs_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "logs"))
    
    # Data sources
    use_alpha_vantage: bool = True
    use_yahoo_finance: bool = True
    alpha_vantage_api_key: Optional[str] = None
    
    # Model parameters
    default_model_type: str = "ensemble"
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {"lstm": 0.4, "xgboost": 0.3, "lightgbm": 0.3})
    
    # Training parameters
    train_test_split: float = 0.8
    sequence_length: int = 20
    prediction_horizon: int = 1
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Feature engineering
    use_technical_indicators: bool = True
    use_sentiment_analysis: bool = False
    
    # Cloud training
    use_cloud_training: bool = False
    thunder_compute_api_key: Optional[str] = None
    
    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    enable_cors: bool = True
    
    # Caching
    enable_cache: bool = True
    cache_expiry_seconds: int = 3600
    
    # Database configuration
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # AWS configuration (optional)
    aws: Optional[AWSConfig] = None
    
    def __post_init__(self):
        """Initialize directories and load environment variables."""
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Load environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # API keys
        if os.environ.get("ALPHA_VANTAGE_API_KEY"):
            self.alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        
        if os.environ.get("THUNDER_COMPUTE_API_KEY"):
            self.thunder_compute_api_key = os.environ.get("THUNDER_COMPUTE_API_KEY")
            self.use_cloud_training = True
        
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
    
    @classmethod
    def from_file(cls, file_path: str) -> 'StockerConfig':
        """
        Load configuration from a file (JSON or YAML).
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            StockerConfig instance
        """
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            logger.warning(f"Config file not found: {file_path}, using defaults")
            return cls()
        
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    config_dict = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    config_dict = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
            
            # Create config with file values
            config = cls(**config_dict)
            logger.info(f"Loaded configuration from {file_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def save(self, file_path: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to save configuration
        """
        try:
            config_dict = self.to_dict()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                elif file_path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_dict, f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
                    
            logger.info(f"Saved configuration to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")

def load_config(config_path: Optional[str] = None) -> StockerConfig:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        StockerConfig instance
    """
    if config_path and os.path.exists(config_path):
        return StockerConfig.from_file(config_path)
    
    # Try to find config in default locations
    default_locations = [
        os.path.join(os.getcwd(), "config.json"),
        os.path.join(os.getcwd(), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"),
    ]
    
    for loc in default_locations:
        if os.path.exists(loc):
            logger.info(f"Found config at {loc}")
            return StockerConfig.from_file(loc)
    
    # No config found, use defaults
    logger.info("No config file found, using defaults")
    return StockerConfig()