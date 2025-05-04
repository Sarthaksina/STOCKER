"""
Pydantic models for STOCKER Pro data validation.

This module defines Pydantic models for validating configuration, user input,
and other structured data throughout the application.
"""
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from enum import Enum
import os

class RiskAppetite(str, Enum):
    """Risk appetite levels for user profiles."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class UserProfile(BaseModel):
    """User profile model for personalized recommendations."""
    
    user_id: str = Field(..., description="Unique user identifier")
    risk_appetite: RiskAppetite = Field(..., description="User's risk appetite")
    age: int = Field(..., ge=18, le=120, description="User's age")
    income: float = Field(..., ge=0, description="User's annual income")
    investment_horizon: Optional[int] = Field(None, ge=1, description="Investment horizon in years")
    existing_investments: Optional[Dict[str, float]] = Field(None, description="Existing investments by asset class")
    
    @validator('age')
    def validate_age(cls, v):
        """Validate that age is reasonable."""
        if v < 18:
            raise ValueError("User must be at least 18 years old")
        return v

class ModelConfig(BaseModel):
    """Configuration for ML models."""
    
    model_type: str = Field(..., description="Type of model (e.g., 'sklearn_rf', 'xgboost', 'lightgbm', 'keras')")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    target_column: str = Field(..., description="Target column for prediction")
    train_test_split: float = Field(0.8, ge=0.5, le=0.95, description="Train/test split ratio")
    validation_size: Optional[float] = Field(None, ge=0.0, le=0.5, description="Validation set size")
    random_state: int = Field(42, description="Random state for reproducibility")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        """Validate that model type is supported."""
        supported_models = ['sklearn_rf', 'xgboost', 'lightgbm', 'keras', 'lstm', 'arima', 'prophet']
        if v not in supported_models:
            raise ValueError(f"Model type must be one of {supported_models}")
        return v

class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    
    source_type: str = Field(..., description="Type of data source (e.g., 'csv', 'api', 'database')")
    connection_params: Dict[str, Any] = Field(default_factory=dict, description="Connection parameters")
    symbols: List[str] = Field(..., min_items=1, description="List of stock symbols")
    start_date: Union[str, date, datetime] = Field(..., description="Start date for data")
    end_date: Optional[Union[str, date, datetime]] = Field(None, description="End date for data")
    
    @validator('start_date', 'end_date', pre=True)
    def parse_date(cls, v):
        """Parse date strings to datetime objects."""
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError("Date must be in format YYYY-MM-DD")
        return v
    
    @root_validator
    def validate_dates(cls, values):
        """Validate that start_date is before end_date."""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be before end_date")
        return values

class StockerConfigSchema(BaseModel):
    """Main configuration schema for STOCKER Pro."""
    
    # Basic configuration
    app_name: str = Field("STOCKER Pro", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    log_level: str = Field("INFO", description="Logging level")
    
    # Paths configuration
    data_dir: str = Field("data", description="Data directory")
    models_dir: str = Field("models", description="Models directory")
    logs_dir: str = Field("logs", description="Logs directory")
    
    # Data sources
    data_sources: List[DataSourceConfig] = Field(default_factory=list, description="Data sources configuration")
    
    # Model configuration
    default_model_type: str = Field("lightgbm", description="Default model type")
    model_config: ModelConfig = Field(..., description="Model configuration")
    
    # Training configuration
    training_symbols: List[str] = Field(default_factory=list, description="Symbols to train on")
    training_start_date: Union[str, date, datetime] = Field(..., description="Training start date")
    training_end_date: Optional[Union[str, date, datetime]] = Field(None, description="Training end date")
    
    # Prediction configuration
    prediction_symbols: List[str] = Field(default_factory=list, description="Symbols to predict")
    prediction_days: int = Field(1, ge=1, le=365, description="Number of days to predict")
    prediction_output_path: str = Field("predictions", description="Path to save predictions")
    
    # API configuration
    api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for data sources")
    
    # MongoDB configuration
    mongo_uri: Optional[str] = Field(None, description="MongoDB connection URI")
    
    @validator('data_dir', 'models_dir', 'logs_dir', 'prediction_output_path')
    def create_directory(cls, v):
        """Create directory if it doesn't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator('default_model_type')
    def validate_default_model(cls, v):
        """Validate that default model type is supported."""
        supported_models = ['sklearn_rf', 'xgboost', 'lightgbm', 'keras', 'lstm', 'arima', 'prophet']
        if v not in supported_models:
            raise ValueError(f"Model type must be one of {supported_models}")
        return v

# Additional models can be added as needed for specific components
class PortfolioItem(BaseModel):
    symbol: str
    weight: float = Field(..., ge=0, le=1)

class UserPortfolio(BaseModel):
    items: List[PortfolioItem]

    @root_validator(skip_on_failure=True)
    def check_weights_sum(cls, values):
        items = values.get('items')
        if items and abs(sum(i.weight for i in items) - 1) > 1e-4:
            raise ValueError('Portfolio weights must sum to 1')
        return values
