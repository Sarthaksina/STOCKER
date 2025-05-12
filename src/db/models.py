"""
Database models for STOCKER Pro.

This module provides Pydantic models for database entities and
data transfer objects used throughout the application.
"""
from pydantic import BaseModel, Field, root_validator, validator
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime, date
from enum import Enum
import json
import uuid

class TimeFrame(str, Enum):
    """Time frame for stock data."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    INTRADAY = "intraday"

class AssetType(str, Enum):
    """Types of financial assets."""
    STOCK = "stock"
    ETF = "etf"
    FOREX = "forex"
    CRYPTO = "crypto"
    INDEX = "index"
    COMMODITY = "commodity"
    BOND = "bond"

class ModelType(str, Enum):
    """Types of prediction models."""
    LSTM = "lstm"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"
    ARIMA = "arima"
    PROPHET = "prophet"
    CUSTOM = "custom"

class PredictionHorizon(str, Enum):
    """Prediction horizons."""
    DAY_1 = "1d"
    DAY_3 = "3d"
    DAY_5 = "5d"
    DAY_10 = "10d"
    WEEK_2 = "2w"
    MONTH_1 = "1m"
    MONTH_3 = "3m"

class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""
    EFFICIENT_FRONTIER = "efficient_frontier"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"

class StockDataFilter(BaseModel):
    """Filter for stock data queries."""
    symbol: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    time_frame: TimeFrame = TimeFrame.DAILY
    include_indicators: bool = False
    indicators: Optional[List[str]] = None

class StockPrice(BaseModel):
    """Stock price data point."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    dividend_amount: Optional[float] = None
    split_coefficient: Optional[float] = None

class StockDataResponse(BaseModel):
    """Response model for stock data API."""
    symbol: str
    time_frame: TimeFrame
    prices: List[StockPrice]
    indicators: Optional[Dict[str, List[Dict[str, Any]]]] = None
    metadata: Optional[Dict[str, Any]] = None

class CompanyInfo(BaseModel):
    """Company information."""
    symbol: str
    name: str
    description: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: Optional[str] = None
    country: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    updated_at: datetime = Field(default_factory=datetime.now)

class Portfolio(BaseModel):
    """Portfolio model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    user_id: Optional[str] = None
    assets: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    risk_profile: Optional[str] = None
    optimization_method: OptimizationMethod = OptimizationMethod.EFFICIENT_FRONTIER
    metrics: Optional[Dict[str, Any]] = None
    
    @validator('assets')
    def validate_assets(cls, v):
        """Validate that assets have required fields."""
        for asset in v:
            if 'symbol' not in asset:
                raise ValueError("Each asset must have a 'symbol' field")
            if 'weight' not in asset:
                raise ValueError("Each asset must have a 'weight' field")
        
        # Check that weights sum to approximately 1
        weights_sum = sum(asset['weight'] for asset in v)
        if not 0.99 <= weights_sum <= 1.01:
            raise ValueError(f"Asset weights must sum to 1, got {weights_sum}")
        
        return v

class PredictionModel(BaseModel):
    """Machine learning model metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    model_type: ModelType
    target_symbol: str
    features: List[str]
    prediction_horizon: PredictionHorizon
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    training_start_date: Optional[date] = None
    training_end_date: Optional[date] = None
    metrics: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    status: str = "created"

class PredictionRequest(BaseModel):
    """Request for a price prediction."""
    symbol: str
    model_id: Optional[str] = None
    model_type: Optional[ModelType] = None
    prediction_horizon: PredictionHorizon = PredictionHorizon.DAY_5
    include_confidence_intervals: bool = False
    confidence_level: float = 0.95

class PredictionResponse(BaseModel):
    """Response for a price prediction."""
    symbol: str
    model_id: str
    model_type: ModelType
    prediction_horizon: PredictionHorizon
    predictions: List[Dict[str, Any]]
    confidence_intervals: Optional[List[Dict[str, Any]]] = None
    predicted_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class User(BaseModel):
    """User model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    hashed_password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    is_admin: bool = False
    preferences: Optional[Dict[str, Any]] = None

class TrainingJob(BaseModel):
    """Model training job."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"
    logs: List[str] = []
    parameters: Dict[str, Any]
    result_metrics: Optional[Dict[str, float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class BacktestRequest(BaseModel):
    """Request for a strategy backtest."""
    portfolio_id: Optional[str] = None
    symbols: Optional[List[str]] = None
    weights: Optional[List[float]] = None
    start_date: date
    end_date: date
    initial_capital: float = 10000.0
    rebalance_frequency: Optional[str] = None
    transaction_cost: float = 0.001
    include_metrics: bool = True

class BacktestResult(BaseModel):
    """Result of a strategy backtest."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    portfolio_id: Optional[str] = None
    symbols: List[str]
    weights: List[float]
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    returns: List[Dict[str, Any]]
    metrics: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('weights')
    def validate_weights(cls, v, values):
        """Validate that weights match the number of symbols."""
        if 'symbols' in values and len(values['symbols']) != len(v):
            raise ValueError(f"Number of weights ({len(v)}) must match number of symbols ({len(values['symbols'])})")
        return v

class OptimizationRequest(BaseModel):
    """Request for portfolio optimization."""
    symbols: List[str]
    start_date: date
    end_date: date
    optimization_method: OptimizationMethod = OptimizationMethod.EFFICIENT_FRONTIER
    constraints: Optional[Dict[str, Any]] = None
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    target_risk: Optional[float] = None

class OptimizationResult(BaseModel):
    """Result of portfolio optimization."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbols: List[str]
    weights: Dict[str, float]
    optimization_method: OptimizationMethod
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    efficient_frontier: Optional[List[Dict[str, float]]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('weights')
    def validate_portfolio_weights(cls, v, values):
        """Validate that weights sum to 1 and match symbols."""
        if 'symbols' in values and set(v.keys()) != set(values['symbols']):
            raise ValueError("Weights keys must match symbols")
        
        weights_sum = sum(v.values())
        if not 0.99 <= weights_sum <= 1.01:
            raise ValueError(f"Weights must sum to 1, got {weights_sum}")
        
        return v

class ModelArtifact(BaseModel):
    """Model artifact for storage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    artifact_type: str  # 'weights', 'config', 'metrics', etc.
    file_path: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelArtifact':
        """Create from JSON string."""
        data = json.loads(json_str)
        
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            
        return cls(**data)

class DatasetMetadata(BaseModel):
    """Metadata for a dataset."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    symbols: List[str]
    start_date: date
    end_date: date
    time_frame: TimeFrame
    features: List[str]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    file_path: Optional[str] = None
    row_count: int
    column_count: int
    has_missing_values: bool = False
    statistics: Optional[Dict[str, Any]] = None

class NewsItem(BaseModel):
    """News item related to a stock."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: Optional[str] = None
    title: str
    source: str
    url: str
    published_at: datetime
    summary: Optional[str] = None
    sentiment: Optional[float] = None
    relevance: Optional[float] = None
    tags: List[str] = []
    collected_at: datetime = Field(default_factory=datetime.now)

class Alert(BaseModel):
    """User alert configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    symbol: str
    alert_type: str  # 'price', 'volume', 'indicator', etc.
    condition: str  # 'above', 'below', 'crossing', etc.
    threshold: float
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    message_template: Optional[str] = None 