"""
Schema definitions for data validation in STOCKER Pro.

This module provides Pandera schemas for validating data structures
used throughout the application, ensuring data consistency and quality.
"""
import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import Series, DataFrame
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

class StockDataSchema(pa.SchemaModel):
    """
    Schema for validating stock price data.
    
    Validates the following columns:
    - open: Opening price (float)
    - high: High price (float)
    - low: Low price (float)
    - close: Closing price (float)
    - volume: Trading volume (int)
    - date: Date (index)
    """
    
    open: Series[float] = pa.Field(gt=0, coerce=True)
    high: Series[float] = pa.Field(gt=0, coerce=True)
    low: Series[float] = pa.Field(gt=0, coerce=True)
    close: Series[float] = pa.Field(gt=0, coerce=True)
    volume: Series[int] = pa.Field(ge=0, coerce=True)
    
    # Additional optional columns that might be present
    adjusted_close: Optional[Series[float]] = pa.Field(gt=0, coerce=True, nullable=True)
    dividend_amount: Optional[Series[float]] = pa.Field(ge=0, coerce=True, nullable=True)
    split_coefficient: Optional[Series[float]] = pa.Field(ge=0, coerce=True, nullable=True)
    
    class Config:
        coerce = True
        strict = False  # Allow additional columns
        index_type = pd.DatetimeIndex

class ReturnsSchema(pa.SchemaModel):
    """
    Schema for validating returns data.
    
    Each column represents returns for a different asset.
    The index is a DatetimeIndex representing the date of the returns.
    """
    
    class Config:
        coerce = True
        strict = False  # Allow any columns
        index_type = pd.DatetimeIndex
    
    # Dynamic column validation
    @pa.check("*", name="valid_returns")
    def check_returns_values(cls, series: Series[float]) -> bool:
        """Check that returns are within reasonable bounds."""
        # Returns should generally be between -100% and +100%
        return ((series >= -1.0) & (series <= 1.0) | pd.isna(series)).all()

class PortfolioSchema(pa.SchemaModel):
    """
    Schema for validating portfolio data.
    
    Contains:
    - asset: Asset identifier (str)
    - weight: Portfolio weight (float)
    """
    
    asset: Series[str] = pa.Field()
    weight: Series[float] = pa.Field(ge=0, le=1.0, coerce=True)
    
    class Config:
        coerce = True
        strict = False  # Allow additional columns

class TechnicalIndicatorSchema(pa.SchemaModel):
    """
    Schema for validating technical indicator data.
    
    Each indicator may have different columns, so this is a base schema
    that can be extended for specific indicators.
    """
    
    # Common metadata columns
    symbol: Optional[Series[str]] = pa.Field(nullable=True)
    indicator: Optional[Series[str]] = pa.Field(nullable=True)
    
    class Config:
        coerce = True
        strict = False  # Allow any columns
        index_type = pd.DatetimeIndex

class SimpleMovingAverageSchema(TechnicalIndicatorSchema):
    """Schema for Simple Moving Average (SMA) indicator."""
    SMA: Series[float] = pa.Field(gt=0, coerce=True)

class ExponentialMovingAverageSchema(TechnicalIndicatorSchema):
    """Schema for Exponential Moving Average (EMA) indicator."""
    EMA: Series[float] = pa.Field(gt=0, coerce=True)

class RelativeStrengthIndexSchema(TechnicalIndicatorSchema):
    """Schema for Relative Strength Index (RSI) indicator."""
    RSI: Series[float] = pa.Field(ge=0, le=100, coerce=True)

class MacdSchema(TechnicalIndicatorSchema):
    """Schema for Moving Average Convergence Divergence (MACD) indicator."""
    MACD: Series[float] = pa.Field(coerce=True)
    MACD_Hist: Series[float] = pa.Field(coerce=True)
    MACD_Signal: Series[float] = pa.Field(coerce=True)

class BollingerBandsSchema(TechnicalIndicatorSchema):
    """Schema for Bollinger Bands indicator."""
    Real_Middle_Band: Series[float] = pa.Field(gt=0, coerce=True)
    Real_Upper_Band: Series[float] = pa.Field(gt=0, coerce=True)
    Real_Lower_Band: Series[float] = pa.Field(gt=0, coerce=True)

# Dictionary mapping indicator names to their schemas
INDICATOR_SCHEMAS = {
    'SMA': SimpleMovingAverageSchema,
    'EMA': ExponentialMovingAverageSchema,
    'RSI': RelativeStrengthIndexSchema,
    'MACD': MacdSchema,
    'BBANDS': BollingerBandsSchema
}

def validate_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate stock price data against StockDataSchema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validated DataFrame
        
    Raises:
        pa.errors.SchemaError: If validation fails
    """
    return StockDataSchema.validate(df)

def validate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate returns data against ReturnsSchema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validated DataFrame
        
    Raises:
        pa.errors.SchemaError: If validation fails
    """
    return ReturnsSchema.validate(df)

def validate_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate portfolio data against PortfolioSchema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validated DataFrame
        
    Raises:
        pa.errors.SchemaError: If validation fails
    """
    return PortfolioSchema.validate(df)

def validate_technical_indicator(df: pd.DataFrame, indicator_type: str) -> pd.DataFrame:
    """
    Validate technical indicator data against the appropriate schema.
    
    Args:
        df: DataFrame to validate
        indicator_type: Type of indicator (e.g., 'SMA', 'EMA', 'RSI')
        
    Returns:
        Validated DataFrame
        
    Raises:
        pa.errors.SchemaError: If validation fails
        ValueError: If indicator type is unknown
    """
    if indicator_type not in INDICATOR_SCHEMAS:
        raise ValueError(f"Unknown indicator type: {indicator_type}")
        
    schema = INDICATOR_SCHEMAS[indicator_type]
    return schema.validate(df)

def check_dataframe_integrity(df: pd.DataFrame, expected_columns: List[str] = None, 
                            min_rows: int = 1, allow_nans: bool = False) -> bool:
    """
    Check general DataFrame integrity.
    
    Args:
        df: DataFrame to check
        expected_columns: List of expected column names
        min_rows: Minimum expected number of rows
        allow_nans: Whether to allow NaN values
        
    Returns:
        True if DataFrame passes all checks, False otherwise
    """
    # Check if DataFrame is valid
    if df is None or not isinstance(df, pd.DataFrame):
        return False
        
    # Check row count
    if len(df) < min_rows:
        return False
        
    # Check columns
    if expected_columns and not all(col in df.columns for col in expected_columns):
        return False
        
    # Check for NaNs
    if not allow_nans and df.isna().any().any():
        return False
        
    return True 