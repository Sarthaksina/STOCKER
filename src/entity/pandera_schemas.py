"""
Pandera schemas for STOCKER Pro data validation.

This module defines schemas for validating DataFrames used throughout the application,
ensuring data quality and consistency before processing.
"""
import pandera as pa
from pandera.typing import Series, DataFrame
import pandas as pd
from typing import Optional
import numpy as np

# Define schema for holdings data
class HoldingsSchema(pa.SchemaModel):
    """Schema for holdings data validation."""
    
    symbol: Series[str] = pa.Field(
        nullable=False,
        description="Stock symbol"
    )
    date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        description="Date of holding"
    )
    public: Series[float] = pa.Field(
        ge=0.0,
        le=100.0,
        nullable=True,
        description="Public holding percentage"
    )
    promoter: Series[float] = pa.Field(
        ge=0.0,
        le=100.0,
        nullable=True,
        description="Promoter holding percentage"
    )
    fii: Series[float] = pa.Field(
        ge=0.0,
        le=100.0,
        nullable=True,
        description="Foreign Institutional Investor holding percentage"
    )
    dii: Series[float] = pa.Field(
        ge=0.0,
        le=100.0,
        nullable=True,
        description="Domestic Institutional Investor holding percentage"
    )
    
    # Define checks that apply to the entire dataframe
    @pa.check("symbol", "date")
    def unique_symbol_date_combo(cls, symbol: Series, date: Series) -> Series:
        """Check that each symbol-date combination is unique."""
        return ~pd.DataFrame({"symbol": symbol, "date": date}).duplicated().values
    
    # Ensure total percentage is valid
    @pa.check("public", "promoter", "fii", "dii")
    def total_percentage_check(cls, public: Series, promoter: Series, 
                              fii: Series, dii: Series) -> Series:
        """Check that percentages sum to approximately 100%."""
        total = public.fillna(0) + promoter.fillna(0) + fii.fillna(0) + dii.fillna(0)
        # Allow some tolerance for rounding errors
        return (total >= 95.0) & (total <= 105.0)

# Define schema for stock price data
class StockPriceSchema(pa.SchemaModel):
    """Schema for stock price data validation."""
    
    date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        description="Trading date"
    )
    open: Series[float] = pa.Field(
        gt=0.0,
        nullable=False,
        description="Opening price"
    )
    high: Series[float] = pa.Field(
        gt=0.0,
        nullable=False,
        description="Highest price"
    )
    low: Series[float] = pa.Field(
        gt=0.0,
        nullable=False,
        description="Lowest price"
    )
    close: Series[float] = pa.Field(
        gt=0.0,
        nullable=False,
        description="Closing price"
    )
    volume: Series[int] = pa.Field(
        ge=0,
        nullable=True,
        description="Trading volume"
    )
    
    # Define logical checks
    @pa.check("high", "low", "open", "close")
    def price_consistency_check(cls, high: Series, low: Series, 
                               open_price: Series, close: Series) -> Series:
        """Check that high >= low and high/low are consistent with open/close."""
        high_low_check = high >= low
        open_range_check = (open_price >= low) & (open_price <= high)
        close_range_check = (close >= low) & (close <= high)
        return high_low_check & open_range_check & close_range_check

# Define schema for feature data
class FeatureDataSchema(pa.SchemaModel):
    """Schema for feature data validation."""
    
    # Dynamic schema that can adapt to different feature sets
    # The minimum required columns are defined here
    date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        description="Date for features"
    )
    symbol: Series[str] = pa.Field(
        nullable=False,
        description="Stock symbol"
    )
    
    # Additional columns will be validated based on their data types
    class Config:
        # Allow columns not explicitly defined in the schema
        strict = False

# Define schema for model prediction data
class PredictionSchema(pa.SchemaModel):
    """Schema for model prediction data validation."""
    
    date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        description="Prediction date"
    )
    symbol: Series[str] = pa.Field(
        nullable=False,
        description="Stock symbol"
    )
    predicted_value: Series[float] = pa.Field(
        nullable=False,
        description="Predicted value"
    )
    confidence: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        nullable=True,
        description="Prediction confidence"
    )
    
    # Define checks for the dataframe
    @pa.check("symbol", "date")
    def unique_prediction_check(cls, symbol: Series, date: Series) -> Series:
        """Check that each symbol-date combination is unique."""
        return ~pd.DataFrame({"symbol": symbol, "date": date}).duplicated().values

# Function to create a dynamic schema based on expected columns
def create_dynamic_schema(
    required_columns: dict,
    optional_columns: Optional[dict] = None,
    allow_unknown: bool = True
) -> pa.DataFrameSchema:
    """
    Create a dynamic Pandera schema based on expected columns.
    
    Args:
        required_columns: Dict of column name -> (type, constraints)
        optional_columns: Optional dict of column name -> (type, constraints)
        allow_unknown: Whether to allow columns not in the schema
        
    Returns:
        Pandera DataFrameSchema
    """
    schema_dict = {}
    
    # Add required columns
    for col_name, (dtype, constraints) in required_columns.items():
        schema_dict[col_name] = pa.Column(dtype, **constraints)
    
    # Add optional columns if provided
    if optional_columns:
        for col_name, (dtype, constraints) in optional_columns.items():
            schema_dict[col_name] = pa.Column(dtype, **constraints, nullable=True)
    
    # Create and return schema
    return pa.DataFrameSchema(
        schema_dict,
        strict=not allow_unknown
    )
