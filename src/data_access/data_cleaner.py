"""
data_cleaner.py
Comprehensive data cleaning, validation, and transformation pipeline for STOCKER.
"""
import logging
import pandas as pd
import numpy as np
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

# ML imports for transformation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Project imports
from src.utils.logger import setup_logging
from src.entity.artifact_entity import ValidationArtifact
from src.components.pandera_schemas import HoldingsSchema, PriceHistorySchema, PortfolioSchema
from src.components.pydantic_models import UserProfile, StockerConfigSchema
from src.utils import validate_dataframe, validate_config, validate_user_input, get_advanced_logger

setup_logging()
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# Custom Exceptions
#------------------------------------------------------------------------------
class DataCleaningError(Exception):
    """Custom exception for data cleaning errors."""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataTransformationError(Exception):
    """Custom exception for data transformation errors."""
    pass

#------------------------------------------------------------------------------
# Validation Functions
#------------------------------------------------------------------------------
def log_validation_report(df, schema, context, errors=None, symbol=None, error_dict=None):
    """
    Validate a dataframe against a schema and log the results.
    
    Args:
        df: DataFrame to validate
        schema: Pandera schema to validate against
        context: Context string for logging
        errors: Optional list to append errors to
        symbol: Optional symbol for error tracking
        error_dict: Optional dictionary to track errors by symbol
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        schema.validate(df, lazy=True)
        logger.info(f"[VALIDATION PASS] {context}")
        if error_dict is not None and symbol is not None:
            error_dict.pop(symbol, None)
        return True, None
    except Exception as e:
        logger.error(f"[VALIDATION FAIL] {context}: {e}\n{traceback.format_exc()}")
        if error_dict is not None and symbol is not None:
            error_dict[symbol] = f"{context} validation failed: {e}"
        return False, str(e)

def compute_drift_metrics(df: pd.DataFrame, baseline: Optional[pd.DataFrame] = None) -> dict:
    """
    Compute schema and distribution drift metrics between baseline and new df.
    
    Args:
        df: Current DataFrame
        baseline: Baseline DataFrame to compare against
        
    Returns:
        Dictionary with drift metrics
    """
    drift = {"schema_drift": False, "distribution_drift": {}}
    if baseline is not None:
        # Schema drift: columns and types
        schema_drift = not (list(df.columns) == list(baseline.columns) and all(df.dtypes == baseline.dtypes))
        drift["schema_drift"] = schema_drift
        # Distribution drift: compare means/stds for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in baseline:
                mean_diff = abs(df[col].mean() - baseline[col].mean())
                std_diff = abs(df[col].std() - baseline[col].std())
                drift["distribution_drift"][col] = {"mean_diff": mean_diff, "std_diff": std_diff}
    return drift

def validate_holdings_df(df: pd.DataFrame, baseline: Optional[pd.DataFrame] = None) -> ValidationArtifact:
    """
    Validate holdings DataFrame structure and content.
    
    Args:
        df: Holdings DataFrame to validate
        baseline: Optional baseline DataFrame for drift detection
        
    Returns:
        ValidationArtifact with validation results
    """
    step = "validate_holdings_df"
    try:
        required_columns = ["symbol", "date", "public", "promoter", "fii", "dii"]
        dtypes = {"symbol": str, "date": 'datetime64[ns]'}
        validate_dataframe(df, required_columns=required_columns, dtypes=dtypes, allow_na=False, name="Holdings DataFrame")
        drift_metrics = compute_drift_metrics(df, baseline)
        return ValidationArtifact(
            step=step,
            status='success',
            drift_metrics=drift_metrics,
            data_hash=ValidationArtifact.hash_dataframe(df)
        )
    except Exception as e:
        return ValidationArtifact(
            step=step,
            status='fail',
            errors=[str(e)],
            data_hash=ValidationArtifact.hash_dataframe(df)
        )

def validate_holdings_df_pandera(df: pd.DataFrame, baseline: Optional[pd.DataFrame] = None) -> ValidationArtifact:
    """
    Validate holdings DataFrame using Pandera schema.
    
    Args:
        df: Holdings DataFrame to validate
        baseline: Optional baseline DataFrame for drift detection
        
    Returns:
        ValidationArtifact with validation results
    """
    step = "validate_holdings_df_pandera"
    try:
        HoldingsSchema.validate(df, lazy=True)
        drift_metrics = compute_drift_metrics(df, baseline)
        return ValidationArtifact(
            step=step,
            status='success',
            drift_metrics=drift_metrics,
            data_hash=ValidationArtifact.hash_dataframe(df)
        )
    except Exception as e:
        return ValidationArtifact(
            step=step,
            status='fail',
            errors=[str(e)],
            data_hash=ValidationArtifact.hash_dataframe(df)
        )

def validate_stocker_config(config: Dict[str, Any]) -> ValidationArtifact:
    """
    Validate Stocker configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        ValidationArtifact with validation results
    """
    step = "validate_stocker_config"
    try:
        required_keys = ["mongo_uri", "symbols"]
        validate_config(config, required_keys=required_keys, name="StockerConfig")
        return ValidationArtifact(step=step, status='success')
    except Exception as e:
        return ValidationArtifact(step=step, status='fail', errors=[str(e)])

def validate_stocker_config_pydantic(config: dict) -> ValidationArtifact:
    """
    Validate Stocker configuration using Pydantic model.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        ValidationArtifact with validation results
    """
    step = "validate_stocker_config_pydantic"
    try:
        StockerConfigSchema(**config)
        return ValidationArtifact(step=step, status='success')
    except Exception as e:
        return ValidationArtifact(step=step, status='fail', errors=[str(e)])

def validate_user_profile(user_info: Dict[str, Any]) -> ValidationArtifact:
    """
    Validate user profile data.
    
    Args:
        user_info: User profile dictionary to validate
        
    Returns:
        ValidationArtifact with validation results
    """
    step = "validate_user_profile"
    try:
        required_fields = ["risk_appetite", "age", "income"]
        validate_user_input(user_info, required_fields=required_fields, name="UserProfile")
        return ValidationArtifact(step=step, status='success')
    except Exception as e:
        return ValidationArtifact(step=step, status='fail', errors=[str(e)])

def validate_user_profile_pydantic(user_info: dict) -> ValidationArtifact:
    """
    Validate user profile using Pydantic model.
    
    Args:
        user_info: User profile dictionary to validate
        
    Returns:
        ValidationArtifact with validation results
    """
    step = "validate_user_profile_pydantic"
    try:
        UserProfile(**user_info)
        return ValidationArtifact(step=step, status='success')
    except Exception as e:
        return ValidationArtifact(step=step, status='fail', errors=[str(e)])

def validate_ml_features(features: Dict[str, Any], expected: list) -> ValidationArtifact:
    """
    Validate machine learning features.
    
    Args:
        features: Features dictionary to validate
        expected: List of expected feature names
        
    Returns:
        ValidationArtifact with validation results
    """
    step = "validate_ml_features"
    try:
        missing = [f for f in expected if f not in features]
        if missing:
            raise ValueError(f"ML features missing: {missing}")
        return ValidationArtifact(step=step, status='success')
    except Exception as e:
        return ValidationArtifact(step=step, status='fail', errors=[str(e)])

#------------------------------------------------------------------------------
# Data Cleaning Functions
#------------------------------------------------------------------------------
def validate_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate and clean stock price data.
    - Drops rows with missing essential fields.
    - Logs anomalies (missing, negative, or zero prices/volumes).
    
    Args:
        df: Stock price DataFrame to validate
        symbol: Stock symbol
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        DataCleaningError: If validation fails
    """
    if df.empty:
        logger.warning(f"Stock data for {symbol} is empty after download.")
        raise DataCleaningError(f"Stock data for {symbol} is empty after download.")
    
    essential_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in essential_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing columns in {symbol} stock data: {missing_cols}")
        raise DataCleaningError(f"Missing columns in {symbol} stock data: {missing_cols}")
    
    df_clean = df.dropna(subset=essential_cols)
    
    if len(df_clean) < len(df):
        logger.warning(f"Dropped {len(df) - len(df_clean)} rows with missing values for {symbol}")
    
    for col in ["Open", "High", "Low", "Close"]:
        if (df_clean[col] <= 0).any():
            logger.warning(f"Non-positive prices detected in {symbol} for column {col}")
    
    if (df_clean["Volume"] < 0).any():
        logger.warning(f"Negative volume detected in {symbol}")
    
    # Correct any non-positive price values by converting to absolute (flagged as anomalies)
    for col in ["Open", "High", "Low", "Close"]:
        df_clean[col] = df_clean[col].abs()
    
    if df_clean.empty:
        raise DataCleaningError(f"All stock data for {symbol} was invalid after cleaning.")
    
    return df_clean

def validate_quarterly_results(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate quarterly results (must have at least sales/revenue, drop rows with missing essential values).
    
    Args:
        df: Quarterly results DataFrame to validate
        symbol: Stock symbol
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        DataCleaningError: If validation fails
    """
    if df.empty:
        logger.warning(f"Quarterly results for {symbol} are empty.")
        raise DataCleaningError(f"Quarterly results for {symbol} are empty.")
    
    essential_cols = [col for col in df.columns if col.lower() in ["sales", "revenue"]]
    
    if not essential_cols:
        logger.error(f"No sales/revenue columns found in quarterly results for {symbol}.")
        raise DataCleaningError(f"No sales/revenue columns found in quarterly results for {symbol}.")
    
    df_clean = df.dropna(subset=essential_cols)
    
    if len(df_clean) < len(df):
        logger.warning(f"Dropped {len(df) - len(df_clean)} rows with missing sales/revenue for {symbol}")
    
    if df_clean.empty:
        raise DataCleaningError(f"All quarterly results for {symbol} were invalid after cleaning.")
    
    return df_clean

def validate_shareholding_pattern(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate shareholding pattern (must have at least one of Promoters/Public/FII/DII, drop rows with all missing).
    
    Args:
        df: Shareholding pattern DataFrame to validate
        symbol: Stock symbol
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        DataCleaningError: If validation fails
    """
    if df.empty:
        logger.warning(f"Shareholding pattern for {symbol} is empty.")
        raise DataCleaningError(f"Shareholding pattern for {symbol} is empty.")
    
    essential_cols = [col for col in df.columns if col in ['Promoters', 'Public', 'FII', 'DII']]
    
    if not essential_cols:
        logger.error(f"No Promoters/Public/FII/DII columns found in shareholding pattern for {symbol}.")
        raise DataCleaningError(f"No Promoters/Public/FII/DII columns found in shareholding pattern for {symbol}.")
    
    df_clean = df.dropna(subset=essential_cols, how="all")
    
    if len(df_clean) < len(df):
        logger.warning(f"Dropped {len(df) - len(df_clean)} rows with all missing shareholding for {symbol}")
    
    if df_clean.empty:
        raise DataCleaningError(f"All shareholding pattern for {symbol} were invalid after cleaning.")
    
    return df_clean

def validate_news_articles(news: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
    """
    Validate and clean a list of news articles.
    - Keeps only articles with non-empty 'title' and 'url'.
    
    Args:
        news: List of news article dictionaries
        symbol: Stock symbol
        
    Returns:
        Cleaned list of news articles
    """
    cleaned: List[Dict[str, Any]] = []
    
    for article in news:
        title = article.get("title")
        url = article.get("url")
        
        if (isinstance(title, str) and title.strip() and 
            isinstance(url, str) and url.strip()):
            cleaned.append(article)
    
    return cleaned

#------------------------------------------------------------------------------
# Data Transformation Functions
#------------------------------------------------------------------------------
def build_transformation_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Build a scikit-learn pipeline for data transformation based on configuration.
    
    Args:
        config: Transformation configuration dictionary
        
    Returns:
        scikit-learn Pipeline or None if no transformations specified
    """
    steps = []
    
    if config.get("imputation", "none") != "none":
        strategy = config["imputation"]
        steps.append(("imputer", SimpleImputer(strategy=strategy)))
    
    if config.get("scaling", "none") == "standard":
        steps.append(("scaler", StandardScaler()))
    elif config.get("scaling") == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    
    return Pipeline(steps) if steps else None

def apply_data_transformation(
    df: pd.DataFrame,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply data transformation pipeline to a DataFrame.
    
    Args:
        df: DataFrame to transform
        config: Transformation configuration dictionary
        logger: Optional logger instance
        
    Returns:
        Tuple of (transformed DataFrame, transformation artifact)
        
    Raises:
        DataTransformationError: If transformation fails
    """
    logger = logger or get_advanced_logger("data_transformation", log_to_file=True, log_dir="logs")
    artifact = {"steps": [], "errors": [], "feature_names": list(df.columns)}
    
    try:
        # Feature selection
        features = config.get("feature_selection", [])
        if features:
            df = df[features]
            artifact["steps"].append(f"Selected features: {features}")
        
        # Apply scikit-learn pipeline
        pipeline = build_transformation_pipeline(config)
        if pipeline:
            df_transformed = pipeline.fit_transform(df)
            if hasattr(pipeline, 'get_feature_names_out'):
                feature_names = pipeline.get_feature_names_out()
            else:
                feature_names = df.columns
            df = pd.DataFrame(df_transformed, columns=feature_names)
            artifact["steps"].append(f"Applied pipeline: {pipeline}")
        
        # Categorical encoding
        if config.get("encoding", "none") == "onehot":
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                encoded = encoder.fit_transform(df[cat_cols])
                encoded_cols = encoder.get_feature_names_out(cat_cols)
                df = pd.concat([
                    df.drop(columns=cat_cols),
                    pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
                ], axis=1)
                artifact["steps"].append(f"OneHot encoded: {cat_cols}")
        elif config.get("encoding") == "label":
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            for col in cat_cols:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
            artifact["steps"].append(f"Label encoded: {cat_cols}")
        
        # Custom pipeline if provided
        if config.get("custom_pipeline") is not None:
            custom_pipeline = config["custom_pipeline"]
            df = custom_pipeline.fit_transform(df)
            artifact["steps"].append("Applied custom pipeline.")
        
        artifact["feature_names"] = list(df.columns)
        logger.info(f"Data transformation completed. Steps: {artifact['steps']}")
    
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        artifact["errors"].append(str(e))
        raise DataTransformationError(f"Data transformation failed: {e}")
    
    return df, artifact

#------------------------------------------------------------------------------
# Market Calendar Functions
#------------------------------------------------------------------------------
def is_trading_day(date, exchange="NSE"):
    """
    Check if a given date is a trading day for the specified exchange.
    
    Args:
        date: Date to check (datetime or string in YYYY-MM-DD format)
        exchange: Exchange code (default: NSE)
        
    Returns:
        Boolean indicating if the date is a trading day
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    
    # Check if weekend
    if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # TODO: Add holiday calendar for different exchanges
    # For now, just using weekday check
    
    return True

def get_trading_days(start_date, end_date, exchange="NSE"):
    """
    Get a list of trading days between start_date and end_date.
    
    Args:
        start_date: Start date (datetime or string in YYYY-MM-DD format)
        end_date: End date (datetime or string in YYYY-MM-DD format)
        exchange: Exchange code (default: NSE)
        
    Returns:
        List of datetime.date objects representing trading days
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    trading_days = []
    current_date = start_date
    
    while current_date <= end_date:
        if is_trading_day(current_date, exchange):
            trading_days.append(current_date)
        current_date = current_date + pd.Timedelta(days=1)
    
    return trading_days

def get_previous_trading_day(date, exchange="NSE"):
    """
    Get the previous trading day before the given date.
    
    Args:
        date: Date (datetime or string in YYYY-MM-DD format)
        exchange: Exchange code (default: NSE)
        
    Returns:
        datetime.date object representing the previous trading day
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    
    current_date = date - pd.Timedelta(days=1)
    
    while not is_trading_day(current_date, exchange):
        current_date = current_date - pd.Timedelta(days=1)
    
    return current_date

def get_next_trading_day(date, exchange="NSE"):
    """
    Get the next trading day after the given date.
    
    Args:
        date: Date (datetime or string in YYYY-MM-DD format)
        exchange: Exchange code (default: NSE)
        
    Returns:
        datetime.date object representing the next trading day
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    
    current_date = date + pd.Timedelta(days=1)
    
    while not is_trading_day(current_date, exchange):
        current_date = current_date + pd.Timedelta(days=1)
    
    return current_date

#------------------------------------------------------------------------------
# Unified Data Cleaning Class
#------------------------------------------------------------------------------
class DataCleaner:
    """
    Unified data cleaning, validation, and transformation class.
    """
    
    @staticmethod
    def clean_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate stock price data.
        
        Args:
            df: Stock price DataFrame
            symbol: Stock symbol
            
        Returns:
            Cleaned DataFrame
        """
        return validate_stock_data(df, symbol)
    
    @staticmethod
    def clean_quarterly_results(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate quarterly results data.
        
        Args:
            df: Quarterly results DataFrame
            symbol: Stock symbol
            
        Returns:
            Cleaned DataFrame
        """
        return validate_quarterly_results(df, symbol)
    
    @staticmethod
    def clean_shareholding_pattern(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate shareholding pattern data.
        
        Args:
            df: Shareholding pattern DataFrame
            symbol: Stock symbol
            
        Returns:
            Cleaned DataFrame
        """
        return validate_shareholding_pattern(df, symbol)
    
    @staticmethod
    def clean_news_articles(news: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
        """
        Clean and validate news articles.
        
        Args:
            news: List of news article dictionaries
            symbol: Stock symbol
            
        Returns:
            Cleaned list of news articles
        """
        return validate_news_articles(news, symbol)
    
    @staticmethod
    def transform_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform data according to configuration.
        
        Args:
            df: DataFrame to transform
            config: Transformation configuration
            
        Returns:
            Tuple of (transformed DataFrame, transformation artifact)
        """
        return apply_data_transformation(df, config)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> ValidationArtifact:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ValidationArtifact with validation results
        """
        return validate_stocker_config_pydantic(config)