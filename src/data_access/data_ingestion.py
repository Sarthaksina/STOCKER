"""
data_ingestion.py
Handles robust stock data ingestion for the STOCKER pipeline.
"""
from src.utils.logger import setup_logging
from src.entity.config_entity import StockerConfig
from src.entity.artifact_entity import IngestionArtifact
from src.constant.constants import (
    NSE_SYMBOLS_URL,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_DATA_DIR,
    MAX_NEWS_ARTICLES
)
import logging
setup_logging()
logger = logging.getLogger(__name__)

import yfinance as yf
import pandas as pd
import os
import requests
from typing import List, Dict, Any
import time
import functools
import datetime
# Import news_agent from ingestion for industry-standard structure
from src.components.news_agent import search_news
from src.components.data_validation import (
    validate_holdings_df_pandera,
    validate_user_profile_pydantic,
    validate_stocker_config_pydantic
)
from src.components.pandera_schemas import HoldingsSchema, PriceHistorySchema, PortfolioSchema
import traceback

# --- Automated validation report utility (extended) ---
def log_validation_report(df, schema, context, errors=None, symbol=None, error_dict=None):
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


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


def get_all_nse_symbols(config: StockerConfig) -> List[str]:
    """
    Download and parse all NSE equity symbols from the official NSE CSV.
    Returns a list of symbols formatted for yfinance (e.g., 'RELIANCE.NS').
    """
    try:
        # If a CSV URL is provided in config, fetch from there; else use config.symbols
        if hasattr(config, 'nse_symbols_url') and config.nse_symbols_url:
            df = pd.read_csv(config.nse_symbols_url)
            symbols = df['SYMBOL'].astype(str).tolist()
            symbols = [s + ".NS" for s in symbols if s.isalnum()]
            logger.info(f"Loaded {len(symbols)} NSE symbols from CSV.")
            return symbols
        else:
            logger.info(f"Using symbols from config: {config.symbols}")
            return config.symbols
    except Exception as e:
        logger.error(f"Error loading NSE symbols: {e}")
        return config.symbols


def validate_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate and clean stock price data.
    - Drops rows with missing essential fields.
    - Logs anomalies (missing, negative, or zero prices/volumes).
    """
    if df.empty:
        logger.warning(f"Stock data for {symbol} is empty after download.")
        raise DataIngestionError(f"Stock data for {symbol} is empty after download.")
    essential_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in {symbol} stock data: {missing_cols}")
        raise DataIngestionError(f"Missing columns in {symbol} stock data: {missing_cols}")
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
        raise DataIngestionError(f"All stock data for {symbol} was invalid after cleaning.")
    return df_clean


def fetch_and_save_stock(config: StockerConfig, symbol: str) -> str:
    """
    Fetch historical stock data for a given symbol from Yahoo Finance (NSE/BSE supported)
    and save it as a CSV file in the data directory (with validation and cleaning).
    """
    os.makedirs(config.data_dir, exist_ok=True)
    start_date = config.start_date
    end_date = config.end_date
    csv_path = os.path.join(config.data_dir, f"{symbol.replace('.', '_')}.csv")
    logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
    df = yf.download(symbol, start=start_date, end=end_date)
    df = validate_stock_data(df, symbol)
    if df.empty:
        logger.warning(f"No valid data found for {symbol} after cleaning!")
        raise DataIngestionError(f"No valid data found for {symbol} after cleaning!")
    df.to_csv(csv_path)
    logger.info(f"Saved cleaned data to {csv_path}")
    return csv_path


def validate_quarterly_results(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate quarterly results (must have at least sales/revenue, drop rows with missing essential values).
    """
    if df.empty:
        logger.warning(f"Quarterly results for {symbol} are empty.")
        raise DataIngestionError(f"Quarterly results for {symbol} are empty.")
    essential_cols = [col for col in df.columns if col.lower() in ["sales", "revenue"]]
    if not essential_cols:
        logger.error(f"No sales/revenue columns found in quarterly results for {symbol}.")
        raise DataIngestionError(f"No sales/revenue columns found in quarterly results for {symbol}.")
    df_clean = df.dropna(subset=essential_cols)
    if len(df_clean) < len(df):
        logger.warning(f"Dropped {len(df) - len(df_clean)} rows with missing sales/revenue for {symbol}")
    if df_clean.empty:
        raise DataIngestionError(f"All quarterly results for {symbol} were invalid after cleaning.")
    return df_clean


def fetch_quarterly_results(config: StockerConfig, symbol: str) -> pd.DataFrame:
    """
    Fetch quarterly financial results for a stock (Indian market).
    Uses screener.in as a free source (publicly available tables).
    """
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Sales' in table.columns or 'Revenue' in table.columns:
                logger.info(f"Fetched quarterly results for {symbol}")
                return validate_quarterly_results(table, symbol)
        logger.warning(f"No quarterly results found for {symbol}")
        raise DataIngestionError(f"No quarterly results found for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching quarterly results for {symbol}: {e}")
        raise DataIngestionError(f"Error fetching quarterly results for {symbol}: {e}")


def validate_shareholding_pattern(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate shareholding pattern (must have at least one of Promoters/Public/FII/DII, drop rows with all missing).
    """
    if df.empty:
        logger.warning(f"Shareholding pattern for {symbol} is empty.")
        raise DataIngestionError(f"Shareholding pattern for {symbol} is empty.")
    essential_cols = [col for col in df.columns if col in ['Promoters', 'Public', 'FII', 'DII']]
    if not essential_cols:
        logger.error(f"No Promoters/Public/FII/DII columns found in shareholding pattern for {symbol}.")
        raise DataIngestionError(f"No Promoters/Public/FII/DII columns found in shareholding pattern for {symbol}.")
    df_clean = df.dropna(subset=essential_cols, how="all")
    if len(df_clean) < len(df):
        logger.warning(f"Dropped {len(df) - len(df_clean)} rows with all missing shareholding for {symbol}")
    if df_clean.empty:
        raise DataIngestionError(f"All shareholding pattern for {symbol} were invalid after cleaning.")
    return df_clean


def validate_news_articles(news: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
    """
    Validate and clean a list of news articles.
    - Keeps only articles with non-empty 'title' and 'url'.
    """
    cleaned: List[Dict[str, Any]] = []
    for article in news:
        title = article.get("title")
        url = article.get("url")
        if (
            isinstance(title, str) and title.strip()
            and isinstance(url, str) and url.strip()
        ):
            cleaned.append(article)
    return cleaned


def fetch_shareholding_pattern(config: StockerConfig, symbol: str) -> pd.DataFrame:
    """
    Fetch latest shareholding pattern (public, promoter, FII, DII) from Screener.in.
    """
    url = f"https://www.screener.in/company/{symbol}/shareholding/"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if any(x in table.columns for x in ['Promoters', 'Public', 'FII', 'DII']):
                logger.info(f"Fetched shareholding pattern for {symbol}")
                return validate_shareholding_pattern(table, symbol)
        logger.warning(f"No shareholding pattern found for {symbol}")
        raise DataIngestionError(f"No shareholding pattern found for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching shareholding for {symbol}: {e}")
        raise DataIngestionError(f"Error fetching shareholding for {symbol}: {e}")


def fetch_news_headlines(config: StockerConfig, symbol: str, max_articles: int = MAX_NEWS_ARTICLES, summarize: bool = True, sentiment: bool = True):
    """
    Fetch latest news headlines for a stock using the news_agent (Google News RSS, no API key required).
    Also provides LLM-powered summarization and sentiment analysis.
    Returns a list of dicts: [{title, url, publishedAt, summary, sentiment}]
    """
    try:
        news = search_news(symbol, max_articles=max_articles, summarize=summarize, sentiment=sentiment)
        if not news:
            raise DataIngestionError(f"No news found for {symbol}")
        return [n for n in news if n.get('title') and n.get('url')]
    except Exception as e:
        logger.error(f"Error fetching news for {symbol} in fetch_news_headlines: {e}")
        raise DataIngestionError(f"Error fetching news for {symbol} in fetch_news_headlines: {e}")


# --- Retry decorator for robust web requests ---
def retry_on_exception(max_retries=3, delay=2, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(delay)
            raise
        return wrapper
    return decorator

# --- Strict config/user validation at pipeline entry ---
def validate_pipeline_entry(config):
    try:
        validate_stocker_config_pydantic(config.__dict__)
        logger.info("Config validation passed.")
    except Exception as e:
        logger.critical(f"Config validation failed: {e}")
        raise

# --- Modular fetch/validate/save/report for each data type ---
@retry_on_exception(max_retries=3, delay=2, exceptions=(requests.RequestException,))
def robust_fetch_quarterly_results(config, symbol):
    return fetch_quarterly_results(config, symbol)

@retry_on_exception(max_retries=3, delay=2, exceptions=(requests.RequestException,))
def robust_fetch_shareholding_pattern(config, symbol):
    return fetch_shareholding_pattern(config, symbol)

@retry_on_exception(max_retries=3, delay=2, exceptions=(requests.RequestException,))
def robust_fetch_news(config, symbol):
    return fetch_news_headlines(config, symbol)

# --- Main pipeline orchestrator ---
def ingest_stock_data(config: StockerConfig) -> IngestionArtifact:
    from src.components.data_transformation import DataTransformation
    validate_pipeline_entry(config)
    raw_paths = []
    errors = {}
    timing = {}
    symbols = get_all_nse_symbols(config)
    transform_artifacts = {}
    for symbol in symbols:
        start_time = datetime.datetime.now()
        try:
            logger.info(f"--- Processing {symbol} ---")
            # Fetch and validate stock prices
            raw_path = fetch_and_save_stock(config, symbol)
            raw_paths.append(raw_path)
            df = pd.read_csv(raw_path)
            # Data transformation (robust, config-driven)
            try:
                df_transformed, transform_artifact = DataTransformation.transform(df, config.data_transformation)
                transform_artifacts[symbol] = transform_artifact
                # Optionally save transformed data
                transformed_path = raw_path.replace('.csv', '_transformed.csv')
                df_transformed.to_csv(transformed_path, index=False)
            except Exception as e:
                logger.error(f"Transformation failed for {symbol}: {e}")
                errors[symbol] = f"Transformation failed: {e}"
                continue
            valid, report = log_validation_report(df_transformed, PriceHistorySchema, f"Price history for {symbol}", errors, symbol)
            if not valid:
                logger.error(f"Validation failed for {symbol} price history. Skipping further processing. Error: {report}")
                continue
            # Quarterly results
            try:
                qtr = robust_fetch_quarterly_results(config, symbol.split('.')[0])
                if not qtr.empty:
                    valid, report = log_validation_report(qtr, PriceHistorySchema, f"Quarterly results for {symbol}", errors, symbol)
                    if valid:
                        qtr.to_csv(f"{config.data_dir}/{symbol.replace('.', '_')}_quarterly.csv", index=False)
            except Exception as e:
                logger.warning(f"Quarterly results unavailable for {symbol}: {e}")
            # Shareholding pattern
            try:
                shp = robust_fetch_shareholding_pattern(config, symbol.split('.')[0])
                if not shp.empty:
                    valid, report = log_validation_report(shp, HoldingsSchema, f"Shareholding pattern for {symbol}", errors, symbol)
                    if valid:
                        shp.to_csv(f"{config.data_dir}/{symbol.replace('.', '_')}_holdings.csv", index=False)
            except Exception as e:
                logger.warning(f"Shareholding pattern unavailable for {symbol}: {e}")
            # Portfolio validation (if portfolio data is available)
            portfolio_path = f"{config.data_dir}/{symbol.replace('.', '_')}_portfolio.csv"
            if os.path.exists(portfolio_path):
                df_portfolio = pd.read_csv(portfolio_path)
                valid, report = log_validation_report(df_portfolio, PortfolioSchema, f"Portfolio for {symbol}", errors, symbol)
                if not valid:
                    logger.error(f"Validation failed for {symbol} portfolio. Skipping further processing. Error: {report}")
            # News
            try:
                news = robust_fetch_news(config, symbol)
                if news:
                    df_news = pd.DataFrame(validate_news_articles(news, symbol))
                    df_news.to_csv(f"{config.data_dir}/{symbol.replace('.', '_')}_news.csv", index=False)
            except Exception as e:
                logger.warning(f"News unavailable for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Data ingestion error for {symbol}: {e}")
            errors[symbol] = str(e)
        end_time = datetime.datetime.now()
        timing[symbol] = (end_time - start_time).total_seconds()
        time.sleep(1)  # Respectful delay
    status = "success" if not errors else "partial_success"
    # Save validation summary report
    validation_report_path = os.path.join(config.data_dir, "validation_report.json")
    import json
    with open(validation_report_path, "w") as f:
        json.dump(errors, f, indent=2)
    logger.info(f"Validation report saved to {validation_report_path}")
    # Save timing report
    timing_report_path = os.path.join(config.data_dir, "timing_report.json")
    with open(timing_report_path, "w") as f:
        json.dump(timing, f, indent=2)
    logger.info(f"Timing report saved to {timing_report_path}")
    # Save transformation artifact report
    transform_report_path = os.path.join(config.data_dir, "transform_artifacts.json")
    with open(transform_report_path, "w") as f:
        json.dump(transform_artifacts, f, indent=2)
    logger.info(f"Transformation artifact report saved to {transform_report_path}")
    return IngestionArtifact(symbols=symbols, raw_data_paths=raw_paths, status=status, data_transformation=config.data_transformation, metadata={"errors": errors, "timing": timing, "transform_artifacts": transform_artifacts})

if __name__ == "__main__":
    config = StockerConfig()
    ingestion_artifact = ingest_stock_data(config)
    logger.info(f"Data ingestion completed with status: {ingestion_artifact.status}")