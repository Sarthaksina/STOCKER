\"""General helper functions for STOCKER Pro"""

import pandas as pd
import logging
from typing import List

# Configure logging
logger = logging.getLogger(__name__)

def top_n_recommender(df: pd.DataFrame, score_col: str = "score", n: int = 5) -> List[str]:
    """
    Returns top-N items by score.
    
    Args:
        df: DataFrame with scores
        score_col: Column name for scores
        n: Number of items to return
        
    Returns:
        List of top-N item indices
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Input 'df' must be a pandas DataFrame, got {type(df)}")
        return []
        
    if score_col not in df.columns:
        logger.warning(f"Score column '{score_col}' not found in DataFrame columns: {df.columns.tolist()}")
        return []
        
    try:
        # Ensure the score column is numeric before sorting
        if not pd.api.types.is_numeric_dtype(df[score_col]):
            logger.warning(f"Score column '{score_col}' is not numeric. Attempting conversion.")
            # Attempt conversion, coercing errors to NaN
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            # Drop rows where conversion failed
            df = df.dropna(subset=[score_col])

        if df.empty:
            logger.warning("DataFrame is empty after handling non-numeric scores.")
            return []

        # Sort and get top N
        top_items = df.sort_values(score_col, ascending=False).head(n)
        return top_items.index.tolist()
    except Exception as e:
        logger.error(f"Error in top_n_recommender while sorting/selecting top N: {e}")
        return []