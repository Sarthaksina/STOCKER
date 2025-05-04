"""
Anomaly detection component for STOCKER.
Modular, production-grade, and ready for pipeline integration.
"""
from src.utils import get_advanced_logger
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = get_advanced_logger("anomaly_detection", log_to_file=True, log_dir="logs")

def detect_anomalies(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Detect anomalies using configurable method (z-score, IQR, etc.).
    Returns DataFrame with anomaly flags.
    """
    method = config.get("method", "zscore")
    threshold = config.get("threshold", 3.0)
    df = df.copy()
    if method == "zscore":
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
        df["anomaly"] = (z_scores > threshold).any(axis=1)
    elif method == "iqr":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        mask = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        df["anomaly"] = mask
    else:
        logger.warning(f"Unknown anomaly detection method: {method}. No anomalies flagged.")
        df["anomaly"] = False
    logger.info(f"Anomaly detection complete. Method: {method}, threshold: {threshold}")
    return df
