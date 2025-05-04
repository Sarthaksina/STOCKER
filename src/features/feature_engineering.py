"""
Unified feature engineering and transformation component for STOCKER.
- Handles all feature creation, selection, outlier removal, SMOTE, etc.
- Use this for all model prep steps.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.components.data_transformation import build_transformation_pipeline
from imblearn.over_sampling import SMOTE
import logging
from src.utils import get_advanced_logger

def remove_outliers(df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
    if method == "zscore":
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
        return df[(z_scores < threshold).all(axis=1)]
    elif method == "iqr":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    else:
        return df

def feature_engineer(
    df: pd.DataFrame,
    config: Dict[str, Any],
    target_col: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logger = logger or get_advanced_logger("feature_engineering", log_to_file=True, log_dir="logs")
    artifact = {"steps": [], "errors": [], "feature_names": list(df.columns)}
    try:
        pipeline = build_transformation_pipeline(config.get("data_transformation", {}))
        df = pd.DataFrame(pipeline.fit_transform(df), columns=df.columns)
        artifact["steps"].append("Applied data transformation pipeline.")
        if config.get("outlier_removal", {}).get("enabled", False):
            method = config["outlier_removal"].get("method", "zscore")
            threshold = config["outlier_removal"].get("threshold", 3.0)
            before = len(df)
            df = remove_outliers(df, method, threshold)
            after = len(df)
            artifact["steps"].append(f"Removed outliers using {method}, threshold={threshold} (rows: {before} -> {after})")
        if target_col and config.get("imbalance_handling", {}).get("enabled", False):
            smote = SMOTE()
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_res, y_res = smote.fit_resample(X, y)
            df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)
            artifact["steps"].append("Applied SMOTE for class imbalance.")
        if config.get("custom_features"):
            for func in config["custom_features"]:
                df = func(df)
                artifact["steps"].append(f"Applied custom feature: {func.__name__}")
        artifact["feature_names"] = list(df.columns)
        logger.info(f"Feature engineering completed. Steps: {artifact['steps']}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        artifact["errors"].append(str(e))
        raise
    return df, artifact
