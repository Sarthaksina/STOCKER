"""
Robust, extensible model training utility for STOCKER.
Supports: sklearn, XGBoost, LightGBM, and deep learning (Keras).
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional
import logging
from src.utils import get_advanced_logger
from src.pipeline.training_pipeline import TrainingPipeline

def train_model(features: pd.DataFrame, config: Dict[str, Any], best_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, float]]:
    pipeline = TrainingPipeline(config)
    return pipeline.run()

def save_model(model: Any, version: str, config: Dict[str, Any]):
    """
    Save the trained model to disk (format depends on type).
    """
    import os
    logger = get_advanced_logger("model_saving", log_to_file=True, log_dir="logs")
    model_type = config.get('model_type', 'sklearn_rf')
    save_dir = config.get('model_save_dir', 'models')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"model_{model_type}_{version}.pkl")
    try:
        if model_type in ['sklearn_rf', 'xgboost', 'lightgbm']:
            import joblib
            joblib.dump(model, path)
        elif model_type == 'keras':
            model.save(os.path.join(save_dir, f"model_{model_type}_{version}.h5"))
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        logger.info(f"Model saved at {path}")
    except Exception as e:
        logger.error(f"Model saving failed: {e}")
        raise

def get_model_version(config: Dict[str, Any]) -> str:
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')
