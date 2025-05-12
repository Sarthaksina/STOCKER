"""
Machine learning module for STOCKER Pro.

This module provides machine learning models, evaluation metrics, and pipelines
for financial forecasting and analysis.
"""

from src.ml.base import BaseModel, ModelConfig, load_model, save_model
from src.ml.models import (
    EnsembleModel,
    LSTMModel,
    XGBoostModel,
    LightGBMModel
)
from src.ml.evaluation import (
    evaluate_model,
    calculate_metrics,
    plot_predictions,
    plot_feature_importance
)
from src.ml.pipelines import (
    TrainingPipeline,
    PredictionPipeline,
    create_training_pipeline,
    create_prediction_pipeline
)

__all__ = [
    # Base classes and functions
    'BaseModel',
    'ModelConfig',
    'load_model',
    'save_model',
    
    # Model implementations
    'EnsembleModel',
    'LSTMModel', 
    'XGBoostModel',
    'LightGBMModel',
    
    # Evaluation
    'evaluate_model',
    'calculate_metrics',
    'plot_predictions',
    'plot_feature_importance',
    
    # Pipelines
    'TrainingPipeline',
    'PredictionPipeline',
    'create_training_pipeline',
    'create_prediction_pipeline'
]
