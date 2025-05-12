"""
Consolidated model implementations for STOCKER Pro.

This module provides implementations of various machine learning models used
for financial prediction tasks, including LSTM, XGBoost, LightGBM, and ensemble models.
"""
from typing import Dict, List, Union, Optional, Tuple, Any
import numpy as np
import pandas as pd
import os
import logging

from src.ml.base import BaseModel

# Import the individual model implementations
from src.ml.lstm_model import LSTMModel
from src.ml.xgboost_model import XGBoostModel
from src.ml.lightgbm_model import LightGBMModel
from src.ml.ensemble_model import EnsembleModel

# Re-export the model classes
__all__ = [
    'LSTMModel',
    'XGBoostModel',
    'LightGBMModel',
    'EnsembleModel'
] 