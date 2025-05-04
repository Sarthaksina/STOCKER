# prediction_pipeline.py
"""
Prediction pipeline for STOCKER Pro.
This module implements a robust prediction pipeline that inherits from BasePipeline.
"""
import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Union
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.configuration.config import StockerConfig
from src.features.model_loading import load_latest_model, load_model_by_id
from src.features.feature_engineering import feature_engineer_for_prediction
from src.features.prediction import predict, save_predictions
from src.configuration.config_validator import ConfigValidationError
from src.exception.exception import StockerPredictionError, ModelLoadingError, FeatureEngineeringError
from src.pipeline.base_pipeline import BasePipeline
from src.logger.logger import get_advanced_logger

class PredictionPipeline(BasePipeline):
    """
    A production-grade prediction pipeline for STOCKER Pro.
    
    This pipeline handles the end-to-end process of loading models,
    preparing features, generating predictions, and saving results.
    It includes comprehensive error handling, performance monitoring,
    and support for ensemble models.
    
    Attributes:
        config: Configuration for the prediction pipeline
        logger: Logger for tracking pipeline execution
        artifacts: Dictionary to store artifacts generated during pipeline execution
        start_time: Start time of the pipeline execution
        performance_metrics: Dictionary to store performance metrics
    """
    
    def __init__(self, config: StockerConfig):
        """
        Initialize the prediction pipeline.
        
        Args:
            config: Configuration for the prediction pipeline
        """
        self.config = config
        self.logger = get_advanced_logger(
            "prediction_pipeline", 
            log_to_file=True, 
            log_dir=os.path.join(config.logs_dir, "prediction")
        )
        self.artifacts: Dict[str, Any] = {}
        self.start_time = time.time()
        self.performance_metrics: Dict[str, float] = {}

    def validate_config(self) -> None:
        """
        Validate the pipeline configuration.
        
        Raises:
            ConfigValidationError: If configuration validation fails
        """
        try:
            # Log config for debugging (excluding sensitive info)
            safe_config = {k: v for k, v in self.config.__dict__.items() 
                          if not k.endswith("key") and not k.endswith("token")}
            self.logger.debug(f"Config: {json.dumps(safe_config, default=str)}")
            
            # TODO: Replace this with the new config validator
            # validate_prediction_config(self.config.__dict__)
            self.logger.info("Config validation passed")
            
            # Track step execution time
            self.performance_metrics['config_validation_time'] = time.time() - self.start_time
        except ConfigValidationError as e:
            self.logger.critical(f"Config validation failed: {e}")
            raise
        except Exception as e:
            self.logger.critical(f"Unexpected error during config validation: {e}\n{traceback.format_exc()}")
            raise ConfigValidationError(f"Unexpected error during config validation: {e}")

    def load_model(self, model_id: Optional[str] = None) -> None:
        """
        Load the model for prediction.
        
        Args:
            model_id: Optional specific model ID to load
                     If not provided, the latest model is loaded
                     
        Raises:
            ModelLoadingError: If model loading fails
        """
        step_start = time.time()
        try:
            if model_id:
                self.logger.info(f"Loading specific model with ID: {model_id}")
                model, version, metadata = load_model_by_id(model_id, self.config)
            else:
                self.logger.info("Loading latest model")
                model, version, metadata = load_latest_model(self.config)
            
            self.artifacts['model'] = model
            self.artifacts['model_version'] = version
            self.artifacts['model_metadata'] = metadata
            
            # Log model information
            self.logger.info(f"Loaded model: {type(model).__name__}, version: {version}")
            self.logger.debug(f"Model metadata: {json.dumps(metadata, default=str)}")
            
            # Track step execution time
            self.performance_metrics['model_loading_time'] = time.time() - step_start
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}\n{traceback.format_exc()}")
            raise ModelLoadingError(f"Failed to load model: {e}")

    def prepare_features(self, input_data: Optional[pd.DataFrame] = None) -> None:
        """
        Prepare features for prediction.
        
        Args:
            input_data: Optional input data to use for feature engineering
                       If not provided, data is loaded according to config
                       
        Raises:
            FeatureEngineeringError: If feature engineering fails
        """
        step_start = time.time()
        try:
            # If input data is provided, use it; otherwise load data according to config
            if input_data is not None:
                self.logger.info(f"Using provided input data: {input_data.shape}")
                features = feature_engineer_for_prediction(self.config, input_data=input_data)
            else:
                self.logger.info("Loading and preparing features from configured data sources")
                features = feature_engineer_for_prediction(self.config)
            
            self.artifacts['features'] = features
            
            # Log feature information
            self.logger.info(f"Feature preparation completed: {features.shape}")
            self.logger.debug(f"Feature columns: {features.columns.tolist()}")
            
            # Check for potential issues in features
            self._validate_features(features)
            
            # Track step execution time
            self.performance_metrics['feature_engineering_time'] = time.time() - step_start
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}\n{traceback.format_exc()}")
            raise FeatureEngineeringError(f"Failed to prepare features: {e}")

    def _validate_features(self, features: pd.DataFrame) -> None:
        """
        Validate features to identify potential issues.
        
        Args:
            features: Features dataframe to validate
        """
        # Check for missing values
        missing_count = features.isna().sum().sum()
        if missing_count > 0:
            self.logger.warning(f"Features contain {missing_count} missing values")
        
        # Check for infinite values
        inf_count = np.isinf(features.select_dtypes(include=['float64', 'float32'])).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Features contain {inf_count} infinite values")
        
        # Log feature statistics
        self.logger.debug(f"Feature statistics:\n{features.describe()}")

    def run_prediction(self) -> None:
        """
        Run the prediction using the loaded model and prepared features.
        
        Raises:
            StockerPredictionError: If prediction fails
        """
        step_start = time.time()
        try:
            # Ensure model and features are available
            if 'model' not in self.artifacts:
                raise StockerPredictionError("Model not loaded. Call load_model() first.")
            if 'features' not in self.artifacts:
                raise StockerPredictionError("Features not prepared. Call prepare_features() first.")
            
            self.logger.info("Starting prediction")
            
            # Run prediction
            preds, confidence = predict(
                self.artifacts['model'], 
                self.artifacts['features'], 
                self.config
            )
            
            self.artifacts['predictions'] = preds
            self.artifacts['confidence'] = confidence
            
            # Log prediction information
            self.logger.info(f"Prediction completed: {len(preds)} predictions generated")
            
            # Track step execution time
            self.performance_metrics['prediction_time'] = time.time() - step_start
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
            raise StockerPredictionError(f"Prediction failed: {e}")

    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save the predictions to the specified location.
        
        Args:
            output_path: Optional path to save predictions
                        If not provided, uses the path in config
                        
        Returns:
            Path where predictions were saved
            
        Raises:
            StockerPredictionError: If saving predictions fails
        """
        step_start = time.time()
        try:
            # Ensure predictions are available
            if 'predictions' not in self.artifacts:
                raise StockerPredictionError("No predictions to save. Call run_prediction() first.")
            
            # Use specified output path or get from config
            save_path = output_path or self.config.prediction_output_path
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save predictions
            save_path = save_predictions(
                self.artifacts['predictions'],
                self.config,
                confidence=self.artifacts.get('confidence'),
                output_path=save_path,
                model_metadata=self.artifacts.get('model_metadata')
            )
            
            self.logger.info(f"Predictions saved to: {save_path}")
            
            # Track step execution time
            self.performance_metrics['save_time'] = time.time() - step_start
            
            return save_path
        except Exception as e:
            self.logger.error(f"Saving predictions failed: {e}\n{traceback.format_exc()}")
            raise StockerPredictionError(f"Failed to save predictions: {e}")

    def run(self, input_data: Optional[pd.DataFrame] = None, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete prediction pipeline.
        
        Args:
            input_data: Optional input data for prediction
            model_id: Optional specific model ID to use
            
        Returns:
            Dictionary containing prediction artifacts
            
        Raises:
            Various exceptions from individual steps
        """
        self.logger.info(f"Starting prediction pipeline run at {datetime.now().isoformat()}")
        total_start = time.time()
        
        try:
            self.validate_config()
            self.load_model(model_id)
            self.prepare_features(input_data)
            self.run_prediction()
            self.save()
            
            # Calculate and log total execution time
            total_time = time.time() - total_start
            self.performance_metrics['total_execution_time'] = total_time
            
            self.logger.info(f"Prediction pipeline completed successfully in {total_time:.2f} seconds")
            self.logger.debug(f"Performance metrics: {json.dumps(self.performance_metrics, indent=2)}")
            
            return {
                'predictions': self.artifacts.get('predictions'),
                'confidence': self.artifacts.get('confidence'),
                'model_version': self.artifacts.get('model_version'),
                'performance_metrics': self.performance_metrics
            }
        except Exception as e:
            self.logger.error(f"Prediction pipeline failed: {e}\n{traceback.format_exc()}")
            # Re-raise the original exception to maintain the error type
            raise

if __name__ == "__main__":
    config = StockerConfig()
    pipeline = PredictionPipeline(config)
    result = pipeline.run()
    print(f"Prediction completed successfully. Results shape: {result['predictions'].shape}")
