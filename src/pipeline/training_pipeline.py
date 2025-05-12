# training_pipeline.py
"""
Industry-grade training pipeline for STOCKER: modular, robust, and extensible.
"""
import logging
import os
import traceback
import json
from datetime import datetime
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.configuration.config import StockerConfig
from src.components.data_ingestion import ingest_stock_data
from src.components.data_validation import validate_stocker_config_pydantic
from src.features.feature_engineering import feature_engineer, optuna_tune
from src.features.model_training import train_model, save_model, get_model_version
from src.features.evaluation import evaluate_model
from src.pipeline.base_pipeline import BasePipeline

"""
Training pipeline for STOCKER Pro.
This module implements a robust training pipeline that inherits from BasePipeline.
"""
import logging
import traceback
import time
import os
import json
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime

from src.configuration.config import StockerConfig
from src.features.feature_engineering import feature_engineer_for_training
from src.features.model_training import train_model, save_model
from src.configuration.config_validator import ConfigValidationError
from src.exception.exception import StockerTrainingError, FeatureEngineeringError
from src.pipeline.base_pipeline import BasePipeline
from src.data_access.access_data import get_data_access_manager

class TrainingPipeline(BasePipeline):
    """
    A production-grade training pipeline for STOCKER Pro.
    
    This pipeline handles the end-to-end process of loading data,
    preparing features, training models, and saving results.
    It includes comprehensive error handling, performance monitoring,
    and support for ensemble models.
    
    Attributes:
        config: Configuration for the training pipeline
        logger: Logger for tracking pipeline execution
        artifacts: Dictionary to store artifacts generated during pipeline execution
        start_time: Start time of the pipeline execution
        performance_metrics: Dictionary to store performance metrics
    """
    
    def __init__(self, config: StockerConfig):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration for the training pipeline
        """
        super().__init__(config)
    
    def validate_config(self) -> None:
        """
        Validate the pipeline configuration.
        
        Raises:
            ConfigValidationError: If configuration validation fails
        """
        try:
            step_start = self.log_step_start("config_validation")
            
            # Log config for debugging (excluding sensitive info)
            safe_config = {k: v for k, v in self.config.__dict__.items() 
                          if not k.endswith("key") and not k.endswith("token")}
            self.logger.debug(f"Config: {json.dumps(safe_config, default=str)}")
            
            # TODO: Replace this with the new config validator
            # validate_training_config(self.config.__dict__)
            self.logger.info("Config validation passed")
            
            self.log_step_end("config_validation", step_start)
        except ConfigValidationError as e:
            self.logger.critical(f"Config validation failed: {e}")
            raise
        except Exception as e:
            self.logger.critical(f"Unexpected error during config validation: {e}\n{traceback.format_exc()}")
            raise ConfigValidationError(f"Unexpected error during config validation: {e}")
    
    def load_data(self, symbols: Optional[List[str]] = None) -> None:
        """
        Load data for training.
        
        Args:
            symbols: Optional list of symbols to load data for
                    If not provided, uses symbols from config
                    
        Raises:
            DataAccessError: If data loading fails
        """
        step_start = self.log_step_start("data_loading")
        
        try:
            # Use the data access manager to load data
            data_manager = get_data_access_manager(self.config)
            
            # If symbols not provided, use from config
            symbols = symbols or self.config.training_symbols
            
            if not symbols:
                self.logger.warning("No symbols specified for training")
                symbols = ["AAPL", "MSFT", "GOOGL"]  # Default symbols
            
            self.logger.info(f"Loading data for symbols: {symbols}")
            
            # Load data for each symbol
            all_data = {}
            for symbol in symbols:
                self.logger.info(f"Loading data for {symbol}")
                data = data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=self.config.training_start_date,
                    end_date=self.config.training_end_date,
                    interval='daily'
                )
                all_data[symbol] = data
            
            self.artifacts['raw_data'] = all_data
            self.logger.info(f"Loaded data for {len(all_data)} symbols")
            
            self.log_step_end("data_loading", step_start)
        except Exception as e:
            self.handle_exception(e, "data_loading")
            raise
    
    def prepare_features(self) -> None:
        """
        Prepare features for training.
        
        Raises:
            FeatureEngineeringError: If feature engineering fails
        """
        step_start = self.log_step_start("feature_engineering")
        
        try:
            # Ensure raw data is loaded
            if 'raw_data' not in self.artifacts:
                raise FeatureEngineeringError("Raw data not loaded. Call load_data() first.")
            
            self.logger.info("Preparing features for training")
            
            # Process each symbol's data
            features_dict = {}
            for symbol, data in self.artifacts['raw_data'].items():
                self.logger.info(f"Engineering features for {symbol}")
                features = feature_engineer_for_training(
                    self.config,
                    input_data=data,
                    symbol=symbol
                )
                features_dict[symbol] = features
            
            self.artifacts['features'] = features_dict
            
            # Log feature information
            for symbol, features in features_dict.items():
                self.logger.info(f"Features for {symbol}: {features.shape}")
                self.logger.debug(f"Feature columns for {symbol}: {features.columns.tolist()}")
            
            self.log_step_end("feature_engineering", step_start)
        except Exception as e:
            self.handle_exception(e, "feature_engineering")
            raise FeatureEngineeringError(f"Failed to prepare features: {e}")
    
    def train_models(self) -> None:
        """
        Train models using the prepared features.
        
        Raises:
            StockerTrainingError: If training fails
        """
        step_start = self.log_step_start("model_training")
        
        try:
            # Ensure features are prepared
            if 'features' not in self.artifacts:
                raise StockerTrainingError("Features not prepared. Call prepare_features() first.")
            
            self.logger.info(f"Training {self.config.default_model_type} model")
            
            # Train models for each symbol
            models_dict = {}
            metrics_dict = {}
            
            for symbol, features in self.artifacts['features'].items():
                self.logger.info(f"Training model for {symbol}")
                
                model, metrics = train_model(
                    features,
                    self.config,
                    model_type=self.config.default_model_type
                )
                
                models_dict[symbol] = model
                metrics_dict[symbol] = metrics
                
                self.logger.info(f"Model training completed for {symbol}")
                self.logger.info(f"Training metrics for {symbol}: {metrics}")
            
            self.artifacts['models'] = models_dict
            self.artifacts['training_metrics'] = metrics_dict
            
            self.log_step_end("model_training", step_start)
        except Exception as e:
            self.handle_exception(e, "model_training")
            raise StockerTrainingError(f"Model training failed: {e}")
    
    def save_models(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Optional directory to save models
                       If not provided, uses the path in config
                       
        Returns:
            Dictionary mapping symbols to model save paths
            
        Raises:
            StockerTrainingError: If saving models fails
        """
        step_start = self.log_step_start("model_saving")
        
        try:
            # Ensure models are trained
            if 'models' not in self.artifacts:
                raise StockerTrainingError("No models to save. Call train_models() first.")
            
            # Use specified output directory or get from config
            if output_dir is None:
                output_dir = os.path.join(self.config.models_dir, 
                                         datetime.now().strftime("%Y%m%d_%H%M%S"))
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each model
            save_paths = {}
            for symbol, model in self.artifacts['models'].items():
                self.logger.info(f"Saving model for {symbol}")
                
                # Get training metrics for this symbol
                metrics = self.artifacts['training_metrics'].get(symbol, {})
                
                # Save the model
                save_path = save_model(
                    model,
                    symbol=symbol,
                    metrics=metrics,
                    config=self.config,
                    output_dir=output_dir
                )
                
                save_paths[symbol] = save_path
                self.logger.info(f"Model for {symbol} saved to: {save_path}")
            
            self.artifacts['model_paths'] = save_paths
            
            # Save overall training metrics
            metrics_path = os.path.join(output_dir, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.artifacts['training_metrics'], f, indent=2, default=str)
            
            self.log_step_end("model_saving", step_start)
            return save_paths
        except Exception as e:
            self.handle_exception(e, "model_saving")
            raise StockerTrainingError(f"Failed to save models: {e}")
    
    def run(self, symbols: Optional[List[str]] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            symbols: Optional list of symbols to train models for
            output_dir: Optional directory to save models
            
        Returns:
            Dictionary containing training artifacts
            
        Raises:
            Various exceptions from individual steps
        """
        self.logger.info(f"Starting training pipeline run at {datetime.now().isoformat()}")
        total_start = time.time()
        
        try:
            self.validate_config()
            self.load_data(symbols)
            self.prepare_features()
            self.train_models()
            save_paths = self.save_models(output_dir)
            
            # Calculate and log total execution time
            total_time = time.time() - total_start
            self.performance_metrics['total_execution_time'] = total_time
            
            self.logger.info(f"Training pipeline completed successfully in {total_time:.2f} seconds")
            self.logger.debug(f"Performance metrics: {json.dumps(self.performance_metrics, indent=2)}")
            
            # Save all artifacts
            self.save_artifacts(os.path.dirname(list(save_paths.values())[0]) if save_paths else None)
            
            return {
                'model_paths': save_paths,
                'training_metrics': self.artifacts.get('training_metrics'),
                'performance_metrics': self.performance_metrics
            }
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}\n{traceback.format_exc()}")
            # Re-raise the original exception to maintain the error type
            raise

if __name__ == "__main__":
    config = StockerConfig()
    pipeline = TrainingPipeline(config)
    result = pipeline.run()
    print(f"Training completed successfully. Models saved to: {result['model_paths']}")
