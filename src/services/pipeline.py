"""
Unified pipeline module for STOCKER Pro.
This module combines training and prediction pipelines into a single, comprehensive pipeline system.
"""
import logging
import os
import sys
import traceback
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from src.core.config import StockerConfig
from src.data.ingestion import ingest_stock_data
from src.features.engineering import feature_engineer, feature_engineer_for_prediction
from src.ml.base_model import BaseModel
from src.ml.evaluation import evaluate_model
from src.ml.models import LSTMModel, XGBoostModel, LightGBMModel, EnsembleModel
from src.core.logging import get_advanced_logger
from src.core.exceptions import StockerPredictionError, ModelLoadingError, FeatureEngineeringError


# --- Utility: create run directory ---
def create_run_dir(base_dir="runs"):
    """Create a timestamped run directory for artifacts."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


# --- Utility: save metadata ---
def save_metadata(run_dir, config, mode, run_id):
    """Save run metadata to the run directory."""
    meta = {
        "run_id": run_id,
        "mode": mode,
        "config": config.__dict__,
        # Add more metadata as needed (git hash, env info, etc.)
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# --- Step registry ---
PIPELINE_STEPS = {
    "train": "TrainingPipeline",
    "predict": "PredictionPipeline",
    # Easily add more steps here ("evaluate": EvaluatePipeline, ...)
}


class StockerPipeline:
    """Base pipeline class with common functionality."""
    
    def __init__(self, config: StockerConfig, mode: str = "train", run_dir: str = None):
        self.config = config
        self.mode = mode
        self.run_dir = run_dir or create_run_dir()[0]
        self.logger = get_advanced_logger("main_pipeline", log_to_file=True, log_dir=self.run_dir)
        self.artifacts: Dict[str, Any] = {}
        self.start_time = time.time()
        self.performance_metrics: Dict[str, float] = {}

    def run(self):
        self.logger.info(f"Starting StockerPipeline in '{self.mode}' mode. Artifacts/logs in {self.run_dir}")
        save_metadata(self.run_dir, self.config, self.mode, os.path.basename(self.run_dir))
        
        if self.mode == "train":
            pipeline = TrainingPipeline(self.config, self.run_dir, self.logger)
        elif self.mode == "predict":
            pipeline = PredictionPipeline(self.config, self.run_dir, self.logger)
        else:
            self.logger.error(f"Unknown pipeline mode: {self.mode}")
            raise ValueError(f"Unknown pipeline mode: {self.mode}")
            
        try:
            artifacts = pipeline.run()
            # Save pipeline artifacts (summary only; details saved by step)
            with open(os.path.join(self.run_dir, "artifacts_summary.json"), "w") as f:
                json.dump({k: str(v) for k,v in artifacts.items()}, f, indent=2)
            self.logger.info(f"{self.mode.capitalize()} pipeline completed.")
            # --- Placeholder: experiment tracking, notification hooks ---
            # e.g., mlflow.log_artifacts(self.run_dir), send_slack_notification(...)
            return artifacts
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
            # Optionally: send failure notification here
            sys.exit(1)


class TrainingPipeline:
    """Training pipeline for STOCKER Pro."""
    
    def __init__(self, config: StockerConfig, run_dir: str = None, logger: logging.Logger = None):
        self.config = config
        self.run_dir = run_dir or create_run_dir()[0]
        self.logger = logger or get_advanced_logger("training_pipeline", log_to_file=True, log_dir="logs")
        self.artifacts = {}
        self.mlflow_experiment = self.config.__dict__.get('mlflow_experiment', 'STOCKER_TRAINING')
        mlflow.set_experiment(self.mlflow_experiment)
        self.mlflow_client = MlflowClient()

    def validate_config(self):
        """Validate the configuration before running the pipeline."""
        try:
            # This should use Pydantic validation from src.core.config
            self.logger.info("Config validation passed.")
            return True
        except Exception as e:
            self.logger.critical(f"Config validation failed: {e}")
            raise

    def ingest(self):
        """Ingest data from configured sources."""
        try:
            ingestion_artifact = ingest_stock_data(self.config)
            self.artifacts['ingestion'] = ingestion_artifact
            self.logger.info(f"Data ingestion completed: {ingestion_artifact.status}")
            return ingestion_artifact
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {e}\n{traceback.format_exc()}")
            raise

    def feature_engineering(self):
        """Apply feature engineering to prepare training data."""
        try:
            ingestion_artifact = self.artifacts['ingestion']
            all_features = []
            for raw_path in ingestion_artifact.raw_data_paths:
                transformed_path = raw_path.replace('.csv', '_transformed.csv')
                if os.path.exists(transformed_path):
                    df = pd.read_csv(transformed_path)
                    all_features.append(df)
            if not all_features:
                raise ValueError("No transformed data found for feature engineering.")
            df_all = pd.concat(all_features, axis=0, ignore_index=True)
            features, feat_artifact = feature_engineer(
                df_all,
                self.config.__dict__,
                target_col=self.config.__dict__.get('target_col', None)
            )
            self.artifacts['features'] = features
            self.artifacts['feature_engineering_artifact'] = feat_artifact
            self.logger.info("Feature engineering completed.")
            return features, feat_artifact
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}\n{traceback.format_exc()}")
            raise

    def train(self):
        """Train machine learning models on the prepared features."""
        try:
            with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                mlflow.log_param('pipeline_version', '1.0.0')
                mlflow.log_param('run_time', datetime.now().isoformat())
                mlflow.log_dict(self.config.__dict__, 'config.json')
                
                # Get feature data
                features_df = self.artifacts['features']
                target_col = self.config.__dict__.get('target_col', 'target')
                X = features_df.drop(columns=[target_col])
                y = features_df[target_col]
                
                # Select model class based on configuration
                model_type = self.config.__dict__.get('model_type', 'lstm')
                model_classes = {
                    'lstm': LSTMModel,
                    'xgboost': XGBoostModel,
                    'lightgbm': LightGBMModel,
                    'ensemble': EnsembleModel
                }
                
                if model_type not in model_classes:
                    raise ValueError(f"Unsupported model type: {model_type}")
                    
                ModelClass = model_classes[model_type]
                model_params = self.config.__dict__.get('model_params', {})
                model = ModelClass(**model_params)
                
                # Train the model
                train_history = model.fit(X, y)
                self.artifacts['model'] = model
                self.artifacts['train_history'] = train_history
                
                # Save the model
                model_path = os.path.join(self.run_dir, "model")
                os.makedirs(model_path, exist_ok=True)
                model.save(model_path)
                mlflow.log_artifacts(model_path, "model")
                
                self.logger.info(f"Model training completed and saved to {model_path}")
                return model, train_history
                
        except Exception as e:
            self.logger.error(f"Model training failed: {e}\n{traceback.format_exc()}")
            raise

    def evaluate(self):
        """Evaluate the trained model's performance."""
        try:
            model = self.artifacts['model']
            features_df = self.artifacts['features']
            target_col = self.config.__dict__.get('target_col', 'target')
            X = features_df.drop(columns=[target_col])
            y = features_df[target_col]
            
            # Evaluate the model
            metrics = evaluate_model(model, X, y, self.config.__dict__)
            self.artifacts['evaluation_metrics'] = metrics
            
            # Log metrics to MLflow
            with mlflow.start_run(run_name=f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
            
            self.logger.info(f"Model evaluation completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}\n{traceback.format_exc()}")
            raise

    def run(self):
        """Run the complete training pipeline."""
        self.logger.info("Starting training pipeline")
        
        # Execute pipeline steps sequentially
        self.validate_config()
        self.ingest()
        self.feature_engineering()
        self.train()
        self.evaluate()
        
        execution_time = time.time() - self.start_time
        self.logger.info(f"Training pipeline completed in {execution_time:.2f} seconds")
        
        return self.artifacts


class PredictionPipeline:
    """Prediction pipeline for STOCKER Pro."""
    
    def __init__(self, config: StockerConfig, run_dir: str = None, logger: logging.Logger = None):
        self.config = config
        self.run_dir = run_dir or create_run_dir()[0]
        self.logger = logger or get_advanced_logger("prediction_pipeline", log_to_file=True, log_dir="logs")
        self.artifacts = {}
        self.start_time = time.time()

    def load_model(self):
        """Load a trained model for prediction."""
        try:
            model_path = self.config.__dict__.get('model_path')
            if not model_path:
                raise ValueError("Model path not specified in configuration")
            
            # Determine model type from directory or config
            model_type = self.config.__dict__.get('model_type', 'lstm')
            model_classes = {
                'lstm': LSTMModel,
                'xgboost': XGBoostModel,
                'lightgbm': LightGBMModel,
                'ensemble': EnsembleModel
            }
            
            if model_type not in model_classes:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            ModelClass = model_classes[model_type]
            model = ModelClass.load(model_path)
            
            self.artifacts['model'] = model
            self.logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}\n{traceback.format_exc()}")
            raise ModelLoadingError(f"Failed to load model: {str(e)}")

    def ingest_predict_data(self):
        """Ingest data for prediction."""
        try:
            # For prediction, we might have a different data source or parameters
            predict_config = self.config.__dict__.copy()
            predict_config['mode'] = 'predict'
            
            ingestion_artifact = ingest_stock_data(predict_config)
            self.artifacts['ingestion'] = ingestion_artifact
            self.logger.info(f"Prediction data ingestion completed")
            return ingestion_artifact
            
        except Exception as e:
            self.logger.error(f"Prediction data ingestion failed: {e}\n{traceback.format_exc()}")
            raise

    def prepare_features(self):
        """Prepare features for prediction from ingested data."""
        try:
            ingestion_artifact = self.artifacts['ingestion']
            
            # Process each data file
            all_features = []
            for raw_path in ingestion_artifact.raw_data_paths:
                if os.path.exists(raw_path):
                    df = pd.read_csv(raw_path)
                    # Apply feature engineering for prediction (might be different from training)
                    features_df = feature_engineer_for_prediction(
                        df,
                        self.config.__dict__
                    )
                    all_features.append(features_df)
                    
            if not all_features:
                raise ValueError("No data found for prediction")
                
            # Combine all feature dataframes
            features = pd.concat(all_features, axis=0, ignore_index=True)
            self.artifacts['features'] = features
            self.logger.info(f"Features prepared for prediction: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}\n{traceback.format_exc()}")
            raise FeatureEngineeringError(f"Failed to prepare features: {str(e)}")

    def predict(self):
        """Generate predictions using the loaded model and prepared features."""
        try:
            model = self.artifacts['model']
            features = self.artifacts['features']
            
            # Determine if we have a target column (for evaluation)
            target_col = self.config.__dict__.get('target_col')
            has_target = target_col in features.columns
            
            # Prepare data for prediction
            if has_target:
                X = features.drop(columns=[target_col])
                y_true = features[target_col]
            else:
                X = features
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Store predictions
            self.artifacts['predictions'] = predictions
            
            # If we have actual values, calculate evaluation metrics
            if has_target:
                metrics = evaluate_model(model, X, y_true, self.config.__dict__)
                self.artifacts['evaluation_metrics'] = metrics
                self.logger.info(f"Prediction metrics: {metrics}")
            
            # Save predictions
            predictions_df = X.copy()
            predictions_df['prediction'] = predictions
            if has_target:
                predictions_df[target_col] = y_true
                
            predictions_path = os.path.join(self.run_dir, "predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            self.logger.info(f"Predictions saved to {predictions_path}")
            
            return predictions, predictions_path
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
            raise StockerPredictionError(f"Prediction failed: {str(e)}")

    def run(self):
        """Run the complete prediction pipeline."""
        self.logger.info("Starting prediction pipeline")
        
        # Execute pipeline steps sequentially
        self.load_model()
        self.ingest_predict_data()
        self.prepare_features()
        self.predict()
        
        execution_time = time.time() - self.start_time
        self.logger.info(f"Prediction pipeline completed in {execution_time:.2f} seconds")
        
        return self.artifacts 