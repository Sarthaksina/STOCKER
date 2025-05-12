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

from src.configuration.config import StockerConfig
from src.components.data_ingestion import ingest_stock_data
from src.components.data_validation import validate_stocker_config_pydantic
from src.features.feature_engineering import feature_engineer, optuna_tune, feature_engineer_for_prediction
from src.features.model_training import train_model, save_model, get_model_version
from src.features.model_loading import load_latest_model, load_model_by_id
from src.features.evaluation import evaluate_model
from src.features.prediction import predict, save_predictions
from src.utils import get_advanced_logger
from src.exception.exception import StockerPredictionError, ModelLoadingError, FeatureEngineeringError

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
        try:
            validate_stocker_config_pydantic(self.config.__dict__)
            self.logger.info("Config validation passed.")
        except Exception as e:
            self.logger.critical(f"Config validation failed: {e}")
            raise

    def ingest(self):
        try:
            ingestion_artifact = ingest_stock_data(self.config)
            self.artifacts['ingestion'] = ingestion_artifact
            self.logger.info(f"Data ingestion completed: {ingestion_artifact.status}")
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {e}\n{traceback.format_exc()}")
            raise

    def feature_engineering(self):
        try:
            ingestion_artifact = self.artifacts['ingestion']
            import pandas as pd
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
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}\n{traceback.format_exc()}")
            raise

    def train(self):
        try:
            with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                mlflow.log_param('pipeline_version', '1.0.0')
                mlflow.log_param('run_time', datetime.now().isoformat())
                mlflow.log_dict(self.config.__dict__, 'config.json')
                # Optuna tuning (if enabled)
                best_params = None
                if self.config.__dict__.get('optuna_tuning', {}).get('enabled', False):
                    def objective(trial, config_dict):
                        model_type = self.config.__dict__.get('model_type', 'sklearn_rf')
                        X = self.artifacts['features'].drop(columns=[self.config.__dict__.get('target_col', 'target')])
                        y = self.artifacts['features'][self.config.__dict__.get('target_col', 'target')]
                        from sklearn.model_selection import cross_val_score
                        if model_type == 'sklearn_rf':
                            from sklearn.ensemble import RandomForestClassifier
                            n_estimators = trial.suggest_int('n_estimators', 10, 200)
                            max_depth = trial.suggest_int('max_depth', 2, 20)
                            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
                        elif model_type == 'xgboost':
                            import xgboost as xgb
                            n_estimators = trial.suggest_int('n_estimators', 10, 200)
                            max_depth = trial.suggest_int('max_depth', 2, 20)
                            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                            subsample = trial.suggest_float('subsample', 0.5, 1.0)
                            clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, use_label_encoder=False, eval_metric='logloss')
                        elif model_type == 'lightgbm':
                            import lightgbm as lgb
                            n_estimators = trial.suggest_int('n_estimators', 10, 200)
                            max_depth = trial.suggest_int('max_depth', 2, 20)
                            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                            subsample = trial.suggest_float('subsample', 0.5, 1.0)
                            clf = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)
                        elif model_type == 'keras':
                            from tensorflow import keras
                            input_dim = X.shape[1]
                            n_units = trial.suggest_int('n_units', 16, 128)
                            n_layers = trial.suggest_int('n_layers', 1, 3)
                            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
                            model = keras.Sequential()
                            model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
                            for _ in range(n_layers):
                                model.add(keras.layers.Dense(n_units, activation='relu'))
                            model.add(keras.layers.Dense(1, activation='sigmoid'))
                            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                            _, acc = model.evaluate(X, y, verbose=0)
                            mlflow.log_metric('optuna_cv_score', acc)
                            return acc
                        else:
                            raise ValueError(f"Unsupported model_type for Optuna: {model_type}")
                        score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
                        mlflow.log_metric('optuna_cv_score', score)
                        return score
                    tune_result = optuna_tune(objective, self.config.__dict__, n_trials=self.config.__dict__.get('optuna_tuning', {}).get('n_trials', 30))
                    best_params = tune_result['best_params']
                    mlflow.log_params(best_params)
                    self.artifacts['optuna'] = tune_result
                    with open('optuna_study.json', 'w') as f:
                        json.dump({"best_params": tune_result['best_params']}, f, indent=2)
                    mlflow.log_artifact('optuna_study.json')
                # Train model using best params if available
                from src.features.model_training import train_model
                model, train_report = train_model(self.artifacts['features'], self.config.__dict__, best_params=best_params)
                self.artifacts['model'] = model
                self.artifacts['train_report'] = train_report
                mlflow.log_metrics(train_report)
                self.logger.info(f"Model training completed. Metrics: {train_report}")
                mlflow.log_dict(self.artifacts['feature_engineering_artifact'], "feature_engineering_artifact.json")
                mlflow.log_dict(self.artifacts['ingestion'].to_dict() if hasattr(self.artifacts['ingestion'], 'to_dict') else {}, "ingestion_artifact.json")
                # Log the model using MLflow Model Registry
                try:
                    import joblib
                    model_type = self.config.__dict__.get('model_type', 'sklearn_rf')
                    if model_type in ['sklearn_rf', 'xgboost', 'lightgbm']:
                        joblib.dump(model, 'model.joblib')
                        import mlflow.sklearn
                        mlflow.sklearn.log_model(model, "model", registered_model_name="StockerModel")
                    elif model_type == 'keras':
                        import mlflow.keras
                        mlflow.keras.log_model(model, "model", registered_model_name="StockerModelKeras")
                except Exception as model_log_err:
                    self.logger.warning(f"Model logging to MLflow failed: {model_log_err}")
        except Exception as e:
            self.logger.error(f"Model training failed: {e}\n{traceback.format_exc()}")
            raise

    def evaluate(self):
        try:
            metrics = evaluate_model(self.artifacts['model'], self.artifacts['features'], self.config)
            self.artifacts['metrics'] = metrics
            self.logger.info(f"Model evaluation completed. Metrics: {metrics}")
            # Log evaluation metrics to MLflow
            mlflow.log_metrics(metrics)
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}\n{traceback.format_exc()}")
            raise

    def save(self):
        try:
            version = get_model_version(self.config)
            save_model(self.artifacts['model'], version, self.config)
            self.logger.info(f"Model saved with version: {version}")
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}\n{traceback.format