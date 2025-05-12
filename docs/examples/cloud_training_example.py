"""
Example script demonstrating cloud-based model training with ThunderCompute.

This script shows how to:
1. Set up ThunderCompute client and job manager
2. Prepare and upload training data
3. Configure and submit cloud training jobs
4. Monitor job progress
5. Create and train ensembles in the cloud
6. Load trained models
"""
import os
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom modules
from src.cloud_training.thunder_compute import ThunderComputeClient, ThunderComputeConfig
from src.cloud_training.cloud_optimizer import CloudTrainingOptimizer
from src.cloud_training.data_manager import CloudDataManager
from src.cloud_training.job_manager import CloudJobManager
from src.ml import LSTMModel, XGBoostModel, LightGBMModel, EnsembleModel
from src.ml.evaluation import evaluate_financial_model, plot_financial_evaluation

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with stock data
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Add some basic features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Rolling_Vol_10'] = data['Returns'].rolling(window=10).std()
    data['Rolling_Vol_30'] = data['Returns'].rolling(window=30).std()
    
    # Drop rows with NaN values
    data = data.dropna()
    
    logger.info(f"Fetched {len(data)} data points")
    return data

def save_data_for_training(data, output_path):
    """
    Save data for training in CSV format.
    
    Args:
        data: DataFrame with stock data
        output_path: Path to save data
        
    Returns:
        Path to saved data file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    data.to_csv(output_path)
    logger.info(f"Saved {len(data)} data points to {output_path}")
    
    return output_path

def setup_thunder_compute():
    """
    Set up ThunderCompute client and related managers.
    
    Returns:
        Tuple of (client, data_manager, job_manager)
    """
    logger.info("Setting up ThunderCompute client and managers")
    
    # Check for API key in environment variable
    api_key = os.environ.get("THUNDER_COMPUTE_API_KEY")
    if not api_key:
        # For demo purposes, use a mock API key
        api_key = "demo_api_key_12345"
        logger.warning("Using mock API key. Set THUNDER_COMPUTE_API_KEY environment variable for actual usage.")
    
    # Create configuration
    config = ThunderComputeConfig(
        api_key=api_key,
        api_url="https://api.thundercompute.ai",
        storage_bucket="thundercompute-stocker-pro",
        region="us-west-2",
        default_instance_type="ml.g4dn.xlarge",
        use_spot_instances=True,
        max_runtime_hours=8
    )
    
    # Initialize client
    client = ThunderComputeClient(config)
    
    # Test connection
    if not client.test_connection():
        logger.warning("Could not connect to ThunderCompute API. Running in demo mode.")
    
    # Initialize managers
    data_manager = CloudDataManager(client)
    job_manager = CloudJobManager(client)
    
    return client, data_manager, job_manager

def cloud_train_models(data_path, job_manager, data_manager):
    """
    Train individual models in the cloud.
    
    Args:
        data_path: Local path to data file
        job_manager: Cloud job manager
        data_manager: Cloud data manager
        
    Returns:
        Dictionary with job IDs for each model
    """
    # Upload data to cloud
    data_version = "stock_data_v1"
    remote_path = data_manager.upload_data_version(
        data_path=data_path,
        version_name=data_version,
        description="Stock price data for cloud training",
        preprocess=True
    )
    
    if not remote_path:
        logger.error("Failed to upload data to cloud")
        return {}
    
    # Configure LSTM model
    lstm_config = {
        "input_dim": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "sequence_length": 20,
        "epochs": 50,
        "learning_rate": 0.001,
        "batch_size": 32,
        "save_checkpoints": True,
        "checkpoint_interval": 5  # Save every 5 epochs
    }
    
    # Configure XGBoost model
    xgb_config = {
        "objective": "reg:squarederror",
        "learning_rate": 0.01,
        "max_depth": 6,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "sequence_length": 20,
        "prediction_length": 1,
        "early_stopping_rounds": 30
    }
    
    # Configure LightGBM model
    lgb_config = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 300,
        "sequence_length": 20,
        "prediction_length": 1,
        "early_stopping_rounds": 30
    }
    
    # Submit jobs for all models
    job_ids = {}
    
    # Submit LSTM job
    lstm_job_id = job_manager.submit_training_job(
        model_type="lstm",
        data_path=remote_path,
        config=lstm_config,
        job_name="stocker_lstm_cloud",
        optimize_cost=True,
        monitor=True
    )
    job_ids["lstm"] = lstm_job_id
    
    # Submit XGBoost job
    xgb_job_id = job_manager.submit_training_job(
        model_type="xgboost",
        data_path=remote_path,
        config=xgb_config,
        job_name="stocker_xgboost_cloud",
        optimize_cost=True,
        monitor=True
    )
    job_ids["xgboost"] = xgb_job_id
    
    # Submit LightGBM job
    lgb_job_id = job_manager.submit_training_job(
        model_type="lightgbm",
        data_path=remote_path,
        config=lgb_config,
        job_name="stocker_lightgbm_cloud",
        optimize_cost=True,
        monitor=True
    )
    job_ids["lightgbm"] = lgb_job_id
    
    return job_ids

def cloud_train_ensemble(data_path, job_manager, data_manager, wait_for_base_models=False):
    """
    Train an ensemble model in the cloud.
    
    Args:
        data_path: Local path to data file
        job_manager: Cloud job manager
        data_manager: Cloud data manager
        wait_for_base_models: Whether to wait for base models to complete
        
    Returns:
        Dictionary with ensemble training information
    """
    # Upload data to cloud
    data_version = "stock_data_ensemble_v1"
    remote_path = data_manager.upload_data_version(
        data_path=data_path,
        version_name=data_version,
        description="Stock price data for ensemble training",
        preprocess=True
    )
    
    if not remote_path:
        logger.error("Failed to upload data to cloud")
        return {}
    
    # Configure models
    base_models = {
        "lstm": {
            "input_dim": 1,
            "hidden_dim": 64,
            "num_layers": 2,
            "sequence_length": 20,
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        "xgboost": {
            "objective": "reg:squarederror",
            "learning_rate": 0.01,
            "max_depth": 6,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "sequence_length": 20,
            "prediction_length": 1
        },
        "lightgbm": {
            "objective": "regression",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": -1,
            "n_estimators": 300,
            "sequence_length": 20,
            "prediction_length": 1
        }
    }
    
    # Configure ensemble
    ensemble_config = {
        "ensemble_method": "weighted_avg",
        "weights": {
            "lstm": 0.4,
            "xgboost": 0.3,
            "lightgbm": 0.3
        }
    }
    
    # Submit ensemble training
    ensemble_result = job_manager.submit_ensemble_training(
        ensemble_config=ensemble_config,
        data_path=remote_path,
        base_models=base_models,
        job_name="stocker_ensemble_cloud",
        optimize_cost=True,
        max_budget=20.0,  # $20 max budget
        wait_for_completion=wait_for_base_models
    )
    
    return ensemble_result

def wait_for_models_and_load(job_ids, job_manager, test_data, max_wait_time=3600):
    """
    Wait for models to complete training and load them.
    
    Args:
        job_ids: Dictionary with job IDs
        job_manager: Cloud job manager
        test_data: Test data for evaluation
        max_wait_time: Maximum wait time in seconds
        
    Returns:
        Dictionary with loaded models
    """
    logger.info("Waiting for models to complete training...")
    
    models = {}
    start_time = time.time()
    
    while len(models) < len(job_ids) and (time.time() - start_time) < max_wait_time:
        # Check if any new status updates are available
        status_update = job_manager.get_next_status_update(timeout=10)
        
        if status_update:
            job_id, status_info = status_update
            
            # Check if job completed
            if status_info.get("status") == "COMPLETED":
                model_type = None
                
                # Find which model this job corresponds to
                for mtype, jid in job_ids.items():
                    if jid == job_id:
                        model_type = mtype
                        break
                
                if model_type and model_type not in models:
                    try:
                        # Load model
                        model = job_manager.load_trained_model(job_id)
                        models[model_type] = model
                        logger.info(f"Loaded {model_type} model from job {job_id}")
                        
                        # Evaluate model if test data is provided
                        if test_data is not None:
                            predictions = model.predict(test_data)
                            metrics = model.evaluate(test_data, test_data)
                            logger.info(f"{model_type} evaluation: MSE={metrics['mse']:.4f}, Dir. Acc={metrics.get('directional_accuracy', 0):.4f}")
                    except Exception as e:
                        logger.error(f"Failed to load model for job {job_id}: {e}")
        
        # Check if we have all models
        if len(models) == len(job_ids):
            break
    
    logger.info(f"Loaded {len(models)}/{len(job_ids)} models")
    return models

def load_ensemble_from_cloud(ensemble_job_id, job_manager, test_data=None):
    """
    Load an ensemble model from the cloud.
    
    Args:
        ensemble_job_id: Ensemble job ID
        job_manager: Cloud job manager
        test_data: Test data for evaluation
        
    Returns:
        Loaded ensemble model
    """
    try:
        # Load ensemble model
        ensemble = job_manager.load_ensemble_model(ensemble_job_id)
        logger.info(f"Loaded ensemble model from job {ensemble_job_id}")
        
        # Evaluate if test data is provided
        if test_data is not None:
            predictions = ensemble.predict(test_data)
            metrics = ensemble.evaluate(test_data, test_data)
            logger.info(f"Ensemble evaluation: MSE={metrics['mse']:.4f}, Dir. Acc={metrics.get('directional_accuracy', 0):.4f}")
        
        return ensemble
    except Exception as e:
        logger.error(f"Failed to load ensemble model: {e}")
        return None

def main(args):
    """Main function for cloud training example."""
    # Set parameters
    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date if args.end_date else datetime.now().strftime("%Y-%m-%d")
    
    # Create output directory
    output_dir = os.path.join('data', 'cloud_training_example')
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch and save data
    data = fetch_stock_data(ticker, start_date, end_date)
    data_path = save_data_for_training(data, os.path.join(output_dir, f"{ticker}_data.csv"))
    
    # Set up ThunderCompute
    client, data_manager, job_manager = setup_thunder_compute()
    
    # Choose action based on mode
    if args.mode == "individual":
        # Train individual models
        job_ids = cloud_train_models(data_path, job_manager, data_manager)
        
        if args.wait:
            # Prepare test data
            test_size = int(len(data) * 0.2)
            test_data = data['Close'].values[-test_size:]
            
            # Wait for models and load them
            models = wait_for_models_and_load(job_ids, job_manager, test_data)
            
            # Save job information
            job_info_path = os.path.join(output_dir, "job_registry.json")
            job_manager.save_job_registry(job_info_path)
            logger.info(f"Saved job registry to {job_info_path}")
        
    elif args.mode == "ensemble":
        # Train ensemble
        ensemble_result = cloud_train_ensemble(data_path, job_manager, data_manager, args.wait)
        
        if args.wait:
            # Prepare test data
            test_size = int(len(data) * 0.2)
            test_data = data['Close'].values[-test_size:]
            
            # Get ensemble job ID
            ensemble_job_id = ensemble_result["ensemble_job"]["job_id"]
            
            # Wait for it to complete
            client.wait_for_job(ensemble_job_id)
            
            # Load ensemble
            ensemble = load_ensemble_from_cloud(ensemble_job_id, job_manager, test_data)
            
            # Save job information
            job_info_path = os.path.join(output_dir, "ensemble_job_registry.json")
            job_manager.save_job_registry(job_info_path)
            logger.info(f"Saved job registry to {job_info_path}")
    
    logger.info("Cloud training example completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Training Example")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start-date", type=str, default="2019-01-01", help="Start date for data")
    parser.add_argument("--end-date", type=str, default=None, help="End date for data")
    parser.add_argument("--mode", type=str, choices=["individual", "ensemble"], default="ensemble", 
                       help="Training mode: individual models or ensemble")
    parser.add_argument("--wait", action="store_true", help="Wait for training to complete")
    
    args = parser.parse_args()
    main(args) 