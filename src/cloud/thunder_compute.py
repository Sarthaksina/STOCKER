"""
ThunderCompute integration for cloud-based model training.

This module provides functionality for training models on ThunderCompute's
cloud infrastructure, handling job submission, monitoring, and model retrieval.
"""

import os
import json
import time
import shutil
import logging
import tempfile
import requests
from typing import Dict, Any, Union, Optional, List
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import zipfile
import io
import yaml

from src.ml.base_model import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class ThunderComputeConfig:
    """Configuration for ThunderCompute integration."""
    api_key: str
    api_url: str = "https://api.thundercompute.ai"
    storage_bucket: str = "thundercompute-stocker-pro"
    region: str = "us-west-2"
    default_instance_type: str = "ml.p3.2xlarge"
    use_spot_instances: bool = True
    max_runtime_hours: int = 8


class ThunderComputeClient:
    """
    Client for interacting with ThunderCompute's cloud training platform.
    """
    
    def __init__(self, config: Optional[ThunderComputeConfig] = None):
        """
        Initialize ThunderCompute client.
        
        Args:
            config: ThunderCompute configuration (if None, load from env vars)
        """
        if config is None:
            # Load from environment variables
            self.config = ThunderComputeConfig(
                api_key=os.environ.get("THUNDER_COMPUTE_API_KEY", ""),
                api_url=os.environ.get("THUNDER_COMPUTE_API_URL", "https://api.thundercompute.ai"),
                storage_bucket=os.environ.get("THUNDER_COMPUTE_BUCKET", "thundercompute-stocker-pro"),
                region=os.environ.get("THUNDER_COMPUTE_REGION", "us-west-2"),
                default_instance_type=os.environ.get("THUNDER_COMPUTE_INSTANCE", "ml.p3.2xlarge"),
                use_spot_instances=os.environ.get("THUNDER_COMPUTE_SPOT", "true").lower() == "true",
                max_runtime_hours=int(os.environ.get("THUNDER_COMPUTE_MAX_HOURS", "8"))
            )
        else:
            self.config = config
        
        # Validate configuration
        if not self.config.api_key:
            raise ValueError("ThunderCompute API key not provided. Set THUNDER_COMPUTE_API_KEY environment variable.")
        
        # Initialize S3 client for data transfer
        self.s3_client = boto3.client(
            's3',
            region_name=self.config.region,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        )
        
        # Set up API headers
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized ThunderCompute client (API URL: {self.config.api_url})")
    
    def test_connection(self) -> bool:
        """
        Test connection to ThunderCompute API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.config.api_url}/auth/test",
                headers=self.headers
            )
            response.raise_for_status()
            logger.info("Successfully connected to ThunderCompute API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ThunderCompute API: {e}")
            return False
    
    def upload_data(self, local_path: str, remote_key: str) -> bool:
        """
        Upload data to S3 storage bucket.
        
        Args:
            local_path: Local file or directory path
            remote_key: Remote S3 key
        
        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Check if local_path is a directory
            if os.path.isdir(local_path):
                # Create a zip file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for root, _, files in os.walk(local_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.dirname(local_path))
                            zip_file.write(file_path, arcname)
                
                # Upload the zip file
                zip_buffer.seek(0)
                zip_key = f"{remote_key}.zip"
                self.s3_client.upload_fileobj(
                    zip_buffer, 
                    self.config.storage_bucket, 
                    zip_key
                )
                logger.info(f"Uploaded directory {local_path} to s3://{self.config.storage_bucket}/{zip_key}")
                return True
            else:
                # Upload a single file
                self.s3_client.upload_file(
                    local_path, 
                    self.config.storage_bucket, 
                    remote_key
                )
                logger.info(f"Uploaded file {local_path} to s3://{self.config.storage_bucket}/{remote_key}")
                return True
        except Exception as e:
            logger.error(f"Failed to upload data: {e}")
            return False
    
    def download_data(self, remote_key: str, local_path: str) -> bool:
        """
        Download data from S3 storage bucket.
        
        Args:
            remote_key: Remote S3 key
            local_path: Local file or directory path
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Check if remote_key is a zip file
            if remote_key.endswith('.zip'):
                # Create temporary file for zip download
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                    # Download to temporary file
                    self.s3_client.download_file(
                        self.config.storage_bucket,
                        remote_key,
                        temp_file.name
                    )
                    
                    # Create directory if it doesn't exist
                    os.makedirs(local_path, exist_ok=True)
                    
                    # Extract zip file
                    with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                        zip_ref.extractall(local_path)
                    
                    # Remove temporary file
                    os.unlink(temp_file.name)
                
                logger.info(f"Downloaded and extracted s3://{self.config.storage_bucket}/{remote_key} to {local_path}")
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download a single file
                self.s3_client.download_file(
                    self.config.storage_bucket,
                    remote_key,
                    local_path
                )
                logger.info(f"Downloaded s3://{self.config.storage_bucket}/{remote_key} to {local_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return False
    
    def submit_job(self, 
                  job_name: str,
                  model_type: str,
                  data_path: str,
                  config: Dict[str, Any],
                  instance_type: Optional[str] = None,
                  use_spot: Optional[bool] = None,
                  max_runtime_hours: Optional[int] = None,
                  distributed: bool = False,
                  num_instances: int = 1) -> str:
        """
        Submit a training job to ThunderCompute.
        
        Args:
            job_name: Unique name for the job
            model_type: Type of model to train (lstm, xgboost, lightgbm, ensemble)
            data_path: S3 path to training data
            config: Model configuration
            instance_type: EC2 instance type
            use_spot: Whether to use spot instances
            max_runtime_hours: Maximum runtime in hours
            distributed: Whether to use distributed training
            num_instances: Number of instances for distributed training
        
        Returns:
            Job ID if submission successful
        """
        # Use default values if not provided
        instance_type = instance_type or self.config.default_instance_type
        use_spot = use_spot if use_spot is not None else self.config.use_spot_instances
        max_runtime_hours = max_runtime_hours or self.config.max_runtime_hours
        
        # Prepare job configuration
        job_config = {
            "job_name": job_name,
            "model_type": model_type,
            "data_path": data_path,
            "model_config": config,
            "training_config": {
                "instance_type": instance_type,
                "use_spot": use_spot,
                "max_runtime_hours": max_runtime_hours,
                "distributed": distributed,
                "num_instances": num_instances
            },
            "output_path": f"s3://{self.config.storage_bucket}/models/{job_name}/",
            "save_checkpoints": True,
            "checkpoint_interval": 15  # minutes
        }
        
        try:
            # Submit job
            response = requests.post(
                f"{self.config.api_url}/jobs/train",
                headers=self.headers,
                json=job_config
            )
            response.raise_for_status()
            result = response.json()
            job_id = result["job_id"]
            
            logger.info(f"Submitted job {job_name} (ID: {job_id})")
            return job_id
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a training job.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job status information
        """
        try:
            response = requests.get(
                f"{self.config.api_url}/jobs/status/{job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise
    
    def wait_for_job(self, 
                    job_id: str, 
                    poll_interval: int = 30,
                    max_wait_time: Optional[int] = None) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID
            poll_interval: Polling interval in seconds
            max_wait_time: Maximum wait time in seconds (None for unlimited)
            
        Returns:
            Final job status
        """
        start_time = time.time()
        logger.info(f"Waiting for job {job_id} to complete...")
        
        while True:
            # Check if max wait time exceeded
            if max_wait_time and (time.time() - start_time) > max_wait_time:
                logger.warning(f"Maximum wait time ({max_wait_time}s) exceeded for job {job_id}")
                break
                
            # Get job status
            status = self.get_job_status(job_id)
            job_status = status.get("status", "")
            
            # Log progress
            if "metrics" in status:
                metrics = status["metrics"]
                if metrics:
                    # Format latest metrics
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics[-1].items()])
                    logger.info(f"Job {job_id} - Status: {job_status}, Metrics: {metrics_str}")
                else:
                    logger.info(f"Job {job_id} - Status: {job_status}")
            else:
                logger.info(f"Job {job_id} - Status: {job_status}")
            
            # Check if job completed
            if job_status in ["COMPLETED", "FAILED", "STOPPED"]:
                if job_status == "COMPLETED":
                    logger.info(f"Job {job_id} completed successfully!")
                else:
                    logger.warning(f"Job {job_id} {job_status.lower()}")
                break
                
            # Wait before polling again
            time.sleep(poll_interval)
        
        return status
    
    def download_model(self, job_id: str, local_path: str) -> bool:
        """
        Download trained model from completed job.
        
        Args:
            job_id: Job ID
            local_path: Local path to save model
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Get job details
            status = self.get_job_status(job_id)
            
            if status.get("status") != "COMPLETED":
                logger.error(f"Cannot download model: Job {job_id} is not completed")
                return False
                
            # Get model path
            model_path = status.get("output", {}).get("model_path", "")
            
            if not model_path:
                logger.error(f"Cannot download model: No model path in job {job_id} output")
                return False
                
            # Extract key from S3 path
            model_key = model_path.replace(f"s3://{self.config.storage_bucket}/", "")
            
            # Download model
            return self.download_data(model_key, local_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def load_trained_model(self, 
                         job_id: str, 
                         model_class: type, 
                         model_name: str = None) -> BaseModel:
        """
        Load a trained model from ThunderCompute.
        
        Args:
            job_id: Job ID
            model_class: Model class to instantiate
            model_name: Name for the loaded model
            
        Returns:
            Loaded model instance
        """
        try:
            # Create temporary directory for model download
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the model
                success = self.download_model(job_id, temp_dir)
                
                if not success:
                    raise ValueError(f"Failed to download model for job {job_id}")
                
                # Get model name from job if not provided
                if model_name is None:
                    status = self.get_job_status(job_id)
                    model_name = status.get("job_name", f"thunder_model_{job_id}")
                
                # Load model configuration
                config_path = os.path.join(temp_dir, "model_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                else:
                    config = {}
                
                # Instantiate model
                model = model_class(name=model_name, config=config)
                
                # Load model from files
                model.load(temp_dir)
                
                logger.info(f"Successfully loaded model from job {job_id}")
                return model
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            raise
            
    def submit_ensemble_job(self,
                           job_name: str,
                           base_model_ids: List[str],
                           ensemble_config: Dict[str, Any],
                           instance_type: Optional[str] = None) -> str:
        """
        Submit a job to create an ensemble model from trained models.
        
        Args:
            job_name: Unique name for the job
            base_model_ids: List of job IDs for base models
            ensemble_config: Ensemble configuration
            instance_type: EC2 instance type
            
        Returns:
            Job ID if submission successful
        """
        # Use default values if not provided
        instance_type = instance_type or "ml.c5.xlarge"  # Smaller instance for ensemble creation
        
        # Prepare job configuration
        job_config = {
            "job_name": job_name,
            "model_type": "ensemble",
            "base_model_job_ids": base_model_ids,
            "model_config": ensemble_config,
            "training_config": {
                "instance_type": instance_type,
                "use_spot": True,
                "max_runtime_hours": 1
            },
            "output_path": f"s3://{self.config.storage_bucket}/models/{job_name}/",
        }
        
        try:
            # Submit job
            response = requests.post(
                f"{self.config.api_url}/jobs/ensemble",
                headers=self.headers,
                json=job_config
            )
            response.raise_for_status()
            result = response.json()
            job_id = result["job_id"]
            
            logger.info(f"Submitted ensemble job {job_name} (ID: {job_id})")
            return job_id
        except Exception as e:
            logger.error(f"Failed to submit ensemble job: {e}")
            raise
            
    def create_training_script(self, 
                              model_type: str, 
                              config: Dict[str, Any], 
                              output_file: str) -> None:
        """
        Create a training script for ThunderCompute.
        
        Args:
            model_type: Type of model to train
            config: Model configuration
            output_file: Output script file path
        """
        # Template for training script
        script_template = """#!/usr/bin/env python3
'''
ThunderCompute training script for {model_type} model.
This script is automatically generated.
'''
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add STOCKER Pro to Python path
sys.path.append('/opt/ml/code')

# Import STOCKER Pro modules
from src.ml import {model_class}
{extra_imports}

def load_data(data_path):
    '''Load training data from S3 or local path'''
    logger.info(f"Loading data from {{data_path}}")
    
    # Check if data path is a directory or file
    if os.path.isdir(data_path):
        # Look for CSV files
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if csv_files:
            data_file = os.path.join(data_path, csv_files[0])
            logger.info(f"Loading CSV file: {{data_file}}")
            return pd.read_csv(data_file)
        
        # Look for Parquet files
        parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
        if parquet_files:
            data_file = os.path.join(data_path, parquet_files[0])
            logger.info(f"Loading Parquet file: {{data_file}}")
            return pd.read_parquet(data_file)
            
        # Look for NumPy files
        numpy_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        if numpy_files:
            data_file = os.path.join(data_path, numpy_files[0])
            logger.info(f"Loading NumPy file: {{data_file}}")
            return np.load(data_file)
            
        raise ValueError(f"No supported data files found in {{data_path}}")
    else:
        # Load single file
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.npy'):
            return np.load(data_path)
        else:
            raise ValueError(f"Unsupported data file format: {{data_path}}")

def prepare_data(data):
    '''Prepare data for model training'''
    logger.info("Preparing data for training")
    
    # Handle different data formats
    if isinstance(data, pd.DataFrame):
        # Use 'Close' column if available
        if 'Close' in data.columns:
            target_col = 'Close'
        # Otherwise use the first numeric column
        else:
            numeric_cols = data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found in data")
                
        # Extract target values
        target_values = data[target_col].values
        
        # Split into training and validation sets (80/20)
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        train_values = train_data[target_col].values
        val_values = val_data[target_col].values
        
        return train_values, val_values
    elif isinstance(data, np.ndarray):
        # For NumPy arrays, assume it's already the target values
        # Split into training and validation sets (80/20)
        split_idx = int(len(data) * 0.8)
        train_values = data[:split_idx]
        val_values = data[split_idx:]
        
        return train_values, val_values
    else:
        raise ValueError(f"Unsupported data type: {{type(data)}}")

def main():
    '''Main training function'''
    logger.info("Starting {model_type} model training")
    
    # Get environment variables
    data_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    
    # Load model configuration
    config = {config_json}
    
    # Load and prepare data
    data = load_data(data_path)
    train_data, val_data = prepare_data(data)
    
    logger.info(f"Training data shape: {{train_data.shape}}")
    logger.info(f"Validation data shape: {{val_data.shape}}")
    
    # Create and train model
    model = {model_class}(name="{model_name}", config=config)
    
    # Train model
    logger.info("Training model...")
    start_time = datetime.now()
    history = model.fit(train_data, train_data, validation_data=(val_data, val_data))
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    logger.info(f"Model training completed in {{training_time:.2f}} seconds")
    
    # Save model
    logger.info(f"Saving model to {{model_dir}}")
    model.save(model_dir)
    
    # Save configuration
    with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f)
        
    # Save training metrics
    with open(os.path.join(model_dir, 'training_metrics.json'), 'w') as f:
        json.dump({{"training_time": training_time, "history": history}}, f)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()
"""
        
        # Model-specific customizations
        model_class_map = {
            "lstm": "LSTMModel",
            "xgboost": "XGBoostModel",
            "lightgbm": "LightGBMModel",
            "ensemble": "EnsembleModel"
        }
        
        extra_imports_map = {
            "lstm": "",
            "xgboost": "import xgboost as xgb",
            "lightgbm": "import lightgbm as lgb",
            "ensemble": "from src.ml import LSTMModel, XGBoostModel, LightGBMModel"
        }
        
        model_class = model_class_map.get(model_type, "BaseModel")
        extra_imports = extra_imports_map.get(model_type, "")
        
        # Ensure config has a valid JSON representation
        config_json = json.dumps(config, indent=4)
        model_name = config.get("name", f"{model_type}_model")
        
        # Format the script
        script_content = script_template.format(
            model_type=model_type,
            model_class=model_class,
            extra_imports=extra_imports,
            config_json=config_json,
            model_name=model_name
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(output_file, 0o755)
        
        logger.info(f"Created training script at {output_file}")
    
    def prepare_training_package(self, 
                               script_file: str, 
                               output_dir: str,
                               requirements_file: Optional[str] = None) -> str:
        """
        Prepare a training package for ThunderCompute.
        
        Args:
            script_file: Training script file
            output_dir: Output directory for package
            requirements_file: Optional requirements.txt file
            
        Returns:
            Path to the created package zip file
        """
        # Create a temporary directory for the package
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the training script
            script_dst = os.path.join(temp_dir, "train.py")
            shutil.copy(script_file, script_dst)
            
            # Copy requirements file if provided
            if requirements_file and os.path.exists(requirements_file):
                req_dst = os.path.join(temp_dir, "requirements.txt")
                shutil.copy(requirements_file, req_dst)
            else:
                # Create a basic requirements file
                req_dst = os.path.join(temp_dir, "requirements.txt")
                with open(req_dst, 'w') as f:
                    f.write("numpy>=1.19.0\n")
                    f.write("pandas>=1.1.0\n")
                    f.write("scikit-learn>=0.23.2\n")
                    if "lightgbm" in script_file:
                        f.write("lightgbm>=3.0.0\n")
                    if "xgboost" in script_file:
                        f.write("xgboost>=1.2.0\n")
                    if "lstm" in script_file:
                        f.write("torch>=1.7.0\n")
            
            # Create package zip file
            os.makedirs(output_dir, exist_ok=True)
            package_file = os.path.join(output_dir, "training_package.zip")
            
            with zipfile.ZipFile(package_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(script_dst, "train.py")
                zipf.write(req_dst, "requirements.txt")
            
            logger.info(f"Created training package at {package_file}")
            return package_file 