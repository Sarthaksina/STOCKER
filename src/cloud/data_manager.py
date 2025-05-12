"""
Data management utilities for cloud-based model training.

This module provides functionality for efficient data handling, preprocessing,
chunking, and versioning for cloud-based training in STOCKER Pro.
"""

import os
import json
import hashlib
import shutil
import zipfile
import tempfile
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from src.cloud_training.thunder_compute import ThunderComputeClient

logger = logging.getLogger(__name__)

class CloudDataManager:
    """
    Manages data for cloud-based training, including versioning, chunking, and preprocessing.
    """
    
    def __init__(self, thunder_client: ThunderComputeClient, cache_dir: str = '.cache/cloud_data'):
        """
        Initialize the cloud data manager.
        
        Args:
            thunder_client: ThunderCompute client for cloud operations
            cache_dir: Local cache directory for data files
        """
        self.client = thunder_client
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data registry for tracking uploaded data
        self.registry_file = os.path.join(cache_dir, 'data_registry.json')
        self.data_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load data registry from file.
        
        Returns:
            Data registry dictionary
        """
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load data registry: {e}")
                return {"data_versions": {}}
        else:
            return {"data_versions": {}}
    
    def _save_registry(self) -> None:
        """Save data registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.data_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save data registry: {e}")
    
    def _calculate_data_hash(self, data_path: str) -> str:
        """
        Calculate hash of data file or directory.
        
        Args:
            data_path: Path to data file or directory
            
        Returns:
            Hash string
        """
        hash_md5 = hashlib.md5()
        
        if os.path.isfile(data_path):
            # Hash file contents
            with open(data_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            # Hash directory contents
            for root, _, files in os.walk(data_path):
                for file in sorted(files):  # Sort to ensure consistent ordering
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def preprocess_data(self, 
                      data_path: str, 
                      output_path: Optional[str] = None,
                      preprocessing_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Preprocess data for cloud training.
        
        Args:
            data_path: Path to input data
            output_path: Path to save preprocessed data
            preprocessing_config: Configuration for preprocessing
            
        Returns:
            Path to preprocessed data
        """
        if preprocessing_config is None:
            preprocessing_config = {}
        
        # Set default output path if not provided
        if output_path is None:
            output_dir = os.path.join(self.cache_dir, 'processed')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"processed_{os.path.basename(data_path)}")
        
        logger.info(f"Preprocessing data from {data_path} to {output_path}")
        
        # Load data based on file type
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Apply preprocessing steps
        original_shape = df.shape
        
        # Handle missing values
        if preprocessing_config.get('handle_missing', True):
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # Forward fill remaining NaNs
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicates
        if preprocessing_config.get('remove_duplicates', True):
            df = df.drop_duplicates()
        
        # Handle outliers
        if preprocessing_config.get('handle_outliers', False):
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Normalize data (0-1 scaling)
        if preprocessing_config.get('normalize', False):
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        # Save preprocessed data
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        else:
            # Default to parquet for better compression
            if not output_path.endswith('.parquet'):
                output_path += '.parquet'
            df.to_parquet(output_path, index=False)
        
        logger.info(f"Preprocessing complete: {original_shape} -> {df.shape}")
        
        return output_path
    
    def chunk_data(self, 
                 data_path: str, 
                 num_chunks: int = 3,
                 output_dir: Optional[str] = None) -> List[str]:
        """
        Split data into chunks for parallel training.
        
        Args:
            data_path: Path to data file
            num_chunks: Number of chunks to create
            output_dir: Directory to save chunks
            
        Returns:
            List of chunk file paths
        """
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(self.cache_dir, 'chunks', os.path.basename(data_path))
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Chunking data from {data_path} into {num_chunks} parts")
        
        # Load data based on file type
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format for chunking: {data_path}")
        
        # Calculate chunk size
        chunk_size = len(df) // num_chunks
        
        # Create and save chunks
        chunk_paths = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(df)
            
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Determine output format based on input format
            if data_path.endswith('.csv'):
                chunk_path = os.path.join(output_dir, f"chunk_{i}.csv")
                chunk_df.to_csv(chunk_path, index=False)
            else:
                chunk_path = os.path.join(output_dir, f"chunk_{i}.parquet")
                chunk_df.to_parquet(chunk_path, index=False)
                
            chunk_paths.append(chunk_path)
            
        logger.info(f"Created {len(chunk_paths)} data chunks in {output_dir}")
        
        return chunk_paths
    
    def upload_data_version(self, 
                         data_path: str, 
                         version_name: str,
                         description: str = "",
                         preprocess: bool = True,
                         preprocessing_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a versioned dataset to cloud storage.
        
        Args:
            data_path: Path to data file or directory
            version_name: Name for this data version
            description: Description of this data version
            preprocess: Whether to preprocess the data before uploading
            preprocessing_config: Configuration for preprocessing
            
        Returns:
            Remote path to uploaded data
        """
        # Calculate data hash
        data_hash = self._calculate_data_hash(data_path)
        
        # Check if already uploaded
        if version_name in self.data_registry["data_versions"]:
            existing_info = self.data_registry["data_versions"][version_name]
            if existing_info.get("hash") == data_hash:
                logger.info(f"Data version '{version_name}' already exists with same hash")
                return existing_info.get("remote_path", "")
        
        # Preprocess if requested
        if preprocess:
            processed_path = self.preprocess_data(data_path, preprocessing_config=preprocessing_config)
            upload_path = processed_path
        else:
            upload_path = data_path
        
        # Create remote path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        remote_key = f"data/{version_name}/{timestamp}"
        
        # Upload to cloud storage
        success = self.client.upload_data(upload_path, remote_key)
        
        if success:
            # Update registry
            self.data_registry["data_versions"][version_name] = {
                "hash": data_hash,
                "timestamp": timestamp,
                "description": description,
                "remote_path": remote_key,
                "original_path": data_path,
                "processed": preprocess
            }
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Uploaded data version '{version_name}' to {remote_key}")
            return remote_key
        else:
            logger.error(f"Failed to upload data version '{version_name}'")
            return ""
    
    def get_data_version(self, version_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a data version.
        
        Args:
            version_name: Name of the data version
            
        Returns:
            Data version information or None if not found
        """
        return self.data_registry["data_versions"].get(version_name)
    
    def download_data_version(self, 
                           version_name: str, 
                           local_path: Optional[str] = None) -> str:
        """
        Download a data version from cloud storage.
        
        Args:
            version_name: Name of the data version
            local_path: Path to save the downloaded data
            
        Returns:
            Path to downloaded data
        """
        # Get version info
        version_info = self.get_data_version(version_name)
        if not version_info:
            logger.error(f"Data version '{version_name}' not found")
            return ""
        
        # Set default local path if not provided
        if local_path is None:
            local_dir = os.path.join(self.cache_dir, 'downloads')
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"{version_name}_{version_info['timestamp']}")
        
        # Download from cloud storage
        remote_key = version_info.get("remote_path", "")
        success = self.client.download_data(remote_key, local_path)
        
        if success:
            logger.info(f"Downloaded data version '{version_name}' to {local_path}")
            return local_path
        else:
            logger.error(f"Failed to download data version '{version_name}'")
            return ""
    
    def prepare_distributed_training(self, 
                                  data_path: str, 
                                  num_workers: int = 3,
                                  preprocess: bool = True,
                                  preprocessing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare data for distributed training across multiple workers.
        
        Args:
            data_path: Path to data file
            num_workers: Number of worker instances
            preprocess: Whether to preprocess the data before chunking
            preprocessing_config: Configuration for preprocessing
            
        Returns:
            Dictionary with worker chunk information
        """
        # Preprocess if requested
        if preprocess:
            processed_path = self.preprocess_data(data_path, preprocessing_config=preprocessing_config)
        else:
            processed_path = data_path
        
        # Split data into chunks
        chunk_paths = self.chunk_data(processed_path, num_workers)
        
        # Upload each chunk to cloud storage
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_remote_key = f"data/distributed_{timestamp}"
        
        worker_data = {}
        for i, chunk_path in enumerate(chunk_paths):
            worker_id = f"worker_{i}"
            remote_key = f"{base_remote_key}/{worker_id}"
            
            # Upload chunk
            success = self.client.upload_data(chunk_path, remote_key)
            
            if success:
                worker_data[worker_id] = {
                    "chunk_path": chunk_path,
                    "remote_key": remote_key
                }
            else:
                logger.error(f"Failed to upload chunk for worker {worker_id}")
        
        result = {
            "num_workers": len(worker_data),
            "timestamp": timestamp,
            "base_remote_key": base_remote_key,
            "worker_data": worker_data
        }
        
        logger.info(f"Prepared distributed training data for {len(worker_data)} workers")
        
        return result 