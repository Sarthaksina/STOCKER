"""
Cloud training optimization utilities for STOCKER Pro.

This module provides cost optimization strategies and checkpointing
mechanisms for efficient training on ThunderCompute cloud infrastructure.
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from src.cloud_training.thunder_compute import ThunderComputeClient

logger = logging.getLogger(__name__)

class CloudServiceError(Exception):
    """Exception raised for cloud service errors"""
    pass

class CloudQuotaExceededError(CloudServiceError):
    """Exception raised when cloud quota is exceeded"""
    pass

class CloudAuthenticationError(CloudServiceError):
    """Exception raised for authentication errors"""
    pass

class CloudTrainingOptimizer:
    """
    Optimizes cloud-based model training for cost efficiency and reliability.
    """
    
    def __init__(self, thunder_client: ThunderComputeClient):
        """
        Initialize the cloud training optimizer.
        
        Args:
            thunder_client: ThunderCompute client for cloud operations
        """
        self.client = thunder_client
        
        # Default cost configurations
        self.instance_costs = {
            "ml.c5.xlarge": 0.17,      # CPU instance
            "ml.m5.xlarge": 0.19,      # General purpose
            "ml.g4dn.xlarge": 0.53,    # GPU (1 NVIDIA T4)
            "ml.p3.2xlarge": 3.06,     # GPU (1 NVIDIA V100)
            "ml.g5.xlarge": 1.01       # GPU (1 NVIDIA A10G)
        }
        
        # Default instance recommendations for different model types
        self.model_instance_recommendations = {
            "lstm": "ml.g4dn.xlarge",  # GPU for deep learning
            "xgboost": "ml.c5.xlarge", # CPU for tree-based models
            "lightgbm": "ml.c5.xlarge", # CPU for tree-based models
            "ensemble": "ml.c5.xlarge"  # CPU for ensemble operations
        }
        
        # Fallback options for when cloud services are unavailable
        self.fallback_enabled = True
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    def _execute_with_fallback(self, cloud_func, fallback_func, *args, **kwargs):
        """
        Execute a cloud function with fallback to local execution
        
        Args:
            cloud_func: Cloud function to execute
            fallback_func: Fallback function to execute if cloud fails
            *args, **kwargs: Arguments to pass to both functions
            
        Returns:
            Result of either cloud_func or fallback_func
        """
        if not self.fallback_enabled:
            # Just execute cloud function without fallback
            return cloud_func(*args, **kwargs)
            
        # Try cloud execution with retries
        for attempt in range(self.max_retries):
            try:
                return cloud_func(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Cloud connection error (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
            except CloudAuthenticationError as e:
                logger.error(f"Cloud authentication error: {e}")
                break  # Don't retry auth errors
            except CloudQuotaExceededError as e:
                logger.error(f"Cloud quota exceeded: {e}")
                break  # Don't retry quota errors
            except Exception as e:
                logger.error(f"Unexpected cloud error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
        # If we get here, all cloud attempts failed
        logger.info("Falling back to local execution")
        return fallback_func(*args, **kwargs)

    def recommend_instance(self, 
                          model_type: str, 
                          data_size_mb: float, 
                          budget_constraint: Optional[float] = None) -> str:
        """
        Recommend the most cost-effective instance type for a training job.
        
        Args:
            model_type: Type of model to train
            data_size_mb: Size of training data in MB
            budget_constraint: Optional maximum hourly cost
            
        Returns:
            Recommended instance type
        """
        # Get default recommendation for model type
        default_recommendation = self.model_instance_recommendations.get(
            model_type, "ml.c5.xlarge")
        
        # For small datasets (<100MB), use cheaper instances
        if data_size_mb < 100:
            # For small LSTM models, CPU might be sufficient
            if model_type == "lstm" and data_size_mb < 50:
                return "ml.c5.xlarge"
            else:
                return default_recommendation
        
        # For medium datasets (100MB-1GB)
        elif data_size_mb < 1000:
            # Use default recommendations
            if budget_constraint and self.instance_costs.get(default_recommendation, 0) > budget_constraint:
                # Find cheapest instance within budget
                affordable_instances = [
                    inst for inst, cost in self.instance_costs.items()
                    if cost <= budget_constraint
                ]
                if affordable_instances:
                    return affordable_instances[0]  # Return cheapest
                else:
                    logger.warning("No instance types within budget constraint")
                    return "ml.c5.xlarge"  # Fallback to cheapest
            else:
                return default_recommendation
        
        # For large datasets (>1GB)
        else:
            # For LSTM, recommend more powerful GPU
            if model_type == "lstm":
                if budget_constraint and budget_constraint >= self.instance_costs.get("ml.p3.2xlarge", 999):
                    return "ml.p3.2xlarge"
                else:
                    return "ml.g4dn.xlarge"
            # For tree-based models, more CPU/memory
            elif model_type in ["xgboost", "lightgbm"]:
                return "ml.m5.xlarge"
            else:
                return default_recommendation
    
    def estimate_training_cost(self, 
                             instance_type: str, 
                             estimated_hours: float,
                             use_spot: bool = True) -> float:
        """
        Estimate the cost of a training job.
        
        Args:
            instance_type: EC2 instance type
            estimated_hours: Estimated training time in hours
            use_spot: Whether to use spot instances
            
        Returns:
            Estimated cost in USD
        """
        hourly_cost = self.instance_costs.get(instance_type, 0.0)
        
        # Apply spot discount if applicable (typically ~70% cheaper)
        if use_spot:
            hourly_cost *= 0.3
        
        total_cost = hourly_cost * estimated_hours
        return total_cost
    
    def optimize_job_config(self, 
                          model_type: str, 
                          config: Dict[str, Any],
                          data_size_mb: float,
                          max_budget: Optional[float] = None,
                          max_runtime_hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize job configuration for cost and performance.
        
        Args:
            model_type: Type of model to train
            config: Model configuration
            data_size_mb: Size of training data in MB
            max_budget: Maximum budget for training
            max_runtime_hours: Maximum runtime hours
            
        Returns:
            Optimized configuration
        """
        # Make a copy of the configuration to avoid modifying the original
        optimized_config = config.copy()
        
        # Recommend instance type
        instance_type = self.recommend_instance(model_type, data_size_mb, 
                                              max_budget / max_runtime_hours if max_budget and max_runtime_hours else None)
        
        # Optimize model-specific parameters
        if model_type == "lstm":
            # If on cheaper instance, reduce model complexity
            if instance_type == "ml.c5.xlarge":
                # Reduce model size for CPU training
                optimized_config["hidden_dim"] = min(optimized_config.get("hidden_dim", 64), 32)
                optimized_config["num_layers"] = min(optimized_config.get("num_layers", 2), 1)
                optimized_config["batch_size"] = min(optimized_config.get("batch_size", 32), 16)
            
            # Enable checkpointing
            optimized_config["save_checkpoints"] = True
            optimized_config["checkpoint_interval"] = 5  # Save every 5 epochs
            
        elif model_type in ["xgboost", "lightgbm"]:
            # For tree-based models, optimize iterations/trees
            if data_size_mb > 1000:
                # For large datasets, use fewer estimators
                optimized_config["n_estimators"] = min(optimized_config.get("n_estimators", 1000), 500)
            
            # Enable early stopping to save compute time
            optimized_config["early_stopping_rounds"] = optimized_config.get("early_stopping_rounds", 50)
            
        # Add cost-optimized training config
        training_config = {
            "instance_type": instance_type,
            "use_spot": True,  # Use spot instances for cost savings
            "max_runtime_hours": max_runtime_hours or 8,
            "checkpoint_interval": 15,  # minutes
            "save_checkpoints": True
        }
        
        # Estimate cost
        estimated_cost = self.estimate_training_cost(
            instance_type, 
            training_config["max_runtime_hours"],
            training_config["use_spot"]
        )
        
        logger.info(f"Optimized configuration for {model_type} model:")
        logger.info(f"  Instance: {instance_type}")
        logger.info(f"  Estimated cost: ${estimated_cost:.2f}")
        
        return {
            "model_config": optimized_config,
            "training_config": training_config,
            "estimated_cost": estimated_cost
        }
    
    def submit_cost_optimized_job(self,
                                job_name: str,
                                model_type: str,
                                data_path: str,
                                config: Dict[str, Any],
                                data_size_mb: Optional[float] = None,
                                max_budget: Optional[float] = None,
                                max_runtime_hours: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Submit a cost-optimized training job.
        
        Args:
            job_name: Unique name for the job
            model_type: Type of model to train
            data_path: Path to training data
            config: Model configuration
            data_size_mb: Size of training data in MB (estimated if not provided)
            max_budget: Maximum budget for training
            max_runtime_hours: Maximum runtime hours
            
        Returns:
            Tuple of (job_id, optimized_config)
        """
        # Estimate data size if not provided
        if data_size_mb is None:
            # Try to get data size from file
            if os.path.exists(data_path):
                if os.path.isfile(data_path):
                    data_size_mb = os.path.getsize(data_path) / (1024 * 1024)
                else:
                    # Sum size of all files in directory
                    total_size = 0
                    for dirpath, _, filenames in os.walk(data_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)
                    data_size_mb = total_size / (1024 * 1024)
            else:
                # Default to medium size if cannot determine
                data_size_mb = 500
                logger.warning(f"Cannot determine data size, assuming {data_size_mb}MB")
        
        # Optimize configuration
        optimized = self.optimize_job_config(
            model_type, 
            config, 
            data_size_mb,
            max_budget,
            max_runtime_hours
        )
        
        # Submit job with optimized configuration
        job_id = self.client.submit_job(
            job_name=job_name,
            model_type=model_type,
            data_path=data_path,
            config=optimized["model_config"],
            instance_type=optimized["training_config"]["instance_type"],
            use_spot=optimized["training_config"]["use_spot"],
            max_runtime_hours=optimized["training_config"]["max_runtime_hours"]
        )
        
        # Return job ID and optimized configuration
        return job_id, optimized
    
    def resume_from_checkpoint(self, 
                             job_id: str, 
                             max_runtime_hours: Optional[int] = None) -> str:
        """
        Resume training from the latest checkpoint.
        
        Args:
            job_id: ID of the original job
            max_runtime_hours: Maximum runtime hours for resumed job
            
        Returns:
            New job ID
        """
        try:
            # Get original job details
            status = self.client.get_job_status(job_id)
            
            if status.get("status") not in ["FAILED", "STOPPED"]:
                logger.warning(f"Job {job_id} is not failed or stopped (status: {status.get('status')})")
                return job_id
            
            # Check if job has checkpoints
            checkpoints = status.get("checkpoints", [])
            if not checkpoints:
                logger.error(f"No checkpoints found for job {job_id}")
                raise ValueError(f"No checkpoints found for job {job_id}")
            
            # Get latest checkpoint
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = latest_checkpoint.get("path")
            
            if not checkpoint_path:
                logger.error(f"No valid checkpoint path for job {job_id}")
                raise ValueError(f"No valid checkpoint path for job {job_id}")
            
            # Get original job configuration
            original_config = status.get("config", {})
            model_type = original_config.get("model_type", "unknown")
            data_path = original_config.get("data_path", "")
            model_config = original_config.get("model_config", {})
            
            # Update configuration with checkpoint
            model_config["resume_from_checkpoint"] = checkpoint_path
            
            # Create new job name
            new_job_name = f"{status.get('job_name', 'job')}_resumed_{int(time.time())}"
            
            # Submit new job
            new_job_id = self.client.submit_job(
                job_name=new_job_name,
                model_type=model_type,
                data_path=data_path,
                config=model_config,
                instance_type=status.get("instance_type"),
                max_runtime_hours=max_runtime_hours or status.get("max_runtime_hours", 8)
            )
            
            logger.info(f"Resumed job {job_id} as new job {new_job_id}")
            return new_job_id
        
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            raise
    
    def batch_optimize_ensemble_training(self, 
                                      ensemble_config: Dict[str, Any],
                                      data_path: str,
                                      base_model_configs: Dict[str, Dict[str, Any]],
                                      job_prefix: str = "stocker_ensemble",
                                      max_budget: Optional[float] = None,
                                      sequential: bool = False) -> Dict[str, Any]:
        """
        Optimize and train an ensemble model with cost constraints.
        
        Args:
            ensemble_config: Ensemble model configuration
            data_path: Path to training data
            base_model_configs: Configurations for base models
            job_prefix: Prefix for job names
            max_budget: Maximum budget for all training
            sequential: Whether to train models sequentially
            
        Returns:
            Dict with job IDs and configurations
        """
        # Estimate data size
        if os.path.exists(data_path):
            if os.path.isfile(data_path):
                data_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            else:
                # Sum size of all files in directory
                total_size = 0
                for dirpath, _, filenames in os.walk(data_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                data_size_mb = total_size / (1024 * 1024)
        else:
            # Default to medium size if cannot determine
            data_size_mb = 500
        
        result = {
            "base_model_jobs": {},
            "ensemble_job": None,
            "total_estimated_cost": 0.0
        }
        
        timestamp = int(time.time())
        
        # Split budget among models if provided
        model_budget = None
        if max_budget:
            # Allocate 70% to base models, 30% to ensemble
            base_budget = max_budget * 0.7
            ensemble_budget = max_budget * 0.3
            
            # Split base budget among models
            model_budget = base_budget / len(base_model_configs)
        
        # Train base models
        base_model_ids = []
        for model_type, model_config in base_model_configs.items():
            # Create job name
            job_name = f"{job_prefix}_{model_type}_{timestamp}"
            
            # Optimize and submit job
            job_id, optimized = self.submit_cost_optimized_job(
                job_name=job_name,
                model_type=model_type,
                data_path=data_path,
                config=model_config,
                data_size_mb=data_size_mb,
                max_budget=model_budget
            )
            
            base_model_ids.append(job_id)
            result["base_model_jobs"][model_type] = {
                "job_id": job_id,
                "config": optimized
            }
            result["total_estimated_cost"] += optimized["estimated_cost"]
            
            # Wait for completion if sequential
            if sequential:
                self.client.wait_for_job(job_id)
        
        # Train ensemble model
        ensemble_job_name = f"{job_prefix}_ensemble_{timestamp}"
        
        # Submit ensemble job
        ensemble_job_id = self.client.submit_ensemble_job(
            job_name=ensemble_job_name,
            base_model_ids=base_model_ids,
            ensemble_config=ensemble_config
        )
        
        # Estimate ensemble job cost
        ensemble_instance = "ml.c5.xlarge"  # Default for ensemble
        ensemble_cost = self.estimate_training_cost(ensemble_instance, 1, True)
        
        result["ensemble_job"] = {
            "job_id": ensemble_job_id,
            "estimated_cost": ensemble_cost
        }
        result["total_estimated_cost"] += ensemble_cost
        
        logger.info(f"Submitted ensemble training with {len(base_model_ids)} base models")
        logger.info(f"Total estimated cost: ${result['total_estimated_cost']:.2f}")
        
        return result