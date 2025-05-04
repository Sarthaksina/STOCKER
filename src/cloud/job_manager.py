"""
Cloud training job management utilities for STOCKER Pro.

This module provides job management functionality for cloud-based model training,
including job scheduling, monitoring, and ensemble coordination.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import threading
import queue

from src.cloud_training.thunder_compute import ThunderComputeClient
from src.cloud_training.cloud_optimizer import CloudTrainingOptimizer
from src.ml import BaseModel, LSTMModel, XGBoostModel, LightGBMModel, EnsembleModel

logger = logging.getLogger(__name__)

class JobStatus:
    """Status constants for cloud training jobs."""
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"
    FALLBACK_LOCAL = "FALLBACK_LOCAL"  # New status for local fallback execution

class CloudJobManager:
    """
    Manages cloud-based model training jobs, monitoring, and ensemble coordination.
    """
    
    def __init__(self, thunder_client: ThunderComputeClient):
        """
        Initialize the cloud job manager.
        
        Args:
            thunder_client: ThunderCompute client for cloud operations
        """
        self.client = thunder_client
        self.optimizer = CloudTrainingOptimizer(thunder_client)
        
        # Job registry for tracking submitted jobs
        self.job_registry = {}
        
        # Thread-safe queue for async job monitoring
        self.status_queue = queue.Queue()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Fallback configuration
        self.enable_fallback = True
        self.fallback_timeout = 30  # seconds to wait before falling back to local
        self.max_retries = 3
        
    def submit_training_job(self,
                         model_type: str,
                         data_path: str,
                         config: Dict[str, Any],
                         job_name: Optional[str] = None,
                         optimize_cost: bool = True,
                         data_size_mb: Optional[float] = None,
                         max_budget: Optional[float] = None,
                         max_runtime_hours: Optional[int] = None,
                         monitor: bool = False,
                         allow_fallback: bool = True) -> str:
        """
        Submit a model training job to the cloud.
        
        Args:
            model_type: Type of model to train
            data_path: Path to training data
            config: Model configuration
            job_name: Optional job name (generated if not provided)
            optimize_cost: Whether to optimize for cost
            data_size_mb: Size of training data in MB
            max_budget: Maximum budget for training
            max_runtime_hours: Maximum runtime hours
            monitor: Whether to start monitoring the job
            allow_fallback: Whether to allow fallback to local execution
            
        Returns:
            Job ID
        """
        # Generate job name if not provided
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name = f"stocker_{model_type}_{timestamp}"
            
        # Try to submit to cloud with retries
        job_id = None
        for attempt in range(self.max_retries):
            try:
                # Submit job with cost optimization if requested
                if optimize_cost:
                    job_id, optimized = self.optimizer.submit_cost_optimized_job(
                        job_name=job_name,
                        model_type=model_type,
                        data_path=data_path,
                        config=config,
                        data_size_mb=data_size_mb,
                        max_budget=max_budget,
                        max_runtime_hours=max_runtime_hours
                    )
                    
                    # Store optimized configuration
                    self.job_registry[job_id] = {
                        "job_id": job_id,
                        "job_name": job_name,
                        "model_type": model_type,
                        "data_path": data_path,
                        "config": optimized["model_config"],
                        "training_config": optimized["training_config"],
                        "estimated_cost": optimized["estimated_cost"],
                        "submit_time": datetime.now().isoformat(),
                        "status": JobStatus.PENDING
                    }
                else:
                    # Submit job directly
                    job_id = self.client.submit_job(
                        job_name=job_name,
                        model_type=model_type,
                        data_path=data_path,
                        config=config,
                        instance_type=None,  # Use default
                        max_runtime_hours=max_runtime_hours
                    )
                    
                    # Store job information
                    self.job_registry[job_id] = {
                        "job_id": job_id,
                        "job_name": job_name,
                        "model_type": model_type,
                        "data_path": data_path,
                        "config": config,
                        "submit_time": datetime.now().isoformat(),
                        "status": JobStatus.PENDING
                    }
                
                logger.info(f"Submitted {model_type} training job: {job_name} (ID: {job_id})")
                break  # Success, exit retry loop
                
            except ConnectionError as e:
                logger.warning(f"Connection error submitting job (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Error submitting job: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # All retries failed
                    if allow_fallback and self.enable_fallback:
                        logger.info(f"Falling back to local execution for {model_type} job: {job_name}")
                        return self._execute_job_locally(model_type, data_path, config, job_name)
                    else:
                        raise
        
        # Start monitoring if requested
        if monitor and job_id:
            self.start_job_monitoring([job_id])
        
        return job_id
    
    def _execute_job_locally(self, model_type: str, data_path: str, 
                           config: Dict[str, Any], job_name: str) -> str:
        """
        Execute a training job locally as fallback
        
        Args:
            model_type: Type of model to train
            data_path: Path to training data
            config: Model configuration
            job_name: Job name
            
        Returns:
            Job ID (generated for tracking)
        """
        # Generate a local job ID
        job_id = f"local_{job_name}_{int(time.time())}"
        
        # Store job information
        self.job_registry[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "model_type": model_type,
            "data_path": data_path,
            "config": config,
            "submit_time": datetime.now().isoformat(),
            "status": JobStatus.FALLBACK_LOCAL,
            "is_local": True
        }
        
        # Start a thread to execute the job locally
        thread = threading.Thread(
            target=self._run_local_training,
            args=(job_id, model_type, data_path, config),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started local fallback training for {model_type} job: {job_name} (ID: {job_id})")
        return job_id
    
    def _run_local_training(self, job_id: str, model_type: str, 
                          data_path: str, config: Dict[str, Any]) -> None:
        """
        Run a training job locally
        
        Args:
            job_id: Job ID
            model_type: Type of model to train
            data_path: Path to training data
            config: Model configuration
        """
        try:
            # Update job status
            self.job_registry[job_id]["status"] = JobStatus.RUNNING
            
            # Load data
            data = pd.read_csv(data_path) if isinstance(data_path, str) else data_path
            
            # Initialize model based on type
            if model_type == "lstm":
                model = LSTMModel(**config)
            elif model_type == "xgboost":
                model = XGBoostModel(**config)
            elif model_type == "lightgbm":
                model = LightGBMModel(**config)
            elif model_type == "ensemble":
                model = EnsembleModel(**config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.train(data)
            
            # Save model
            output_dir = os.path.join("models", "local_fallback", job_id)
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "model.pkl")
            model.save(model_path)
            
            # Update job status
            self.job_registry[job_id]["status"] = JobStatus.COMPLETED
            self.job_registry[job_id]["completion_time"] = datetime.now().isoformat()
            self.job_registry[job_id]["output_path"] = model_path
            
            logger.info(f"Completed local training for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error in local training for job {job_id}: {e}")
            self.job_registry[job_id]["status"] = JobStatus.FAILED
            self.job_registry[job_id]["error"] = str(e)

    def submit_ensemble_training(self,
                               ensemble_config: Dict[str, Any],
                               data_path: str,
                               base_models: Dict[str, Dict[str, Any]],
                               job_name: Optional[str] = None,
                               optimize_cost: bool = True,
                               max_budget: Optional[float] = None,
                               wait_for_completion: bool = False) -> Dict[str, Any]:
        """
        Submit training jobs for an ensemble model and its base models.
        
        Args:
            ensemble_config: Ensemble model configuration
            data_path: Path to training data
            base_models: Configurations for base models
            job_name: Optional job name prefix
            optimize_cost: Whether to optimize for cost
            max_budget: Maximum budget for all training
            wait_for_completion: Whether to wait for all jobs to complete
            
        Returns:
            Dictionary with job IDs and configurations
        """
        # Generate job name prefix if not provided
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name = f"stocker_ensemble_{timestamp}"
        
        if optimize_cost:
            # Use optimizer for batch optimization
            result = self.optimizer.batch_optimize_ensemble_training(
                ensemble_config=ensemble_config,
                data_path=data_path,
                base_model_configs=base_models,
                job_prefix=job_name,
                max_budget=max_budget,
                sequential=wait_for_completion
            )
            
            # Update job registry with base model jobs
            for model_type, job_info in result["base_model_jobs"].items():
                job_id = job_info["job_id"]
                self.job_registry[job_id] = {
                    "job_id": job_id,
                    "job_name": f"{job_name}_{model_type}",
                    "model_type": model_type,
                    "data_path": data_path,
                    "config": job_info["config"]["model_config"],
                    "training_config": job_info["config"]["training_config"],
                    "estimated_cost": job_info["config"]["estimated_cost"],
                    "submit_time": datetime.now().isoformat(),
                    "status": JobStatus.PENDING,
                    "part_of_ensemble": True,
                    "ensemble_job_name": job_name
                }
            
            # Update job registry with ensemble job
            ensemble_job_id = result["ensemble_job"]["job_id"]
            self.job_registry[ensemble_job_id] = {
                "job_id": ensemble_job_id,
                "job_name": f"{job_name}_ensemble",
                "model_type": "ensemble",
                "config": ensemble_config,
                "estimated_cost": result["ensemble_job"]["estimated_cost"],
                "submit_time": datetime.now().isoformat(),
                "status": JobStatus.PENDING,
                "base_model_job_ids": [job_info["job_id"] for job_info in result["base_model_jobs"].values()]
            }
            
            # Start monitoring the jobs
            all_job_ids = [job_info["job_id"] for job_info in result["base_model_jobs"].values()] + [ensemble_job_id]
            self.start_job_monitoring(all_job_ids)
            
            if wait_for_completion:
                # Wait for all jobs to complete
                for job_id in all_job_ids:
                    self.client.wait_for_job(job_id)
            
            return result
        else:
            # Submit base model jobs manually
            base_model_jobs = {}
            base_model_ids = []
            
            for model_type, model_config in base_models.items():
                # Submit job
                job_id = self.client.submit_job(
                    job_name=f"{job_name}_{model_type}",
                    model_type=model_type,
                    data_path=data_path,
                    config=model_config
                )
                
                base_model_ids.append(job_id)
                base_model_jobs[model_type] = {"job_id": job_id}
                
                # Update job registry
                self.job_registry[job_id] = {
                    "job_id": job_id,
                    "job_name": f"{job_name}_{model_type}",
                    "model_type": model_type,
                    "data_path": data_path,
                    "config": model_config,
                    "submit_time": datetime.now().isoformat(),
                    "status": JobStatus.PENDING,
                    "part_of_ensemble": True,
                    "ensemble_job_name": job_name
                }
                
                if wait_for_completion:
                    # Wait for job to complete
                    self.client.wait_for_job(job_id)
            
            # Submit ensemble job
            ensemble_job_id = self.client.submit_ensemble_job(
                job_name=f"{job_name}_ensemble",
                base_model_ids=base_model_ids,
                ensemble_config=ensemble_config
            )
            
            # Update job registry
            self.job_registry[ensemble_job_id] = {
                "job_id": ensemble_job_id,
                "job_name": f"{job_name}_ensemble",
                "model_type": "ensemble",
                "config": ensemble_config,
                "submit_time": datetime.now().isoformat(),
                "status": JobStatus.PENDING,
                "base_model_job_ids": base_model_ids
            }
            
            if wait_for_completion:
                # Wait for ensemble job to complete
                self.client.wait_for_job(ensemble_job_id)
            
            result = {
                "base_model_jobs": base_model_jobs,
                "ensemble_job": {"job_id": ensemble_job_id}
            }
            
            # Start monitoring the jobs
            all_job_ids = base_model_ids + [ensemble_job_id]
            self.start_job_monitoring(all_job_ids)
            
            return result
    
    def get_job_status(self, job_id: str, refresh: bool = True) -> Dict[str, Any]:
        """
        Get status information for a job.
        
        Args:
            job_id: Job ID
            refresh: Whether to refresh status from cloud
            
        Returns:
            Job status information
        """
        if refresh:
            try:
                # Get updated status from cloud
                cloud_status = self.client.get_job_status(job_id)
                
                # Update job registry
                if job_id in self.job_registry:
                    self.job_registry[job_id]["status"] = cloud_status.get("status", JobStatus.UNKNOWN)
                    self.job_registry[job_id]["last_updated"] = datetime.now().isoformat()
                    
                    # Add metrics if available
                    if "metrics" in cloud_status:
                        self.job_registry[job_id]["metrics"] = cloud_status["metrics"]
                    
                    # Add completion time if job completed
                    if cloud_status.get("status") in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED]:
                        self.job_registry[job_id]["completion_time"] = cloud_status.get("completion_time", datetime.now().isoformat())
                else:
                    # Create new entry in registry
                    self.job_registry[job_id] = {
                        "job_id": job_id,
                        "job_name": cloud_status.get("job_name", f"job_{job_id}"),
                        "status": cloud_status.get("status", JobStatus.UNKNOWN),
                        "last_updated": datetime.now().isoformat()
                    }
                
                return cloud_status
            except Exception as e:
                logger.error(f"Failed to get job status for {job_id}: {e}")
                if job_id in self.job_registry:
                    return self.job_registry[job_id]
                else:
                    return {"job_id": job_id, "status": JobStatus.UNKNOWN, "error": str(e)}
        else:
            # Return cached status
            if job_id in self.job_registry:
                return self.job_registry[job_id]
            else:
                return {"job_id": job_id, "status": JobStatus.UNKNOWN}
    
    def start_job_monitoring(self, job_ids: List[str], poll_interval: int = 60) -> None:
        """
        Start asynchronous monitoring of jobs.
        
        Args:
            job_ids: List of job IDs to monitor
            poll_interval: Polling interval in seconds
        """
        # Stop existing monitoring if running
        self.stop_job_monitoring()
        
        # Start new monitoring thread
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_jobs,
            args=(job_ids, poll_interval),
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info(f"Started monitoring {len(job_ids)} jobs")
    
    def stop_job_monitoring(self) -> None:
        """Stop job monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=10)
            logger.info("Stopped job monitoring")
    
    def _monitor_jobs(self, job_ids: List[str], poll_interval: int) -> None:
        """
        Monitor jobs in a background thread.
        
        Args:
            job_ids: List of job IDs to monitor
            poll_interval: Polling interval in seconds
        """
        # Track which jobs are still active
        active_jobs = set(job_ids)
        
        while active_jobs and not self._stop_monitoring.is_set():
            # Check each active job
            completed_jobs = set()
            
            for job_id in active_jobs:
                try:
                    status = self.get_job_status(job_id)
                    
                    # Put status in queue for listeners
                    self.status_queue.put((job_id, status))
                    
                    # Check if job completed
                    if status.get("status") in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED]:
                        completed_jobs.add(job_id)
                        
                        # Check if job is part of ensemble
                        job_info = self.job_registry.get(job_id, {})
                        if job_info.get("part_of_ensemble") and status.get("status") != JobStatus.COMPLETED:
                            logger.warning(f"Base model job {job_id} {status.get('status').lower()}, ensemble may fail")
                except Exception as e:
                    logger.error(f"Error monitoring job {job_id}: {e}")
            
            # Remove completed jobs
            active_jobs -= completed_jobs
            
            # Wait before next poll
            for _ in range(poll_interval):
                if self._stop_monitoring.is_set():
                    break
                time.sleep(1)
    
    def get_next_status_update(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the next status update from the monitoring queue.
        
        Args:
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Tuple of (job_id, status) or None if queue empty or timeout
        """
        try:
            return self.status_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def load_trained_model(self, job_id: str) -> BaseModel:
        """
        Load a trained model from a completed job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Loaded model instance
        """
        # Get job information
        job_info = self.get_job_status(job_id)
        
        if job_info.get("status") != JobStatus.COMPLETED:
            raise ValueError(f"Cannot load model: Job {job_id} is not completed")
        
        # Determine model class based on model type
        model_type = job_info.get("model_type", self.job_registry.get(job_id, {}).get("model_type"))
        
        if not model_type:
            raise ValueError(f"Cannot determine model type for job {job_id}")
        
        # Select appropriate model class
        if model_type == "lstm":
            model_class = LSTMModel
        elif model_type == "xgboost":
            model_class = XGBoostModel
        elif model_type == "lightgbm":
            model_class = LightGBMModel
        elif model_type == "ensemble":
            model_class = EnsembleModel
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load the model
        model = self.client.load_trained_model(job_id, model_class)
        
        return model
    
    def load_ensemble_model(self, ensemble_job_id: str) -> EnsembleModel:
        """
        Load a complete ensemble model with all its base models.
        
        Args:
            ensemble_job_id: Ensemble job ID
            
        Returns:
            Loaded ensemble model instance
        """
        # Get ensemble job information
        ensemble_info = self.get_job_status(ensemble_job_id)
        
        if ensemble_info.get("status") != JobStatus.COMPLETED:
            raise ValueError(f"Cannot load ensemble: Job {ensemble_job_id} is not completed")
        
        # Get base model job IDs
        base_model_ids = ensemble_info.get("base_model_job_ids") or self.job_registry.get(ensemble_job_id, {}).get("base_model_job_ids", [])
        
        if not base_model_ids:
            raise ValueError(f"No base model jobs found for ensemble {ensemble_job_id}")
        
        # Load ensemble model shell first
        ensemble = self.load_trained_model(ensemble_job_id)
        
        # Load and add base models
        for job_id in base_model_ids:
            # Get base model information
            base_job_info = self.get_job_status(job_id)
            
            if base_job_info.get("status") != JobStatus.COMPLETED:
                logger.warning(f"Base model job {job_id} is not completed, ensemble may be incomplete")
                continue
            
            # Load base model
            base_model = self.load_trained_model(job_id)
            
            # Add to ensemble
            ensemble.add_model(base_model)
        
        return ensemble
    
    def resume_failed_job(self, job_id: str, max_runtime_hours: Optional[int] = None) -> str:
        """
        Resume a failed job from its latest checkpoint.
        
        Args:
            job_id: Failed job ID
            max_runtime_hours: Maximum runtime hours for resumed job
            
        Returns:
            New job ID
        """
        # Check if job has failed
        status = self.get_job_status(job_id)
        if status.get("status") not in [JobStatus.FAILED, JobStatus.STOPPED]:
            raise ValueError(f"Job {job_id} is not failed or stopped (status: {status.get('status')})")
        
        # Resume from checkpoint
        new_job_id = self.optimizer.resume_from_checkpoint(job_id, max_runtime_hours)
        
        # Update job registry
        original_info = self.job_registry.get(job_id, {})
        self.job_registry[new_job_id] = {
            "job_id": new_job_id,
            "job_name": f"{original_info.get('job_name', 'job')}_resumed",
            "model_type": original_info.get("model_type"),
            "data_path": original_info.get("data_path"),
            "config": original_info.get("config"),
            "submit_time": datetime.now().isoformat(),
            "status": JobStatus.PENDING,
            "resumed_from": job_id
        }
        
        # Start monitoring the new job
        self.start_job_monitoring([new_job_id])
        
        return new_job_id
    
    def save_job_registry(self, file_path: str) -> None:
        """
        Save job registry to file.
        
        Args:
            file_path: Path to save registry
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.job_registry, f, indent=2)
            logger.info(f"Saved job registry to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save job registry: {e}")
    
    def load_job_registry(self, file_path: str) -> None:
        """
        Load job registry from file.
        
        Args:
            file_path: Path to registry file
        """
        try:
            with open(file_path, 'r') as f:
                loaded_registry = json.load(f)
            
            # Update registry
            self.job_registry.update(loaded_registry)
            logger.info(f"Loaded job registry from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load job registry: {e}")
            
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get job status
            status = self.get_job_status(job_id)
            
            if status.get("status") not in [JobStatus.PENDING, JobStatus.STARTING, JobStatus.RUNNING]:
                logger.warning(f"Job {job_id} is not running (status: {status.get('status')})")
                return False
            
            # Make API call to cancel job
            response = self.client.cancel_job(job_id)
            
            # Update job registry
            if job_id in self.job_registry:
                self.job_registry[job_id]["status"] = JobStatus.STOPPED
                self.job_registry[job_id]["completion_time"] = datetime.now().isoformat()
            
            logger.info(f"Cancelled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False