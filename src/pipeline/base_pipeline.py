"""
Base pipeline for STOCKER Pro.
This module defines the abstract base class for all pipelines.
"""
import logging
import time
import traceback
import os
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from src.configuration.config import StockerConfig
from src.logger.logger import get_advanced_logger

class BasePipeline(ABC):
    """
    Abstract base class for all STOCKER Pro pipelines.
    
    This class defines the common interface and functionality
    for all pipeline implementations, including training and
    prediction pipelines.
    
    Attributes:
        config: Configuration for the pipeline
        logger: Logger for tracking pipeline execution
        artifacts: Dictionary to store artifacts generated during pipeline execution
        start_time: Start time of the pipeline execution
        performance_metrics: Dictionary to store performance metrics
    """
    
    def __init__(self, config: StockerConfig):
        """
        Initialize the base pipeline.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config
        self.logger = get_advanced_logger(
            self.__class__.__name__, 
            log_to_file=True, 
            log_dir=os.path.join(config.logs_dir, "pipeline")
        )
        self.artifacts: Dict[str, Any] = {}
        self.start_time = time.time()
        self.performance_metrics: Dict[str, float] = {}
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the pipeline configuration.
        
        Raises:
            ConfigValidationError: If configuration validation fails
        """
        pass
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
            Dictionary containing pipeline artifacts
        """
        pass
    
    def log_step_start(self, step_name: str) -> float:
        """
        Log the start of a pipeline step and return the start time.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            Start time of the step
        """
        self.logger.info(f"Starting step: {step_name}")
        return time.time()
    
    def log_step_end(self, step_name: str, start_time: float) -> None:
        """
        Log the end of a pipeline step and record the execution time.
        
        Args:
            step_name: Name of the pipeline step
            start_time: Start time of the step
        """
        execution_time = time.time() - start_time
        self.performance_metrics[f"{step_name}_time"] = execution_time
        self.logger.info(f"Completed step: {step_name} in {execution_time:.2f} seconds")
    
    def save_artifacts(self, output_dir: Optional[str] = None) -> str:
        """
        Save pipeline artifacts to disk.
        
        Args:
            output_dir: Directory to save artifacts
            
        Returns:
            Path where artifacts were saved
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.project_root, "artifacts", 
                                     self.__class__.__name__, 
                                     datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance metrics
        metrics_path = os.path.join(output_dir, "performance_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Save pipeline configuration
        config_path = os.path.join(output_dir, "pipeline_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved pipeline artifacts to {output_dir}")
        return output_dir
    
    def handle_exception(self, e: Exception, step_name: str) -> None:
        """
        Handle and log pipeline exceptions.
        
        Args:
            e: Exception that occurred
            step_name: Name of the step where the exception occurred
        """
        self.logger.error(f"Error in {step_name}: {str(e)}")
        self.logger.debug(f"Exception details: {traceback.format_exc()}")
        self.performance_metrics[f"{step_name}_error"] = str(e)