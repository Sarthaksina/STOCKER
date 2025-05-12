"""STOCKER Pro Artifacts Module

This module provides utilities for managing artifacts in the STOCKER Pro application.
Artifacts include models, datasets, portfolios, and other resources used by the application.

The module consolidates functionality from various utility files for better maintainability
and organization, following production-grade practices.

Sections:
    - Artifact Management: Functions for saving and loading artifacts
    - Model Artifacts: Utilities specific to model artifacts
    - Data Artifacts: Utilities for data artifacts
    - Portfolio Artifacts: Utilities for portfolio artifacts
    - Versioning: Functions for versioning artifacts
    - Cleanup: Utilities for cleaning up old artifacts
"""

import os
import json
import yaml
import hashlib
import logging
import shutil
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from src.core.logging import logger

# ===== Artifact Base Functions =====

def get_artifact_path(artifact_type: str, name: str, version: Optional[str] = None) -> Path:
    """
    Generate a path for an artifact.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        version: Optional version string
        
    Returns:
        Path object for the artifact
    """
    base_dir = Path("artifacts") / artifact_type
    if version:
        return base_dir / name / version
    return base_dir / name

def save_artifact(artifact: Any, artifact_type: str, name: str, 
                 version: Optional[str] = None, metadata: Optional[Dict] = None) -> Path:
    """
    Save an artifact with optional metadata.
    
    Args:
        artifact: The artifact to save
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        version: Optional version string (defaults to timestamp)
        metadata: Optional metadata dictionary
        
    Returns:
        Path where the artifact was saved
    """
    # Generate version if not provided
    if not version:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Get artifact path
    artifact_path = get_artifact_path(artifact_type, name, version)
    
    # Create directory if it doesn't exist
    os.makedirs(artifact_path, exist_ok=True)
    
    # Save artifact based on type
    if artifact_type == "model":
        _save_model_artifact(artifact, artifact_path)
    elif artifact_type == "data":
        _save_data_artifact(artifact, artifact_path)
    elif artifact_type == "portfolio":
        _save_portfolio_artifact(artifact, artifact_path)
    else:
        # Generic artifact saving
        with open(artifact_path / "artifact.pkl", "wb") as f:
            import pickle
            pickle.dump(artifact, f)
    
    # Save metadata if provided
    if metadata:
        metadata["created_at"] = datetime.now().isoformat()
        metadata["version"] = version
        with open(artifact_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {artifact_type} artifact '{name}' version '{version}'")
    return artifact_path

def load_artifact(artifact_type: str, name: str, 
                version: Optional[str] = None) -> Tuple[Any, Dict]:
    """
    Load an artifact and its metadata.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        version: Optional version string (loads latest if not provided)
        
    Returns:
        Tuple of (artifact, metadata)
    """
    # Get version if not provided
    if not version:
        version = _get_latest_version(artifact_type, name)
        if not version:
            raise FileNotFoundError(f"No versions found for {artifact_type} artifact '{name}'")
    
    # Get artifact path
    artifact_path = get_artifact_path(artifact_type, name, version)
    
    # Check if artifact exists
    if not artifact_path.exists():
        raise FileNotFoundError(f"{artifact_type.capitalize()} artifact '{name}' version '{version}' not found")
    
    # Load metadata if exists
    metadata = {}
    metadata_path = artifact_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Load artifact based on type
    if artifact_type == "model":
        artifact = _load_model_artifact(artifact_path)
    elif artifact_type == "data":
        artifact = _load_data_artifact(artifact_path)
    elif artifact_type == "portfolio":
        artifact = _load_portfolio_artifact(artifact_path)
    else:
        # Generic artifact loading
        with open(artifact_path / "artifact.pkl", "rb") as f:
            import pickle
            artifact = pickle.load(f)
    
    logger.info(f"Loaded {artifact_type} artifact '{name}' version '{version}'")
    return artifact, metadata

# ===== Model Artifact Functions =====

def _save_model_artifact(model: Any, path: Path) -> None:
    """
    Save a model artifact.
    
    Args:
        model: The model to save
        path: Path to save the model
    """
    try:
        # Check if model has a save method (e.g., Keras, PyTorch)
        if hasattr(model, 'save'):
            model.save(path / "model")
        elif hasattr(model, 'save_model'):
            model.save_model(path / "model")
        else:
            # Try joblib for scikit-learn models
            try:
                import joblib
                joblib.dump(model, path / "model.joblib")
            except ImportError:
                # Fallback to pickle
                import pickle
                with open(path / "model.pkl", "wb") as f:
                    pickle.dump(model, f)
    except Exception as e:
        logger.error(f"Error saving model artifact: {e}")
        raise

def _load_model_artifact(path: Path) -> Any:
    """
    Load a model artifact.
    
    Args:
        path: Path to the model
        
    Returns:
        Loaded model
    """
    try:
        # Try different model loading methods
        
        # Check for TensorFlow/Keras model
        if (path / "model").exists() and (path / "model").is_dir():
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(path / "model")
            except ImportError:
                pass
        
        # Check for PyTorch model
        if (path / "model.pt").exists():
            try:
                import torch
                return torch.load(path / "model.pt")
            except ImportError:
                pass
        
        # Check for joblib serialized model
        if (path / "model.joblib").exists():
            try:
                import joblib
                return joblib.load(path / "model.joblib")
            except ImportError:
                pass
        
        # Fallback to pickle
        if (path / "model.pkl").exists():
            import pickle
            with open(path / "model.pkl", "rb") as f:
                return pickle.load(f)
        
        raise FileNotFoundError(f"No model file found in {path}")
        
    except Exception as e:
        logger.error(f"Error loading model artifact: {e}")
        raise

# ===== Data Artifact Functions =====

def _save_data_artifact(data: Any, path: Path) -> None:
    """
    Save a data artifact.
    
    Args:
        data: The data to save
        path: Path to save the data
    """
    try:
        # Check if data is a pandas DataFrame
        if 'pandas' in str(type(data)):
            data.to_csv(path / "data.csv", index=True)
            data.to_pickle(path / "data.pkl")
        # Check if data is a numpy array
        elif 'numpy' in str(type(data)):
            import numpy as np
            np.save(path / "data.npy", data)
        else:
            # Fallback to pickle
            import pickle
            with open(path / "data.pkl", "wb") as f:
                pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving data artifact: {e}")
        raise

def _load_data_artifact(path: Path) -> Any:
    """
    Load a data artifact.
    
    Args:
        path: Path to the data
        
    Returns:
        Loaded data
    """
    try:
        # Try different data loading methods
        
        # Check for pandas DataFrame (pickle version)
        if (path / "data.pkl").exists():
            try:
                import pandas as pd
                return pd.read_pickle(path / "data.pkl")
            except (ImportError, Exception):
                # Fallback to regular pickle
                import pickle
                with open(path / "data.pkl", "rb") as f:
                    return pickle.load(f)
        
        # Check for pandas DataFrame (CSV version)
        if (path / "data.csv").exists():
            try:
                import pandas as pd
                return pd.read_csv(path / "data.csv")
            except ImportError:
                pass
        
        # Check for numpy array
        if (path / "data.npy").exists():
            try:
                import numpy as np
                return np.load(path / "data.npy")
            except ImportError:
                pass
        
        raise FileNotFoundError(f"No data file found in {path}")
        
    except Exception as e:
        logger.error(f"Error loading data artifact: {e}")
        raise

# ===== Portfolio Artifact Functions =====

def _save_portfolio_artifact(portfolio: Any, path: Path) -> None:
    """
    Save a portfolio artifact.
    
    Args:
        portfolio: The portfolio to save
        path: Path to save the portfolio
    """
    try:
        # Check if portfolio is a dictionary
        if isinstance(portfolio, dict):
            with open(path / "portfolio.json", "w") as f:
                json.dump(portfolio, f, indent=2, default=str)
        # Check if portfolio is a pandas DataFrame
        elif 'pandas' in str(type(portfolio)):
            portfolio.to_csv(path / "portfolio.csv", index=True)
            portfolio.to_pickle(path / "portfolio.pkl")
        else:
            # Fallback to pickle
            import pickle
            with open(path / "portfolio.pkl", "wb") as f:
                pickle.dump(portfolio, f)
    except Exception as e:
        logger.error(f"Error saving portfolio artifact: {e}")
        raise

def _load_portfolio_artifact(path: Path) -> Any:
    """
    Load a portfolio artifact.
    
    Args:
        path: Path to the portfolio
        
    Returns:
        Loaded portfolio
    """
    try:
        # Try different portfolio loading methods
        
        # Check for JSON portfolio
        if (path / "portfolio.json").exists():
            with open(path / "portfolio.json", "r") as f:
                portfolio_dict = json.load(f)
            
            # Convert date strings to datetime objects if needed
            for key, value in portfolio_dict.items():
                if isinstance(value, str) and len(value) > 10:
                    try:
                        portfolio_dict[key] = datetime.fromisoformat(value)
                    except ValueError:
                        pass
            
            return portfolio_dict
        
        # Check for pandas DataFrame (pickle version)
        if (path / "portfolio.pkl").exists():
            try:
                import pandas as pd
                return pd.read_pickle(path / "portfolio.pkl")
            except (ImportError, Exception):
                # Fallback to regular pickle
                import pickle
                with open(path / "portfolio.pkl", "rb") as f:
                    return pickle.load(f)
        
        # Check for pandas DataFrame (CSV version)
        if (path / "portfolio.csv").exists():
            try:
                import pandas as pd
                return pd.read_csv(path / "portfolio.csv")
            except ImportError:
                pass
        
        raise FileNotFoundError(f"No portfolio file found in {path}")
        
    except Exception as e:
        logger.error(f"Error loading portfolio artifact: {e}")
        raise

# ===== Version Management Functions =====

def _get_latest_version(artifact_type: str, name: str) -> Optional[str]:
    """
    Get the latest version of an artifact.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        
    Returns:
        Latest version string or None if no versions exist
    """
    artifact_path = get_artifact_path(artifact_type, name)
    
    if not artifact_path.exists():
        return None
    
    versions = []
    for version_dir in artifact_path.iterdir():
        if version_dir.is_dir():
            versions.append(version_dir.name)
    
    if not versions:
        return None
    
    # Sort versions by timestamp (assuming YYYYMMDD_HHMMSS format)
    versions.sort(reverse=True)
    return versions[0]

def list_artifact_versions(artifact_type: str, name: str) -> List[Dict[str, Any]]:
    """
    List all versions of an artifact with metadata.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        
    Returns:
        List of dictionaries with version info and metadata
    """
    artifact_path = get_artifact_path(artifact_type, name)
    
    if not artifact_path.exists():
        return []
    
    versions = []
    for version_dir in artifact_path.iterdir():
        if version_dir.is_dir():
            version_info = {"version": version_dir.name}
            
            # Load metadata if exists
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                version_info.update(metadata)
            
            versions.append(version_info)
    
    # Sort versions by timestamp (assuming YYYYMMDD_HHMMSS format)
    versions.sort(key=lambda x: x["version"], reverse=True)
    return versions

# ===== Cleanup Functions =====

def cleanup_old_artifacts(artifact_type: str, name: str, keep_versions: int = 5) -> int:
    """
    Clean up old artifact versions, keeping only the specified number of recent versions.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        keep_versions: Number of recent versions to keep
        
    Returns:
        Number of versions deleted
    """
    artifact_path = get_artifact_path(artifact_type, name)
    
    if not artifact_path.exists():
        return 0
    
    versions = []
    for version_dir in artifact_path.iterdir():
        if version_dir.is_dir():
            versions.append(version_dir.name)
    
    if len(versions) <= keep_versions:
        return 0
    
    # Sort versions by timestamp (assuming YYYYMMDD_HHMMSS format)
    versions.sort(reverse=True)
    
    # Keep the most recent versions
    versions_to_keep = versions[:keep_versions]
    versions_to_delete = versions[keep_versions:]
    
    # Delete old versions
    for version in versions_to_delete:
        version_path = artifact_path / version
        shutil.rmtree(version_path)
        logger.info(f"Deleted old {artifact_type} artifact '{name}' version '{version}'")
    
    return len(versions_to_delete)

# ===== Artifact Comparison Functions =====

def compare_artifacts(artifact_type: str, name: str, version1: str, version2: str) -> Dict[str, Any]:
    """
    Compare two versions of an artifact.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        version1: First version to compare
        version2: Second version to compare
        
    Returns:
        Dictionary with comparison results
    """
    # Load both artifacts
    artifact1, metadata1 = load_artifact(artifact_type, name, version1)
    artifact2, metadata2 = load_artifact(artifact_type, name, version2)
    
    # Initialize comparison results
    comparison = {
        "artifact_type": artifact_type,
        "name": name,
        "version1": version1,
        "version2": version2,
        "metadata_diff": {},
        "content_diff": {}
    }
    
    # Compare metadata
    all_keys = set(metadata1.keys()) | set(metadata2.keys())
    for key in all_keys:
        if key not in metadata1:
            comparison["metadata_diff"][key] = {"added": metadata2[key]}
        elif key not in metadata2:
            comparison["metadata_diff"][key] = {"removed": metadata1[key]}
        elif metadata1[key] != metadata2[key]:
            comparison["metadata_diff"][key] = {"from": metadata1[key], "to": metadata2[key]}
    
    # Compare content based on artifact type
    if artifact_type == "model":
        # For models, compare architecture if possible
        if hasattr(artifact1, 'get_config') and hasattr(artifact2, 'get_config'):
            config1 = artifact1.get_config()
            config2 = artifact2.get_config()
            
            # Compare model configurations
            if config1 != config2:
                comparison["content_diff"]["architecture_changed"] = True
                
                # Compare specific configuration differences
                if isinstance(config1, dict) and isinstance(config2, dict):
                    config_diff = {}
                    all_config_keys = set(config1.keys()) | set(config2.keys())
                    
                    for key in all_config_keys:
                        if key not in config1:
                            config_diff[key] = {"added": config2[key]}
                        elif key not in config2:
                            config_diff[key] = {"removed": config1[key]}
                        elif config1[key] != config2[key]:
                            config_diff[key] = {"from": config1[key], "to": config2[key]}
                    
                    comparison["content_diff"]["config_diff"] = config_diff
            else:
                comparison["content_diff"]["architecture_changed"] = False
        
    elif artifact_type == "data":
        # For data, compare shape, size, etc.
        if hasattr(artifact1, 'shape') and hasattr(artifact2, 'shape'):
            comparison["content_diff"]["shape1"] = artifact1.shape
            comparison["content_diff"]["shape2"] = artifact2.shape
            comparison["content_diff"]["shape_changed"] = (artifact1.shape != artifact2.shape)
        
        if hasattr(artifact1, 'columns') and hasattr(artifact2, 'columns'):
            cols1 = set(artifact1.columns)
            cols2 = set(artifact2.columns)
            comparison["content_diff"]["columns_added"] = list(cols2 - cols1)
            comparison["content_diff"]["columns_removed"] = list(cols1 - cols2)
    
    elif artifact_type == "portfolio":
        # For portfolios, compare weights, performance metrics, etc.
        if isinstance(artifact1, dict) and isinstance(artifact2, dict):
            # Compare portfolio dictionaries
            all_keys = set(artifact1.keys()) | set(artifact2.keys())
            for key in all_keys:
                if key not in artifact1:
                    comparison["content_diff"][key] = {"added": artifact2[key]}
                elif key not in artifact2:
                    comparison["content_diff"][key] = {"removed": artifact1[key]}
                elif artifact1[key] != artifact2[key]:
                    comparison["content_diff"][key] = {"from": artifact1[key], "to": artifact2[key]}
    
    return comparison

# ===== Artifact Export/Import Functions =====

def export_artifact(artifact_type: str, name: str, version: str, export_path: Path) -> Path:
    """
    Export an artifact to a specified location.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        version: Version to export
        export_path: Path to export the artifact to
        
    Returns:
        Path where the artifact was exported
    """
    # Get artifact path
    artifact_path = get_artifact_path(artifact_type, name, version)
    
    # Check if artifact exists
    if not artifact_path.exists():
        raise FileNotFoundError(f"{artifact_type.capitalize()} artifact '{name}' version '{version}' not found")
    
    # Create export directory if it doesn't exist
    os.makedirs(export_path, exist_ok=True)
    
    # Create a zip file of the artifact
    import zipfile
    export_file = export_path / f"{artifact_type}_{name}_{version}.zip"
    
    with zipfile.ZipFile(export_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(artifact_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, artifact_path)
                zipf.write(file_path, arcname)
    
    logger.info(f"Exported {artifact_type} artifact '{name}' version '{version}' to {export_file}")
    return export_file

def import_artifact(import_file: Path, artifact_type: str = None, name: str = None, version: str = None) -> Tuple[str, str, str]:
    """
    Import an artifact from a file.
    
    Args:
        import_file: Path to the artifact file to import
        artifact_type: Optional type override
        name: Optional name override
        version: Optional version override
        
    Returns:
        Tuple of (artifact_type, name, version)
    """
    # Extract artifact info from filename if not provided
    if not all([artifact_type, name, version]):
        filename = os.path.basename(import_file)
        parts = os.path.splitext(filename)[0].split("_")
        
        if len(parts) >= 3:
            artifact_type = artifact_type or parts[0]
            name = name or parts[1]
            version = version or parts[2]
        else:
            raise ValueError(f"Cannot extract artifact info from filename: {filename}")
    
    # Get artifact path
    artifact_path = get_artifact_path(artifact_type, name, version)
    
    # Create artifact directory if it doesn't exist
    os.makedirs(artifact_path, exist_ok=True)
    
    # Extract the zip file
    import zipfile
    with zipfile.ZipFile(import_file, "r") as zipf:
        zipf.extractall(artifact_path)
    
    logger.info(f"Imported {artifact_type} artifact '{name}' version '{version}'")
    return artifact_type, name, version
