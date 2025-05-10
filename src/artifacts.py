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

# Configure logger
logger = logging.getLogger(__name__)

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
    
    # Load metadata if it exists
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
    # Determine model type and use appropriate saving method
    try:
        # Try TensorFlow/Keras saving
        model.save(path / "model")
    except (AttributeError, ImportError):
        try:
            # Try scikit-learn saving
            import joblib
            joblib.dump(model, path / "model.joblib")
        except (AttributeError, ImportError):
            try:
                # Try PyTorch saving
                import torch
                torch.save(model, path / "model.pt")
            except (AttributeError, ImportError):
                # Fallback to pickle
                import pickle
                with open(path / "model.pkl", "wb") as f:
                    pickle.dump(model, f)

def _load_model_artifact(path: Path) -> Any:
    """
    Load a model artifact.
    
    Args:
        path: Path to the model
        
    Returns:
        Loaded model
    """
    # Try different loading methods based on what files exist
    if (path / "model").exists():
        try:
            # Try TensorFlow/Keras loading
            import tensorflow as tf
            return tf.keras.models.load_model(path / "model")
        except ImportError:
            logger.warning("TensorFlow not available, trying other methods")
    
    if (path / "model.joblib").exists():
        try:
            # Try scikit-learn loading
            import joblib
            return joblib.load(path / "model.joblib")
        except ImportError:
            logger.warning("joblib not available, trying other methods")
    
    if (path / "model.pt").exists():
        try:
            # Try PyTorch loading
            import torch
            return torch.load(path / "model.pt")
        except ImportError:
            logger.warning("PyTorch not available, trying other methods")
    
    if (path / "model.pkl").exists():
        # Fallback to pickle
        import pickle
        with open(path / "model.pkl", "rb") as f:
            return pickle.load(f)
    
    raise FileNotFoundError(f"No model files found in {path}")

# ===== Data Artifact Functions =====

def _save_data_artifact(data: Any, path: Path) -> None:
    """
    Save a data artifact.
    
    Args:
        data: The data to save
        path: Path to save the data
    """
    # Handle different data types
    if hasattr(data, "to_csv"):
        # Pandas DataFrame or Series
        data.to_csv(path / "data.csv")
    elif hasattr(data, "savez"):
        # NumPy array
        import numpy as np
        np.savez(path / "data.npz", data=data)
    else:
        # Fallback to pickle
        import pickle
        with open(path / "data.pkl", "wb") as f:
            pickle.dump(data, f)

def _load_data_artifact(path: Path) -> Any:
    """
    Load a data artifact.
    
    Args:
        path: Path to the data
        
    Returns:
        Loaded data
    """
    # Try different loading methods based on what files exist
    if (path / "data.csv").exists():
        try:
            # Try pandas loading
            import pandas as pd
            return pd.read_csv(path / "data.csv", index_col=0)
        except ImportError:
            logger.warning("pandas not available, trying other methods")
    
    if (path / "data.npz").exists():
        try:
            # Try NumPy loading
            import numpy as np
            return np.load(path / "data.npz")["data"]
        except ImportError:
            logger.warning("NumPy not available, trying other methods")
    
    if (path / "data.pkl").exists():
        # Fallback to pickle
        import pickle
        with open(path / "data.pkl", "rb") as f:
            return pickle.load(f)
    
    raise FileNotFoundError(f"No data files found in {path}")

# ===== Portfolio Artifact Functions =====

def _save_portfolio_artifact(portfolio: Any, path: Path) -> None:
    """
    Save a portfolio artifact.
    
    Args:
        portfolio: The portfolio to save
        path: Path to save the portfolio
    """
    # Save portfolio as JSON if it's a dictionary
    if isinstance(portfolio, dict):
        with open(path / "portfolio.json", "w") as f:
            json.dump(portfolio, f, indent=2)
    # Save portfolio as CSV if it has weights attribute
    elif hasattr(portfolio, "weights") and hasattr(portfolio.weights, "to_csv"):
        portfolio.weights.to_csv(path / "weights.csv")
        # Save additional portfolio attributes if available
        if hasattr(portfolio, "to_dict"):
            with open(path / "portfolio_config.json", "w") as f:
                json.dump(portfolio.to_dict(), f, indent=2)
    # Fallback to pickle for complex objects
    else:
        import pickle
        with open(path / "portfolio.pkl", "wb") as f:
            pickle.dump(portfolio, f)

def _load_portfolio_artifact(path: Path) -> Any:
    """
    Load a portfolio artifact.
    
    Args:
        path: Path to the portfolio
        
    Returns:
        Loaded portfolio
    """
    # Try different loading methods based on what files exist
    if (path / "portfolio.json").exists():
        with open(path / "portfolio.json", "r") as f:
            return json.load(f)
    
    if (path / "weights.csv").exists():
        try:
            import pandas as pd
            weights = pd.read_csv(path / "weights.csv", index_col=0)
            
            # Try to load portfolio configuration if it exists
            if (path / "portfolio_config.json").exists():
                with open(path / "portfolio_config.json", "r") as f:
                    config = json.load(f)
                
                # Try to reconstruct portfolio object if possible
                try:
                    from src.features.portfolio.portfolio_core import PortfolioManager
                    from src.features.portfolio.portfolio_config import PortfolioConfig
                    
                    portfolio_config = PortfolioConfig()
                    portfolio_config.from_dict(config)
                    portfolio = PortfolioManager(config=portfolio_config)
                    portfolio.set_weights(weights)
                    return portfolio
                except (ImportError, AttributeError):
                    # Return weights if portfolio object can't be reconstructed
                    return weights
            else:
                return weights
        except ImportError:
            logger.warning("pandas not available, trying other methods")
    
    if (path / "portfolio.pkl").exists():
        # Fallback to pickle
        import pickle
        with open(path / "portfolio.pkl", "rb") as f:
            return pickle.load(f)
    
    raise FileNotFoundError(f"No portfolio files found in {path}")

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
    base_path = get_artifact_path(artifact_type, name)
    if not base_path.exists():
        return None
    
    # Get all version directories
    versions = [d.name for d in base_path.iterdir() if d.is_dir()]
    if not versions:
        return None
    
    # Sort versions by creation time (assuming timestamp-based versions)
    versions.sort(reverse=True)
    return versions[0]

def list_artifact_versions(artifact_type: str, name: str) -> List[Dict]:
    """
    List all versions of an artifact with metadata.
    
    Args:
        artifact_type: Type of artifact (model, data, portfolio, etc.)
        name: Name of the artifact
        
    Returns:
        List of dictionaries with version info and metadata
    """
    base_path = get_artifact_path(artifact_type, name)
    if not base_path.exists():
        return []
    
    # Get all version directories
    versions = []
    for version_dir in base_path.iterdir():
        if not version_dir.is_dir():
            continue
        
        version_info = {"version": version_dir.name}
        
        # Add metadata if it exists
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                version_info["metadata"] = json.load(f)
        
        versions.append(version_info)
    
    # Sort versions by creation time (newest first)
    versions.sort(key=lambda x: x.get("metadata", {}).get("created_at", ""), reverse=True)
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
    versions = list_artifact_versions(artifact_type, name)
    if len(versions) <= keep_versions:
        return 0
    
    # Delete older versions
    versions_to_delete = versions[keep_versions:]
    deleted_count = 0
    
    for version_info in versions_to_delete:
        version = version_info["version"]
        version_path = get_artifact_path(artifact_type, name, version)
        try:
            shutil.rmtree(version_path)
            deleted_count += 1
            logger.info(f"Deleted old {artifact_type} artifact '{name}' version '{version}'")
        except Exception as e:
            logger.error(f"Failed to delete {artifact_type} artifact '{name}' version '{version}': {e}")
    
    return deleted_count

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
    # Load both artifacts and their metadata
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
        # For models, we can't easily compare the actual model objects
        # Instead, we can compare their serialized size or structure if available
        comparison["content_diff"]["type"] = "model"
        comparison["content_diff"]["note"] = "Model comparison requires specialized tools"
    
    elif artifact_type == "data":
        # For data artifacts, compare shape, columns, etc. if they're DataFrames
        if hasattr(artifact1, "shape") and hasattr(artifact2, "shape"):
            comparison["content_diff"]["shape_1"] = str(artifact1.shape)
            comparison["content_diff"]["shape_2"] = str(artifact2.shape)
        
        if hasattr(artifact1, "columns") and hasattr(artifact2, "columns"):
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