"""
Portfolio Presets Module for STOCKER Pro

This module provides functionality for saving and loading user portfolio presets.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np

from stocker.cloud.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioPresetManager:
    """
    Manager for saving and loading portfolio presets
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize preset manager
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.presets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'presets')
        os.makedirs(self.presets_dir, exist_ok=True)
    
    def save_preset(self, 
                   preset_name: str,
                   preset_data: Dict[str, Any],
                   overwrite: bool = False) -> str:
        """
        Save a portfolio preset
        
        Args:
            preset_name: Name of the preset
            preset_data: Dictionary with preset data
            overwrite: Whether to overwrite existing preset
            
        Returns:
            Path to the saved preset file
        """
        # Sanitize preset name for filename
        preset_filename = preset_name.lower().replace(' ', '_').replace('-', '_')
        preset_path = os.path.join(self.presets_dir, f"{preset_filename}.json")
        
        # Check if preset exists
        if os.path.exists(preset_path) and not overwrite:
            raise ValueError(f"Preset '{preset_name}' already exists. Use overwrite=True to replace it.")
        
        # Add metadata
        preset_data['metadata'] = {
            'name': preset_name,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Convert numpy arrays and pandas objects to serializable format
        serializable_data = self._make_serializable(preset_data)
        
        # Save preset
        with open(preset_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved preset '{preset_name}' to {preset_path}")
        return preset_path
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Load a portfolio preset
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Dictionary with preset data
        """
        # Handle both preset name and filename
        if not preset_name.endswith('.json'):
            preset_filename = preset_name.lower().replace(' ', '_').replace('-', '_')
            preset_path = os.path.join(self.presets_dir, f"{preset_filename}.json")
        else:
            preset_path = os.path.join(self.presets_dir, preset_name)
        
        # Check if preset exists
        if not os.path.exists(preset_path):
            raise ValueError(f"Preset '{preset_name}' not found.")
        
        # Load preset
        with open(preset_path, 'r') as f:
            preset_data = json.load(f)
        
        # Convert serialized data back to original types
        restored_data = self._restore_from_serialized(preset_data)
        
        logger.info(f"Loaded preset '{preset_name}' from {preset_path}")
        return restored_data
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """
        List all available presets
        
        Returns:
            List of dictionaries with preset information
        """
        presets = []
        
        for filename in os.listdir(self.presets_dir):
            if filename.endswith('.json'):
                preset_path = os.path.join(self.presets_dir, filename)
                
                try:
                    with open(preset_path, 'r') as f:
                        preset_data = json.load(f)
                    
                    # Extract metadata
                    metadata = preset_data.get('metadata', {})
                    name = metadata.get('name', filename.replace('.json', '').replace('_', ' ').title())
                    created_at = metadata.get('created_at', 'Unknown')
                    
                    presets.append({
                        'name': name,
                        'filename': filename,
                        'created_at': created_at,
                        'path': preset_path
                    })
                except Exception as e:
                    logger.warning(f"Error loading preset {filename}: {e}")
        
        # Sort by creation date (newest first)
        presets.sort(key=lambda x: x['created_at'], reverse=True)
        
        return presets
    
    def delete_preset(self, preset_name: str) -> bool:
        """
        Delete a portfolio preset
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            True if deleted successfully
        """
        # Handle both preset name and filename
        if not preset_name.endswith('.json'):
            preset_filename = preset_name.lower().replace(' ', '_').replace('-', '_')
            preset_path = os.path.join(self.presets_dir, f"{preset_filename}.json")
        else:
            preset_path = os.path.join(self.presets_dir, preset_name)
        
        # Check if preset exists
        if not os.path.exists(preset_path):
            raise ValueError(f"Preset '{preset_name}' not found.")
        
        # Delete preset
        os.remove(preset_path)
        logger.info(f"Deleted preset '{preset_name}' from {preset_path}")
        
        return True
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return {
                '__type__': 'DataFrame',
                'data': data.to_dict(orient='records'),
                'index': data.index.tolist()
            }
        elif isinstance(data, pd.Series):
            return {
                '__type__': 'Series',
                'data': data.tolist(),
                'index': data.index.tolist(),
                'name': data.name
            }
        elif isinstance(data, (datetime, pd.Timestamp)):
            return data.isoformat()
        else:
            return data
    
    def _restore_from_serialized(self, data: Any) -> Any:
        """Restore original data types from serialized format"""
        if isinstance(data, dict):
            if '__type__' in data:
                if data['__type__'] == 'DataFrame':
                    df = pd.DataFrame(data['data'])
                    df.index = pd.Index(data['index'])
                    return df
                elif data['__type__'] == 'Series':
                    return pd.Series(
                        data=data['data'],
                        index=data['index'],
                        name=data['name']
                    )
            return {k: self._restore_from_serialized(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._restore_from_serialized(item) for item in data]
        else:
            return data