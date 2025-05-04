"""
Portfolio Configuration Module for STOCKER Pro

This module provides configuration settings and preset management for portfolio analytics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PortfolioConfig:
    """Configuration for portfolio analytics"""
    risk_free_rate: float = 0.02
    target_volatility: float = 0.15
    rebalance_threshold: float = 0.05
    max_asset_weight: float = 0.30
    min_asset_weight: float = 0.05
    benchmark_ticker: str = "SPY"
    confidence_level: float = 0.95
    stress_scenarios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "market_crash": {"market": -0.30, "volatility": 2.0},
        "recession": {"market": -0.20, "interest_rates": 0.02, "credit_spread": 0.05},
        "inflation_shock": {"inflation": 0.05, "interest_rates": 0.03, "commodities": 0.15},
        "tech_bubble": {"tech": -0.40, "market": -0.15},
        "recovery": {"market": 0.15, "credit_spread": -0.02}
    })
    factor_model: str = "Fama-French-5"  # Options: "Fama-French-3", "Fama-French-5", "CAPM"
    annualization_factor: int = 252  # Trading days in a year
    target_return_for_sortino: float = 0.0  # Target return for Sortino ratio calculation

    def __init__(self, config_path: Optional[str] = None):
        # Presets directory
        self.presets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'presets')
        os.makedirs(self.presets_dir, exist_ok=True)
    
    def save_preset(self, name: str, preset_data: Optional[Dict[str, Any]] = None, 
                   description: Optional[str] = None, overwrite: bool = False) -> str:
        """
        Save current configuration as a preset
        
        Args:
            name: Name of the preset
            preset_data: Optional dictionary with preset data (if None, uses current config)
            description: Optional description
            overwrite: Whether to overwrite existing preset
            
        Returns:
            Path to the saved preset file
        """
        # Sanitize preset name for filename
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        filename = f"{safe_name}_{int(datetime.now().timestamp())}.json"
        preset_path = os.path.join(self.presets_dir, filename)
        
        # Check if preset exists with similar name
        if not overwrite:
            existing_presets = [f for f in os.listdir(self.presets_dir) 
                              if f.startswith(safe_name) and f.endswith('.json')]
            if existing_presets:
                raise ValueError(f"Preset with name '{name}' already exists. Use overwrite=True to replace it.")
        
        # Create preset data
        if preset_data is None:
            preset_data = self.to_dict()
            
        # Add metadata
        preset_data = {
            "name": name,
            "description": description or f"Preset created on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now().isoformat(),
            "config": preset_data,
            "metadata": {
                "version": "1.0"
            }
        }
        
        # Convert numpy arrays and pandas objects to serializable format
        serializable_data = self._make_serializable(preset_data)
        
        # Save to file
        with open(preset_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
        logger.info(f"Saved preset '{name}' to {preset_path}")
        return preset_path
    
    def load_preset(self, preset_name: str, update_config: bool = True) -> Dict[str, Any]:
        """
        Load configuration from a preset file
        
        Args:
            preset_name: Name or path of the preset file
            update_config: Whether to update the current configuration with loaded values
            
        Returns:
            Dictionary with preset data
        """
        # Handle both preset name and filename
        if not preset_name.endswith('.json'):
            preset_filename = preset_name.lower().replace(' ', '_').replace('-', '_')
            # Look for files that start with the preset name
            matching_files = [f for f in os.listdir(self.presets_dir) 
                            if f.startswith(preset_filename) and f.endswith('.json')]
            
            if not matching_files:
                raise ValueError(f"Preset '{preset_name}' not found.")
                
            # Use the most recent matching file
            preset_path = os.path.join(self.presets_dir, sorted(matching_files)[-1])
        else:
            preset_path = os.path.join(self.presets_dir, preset_name)
        
        # Check if preset exists
        if not os.path.exists(preset_path):
            raise ValueError(f"Preset file not found at {preset_path}")
            
        # Load preset
        with open(preset_path, 'r') as f:
            preset_data = json.load(f)
            
        # Extract configuration
        if 'config' in preset_data:
            config_data = preset_data['config']
        else:
            config_data = preset_data  # For backward compatibility
            
        # Restore from serialized format if needed
        config_data = self._restore_from_serialized(config_data)
            
        # Update configuration if requested
        if update_config:
            self.from_dict(config_data)
            logger.info(f"Updated configuration from preset: {preset_path}")
        
        logger.info(f"Loaded preset from {preset_path}")
        return preset_data
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """
        List all available presets
        
        Returns:
            List of preset metadata
        """
        presets = []
        
        for filename in os.listdir(self.presets_dir):
            if not filename.endswith('.json'):
                continue
                
            preset_path = os.path.join(self.presets_dir, filename)
            try:
                with open(preset_path, 'r') as f:
                    preset_data = json.load(f)
                    
                # Extract metadata
                if 'metadata' in preset_data:
                    metadata = preset_data['metadata']
                else:
                    metadata = {}
                    
                name = preset_data.get("name", filename.replace('.json', '').replace('_', ' ').title())
                created_at = preset_data.get("created_at", "Unknown")
                description = preset_data.get("description", "")
                
                presets.append({
                    "name": name,
                    "filename": filename,
                    "description": description,
                    "created_at": created_at,
                    "path": preset_path,
                    "metadata": metadata
                })
            except Exception as e:
                logger.warning(f"Error reading preset {filename}: {e}")
                
        # Sort by creation date (newest first)
        presets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return presets
    
    def delete_preset(self, preset_name: str) -> bool:
        """
        Delete a preset file
        
        Args:
            preset_name: Name or path of the preset file
            
        Returns:
            True if successful, False otherwise
        """
        # Handle both preset name and filename
        if not preset_name.endswith('.json'):
            preset_filename = preset_name.lower().replace(' ', '_').replace('-', '_')
            # Look for files that start with the preset name
            matching_files = [f for f in os.listdir(self.presets_dir) 
                            if f.startswith(preset_filename) and f.endswith('.json')]
            
            if not matching_files:
                raise ValueError(f"Preset '{preset_name}' not found.")
                
            # Delete all matching files
            success = True
            for filename in matching_files:
                preset_path = os.path.join(self.presets_dir, filename)
                try:
                    os.remove(preset_path)
                    logger.info(f"Deleted preset file: {preset_path}")
                except Exception as e:
                    logger.warning(f"Error deleting preset {filename}: {e}")
                    success = False
            return success
        else:
            preset_path = os.path.join(self.presets_dir, preset_name)
            
        # Check if preset exists
        if not os.path.exists(preset_path):
            raise ValueError(f"Preset file not found at {preset_path}")
            
        # Delete preset
        try:
            os.remove(preset_path)
            logger.info(f"Deleted preset: {preset_path}")
            return True
        except Exception as e:
            logger.warning(f"Error deleting preset: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "risk_free_rate": self.risk_free_rate,
            "target_volatility": self.target_volatility,
            "rebalance_threshold": self.rebalance_threshold,
            "max_asset_weight": self.max_asset_weight,
            "min_asset_weight": self.min_asset_weight,
            "benchmark_ticker": self.benchmark_ticker,
            "confidence_level": self.confidence_level,
            "stress_scenarios": self.stress_scenarios,
            "factor_model": self.factor_model,
            "annualization_factor": self.annualization_factor,
            "target_return_for_sortino": self.target_return_for_sortino
        }
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary
        
        Args:
            config_dict: Dictionary with configuration values
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info("Configuration updated from dictionary")
        
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
                    if 'index' in data:
                        df.index = pd.Index(data['index'])
                    return df
                elif data['__type__'] == 'Series':
                    return pd.Series(
                        data=data['data'],
                        index=data['index'],
                        name=data['name']
                    )
            else:
                return {k: self._restore_from_serialized(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._restore_from_serialized(item) for item in data]
        else:
            return data