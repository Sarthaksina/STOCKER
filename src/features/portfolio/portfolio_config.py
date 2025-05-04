"""
Portfolio Configuration Module for STOCKER Pro

This module provides configuration settings for portfolio analytics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

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

    def __init__(self, config_path: Optional[str] = None):
        # Presets directory
        self.presets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'presets')
        os.makedirs(self.presets_dir, exist_ok=True)
    
    def save_preset(self, name: str, description: Optional[str] = None) -> str:
        """
        Save current configuration as a preset
        
        Args:
            name: Name of the preset
            description: Optional description
            
        Returns:
            Path to the saved preset file
        """
        # Create preset data
        preset_data = {
            "name": name,
            "description": description or f"Preset created on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now().isoformat(),
            "config": self.to_dict(),
        }
        
        # Generate filename
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        filename = f"{safe_name}_{int(datetime.now().timestamp())}.json"
        preset_path = os.path.join(self.presets_dir, filename)
        
        # Save to file
        with open(preset_path, 'w') as f:
            json.dump(preset_data, f, indent=2)
            
        return preset_path
    
    def load_preset(self, preset_path: str) -> None:
        """
        Load configuration from a preset file
        
        Args:
            preset_path: Path to the preset file
        """
        with open(preset_path, 'r') as f:
            preset_data = json.load(f)
            
        # Update configuration
        self.from_dict(preset_data["config"])
    
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
                    
                presets.append({
                    "name": preset_data.get("name", "Unnamed"),
                    "description": preset_data.get("description", ""),
                    "created_at": preset_data.get("created_at", ""),
                    "path": preset_path
                })
            except Exception as e:
                logger.warning(f"Error reading preset {filename}: {e}")
                
        # Sort by creation date (newest first)
        presets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return presets
    
    def delete_preset(self, preset_path: str) -> bool:
        """
        Delete a preset file
        
        Args:
            preset_path: Path to the preset file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.remove(preset_path)
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
            "factor_model": self.factor_model
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