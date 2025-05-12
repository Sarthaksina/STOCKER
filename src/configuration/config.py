"""STOCKER Pro Configuration Module

This module provides backward compatibility for the consolidated configuration module.
All functionality has been moved to src.core.config.

This module imports and re-exports all components from src.core.config to maintain
backward compatibility with existing code that imports from src.configuration.config.
"""

# Import and re-export everything from the new configuration module
from src.core.config import *
