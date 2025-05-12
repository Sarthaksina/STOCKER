"""Unit tests for the consolidated config module.

These tests verify the correctness of the configuration functionality
after consolidation from src/configuration/config.py into src/core/config.py.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import Config, load_config, get_config


class TestConfigModule(unittest.TestCase):
    """Test suite for the consolidated config module."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a sample config for testing
        self.sample_config = {
            'app': {
                'name': 'STOCKER',
                'version': '1.0.0',
                'debug': True
            },
            'database': {
                'host': 'localhost',
                'port': 27017,
                'name': 'stocker_db'
            },
            'api': {
                'key': 'test_api_key',
                'url': 'https://api.example.com'
            }
        }
    
    @patch('src.core.config.yaml.safe_load')
    @patch('src.core.config.open', create=True)
    def test_load_config(self, mock_open, mock_yaml_load):
        """Test loading configuration from a file."""
        # Mock the YAML loading to return our sample config
        mock_yaml_load.return_value = self.sample_config
        
        # Mock file opening
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Call the function
        config = load_config('config.yaml')
        
        # Verify the config was loaded correctly
        self.assertEqual(config['app']['name'], 'STOCKER')
        self.assertEqual(config['database']['port'], 27017)
        self.assertEqual(config['api']['key'], 'test_api_key')
        
        # Verify the mocks were called
        mock_open.assert_called_once_with('config.yaml', 'r')
        mock_yaml_load.assert_called_once_with(mock_file)
    
    def test_config_class(self):
        """Test the Config class functionality."""
        # Create Config instance with our sample config
        config = Config(self.sample_config)
        
        # Test attribute access
        self.assertEqual(config.app.name, 'STOCKER')
        self.assertEqual(config.database.port, 27017)
        self.assertEqual(config.api.key, 'test_api_key')
        
        # Test dictionary-like access
        self.assertEqual(config['app']['name'], 'STOCKER')
        self.assertEqual(config['database']['port'], 27017)
        self.assertEqual(config['api']['key'], 'test_api_key')
        
        # Test getting non-existent attributes
        with self.assertRaises(AttributeError):
            _ = config.nonexistent
        
        # Test getting non-existent keys
        with self.assertRaises(KeyError):
            _ = config['nonexistent']
    
    @patch('src.core.config.load_config')
    def test_get_config(self, mock_load_config):
        """Test the get_config function."""
        # Mock load_config to return our sample config
        mock_load_config.return_value = self.sample_config
        
        # Call get_config
        config = get_config()
        
        # Verify the config was loaded correctly
        self.assertEqual(config.app.name, 'STOCKER')
        self.assertEqual(config.database.port, 27017)
        self.assertEqual(config.api.key, 'test_api_key')
        
        # Verify load_config was called
        mock_load_config.assert_called_once()
    
    def test_config_to_dict(self):
        """Test converting Config object back to dictionary."""
        # Create Config instance with our sample config
        config = Config(self.sample_config)
        
        # Convert back to dict
        config_dict = config.to_dict()
        
        # Verify it matches the original
        self.assertEqual(config_dict, self.sample_config)


if __name__ == '__main__':
    unittest.main()
