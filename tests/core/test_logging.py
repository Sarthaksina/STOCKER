"""Unit tests for the consolidated logging module.

These tests verify the correctness of the logging functionality after
consolidation from src/logger/logger.py into src/core/logging.py.
"""

import os
import sys
import unittest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.logging import get_logger, setup_logging


class TestLoggingModule(unittest.TestCase):
    """Test suite for the consolidated logging module."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Reset the logging configuration before each test
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)
    
    @patch('src.core.logging.logging.FileHandler')
    @patch('src.core.logging.logging.StreamHandler')
    def test_setup_logging(self, mock_stream_handler, mock_file_handler):
        """Test the setup_logging function."""
        # Mock handlers
        mock_stream_handler_instance = MagicMock()
        mock_file_handler_instance = MagicMock()
        mock_stream_handler.return_value = mock_stream_handler_instance
        mock_file_handler.return_value = mock_file_handler_instance
        
        # Call setup_logging
        setup_logging(log_level='INFO', log_file='test.log')
        
        # Verify that the handlers were created and added
        mock_stream_handler.assert_called_once()
        mock_file_handler.assert_called_once_with('test.log')
        
        # Verify that formatters were set
        mock_stream_handler_instance.setFormatter.assert_called_once()
        mock_file_handler_instance.setFormatter.assert_called_once()
        
        # Verify that the root logger was configured
        self.assertEqual(logging.root.level, logging.INFO)
    
    def test_get_logger(self):
        """Test the get_logger function."""
        # Get a logger
        logger = get_logger('test_module')
        
        # Verify that it's a Logger instance
        self.assertIsInstance(logger, logging.Logger)
        
        # Verify the logger name
        self.assertEqual(logger.name, 'test_module')
        
        # Get the same logger again and verify it's the same instance
        logger2 = get_logger('test_module')
        self.assertIs(logger, logger2)
        
        # Get a different logger and verify it's a different instance
        logger3 = get_logger('other_module')
        self.assertIsNot(logger, logger3)
    
    @patch('src.core.logging.setup_logging')
    def test_get_logger_with_setup(self, mock_setup_logging):
        """Test get_logger with setup=True."""
        # Get a logger with setup=True
        logger = get_logger('test_module', setup=True)
        
        # Verify that setup_logging was called
        mock_setup_logging.assert_called_once()
        
        # Verify that it's a Logger instance
        self.assertIsInstance(logger, logging.Logger)
    
    def test_logger_levels(self):
        """Test logger levels."""
        # Set up logging with DEBUG level
        setup_logging(log_level='DEBUG')
        
        # Get a logger
        logger = get_logger('test_module')
        
        # Verify the logger level
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Test with different levels
        setup_logging(log_level='INFO')
        logger = get_logger('info_module')
        self.assertEqual(logger.level, logging.INFO)
        
        setup_logging(log_level='WARNING')
        logger = get_logger('warning_module')
        self.assertEqual(logger.level, logging.WARNING)
        
        setup_logging(log_level='ERROR')
        logger = get_logger('error_module')
        self.assertEqual(logger.level, logging.ERROR)
        
        setup_logging(log_level='CRITICAL')
        logger = get_logger('critical_module')
        self.assertEqual(logger.level, logging.CRITICAL)
    
    def test_invalid_log_level(self):
        """Test with invalid log level."""
        # Set up logging with an invalid level
        # It should default to INFO
        setup_logging(log_level='INVALID')
        
        # Get a logger
        logger = get_logger('test_module')
        
        # Verify the logger level
        self.assertEqual(logger.level, logging.INFO)


if __name__ == '__main__':
    unittest.main()
