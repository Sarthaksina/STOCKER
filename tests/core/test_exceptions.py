"""Unit tests for the consolidated exceptions module.

These tests verify the correctness of the custom exceptions after
consolidation from src/exception/exceptions.py into src/core/exceptions.py.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.exceptions import (
    StockerBaseException,
    ConfigurationError,
    DataAccessError,
    DatabaseConnectionError,
    FeatureEngineeringError,
    ModelError,
    ValidationError
)


class TestExceptionsModule(unittest.TestCase):
    """Test suite for the consolidated exceptions module."""
    
    def test_exception_hierarchy(self):
        """Test the exception class hierarchy."""
        # Test that all exceptions inherit from StockerBaseException
        self.assertTrue(issubclass(ConfigurationError, StockerBaseException))
        self.assertTrue(issubclass(DataAccessError, StockerBaseException))
        self.assertTrue(issubclass(DatabaseConnectionError, StockerBaseException))
        self.assertTrue(issubclass(FeatureEngineeringError, StockerBaseException))
        self.assertTrue(issubclass(ModelError, StockerBaseException))
        self.assertTrue(issubclass(ValidationError, StockerBaseException))
        
        # Test that all exceptions inherit from Exception
        self.assertTrue(issubclass(StockerBaseException, Exception))
    
    def test_exception_messages(self):
        """Test exception messages."""
        # Test default messages
        base_exc = StockerBaseException()
        self.assertEqual(str(base_exc), "An error occurred in the STOCKER application.")
        
        # Test custom messages
        custom_message = "Custom error message"
        config_error = ConfigurationError(custom_message)
        self.assertEqual(str(config_error), custom_message)
        
        data_error = DataAccessError(custom_message)
        self.assertEqual(str(data_error), custom_message)
        
        db_error = DatabaseConnectionError(custom_message)
        self.assertEqual(str(db_error), custom_message)
        
        feature_error = FeatureEngineeringError(custom_message)
        self.assertEqual(str(feature_error), custom_message)
        
        model_error = ModelError(custom_message)
        self.assertEqual(str(model_error), custom_message)
        
        validation_error = ValidationError(custom_message)
        self.assertEqual(str(validation_error), custom_message)
    
    def test_exception_raising(self):
        """Test raising exceptions."""
        # Test raising base exception
        with self.assertRaises(StockerBaseException):
            raise StockerBaseException()
        
        # Test raising specific exceptions
        with self.assertRaises(ConfigurationError):
            raise ConfigurationError("Configuration error")
        
        with self.assertRaises(DataAccessError):
            raise DataAccessError("Data access error")
        
        with self.assertRaises(DatabaseConnectionError):
            raise DatabaseConnectionError("Database connection error")
        
        with self.assertRaises(FeatureEngineeringError):
            raise FeatureEngineeringError("Feature engineering error")
        
        with self.assertRaises(ModelError):
            raise ModelError("Model error")
        
        with self.assertRaises(ValidationError):
            raise ValidationError("Validation error")
    
    def test_exception_catching(self):
        """Test catching exceptions."""
        # Test catching specific exception
        try:
            raise ConfigurationError("Configuration error")
        except ConfigurationError as e:
            self.assertEqual(str(e), "Configuration error")
        
        # Test catching base exception
        try:
            raise FeatureEngineeringError("Feature engineering error")
        except StockerBaseException as e:
            self.assertEqual(str(e), "Feature engineering error")
        
        # Test exception attributes
        try:
            raise ModelError("Model error", model_name="LSTM", details={"epoch": 10})
        except ModelError as e:
            self.assertEqual(str(e), "Model error")
            self.assertEqual(e.model_name, "LSTM")
            self.assertEqual(e.details, {"epoch": 10})


if __name__ == '__main__':
    unittest.main()
