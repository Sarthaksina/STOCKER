"""Unit tests for the consolidated utils module.

These tests verify the correctness of the utility functions after
consolidation from src/utils.py into src/core/utils.py.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.utils import (
    create_directory,
    get_project_root,
    load_json,
    save_json,
    timer,
    memoize,
    validate_dataframe,
    format_currency,
    date_range
)


class TestUtilsModule(unittest.TestCase):
    """Test suite for the consolidated utils module."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=10),
            'price': np.linspace(100, 200, 10),
            'volume': np.random.randint(1000, 5000, 10)
        })
        
        # Create a temporary directory for file operations
        self.temp_dir = os.path.join(get_project_root(), 'tests', 'temp')
        create_directory(self.temp_dir)
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary files and directories
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
    
    def test_create_directory(self):
        """Test the create_directory function."""
        # Test creating a new directory
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        create_directory(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.isdir(test_dir))
        
        # Test creating a directory that already exists
        create_directory(test_dir)  # Should not raise an exception
        self.assertTrue(os.path.exists(test_dir))
        
        # Clean up
        os.rmdir(test_dir)
    
    def test_get_project_root(self):
        """Test the get_project_root function."""
        # Get the project root
        root = get_project_root()
        
        # Verify that it's a Path object
        self.assertIsInstance(root, Path)
        
        # Verify that the path exists
        self.assertTrue(os.path.exists(root))
        
        # Verify that it's a directory
        self.assertTrue(os.path.isdir(root))
    
    def test_load_save_json(self):
        """Test the load_json and save_json functions."""
        # Create a sample dictionary
        sample_dict = {
            'name': 'STOCKER',
            'version': '1.0.0',
            'features': ['technical_indicators', 'portfolio_optimization'],
            'settings': {
                'debug': True,
                'log_level': 'INFO'
            }
        }
        
        # Save the dictionary to a JSON file
        json_file = os.path.join(self.temp_dir, 'test.json')
        save_json(sample_dict, json_file)
        
        # Verify that the file was created
        self.assertTrue(os.path.exists(json_file))
        
        # Load the JSON file
        loaded_dict = load_json(json_file)
        
        # Verify that the loaded dictionary matches the original
        self.assertEqual(loaded_dict, sample_dict)
        
        # Test loading a non-existent file
        with self.assertRaises(FileNotFoundError):
            load_json('nonexistent.json')
    
    def test_timer_decorator(self):
        """Test the timer decorator."""
        # Define a function with the timer decorator
        @timer
        def slow_function():
            # Simulate a slow function
            import time
            time.sleep(0.1)
            return 42
        
        # Call the function and capture its output
        with patch('builtins.print') as mock_print:
            result = slow_function()
        
        # Verify that the function returned the correct result
        self.assertEqual(result, 42)
        
        # Verify that the timer printed a message
        mock_print.assert_called_once()
        # The print should include the function name and execution time
        self.assertIn('slow_function', mock_print.call_args[0][0])
        self.assertIn('seconds', mock_print.call_args[0][0])
    
    def test_memoize_decorator(self):
        """Test the memoize decorator."""
        # Define a function with the memoize decorator
        call_count = 0
        
        @memoize
        def fibonacci(n):
            nonlocal call_count
            call_count += 1
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        # Call the function multiple times
        fib10 = fibonacci(10)
        
        # Verify that the function returned the correct result
        self.assertEqual(fib10, 55)
        
        # Verify that the function was called the correct number of times
        # Without memoization, fibonacci(10) would call the function 177 times
        # With memoization, it should call the function only 11 times (0 through 10)
        self.assertEqual(call_count, 11)
        
        # Call the function again with the same argument
        fib10_again = fibonacci(10)
        
        # Verify that the function returned the same result
        self.assertEqual(fib10_again, 55)
        
        # Verify that the function was not called again
        self.assertEqual(call_count, 11)
    
    def test_validate_dataframe(self):
        """Test the validate_dataframe function."""
        # Test with a valid DataFrame
        required_columns = ['date', 'price']
        validate_dataframe(self.sample_df, required_columns)
        
        # Test with missing columns
        required_columns = ['date', 'price', 'missing_column']
        with self.assertRaises(ValueError):
            validate_dataframe(self.sample_df, required_columns)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            validate_dataframe(empty_df, required_columns)
    
    def test_format_currency(self):
        """Test the format_currency function."""
        # Test with various amounts
        self.assertEqual(format_currency(1234.56), '$1,234.56')
        self.assertEqual(format_currency(1000000), '$1,000,000.00')
        self.assertEqual(format_currency(0), '$0.00')
        self.assertEqual(format_currency(-1234.56), '-$1,234.56')
        
        # Test with different currencies
        self.assertEqual(format_currency(1234.56, currency='EUR'), '€1,234.56')
        self.assertEqual(format_currency(1234.56, currency='GBP'), '£1,234.56')
        self.assertEqual(format_currency(1234.56, currency='JPY'), '¥1,234.56')
        
        # Test with different decimal places
        self.assertEqual(format_currency(1234.56789, decimal_places=3), '$1,234.568')
        self.assertEqual(format_currency(1234.56789, decimal_places=0), '$1,235')
    
    def test_date_range(self):
        """Test the date_range function."""
        # Test with default parameters
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 10)
        dates = date_range(start_date, end_date)
        
        # Verify that the result is a list of dates
        self.assertIsInstance(dates, list)
        self.assertEqual(len(dates), 10)  # 10 days from Jan 1 to Jan 10 (inclusive)
        
        # Verify that the dates are correct
        self.assertEqual(dates[0], start_date)
        self.assertEqual(dates[-1], end_date)
        
        # Test with a different frequency
        dates = date_range(start_date, end_date, freq='2D')
        self.assertEqual(len(dates), 5)  # 5 dates: Jan 1, 3, 5, 7, 9
        
        # Test with a monthly frequency
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        dates = date_range(start_date, end_date, freq='M')
        self.assertEqual(len(dates), 12)  # 12 months in a year


if __name__ == '__main__':
    unittest.main()
