"""Unit tests for the consolidated engineering module.

These tests verify the correctness of the feature engineering functionality after
consolidation from feature_engineering.py into engineering.py.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.engineering import FeatureEngineer, remove_outliers, normalize_data
from src.core.exceptions import FeatureEngineeringError


class TestEngineeringConsolidated(unittest.TestCase):
    """Test suite for the consolidated engineering module."""
    
    def setUp(self):
        """Set up test data for feature engineering tests."""
        # Create a DataFrame with sample price data
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        
        # Create a trending market
        trending_close = np.linspace(100, 200, 100)  # Linear uptrend
        
        # Add some noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 3, 100)
        close_prices = trending_close + noise
        
        # Create high/low prices
        high_prices = close_prices + np.random.uniform(0, 5, 100)
        low_prices = close_prices - np.random.uniform(0, 5, 100)
        open_prices = close_prices - np.random.uniform(-3, 3, 100)
        
        # Create volume data (with occasional spikes)
        volume = np.random.uniform(1000, 5000, 100)
        volume[25] = 15000  # Add volume spike
        volume[75] = 18000  # Add another volume spike
        
        # Create the DataFrame
        self.df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        # Create a FeatureEngineer instance
        self.fe = FeatureEngineer(self.df)
    
    def test_create_lag_features(self):
        """Test creating lag features."""
        # Create lag features for close prices
        result = self.fe.create_lag_features(['close'], [1, 2, 3])
        
        # Verify that the lag features were created
        self.assertTrue('close_lag_1' in result.columns)
        self.assertTrue('close_lag_2' in result.columns)
        self.assertTrue('close_lag_3' in result.columns)
        
        # Verify that the lag values are correct
        for i in range(3, len(result)):
            self.assertEqual(result['close_lag_1'].iloc[i], result['close'].iloc[i-1])
            self.assertEqual(result['close_lag_2'].iloc[i], result['close'].iloc[i-2])
            self.assertEqual(result['close_lag_3'].iloc[i], result['close'].iloc[i-3])
        
        # Verify that the first rows have NaN values for lags
        self.assertTrue(np.isnan(result['close_lag_1'].iloc[0]))
        self.assertTrue(np.isnan(result['close_lag_2'].iloc[0]))
        self.assertTrue(np.isnan(result['close_lag_2'].iloc[1]))
        self.assertTrue(np.isnan(result['close_lag_3'].iloc[0]))
        self.assertTrue(np.isnan(result['close_lag_3'].iloc[1]))
        self.assertTrue(np.isnan(result['close_lag_3'].iloc[2]))
    
    def test_create_rolling_features(self):
        """Test creating rolling features."""
        # Create rolling features for close prices
        result = self.fe.create_rolling_features(['close'], [5, 10], ['mean', 'std'])
        
        # Verify that the rolling features were created
        self.assertTrue('close_rolling_mean_5' in result.columns)
        self.assertTrue('close_rolling_std_5' in result.columns)
        self.assertTrue('close_rolling_mean_10' in result.columns)
        self.assertTrue('close_rolling_std_10' in result.columns)
        
        # Verify that the rolling values are correct
        for i in range(10, len(result)):
            self.assertAlmostEqual(
                result['close_rolling_mean_5'].iloc[i],
                result['close'].iloc[i-4:i+1].mean()
            )
            self.assertAlmostEqual(
                result['close_rolling_std_5'].iloc[i],
                result['close'].iloc[i-4:i+1].std()
            )
            self.assertAlmostEqual(
                result['close_rolling_mean_10'].iloc[i],
                result['close'].iloc[i-9:i+1].mean()
            )
            self.assertAlmostEqual(
                result['close_rolling_std_10'].iloc[i],
                result['close'].iloc[i-9:i+1].std()
            )
    
    def test_create_return_features(self):
        """Test creating return features."""
        # Create return features for close prices
        result = self.fe.create_return_features(['close'], [1, 5, 10])
        
        # Verify that the return features were created
        self.assertTrue('close_return_1' in result.columns)
        self.assertTrue('close_return_5' in result.columns)
        self.assertTrue('close_return_10' in result.columns)
        
        # Verify that the return values are correct
        for i in range(10, len(result)):
            self.assertAlmostEqual(
                result['close_return_1'].iloc[i],
                (result['close'].iloc[i] / result['close'].iloc[i-1]) - 1
            )
            self.assertAlmostEqual(
                result['close_return_5'].iloc[i],
                (result['close'].iloc[i] / result['close'].iloc[i-5]) - 1
            )
            self.assertAlmostEqual(
                result['close_return_10'].iloc[i],
                (result['close'].iloc[i] / result['close'].iloc[i-10]) - 1
            )
    
    def test_create_date_features(self):
        """Test creating date features."""
        # Create date features
        result = self.fe.create_date_features()
        
        # Verify that the date features were created
        self.assertTrue('day_of_week' in result.columns)
        self.assertTrue('day_of_month' in result.columns)
        self.assertTrue('month' in result.columns)
        self.assertTrue('quarter' in result.columns)
        self.assertTrue('year' in result.columns)
        self.assertTrue('is_month_start' in result.columns)
        self.assertTrue('is_month_end' in result.columns)
        
        # Verify that the date values are correct
        for i in range(len(result)):
            date = result.index[i]
            self.assertEqual(result['day_of_week'].iloc[i], date.dayofweek)
            self.assertEqual(result['day_of_month'].iloc[i], date.day)
            self.assertEqual(result['month'].iloc[i], date.month)
            self.assertEqual(result['quarter'].iloc[i], date.quarter)
            self.assertEqual(result['year'].iloc[i], date.year)
            self.assertEqual(result['is_month_start'].iloc[i], date.is_month_start)
            self.assertEqual(result['is_month_end'].iloc[i], date.is_month_end)
    
    def test_remove_outliers(self):
        """Test removing outliers."""
        # Create a DataFrame with outliers
        df_with_outliers = self.df.copy()
        df_with_outliers.loc[df_with_outliers.index[10], 'close'] = 500  # Add an outlier
        df_with_outliers.loc[df_with_outliers.index[20], 'close'] = 50   # Add another outlier
        
        # Remove outliers using the Z-score method
        result = remove_outliers(df_with_outliers, method='zscore', threshold=3.0)
        
        # Verify that the outliers were removed
        self.assertNotEqual(result.loc[result.index[10], 'close'], 500)
        self.assertNotEqual(result.loc[result.index[20], 'close'], 50)
        
        # Verify that the non-outlier values were preserved
        for i in range(len(result)):
            if i not in [10, 20]:
                self.assertEqual(result['close'].iloc[i], df_with_outliers['close'].iloc[i])
    
    def test_normalize_data(self):
        """Test normalizing data."""
        # Normalize the close prices
        result = normalize_data(self.df, columns=['close'])
        
        # Verify that the normalized values are between 0 and 1
        self.assertTrue((result['close'] >= 0).all() and (result['close'] <= 1).all())
        
        # Verify that the minimum value is 0 and the maximum value is 1
        self.assertAlmostEqual(result['close'].min(), 0)
        self.assertAlmostEqual(result['close'].max(), 1)
        
        # Verify that the other columns were preserved
        for col in ['open', 'high', 'low', 'volume']:
            self.assertTrue(np.array_equal(result[col].values, self.df[col].values))
    
    def test_create_all_features(self):
        """Test creating all features."""
        # Create all features
        result = self.fe.create_all_features()
        
        # Verify that all feature types were created
        # Lag features
        self.assertTrue('close_lag_1' in result.columns)
        self.assertTrue('close_lag_5' in result.columns)
        
        # Rolling features
        self.assertTrue('close_rolling_mean_5' in result.columns)
        self.assertTrue('close_rolling_std_5' in result.columns)
        
        # Return features
        self.assertTrue('close_return_1' in result.columns)
        self.assertTrue('close_return_5' in result.columns)
        
        # Date features
        self.assertTrue('day_of_week' in result.columns)
        self.assertTrue('month' in result.columns)
        
        # Verify that the original columns were preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.assertTrue(col in result.columns)
    
    def test_error_handling(self):
        """Test error handling in the feature engineering module."""
        # Test with invalid column
        with self.assertRaises(FeatureEngineeringError):
            self.fe.create_lag_features(['nonexistent_column'], [1, 2])
        
        # Test with invalid window size
        with self.assertRaises(ValueError):
            self.fe.create_rolling_features(['close'], [0], ['mean'])
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        empty_fe = FeatureEngineer(empty_df)
        with self.assertRaises(FeatureEngineeringError):
            empty_fe.create_all_features()


if __name__ == '__main__':
    unittest.main()
