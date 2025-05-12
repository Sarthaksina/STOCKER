"""Unit tests for the consolidated indicators module.

These tests verify the correctness of various technical indicator calculations
from the consolidated indicators module, ensuring that all functionality from
the original indicators.py and technical_indicators.py files is preserved.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.indicators_final_new import TechnicalIndicators, calculate_macd, calculate_rsi, \
    calculate_bollinger_bands, calculate_stochastic_oscillator, calculate_atr, \
    calculate_adx, calculate_obv, calculate_volatility, calculate_trend_indicators, \
    calculate_technical_indicators, add_all_indicators
from src.core.exceptions import FeatureEngineeringError


class TestConsolidatedIndicators(unittest.TestCase):
    """Test suite for the consolidated indicators module."""
    
    def setUp(self):
        """Set up test data for technical indicators."""
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
        
        # Create TechnicalIndicators instance
        self.ti = TechnicalIndicators()
    
    def test_class_instance_methods(self):
        """Test that class instance methods work correctly."""
        # Test calculate_macd instance method
        macd_result = self.ti.calculate_macd(self.df)
        self.assertTrue('macd_line' in macd_result.columns)
        self.assertTrue('macd_signal' in macd_result.columns)
        self.assertTrue('macd_histogram' in macd_result.columns)
        
        # Test calculate_rsi instance method
        rsi_result = self.ti.calculate_rsi(self.df)
        self.assertTrue('rsi' in rsi_result.columns)
        self.assertTrue((rsi_result['rsi'].dropna() >= 0).all() and (rsi_result['rsi'].dropna() <= 100).all())
        
        # Test calculate_bollinger_bands instance method
        bb_result = self.ti.calculate_bollinger_bands(self.df)
        self.assertTrue('bb_upper' in bb_result.columns)
        self.assertTrue('bb_middle' in bb_result.columns)
        self.assertTrue('bb_lower' in bb_result.columns)
        self.assertTrue('bb_pct_b' in bb_result.columns)
        self.assertTrue('bb_bandwidth' in bb_result.columns)
        
        # Test calculate_stochastic_oscillator instance method
        stoch_result = self.ti.calculate_stochastic_oscillator(self.df)
        self.assertTrue('stoch_k' in stoch_result.columns)
        self.assertTrue('stoch_d' in stoch_result.columns)
        self.assertTrue((stoch_result['stoch_k'].dropna() >= 0).all() and 
                       (stoch_result['stoch_k'].dropna() <= 100).all())
        
        # Test calculate_average_directional_index instance method
        adx_result = self.ti.calculate_average_directional_index(self.df)
        self.assertTrue('adx' in adx_result.columns)
        self.assertTrue('plus_di' in adx_result.columns)
        self.assertTrue('minus_di' in adx_result.columns)
        self.assertTrue((adx_result['adx'].dropna() >= 0).all() and 
                       (adx_result['adx'].dropna() <= 100).all())
    
    def test_standalone_functions(self):
        """Test that standalone functions work correctly."""
        # Test calculate_macd function
        macd_result = calculate_macd(self.df)
        self.assertTrue('macd_line' in macd_result.columns)
        self.assertTrue('macd_signal' in macd_result.columns)
        self.assertTrue('macd_histogram' in macd_result.columns)
        
        # Test calculate_rsi function
        rsi_result = calculate_rsi(self.df)
        self.assertTrue('rsi' in rsi_result.columns)
        self.assertTrue((rsi_result['rsi'].dropna() >= 0).all() and (rsi_result['rsi'].dropna() <= 100).all())
        
        # Test calculate_bollinger_bands function
        bb_result = calculate_bollinger_bands(self.df)
        self.assertTrue('bb_upper' in bb_result.columns)
        self.assertTrue('bb_middle' in bb_result.columns)
        self.assertTrue('bb_lower' in bb_result.columns)
        self.assertTrue('bb_pct_b' in bb_result.columns)
        self.assertTrue('bb_bandwidth' in bb_result.columns)
        
        # Test calculate_stochastic_oscillator function
        stoch_result = calculate_stochastic_oscillator(self.df)
        self.assertTrue('stoch_k' in stoch_result.columns)
        self.assertTrue('stoch_d' in stoch_result.columns)
        self.assertTrue((stoch_result['stoch_k'].dropna() >= 0).all() and 
                       (stoch_result['stoch_k'].dropna() <= 100).all())
        
        # Test calculate_atr function
        atr_result = calculate_atr(self.df)
        self.assertTrue('atr' in atr_result.columns)
        self.assertTrue((atr_result['atr'].dropna() > 0).all())
        
        # Test calculate_adx function
        adx_result = calculate_adx(self.df)
        self.assertTrue('adx' in adx_result.columns)
        self.assertTrue('plus_di' in adx_result.columns)
        self.assertTrue('minus_di' in adx_result.columns)
        self.assertTrue((adx_result['adx'].dropna() >= 0).all() and 
                       (adx_result['adx'].dropna() <= 100).all())
        
        # Test calculate_obv function
        obv_result = calculate_obv(self.df)
        self.assertTrue('obv' in obv_result.columns)
    
    def test_utility_functions(self):
        """Test utility functions added in the consolidated module."""
        # Test calculate_volatility function
        vol_result = calculate_volatility(self.df)
        self.assertTrue('volatility_5' in vol_result.columns)
        self.assertTrue('volatility_10' in vol_result.columns)
        self.assertTrue('volatility_20' in vol_result.columns)
        self.assertTrue('volatility_30' in vol_result.columns)
        self.assertTrue('volatility_60' in vol_result.columns)
        
        # Test calculate_trend_indicators function
        trend_result = calculate_trend_indicators(self.df)
        self.assertTrue('sma_20' in trend_result.columns)
        self.assertTrue('sma_50' in trend_result.columns)
        self.assertTrue('ema_12' in trend_result.columns)
        self.assertTrue('ema_26' in trend_result.columns)
        self.assertTrue('price_to_sma_20' in trend_result.columns)
        self.assertTrue('price_to_ema_12' in trend_result.columns)
        self.assertTrue('sma_20_50_cross' in trend_result.columns)
        self.assertTrue('ema_12_26_cross' in trend_result.columns)
    
    def test_comprehensive_functions(self):
        """Test comprehensive functions that calculate multiple indicators."""
        # Test calculate_technical_indicators function
        tech_result = calculate_technical_indicators(self.df)
        self.assertTrue('macd_line' in tech_result.columns)
        self.assertTrue('rsi' in tech_result.columns)
        self.assertTrue('bb_upper' in tech_result.columns)
        self.assertTrue('stoch_k' in tech_result.columns)
        self.assertTrue('adx' in tech_result.columns)
        self.assertTrue('volatility_20' in tech_result.columns)
        
        # Test add_all_indicators function
        all_result = add_all_indicators(self.df)
        self.assertTrue('macd_line' in all_result.columns)
        self.assertTrue('rsi' in all_result.columns)
        self.assertTrue('bb_upper' in all_result.columns)
        self.assertTrue('stoch_k' in all_result.columns)
        self.assertTrue('adx' in all_result.columns)
        self.assertTrue('obv' in all_result.columns)
        self.assertTrue('tenkan_sen' in all_result.columns)
        self.assertTrue('keltner_upper' in all_result.columns)
    
    def test_error_handling(self):
        """Test error handling in the indicators module."""
        # Create DataFrame with missing columns
        df_missing = pd.DataFrame({'close': self.df['close']})
        
        # Test error handling in instance methods
        with self.assertRaises(FeatureEngineeringError):
            self.ti.calculate_stochastic_oscillator(df_missing)
        
        with self.assertRaises(FeatureEngineeringError):
            self.ti.calculate_average_directional_index(df_missing)
        
        # Test error handling in standalone functions
        with self.assertRaises(FeatureEngineeringError):
            calculate_stochastic_oscillator(df_missing)
        
        with self.assertRaises(FeatureEngineeringError):
            calculate_adx(df_missing)
        
        # Test error handling in comprehensive functions
        with self.assertRaises(FeatureEngineeringError):
            calculate_technical_indicators(df_missing)
        
        with self.assertRaises(FeatureEngineeringError):
            add_all_indicators(df_missing)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small DataFrame (just a few rows)
        small_df = self.df.iloc[:5].copy()
        small_result = calculate_macd(small_df)
        self.assertEqual(len(small_result), 5)
        
        # Test with DataFrame containing NaN values
        df_with_nan = self.df.copy()
        df_with_nan.loc[df_with_nan.index[10], 'close'] = np.nan
        nan_result = calculate_rsi(df_with_nan)
        self.assertTrue(np.isnan(nan_result.loc[df_with_nan.index[10], 'rsi']))
        
        # Test with custom parameters
        custom_bb = calculate_bollinger_bands(self.df, period=10, std_dev=3.0)
        self.assertTrue('bb_upper' in custom_bb.columns)
        
        # Test with different price column
        self.df['adjusted_close'] = self.df['close'] * 1.01  # Add adjusted close column
        adj_result = calculate_macd(self.df, price_column='adjusted_close')
        self.assertTrue('macd_line' in adj_result.columns)


if __name__ == '__main__':
    unittest.main()
