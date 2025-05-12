"""
Unit tests for the technical indicators module.

These tests verify the correctness of various technical indicator calculations
across different market conditions and edge cases.
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

from src.features.technical_indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    """Test suite for the TechnicalIndicators class."""
    
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
    
    def test_sma_calculation(self):
        """Test SMA calculation with various periods."""
        # Test SMA with period 20
        sma_20 = TechnicalIndicators.sma(self.df, period=20)
        self.assertEqual(len(sma_20), len(self.df))
        self.assertTrue(np.isnan(sma_20.iloc[0]))
        self.assertTrue(np.isnan(sma_20.iloc[18]))
        self.assertFalse(np.isnan(sma_20.iloc[19]))
        
        # Verify specific value (manual calculation)
        expected_sma = self.df['close'].iloc[0:20].mean()
        self.assertAlmostEqual(sma_20.iloc[19], expected_sma, places=6)
        
        # Test SMA with different period
        sma_5 = TechnicalIndicators.sma(self.df, period=5)
        self.assertEqual(len(sma_5), len(self.df))
        self.assertTrue(np.isnan(sma_5.iloc[3]))
        self.assertFalse(np.isnan(sma_5.iloc[4]))
        
        # Test invalid period
        with self.assertRaises(ValueError):
            TechnicalIndicators.sma(self.df, period=0)
    
    def test_ema_calculation(self):
        """Test EMA calculation with various periods."""
        # Test EMA with period 20
        ema_20 = TechnicalIndicators.ema(self.df, period=20)
        self.assertEqual(len(ema_20), len(self.df))
        
        # First values should be non-NaN (EMA starts with first value)
        self.assertFalse(np.isnan(ema_20.iloc[0]))
        
        # Test different periods
        ema_12 = TechnicalIndicators.ema(self.df, period=12)
        ema_26 = TechnicalIndicators.ema(self.df, period=26)
        
        # EMA should be more responsive than SMA
        sma_20 = TechnicalIndicators.sma(self.df, period=20)
        
        # During uptrend, EMA should be higher than SMA
        self.assertGreater(ema_20.iloc[-1], sma_20.iloc[-1])
        
        # Test invalid period
        with self.assertRaises(ValueError):
            TechnicalIndicators.ema(self.df, period=-1)
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        # Test MACD with default parameters
        macd_df = TechnicalIndicators.macd(self.df)
        self.assertEqual(len(macd_df), len(self.df))
        self.assertTrue('macd_line' in macd_df.columns)
        self.assertTrue('signal_line' in macd_df.columns)
        self.assertTrue('histogram' in macd_df.columns)
        
        # Verify that histogram is the difference between MACD and signal
        hist_check = macd_df['macd_line'] - macd_df['signal_line']
        np.testing.assert_almost_equal(macd_df['histogram'].iloc[-1], hist_check.iloc[-1], decimal=6)
        
        # Test with custom parameters
        custom_macd = TechnicalIndicators.macd(self.df, fast_period=8, slow_period=16, signal_period=5)
        self.assertEqual(len(custom_macd), len(self.df))
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            TechnicalIndicators.macd(self.df, fast_period=0, slow_period=26)
        with self.assertRaises(ValueError):
            TechnicalIndicators.macd(self.df, fast_period=26, slow_period=12)  # Fast > Slow
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        # Test Bollinger Bands with default parameters
        bb_df = TechnicalIndicators.bollinger_bands(self.df)
        self.assertEqual(len(bb_df), len(self.df))
        self.assertTrue('upper_band' in bb_df.columns)
        self.assertTrue('middle_band' in bb_df.columns)
        self.assertTrue('lower_band' in bb_df.columns)
        self.assertTrue('bandwidth' in bb_df.columns)
        self.assertTrue('percent_b' in bb_df.columns)
        
        # Middle band should be equal to SMA
        sma_20 = TechnicalIndicators.sma(self.df, period=20)
        np.testing.assert_almost_equal(bb_df['middle_band'].iloc[-1], sma_20.iloc[-1], decimal=6)
        
        # Upper band should be higher than middle band
        self.assertGreater(bb_df['upper_band'].iloc[-1], bb_df['middle_band'].iloc[-1])
        
        # Lower band should be lower than middle band
        self.assertLess(bb_df['lower_band'].iloc[-1], bb_df['middle_band'].iloc[-1])
        
        # Test custom parameters
        custom_bb = TechnicalIndicators.bollinger_bands(self.df, period=10, std_dev=3.0)
        self.assertEqual(len(custom_bb), len(self.df))
        
        # With higher std_dev, the bands should be wider
        standard_width = bb_df['upper_band'].iloc[-1] - bb_df['lower_band'].iloc[-1]
        custom_width = custom_bb['upper_band'].iloc[-1] - custom_bb['lower_band'].iloc[-1]
        self.assertGreater(custom_width, standard_width)
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            TechnicalIndicators.bollinger_bands(self.df, period=0)
        with self.assertRaises(ValueError):
            TechnicalIndicators.bollinger_bands(self.df, std_dev=-1.0)
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        # Test RSI with default parameters
        rsi = TechnicalIndicators.rsi(self.df)
        self.assertEqual(len(rsi), len(self.df))
        
        # RSI values should be between 0 and 100
        self.assertTrue((rsi.dropna() >= 0).all() and (rsi.dropna() <= 100).all())
        
        # In an uptrend, RSI should be high
        self.assertGreater(rsi.iloc[-1], 50)
        
        # Test with custom period
        custom_rsi = TechnicalIndicators.rsi(self.df, period=7)
        self.assertEqual(len(custom_rsi), len(self.df))
        
        # Test invalid period
        with self.assertRaises(ValueError):
            TechnicalIndicators.rsi(self.df, period=0)
    
    def test_stochastic_oscillator_calculation(self):
        """Test Stochastic Oscillator calculation."""
        # Test Stochastic with default parameters
        stoch_df = TechnicalIndicators.stochastic_oscillator(self.df)
        self.assertEqual(len(stoch_df), len(self.df))
        self.assertTrue('k' in stoch_df.columns)
        self.assertTrue('d' in stoch_df.columns)
        
        # Stochastic values should be between 0 and 100
        self.assertTrue((stoch_df['k'].dropna() >= 0).all() and (stoch_df['k'].dropna() <= 100).all())
        self.assertTrue((stoch_df['d'].dropna() >= 0).all() and (stoch_df['d'].dropna() <= 100).all())
        
        # Test with custom periods
        custom_stoch = TechnicalIndicators.stochastic_oscillator(self.df, k_period=5, d_period=2)
        self.assertEqual(len(custom_stoch), len(self.df))
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            TechnicalIndicators.stochastic_oscillator(self.df, k_period=0)
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        # Test ATR with default parameters
        atr = TechnicalIndicators.atr(self.df)
        self.assertEqual(len(atr), len(self.df))
        
        # ATR should be positive
        self.assertTrue((atr.dropna() > 0).all())
        
        # Test with custom period
        custom_atr = TechnicalIndicators.atr(self.df, period=7)
        self.assertEqual(len(custom_atr), len(self.df))
        
        # Test invalid period
        with self.assertRaises(ValueError):
            TechnicalIndicators.atr(self.df, period=0)
    
    def test_adx_calculation(self):
        """Test ADX calculation."""
        # Test ADX with default parameters
        adx_df = TechnicalIndicators.adx(self.df)
        self.assertEqual(len(adx_df), len(self.df))
        self.assertTrue('adx' in adx_df.columns)
        self.assertTrue('pos_di' in adx_df.columns)
        self.assertTrue('neg_di' in adx_df.columns)
        
        # ADX values should be between 0 and 100
        self.assertTrue((adx_df['adx'].dropna() >= 0).all() and (adx_df['adx'].dropna() <= 100).all())
        self.assertTrue((adx_df['pos_di'].dropna() >= 0).all() and (adx_df['pos_di'].dropna() <= 100).all())
        self.assertTrue((adx_df['neg_di'].dropna() >= 0).all() and (adx_df['neg_di'].dropna() <= 100).all())
        
        # In an uptrend, +DI should be greater than -DI
        self.assertGreater(adx_df['pos_di'].iloc[-1], adx_df['neg_di'].iloc[-1])
        
        # Test with custom period
        custom_adx = TechnicalIndicators.adx(self.df, period=7)
        self.assertEqual(len(custom_adx), len(self.df))
        
        # Test invalid period
        with self.assertRaises(ValueError):
            TechnicalIndicators.adx(self.df, period=0)
    
    def test_obv_calculation(self):
        """Test OBV calculation."""
        # Test OBV calculation
        obv = TechnicalIndicators.obv(self.df)
        self.assertEqual(len(obv), len(self.df))
        
        # First value should be 0
        self.assertEqual(obv.iloc[0], 0)
        
        # OBV should increase when price increases and volume is added
        # OBV should decrease when price decreases and volume is subtracted
        
        # Test with missing volume column
        df_no_volume = self.df[['open', 'high', 'low', 'close']]
        with self.assertRaises(ValueError):
            TechnicalIndicators.obv(df_no_volume)
    
    def test_add_all_indicators(self):
        """Test adding all indicators to a DataFrame."""
        # Test adding all indicators
        result_df = TechnicalIndicators.add_all_indicators(self.df)
        
        # Result should have more columns than original
        self.assertGreater(len(result_df.columns), len(self.df.columns))
        
        # Check that original columns are preserved
        for col in self.df.columns:
            self.assertTrue(col in result_df.columns)
        
        # Check that key indicators are added
        indicator_cols = [
            'sma_20', 'ema_12', 'macd_line', 'bb_upper', 'rsi_14', 
            'stoch_k', 'atr_14', 'adx', 'obv'
        ]
        for col in indicator_cols:
            self.assertTrue(col in result_df.columns)
        
        # Test with invalid DataFrame
        df_missing_cols = pd.DataFrame({'close': self.df['close']})
        with self.assertRaises(ValueError):
            TechnicalIndicators.add_all_indicators(df_missing_cols)

if __name__ == '__main__':
    unittest.main() 