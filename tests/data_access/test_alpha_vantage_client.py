"""
Unit tests for the Alpha Vantage client.

These tests use mock responses to avoid making actual API calls.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, Mock, MagicMock
import pytest
from pathlib import Path

import pandas as pd
from datetime import datetime, timedelta

from src.configuration import Config
from src.data_access.alpha_vantage_client import AlphaVantageClient
from src.exception.exceptions import AlphaVantageAPIException
from src.utils.cache_utils import save_to_cache, load_from_cache, clear_cache

# Mock responses for testing
MOCK_OVERVIEW_RESPONSE = {
    "Symbol": "AAPL",
    "Name": "Apple Inc",
    "Description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
    "Exchange": "NASDAQ",
    "Country": "USA",
    "Sector": "Technology",
    "Industry": "Consumer Electronics",
    "MarketCapitalization": "2500000000",
    "PERatio": "30.5",
    "EPS": "5.25",
    "52WeekHigh": "175.0",
    "52WeekLow": "120.0"
}

MOCK_TIME_SERIES_RESPONSE = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2023-09-18",
        "4. Output Size": "Compact",
        "5. Time Zone": "US/Eastern"
    },
    "Time Series (Daily)": {
        "2023-09-18": {
            "1. open": "175.15",
            "2. high": "176.82",
            "3. low": "174.35",
            "4. close": "176.08",
            "5. volume": "58499387"
        },
        "2023-09-15": {
            "1. open": "176.48",
            "2. high": "177.20",
            "3. low": "175.30",
            "4. close": "175.96",
            "5. volume": "125381116"
        }
    }
}

MOCK_INCOME_STATEMENT_RESPONSE = {
    "symbol": "AAPL",
    "annualReports": [
        {
            "fiscalDateEnding": "2022-09-30",
            "totalRevenue": "394328000000",
            "costOfRevenue": "223546000000",
            "grossProfit": "170782000000",
            "operatingIncome": "119437000000",
            "netIncome": "99803000000",
            "reportedEPS": "6.11"
        }
    ],
    "quarterlyReports": [
        {
            "fiscalDateEnding": "2023-06-30",
            "totalRevenue": "81797000000",
            "costOfRevenue": "45981000000",
            "grossProfit": "35816000000",
            "operatingIncome": "23053000000",
            "netIncome": "19881000000",
            "reportedEPS": "1.26"
        }
    ]
}

class TestAlphaVantageClient(unittest.TestCase):
    """Test suite for the Alpha Vantage client."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.get_alpha_vantage_api_key.return_value = "TEST_API_KEY"
        
        # Create a temp directory for cache
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        
        # Create client with the mock config
        self.client = AlphaVantageClient(self.mock_config, cache_dir=str(self.cache_dir))
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('requests.get')
    def test_get_company_overview(self, mock_get):
        """Test getting company overview data."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = MOCK_OVERVIEW_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the method
        result = self.client.get_company_overview("AAPL")
        
        # Assert expected results
        self.assertEqual(result["Symbol"], "AAPL")
        self.assertEqual(result["Name"], "Apple Inc")
        self.assertEqual(result["Sector"], "Technology")
        
        # Check that the request was called with correct parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][1]
        self.assertEqual(call_args["function"], "OVERVIEW")
        self.assertEqual(call_args["symbol"], "AAPL")
        self.assertEqual(call_args["apikey"], "TEST_API_KEY")
    
    @patch('requests.get')
    def test_get_time_series_daily(self, mock_get):
        """Test getting daily time series data."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = MOCK_TIME_SERIES_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the method
        result = self.client.get_time_series_daily("AAPL")
        
        # Assert expected results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Two days of data
        self.assertTrue(all(col in result.columns for col in ["open", "high", "low", "close", "volume"]))
        
        # Check specific values
        self.assertEqual(result.iloc[1]["close"], 176.08)
        self.assertEqual(result.iloc[0]["close"], 175.96)
    
    @patch('requests.get')
    def test_api_error_handling(self, mock_get):
        """Test error handling when API returns an error message."""
        # Setup mock response with error
        mock_response = Mock()
        mock_response.json.return_value = {"Error Message": "Invalid API call"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the method and assert exception is raised
        with self.assertRaises(AlphaVantageAPIException):
            self.client.get_company_overview("INVALID")
    
    @patch('requests.get')
    def test_rate_limit_handling(self, mock_get):
        """Test handling of rate limit messages."""
        # Setup mock response with rate limit note
        mock_response = Mock()
        mock_response.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day."
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Override sleep to avoid waiting in tests
        with patch('time.sleep'):
            # Should raise exception after retries
            with self.assertRaises(AlphaVantageAPIException):
                self.client.get_company_overview("AAPL")
    
    def test_caching(self):
        """Test caching mechanism."""
        # Save mock data to cache
        params = {
            "function": "OVERVIEW",
            "symbol": "AAPL"
        }
        cache_key = json.dumps(params, sort_keys=True)
        save_to_cache(self.cache_dir, cache_key, MOCK_OVERVIEW_RESPONSE)
        
        # Mock _make_request to ensure it's not called
        with patch.object(self.client, '_make_request') as mock_make_request:
            # Call the method, should use cached data
            result = self.client.get_company_overview("AAPL")
            
            # Check that _make_request was not called
            mock_make_request.assert_not_called()
            
            # Check result is the mock data
            self.assertEqual(result, MOCK_OVERVIEW_RESPONSE)
    
    @patch('requests.get')
    def test_get_income_statement(self, mock_get):
        """Test getting income statement data."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = MOCK_INCOME_STATEMENT_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the method
        result = self.client.get_income_statement("AAPL")
        
        # Assert expected results
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(len(result["annualReports"]), 1)
        self.assertEqual(len(result["quarterlyReports"]), 1)
        
        # Check specific values from annual report
        annual_report = result["annualReports"][0]
        self.assertEqual(annual_report["fiscalDateEnding"], "2022-09-30")
        self.assertEqual(annual_report["totalRevenue"], "394328000000")
        self.assertEqual(annual_report["reportedEPS"], "6.11")


@pytest.fixture
def alpha_vantage_client():
    """Pytest fixture for Alpha Vantage client."""
    mock_config = Mock(spec=Config)
    mock_config.get_alpha_vantage_api_key.return_value = "TEST_API_KEY"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        yield AlphaVantageClient(mock_config, cache_dir=temp_dir)

def test_enforce_rate_limit(alpha_vantage_client):
    """Test rate limiting functionality."""
    with patch('time.sleep') as mock_sleep:
        # Call twice in quick succession
        alpha_vantage_client._enforce_rate_limit()
        alpha_vantage_client._enforce_rate_limit()
        
        # Should have tried to sleep on the second call
        mock_sleep.assert_called_once()

def test_cache_expiry():
    """Test cache expiry functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save data with a timestamp from yesterday
        yesterday = datetime.now() - timedelta(days=1)
        cache_data = {
            'timestamp': yesterday.isoformat(),
            'data': {"test": "data"}
        }
        
        cache_path = Path(temp_dir) / "test.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
        
        # Try to load with 12-hour expiry (should be expired)
        result = load_from_cache(temp_dir, "test", expiry_hours=12)
        assert result is None
        
        # Try to load with 48-hour expiry (should not be expired)
        result = load_from_cache(temp_dir, "test", expiry_hours=48)
        assert result == {"test": "data"}


"""
Unit tests for the Alpha Vantage client.
"""
import pytest
import pandas as pd
import os
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.data_access.alpha_vantage_client import AlphaVantageClient
from src.data_access.exceptions import DataAccessError, DataNotFoundError, DataSourceRateLimitError

# Sample test data
SAMPLE_DAILY_RESPONSE = {
    "Meta Data": {
        "1. Information": "Daily Adjusted Prices",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2023-01-10",
        "4. Output Size": "Compact",
        "5. Time Zone": "US/Eastern"
    },
    "Time Series (Daily)": {
        "2023-01-10": {
            "1. open": "130.465",
            "2. high": "131.2636",
            "3. low": "128.1201",
            "4. close": "130.0729",
            "5. adjusted close": "130.0729",
            "6. volume": "69858304",
            "7. dividend amount": "0.0000",
            "8. split coefficient": "1.0"
        },
        "2023-01-09": {
            "1. open": "129.2201",
            "2. high": "130.9001",
            "3. low": "128.8101",
            "4. close": "130.1501",
            "5. adjusted close": "130.1501",
            "6. volume": "70790803",
            "7. dividend amount": "0.0000",
            "8. split coefficient": "1.0"
        }
    }
}

SAMPLE_COMPANY_INFO = {
    "Symbol": "AAPL",
    "AssetType": "Common Stock",
    "Name": "Apple Inc",
    "Description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Country": "USA",
    "Sector": "Technology",
    "Industry": "Consumer Electronics",
    "MarketCapitalization": "2100000000000"
}


class TestAlphaVantageClient:
    """Test cases for AlphaVantageClient."""
    
    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client with a temporary cache directory."""
        cache_dir = str(tmp_path / "cache")
        return AlphaVantageClient("test_api_key", cache_dir)
    
    @patch('requests.get')
    def test_get_historical_data_success(self, mock_get, client):
        """Test successful retrieval of historical data."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_DAILY_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method
        result = client.get_historical_data("AAPL", interval="daily")
        
        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "open" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs['params']['function'] == 'TIME_SERIES_DAILY_ADJUSTED'
        assert kwargs['params']['symbol'] == 'AAPL'
    
    @patch('requests.get')
    def test_get_historical_data_empty_response(self, mock_get, client):
        """Test handling of empty response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"Meta Data": {}}  # No time series data
        mock_get.return_value = mock_response
        
        # Verify exception is raised
        with pytest.raises(DataNotFoundError):
            client.get_historical_data("AAPL", interval="daily")
    
    @patch('requests.get')
    def test_get_historical_data_rate_limit(self, mock_get, client):
        """Test handling of rate limit error."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day."
        }
        mock_get.return_value = mock_response
        
        # Verify exception is raised
        with pytest.raises(DataSourceRateLimitError):
            client.get_historical_data("AAPL", interval="daily")
    
    @patch('requests.get')
    def test_get_company_info_success(self, mock_get, client):
        """Test successful retrieval of company info."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_COMPANY_INFO
        mock_get.return_value = mock_response
        
        # Call the method
        result = client.get_company_info("AAPL")
        
        # Verify the result
        assert isinstance(result, dict)
        assert result["Symbol"] == "AAPL"
        assert result["Name"] == "Apple Inc"
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs['params']['function'] == 'OVERVIEW'
        assert kwargs['params']['symbol'] == 'AAPL'
    
    @patch('requests.get')
    def test_get_company_info_not_found(self, mock_get, client):
        """Test handling of company info not found."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # Empty response
        mock_get.return_value = mock_response
        
        # Verify exception is raised
        with pytest.raises(DataNotFoundError):
            client.get_company_info("INVALID")
    
    def test_caching(self, client, tmp_path):
        """Test that responses are properly cached and retrieved."""
        # Mock the _make_request method to avoid actual API calls
        with patch.object(client, '_make_request') as mock_make_request:
            mock_make_request.return_value = SAMPLE_DAILY_RESPONSE
            
            # First call should use the mock
            result1 = client.get_historical_data("AAPL", interval="daily")
            assert mock_make_request.call_count == 1
            
            # Second call should use cache
            result2 = client.get_historical_data("AAPL", interval="daily")
            # The mock should not be called again if caching works
            assert mock_make_request.call_count == 1
            
            # Results should be identical
            pd.testing.assert_frame_equal(result1, result2)