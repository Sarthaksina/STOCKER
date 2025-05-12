import pytest
import pandas as pd
import numpy as np
from stocker.src.ml.evaluation import (
    calculate_basic_metrics, directional_accuracy, weighted_directional_accuracy,
    calculate_returns, sharpe_ratio, sortino_ratio, maximum_drawdown,
    calmar_ratio, trading_strategy_returns
)

@pytest.fixture
def sample_true_values():
    """Create sample true values for testing"""
    np.random.seed(42)
    return np.cumsum(np.random.normal(0.001, 0.01, 100))

@pytest.fixture
def sample_pred_values():
    """Create sample predicted values for testing"""
    np.random.seed(43)
    return np.cumsum(np.random.normal(0.001, 0.01, 100))

@pytest.fixture
def sample_returns():
    """Create sample returns for testing"""
    np.random.seed(42)
    return np.random.normal(0.001, 0.01, 100)

class TestBasicMetrics:
    """Test basic evaluation metrics"""
    
    def test_calculate_basic_metrics(self, sample_true_values, sample_pred_values):
        """Test basic metrics calculation"""
        metrics = calculate_basic_metrics(sample_true_values, sample_pred_values)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0
        
        # RMSE should be square root of MSE
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))

class TestDirectionalAccuracy:
    """Test directional accuracy metrics"""
    
    def test_directional_accuracy(self, sample_true_values, sample_pred_values):
        """Test directional accuracy calculation"""
        accuracy = directional_accuracy(sample_true_values, sample_pred_values)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1.0
    
    def test_weighted_directional_accuracy(self, sample_true_values, sample_pred_values):
        """Test weighted directional accuracy calculation"""
        weighted_accuracy = weighted_directional_accuracy(sample_true_values, sample_pred_values)
        
        assert isinstance(weighted_accuracy, float)
        assert 0 <= weighted_accuracy <= 1.0

class TestReturnCalculation:
    """Test return calculation functions"""
    
    def test_calculate_returns(self, sample_true_values):
        """Test returns calculation"""
        # Test simple returns
        simple_returns = calculate_returns(sample_true_values, is_log_return=False)
        assert len(simple_returns) == len(sample_true_values) - 1
        
        # Test log returns
        log_returns = calculate_returns(sample_true_values, is_log_return=True)
        assert len(log_returns) == len(sample_true_values) - 1

class TestRiskAdjustedMetrics:
    """Test risk-adjusted return metrics"""
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation"""
        sharpe = sharpe_ratio(sample_returns, risk_free_rate=0.0, annualization=252)
        
        assert isinstance(sharpe, float)
    
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation"""
        sortino = sortino_ratio(sample_returns, risk_free_rate=0.0, annualization=252)
        
        assert isinstance(sortino, float)
    
    def test_maximum_drawdown(self, sample_true_values):
        """Test maximum drawdown calculation"""
        max_dd = maximum_drawdown(sample_true_values)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1.0  # Drawdown should be between 0 and 1
    
    def test_calmar_ratio(self, sample_returns, sample_true_values):
        """Test Calmar ratio calculation"""
        calmar = calmar_ratio(sample_returns, sample_true_values, annualization=252)
        
        assert isinstance(calmar, float)

class TestTradingStrategy:
    """Test trading strategy evaluation"""
    
    def test_trading_strategy_returns(self, sample_true_values, sample_pred_values):
        """Test trading strategy returns calculation"""
        strategy_returns = trading_strategy_returns(
            y_true=sample_true_values,
            y_pred=sample_pred_values,
            transaction_cost=0.001
        )
        
        assert isinstance(strategy_returns, np.ndarray)
        assert len(strategy_returns) == len(sample_true_values) - 1