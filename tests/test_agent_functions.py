import pytest
import numpy as np
import pandas as pd

from src.features.analytics_agents import (
    performance_analysis,
    sharpe_ratio,
    valuation_metrics,
    alpha_beta,
    attribution_analysis,
    exposure_analysis,
    momentum_analysis,
    var_historical,
    max_drawdown,
    rolling_sharpe
)
# Import from the portfolio facade module which maintains backward compatibility
from src.features.portfolio import (
    recommend_portfolio,
    peer_compare,
    top_n_recommender,
    chart_performance,
    get_portfolio_manager
)


def test_performance_analysis():
    res = performance_analysis([100, 110, 121])
    assert pytest.approx(res['total_return'], rel=1e-3) == 0.21
    assert 'annualized_return' in res
    assert 'volatility' in res


def test_sharpe_ratio():
    returns = [0.01, 0.02, -0.005, 0.0]
    sr = sharpe_ratio(returns, risk_free_rate=0.0)
    assert isinstance(sr, float)


def test_valuation_metrics():
    data = {'pe': 10, 'pb': 2, 'div_yield': 0.05, 'market_cap': 1000}
    res = valuation_metrics(data)
    assert res == data


def test_alpha_beta():
    prices = [100, 110, 121]
    benchmark = [200, 220, 242]
    ab = alpha_beta(prices, benchmark)
    assert pytest.approx(ab['beta'], rel=1e-3) == 1.0
    assert pytest.approx(ab['alpha'], abs=1e-3) == 0.0


def test_attribution_analysis():
    portfolio = {'A': 0.5, 'B': 0.5}
    returns = {'A': 0.1, 'B': 0.2}
    at = attribution_analysis(portfolio, returns)
    assert pytest.approx(at['A'], rel=1e-3) == pytest.approx(0.3333, rel=1e-3)
    assert pytest.approx(at['B'], rel=1e-3) == pytest.approx(0.6667, rel=1e-3)


def test_exposure_analysis():
    portfolio = {'A': 0.7, 'B': 0.3}
    sector_map = {'A': 'Tech', 'B': 'Health'}
    asset_map = {'A': 'Equity', 'B': 'Equity'}
    exp = exposure_analysis(portfolio, sector_map, asset_map)
    assert exp['sector_exposure']['Tech'] == 0.7
    assert exp['sector_exposure']['Health'] == 0.3


def test_momentum_analysis():
    prices = [1, 2, 3, 4, 5, 6, 7]
    m = momentum_analysis(prices)
    expected = 7 / np.mean(prices[-6:])
    assert pytest.approx(m, rel=1e-3) == expected


def test_var_historical():
    returns = [0.1, -0.1, 0.2]
    var = var_historical(returns, confidence=0.05)
    assert var['var'] == pytest.approx(0.1, rel=1e-3)


def test_max_drawdown():
    prices = [100, 80, 120, 60]
    dd = max_drawdown(prices)
    assert dd['max_drawdown'] == pytest.approx(-0.5, rel=1e-3)


def test_rolling_sharpe():
    returns = [0.1, 0.2, 0.15]
    rs = rolling_sharpe(returns, window=2, risk_free_rate=0.0)
    assert len(rs['rolling_sharpe']) == len(returns)
    assert rs['rolling_sharpe'][0] is None
    assert isinstance(rs['rolling_sharpe'][1], float)


def test_top_n_recommender():
    df = pd.DataFrame({'score': [2, 1]}, index=['X', 'Y'])
    top1 = top_n_recommender(df, 'score', 1)
    assert top1 == ['X']


def test_peer_compare():
    price_map = {'A': [1, 2, 3], 'B': [2, 3, 4]}
    res = peer_compare(price_map, 'A', n=1)
    assert 'peers' in res and len(res['peers']) == 1


def test_chart_performance():
    dates = ['2021-01-31', '2021-03-31']
    price_map = {'A': [100, 110]}
    res = chart_performance(dates, price_map)
    assert 'quarterly' in res and 'yearly' in res
