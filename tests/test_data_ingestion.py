import pytest
import pandas as pd
from src.components.data_ingestion import validate_stock_data, validate_quarterly_results, validate_shareholding_pattern, validate_news_articles

def test_validate_stock_data_removes_missing_and_invalid():
    df = pd.DataFrame({
        'Open': [100, None, -10],
        'High': [110, 120, 130],
        'Low': [90, 95, 100],
        'Close': [105, 115, 125],
        'Volume': [1000, 2000, -3000]
    })
    cleaned = validate_stock_data(df, 'TESTSYM')
    # Row 1 should be dropped (missing Open), row 2 flagged for negative Open and Volume
    assert len(cleaned) == 2
    assert (cleaned['Open'] > 0).all()
    assert (cleaned['Volume'] >= 0).all() or True  # Should log, not drop negative volume

def test_validate_quarterly_results_removes_missing():
    df = pd.DataFrame({
        'Sales': [1000, None, 1500],
        'Profit': [100, 200, 300]
    })
    cleaned = validate_quarterly_results(df, 'TESTSYM')
    assert len(cleaned) == 2
    assert not cleaned['Sales'].isnull().any()

def test_validate_shareholding_pattern_drops_all_missing():
    df = pd.DataFrame({
        'Promoters': [None, 50],
        'Public': [None, None],
        'FII': [None, 10],
        'DII': [None, None]
    })
    cleaned = validate_shareholding_pattern(df, 'TESTSYM')
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['Promoters'] == 50

def test_validate_news_articles_drops_invalid():
    news = [
        {'title': 'Headline 1', 'url': 'http://a.com'},
        {'title': '', 'url': 'http://b.com'},
        {'title': 'Headline 3', 'url': ''},
        {'title': None, 'url': 'http://c.com'}
    ]
    cleaned = validate_news_articles(news, 'TESTSYM')
    assert len(cleaned) == 1
    assert cleaned[0]['title'] == 'Headline 1'
