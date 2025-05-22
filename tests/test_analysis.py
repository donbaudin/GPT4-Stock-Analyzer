import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis import StockAnalyzer
import pytest

def test_top_picks():
    """Test that top_picks returns a list of tickers."""
    analyzer = StockAnalyzer()
    picks = analyzer.top_picks()
    assert isinstance(picks, list)
    assert all(isinstance(t, str) for t in picks)

def test_compare():
    """Test that compare returns a dict with the correct keys."""
    analyzer = StockAnalyzer()
    tickers = ["AAPL", "MSFT"]
    result = analyzer.compare(tickers)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(tickers)

def test_portfolio_analysis():
    """Test that portfolio_analysis returns a dict with a summary key."""
    analyzer = StockAnalyzer()
    portfolio = ["AAPL", "GOOGL"]
    result = analyzer.portfolio_analysis(portfolio, amount=10000)
    assert isinstance(result, dict)
    assert "summary" in result
