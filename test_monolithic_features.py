import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import pytest
from analysis import StockAnalyzer
from news import NewsFetcher
from social import SocialMediaFetcher
from export import Exporter
from user import UserSession

def test_stock_analysis_features():
    """Test core stock analysis features"""
    analyzer = StockAnalyzer()
    
    # Test top picks
    print("\nTesting top picks...")
    picks = analyzer.top_picks()
    assert isinstance(picks, list), "Top picks should return a list"
    assert len(picks) > 0, "Top picks should not be empty"
    print(f"Top picks: {picks}")
    
    # Test stock comparison
    print("\nTesting stock comparison...")
    tickers = ["AAPL", "MSFT"]
    comparison = analyzer.compare(tickers)
    assert isinstance(comparison, dict), "Comparison should return a dictionary"
    for ticker in tickers:
        assert ticker in comparison, f"Comparison should include {ticker}"
        assert "name" in comparison[ticker], f"Comparison for {ticker} should include name"
        assert "current_price" in comparison[ticker], f"Comparison for {ticker} should include price"
    print(f"Comparison results: {comparison}")
    
    # Test portfolio analysis
    print("\nTesting portfolio analysis...")
    portfolio = ["AAPL", "MSFT", "GOOGL"]
    amount = 10000
    analysis = analyzer.portfolio_analysis(portfolio, amount)
    assert isinstance(analysis, dict), "Portfolio analysis should return a dictionary"
    assert "summary" in analysis, "Portfolio analysis should include summary"
    assert "stocks" in analysis, "Portfolio analysis should include stocks"
    print(f"Portfolio analysis: {analysis['summary']}")

def test_news_features():
    """Test news fetching and analysis features"""
    fetcher = NewsFetcher()
    
    # Test market news
    print("\nTesting market news...")
    news = fetcher.fetch_market_news(limit=5)
    assert isinstance(news, list), "Market news should return a list"
    print(f"Market news: {len(news)} items fetched")
    
    # Test ticker news
    print("\nTesting ticker news...")
    ticker_news = fetcher.fetch_ticker_news("AAPL", limit=3)
    assert isinstance(ticker_news, list), "Ticker news should return a list"
    print(f"Ticker news: {len(ticker_news)} items fetched")

def test_social_features():
    """Test social media features"""
    fetcher = SocialMediaFetcher()
    
    # Test trusted sources
    print("\nTesting trusted sources...")
    sources = fetcher.list_trusted_sources()
    assert isinstance(sources, dict), "Trusted sources should return a dictionary"
    print(f"Trusted sources: {len(sources)} platforms")
    
    # Test social media insights
    print("\nTesting social media insights...")
    posts = fetcher.fetch_from_trusted("twitter", limit=2)
    assert isinstance(posts, list), "Social posts should return a list"
    if posts:
        insights = fetcher.aggregate_insights(posts)
        assert isinstance(insights, dict), "Insights should return a dictionary"
        print(f"Social insights: {insights.get('summary', 'No summary available')}")

def test_export_features():
    """Test export features"""
    exporter = Exporter()
    
    # Test HTML export
    print("\nTesting HTML export...")
    data = {
        "ticker1": "AAPL",
        "ticker2": "MSFT",
        "comparison": "Test comparison",
        "summary": "Test summary"
    }
    html_file = exporter.export_to_html(data, "test_export.html")
    assert os.path.exists(html_file), "HTML file should be created"
    print(f"HTML export: {html_file}")
    
    # Test CSV export
    print("\nTesting CSV export...")
    csv_file = exporter.export_to_csv(data, "test_export.csv")
    assert os.path.exists(csv_file), "CSV file should be created"
    print(f"CSV export: {csv_file}")

def test_user_session():
    """Test user session management"""
    # Test session creation
    print("\nTesting user session...")
    session = UserSession(
        profile="Beginner",
        goal="growth",
        risk="medium",
        tone="professional",
        accessibility=True,
        mode_type="guided"
    )
    assert session.profile == "Beginner", "Session should have correct profile"
    assert session.goal == "growth", "Session should have correct goal"
    
    # Test session saving and loading
    session.save("test_session.json")
    loaded_session = UserSession.load("test_session.json")
    assert loaded_session is not None, "Session should be loaded"
    assert loaded_session.profile == session.profile, "Loaded session should match saved session"
    print(f"User session: {loaded_session.profile}, {loaded_session.goal}")

def main():
    """Run all tests"""
    print("Starting feature verification...")
    
    try:
        test_stock_analysis_features()
        test_news_features()
        test_social_features()
        test_export_features()
        test_user_session()
        print("\n✅ All features verified successfully!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 