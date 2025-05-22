import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime
import json
from analysis import StockAnalyzer
from news import NewsFetcher
from social import SocialMediaFetcher
from export import Exporter
from user import UserSession
import pytest

# Load environment variables
load_dotenv()

def test_stock_analysis_component():
    """Test the stock analysis functionality"""
    print("\n=== Testing Stock Analysis Component ===")
    analyzer = StockAnalyzer()
    
    # Test basic stock data retrieval
    print("\nTesting stock data retrieval...")
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    for ticker in test_tickers:
        data = analyzer.get_stock_data(ticker)
        assert data is not None, f"Failed to get data for {ticker}"
        print(f"✓ {ticker}: {data['name']}")
    
    # Test top picks
    print("\nTesting top picks...")
    picks = analyzer.top_picks()
    assert len(picks) > 0, "No top picks returned"
    print(f"✓ Top picks: {picks}")
    
    # Test comparison
    print("\nTesting stock comparison...")
    comparison = analyzer.compare(["AAPL", "MSFT"])
    assert "AAPL" in comparison and "MSFT" in comparison, "Comparison failed"
    print("✓ Comparison successful")
    
    # Test portfolio analysis
    print("\nTesting portfolio analysis...")
    portfolio = analyzer.portfolio_analysis(["AAPL", "MSFT", "GOOGL"], 10000)
    assert "summary" in portfolio, "Portfolio analysis failed"
    print("✓ Portfolio analysis successful")

def test_news_component():
    """Test the news fetching functionality"""
    print("\n=== Testing News Component ===")
    fetcher = NewsFetcher()
    
    # Test market news
    print("\nTesting market news...")
    market_news = fetcher.fetch_market_news()
    assert isinstance(market_news, list), "Market news should be a list"
    print(f"✓ Retrieved {len(market_news)} market news items")
    
    # Test ticker news
    print("\nTesting ticker news...")
    ticker_news = fetcher.fetch_ticker_news(["AAPL", "MSFT"])
    assert isinstance(ticker_news, dict), "Ticker news should be a dictionary"
    print(f"✓ Retrieved news for {len(ticker_news)} tickers")

def test_social_media_component():
    """Test the social media functionality"""
    print("\n=== Testing Social Media Component ===")
    fetcher = SocialMediaFetcher()
    
    # Test fetching from trusted sources
    print("\nTesting trusted sources...")
    instagram_posts = fetcher.fetch_from_trusted("instagram", limit=2)
    twitter_posts = fetcher.fetch_from_trusted("twitter", limit=2)
    assert len(instagram_posts) > 0, "No Instagram posts retrieved"
    assert len(twitter_posts) > 0, "No Twitter posts retrieved"
    print("✓ Retrieved posts from both platforms")
    
    # Test insights aggregation
    print("\nTesting insights aggregation...")
    all_posts = instagram_posts + twitter_posts
    insights = fetcher.aggregate_insights(all_posts)
    assert "summary" in insights, "Insights aggregation failed"
    print("✓ Insights aggregation successful")

def test_export_component():
    """Test the export functionality"""
    print("\n=== Testing Export Component ===")
    exporter = Exporter()
    
    # Test data preparation
    print("\nTesting data preparation...")
    test_data = {
        "ticker1": "AAPL",
        "ticker2": "MSFT",
        "stats1": {"return": 10.5, "volatility": 15.2},
        "stats2": {"return": 8.7, "volatility": 12.3},
        "summary": "Test summary"
    }
    
    # Test HTML export
    print("\nTesting HTML export...")
    html_file = "test_export.html"
    exporter.export_html(test_data, html_file)
    assert os.path.exists(os.path.join(exporter.output_dir, html_file)), "HTML export failed"
    print("✓ HTML export successful")
    
    # Test CSV export
    print("\nTesting CSV export...")
    csv_file = "test_export.csv"
    exporter.export_csv(test_data, csv_file)
    assert os.path.exists(os.path.join(exporter.output_dir, csv_file)), "CSV export failed"
    print("✓ CSV export successful")
    
    # Cleanup test files
    for file in [html_file, csv_file]:
        path = os.path.join(exporter.output_dir, file)
        if os.path.exists(path):
            os.remove(path)

def test_user_session_component():
    """Test the user session functionality"""
    print("\n=== Testing User Session Component ===")
    
    # Test session creation
    print("\nTesting session creation...")
    session = UserSession(
        profile="Test",
        goal="growth",
        risk="medium",
        tone="professional",
        accessibility=False,
        mode_type="guided"
    )
    assert session.profile == "Test", "Session creation failed"
    print("✓ Session creation successful")
    
    # Test session saving
    print("\nTesting session saving...")
    test_file = "test_session.json"
    session.save(test_file)
    assert os.path.exists(test_file), "Session save failed"
    print("✓ Session save successful")
    
    # Test session loading
    print("\nTesting session loading...")
    loaded_session = UserSession.load(test_file)
    assert loaded_session.profile == "Test", "Session load failed"
    print("✓ Session load successful")
    
    # Cleanup test file
    if os.path.exists(test_file):
        os.remove(test_file)

def main():
    """Run all tests"""
    print("Starting comprehensive tests...")
    
    try:
        test_stock_analysis_component()
        test_news_component()
        test_social_media_component()
        test_export_component()
        test_user_session_component()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 