from analysis import StockAnalyzer
from news import NewsFetcher
from social import SocialMediaFetcher
from user import UserSession
import json

def test_stock_analysis():
    print("\n=== Testing Stock Analysis ===")
    analyzer = StockAnalyzer()
    
    # Test top picks
    print("\nTesting top picks:")
    picks = analyzer.top_picks()
    print(f"Top picks: {picks}")
    
    # Test stock comparison
    print("\nTesting stock comparison:")
    comparison = analyzer.compare(["AAPL", "MSFT", "GOOGL"])
    print("Comparison results:")
    for ticker, data in comparison.items():
        if "error" not in data:
            print(f"\n{ticker}:")
            print(f"  Price: ${data['current_price']:,.2f}")
            print(f"  P/E: {data['pe_ratio']:.2f}")
            print(f"  Dividend: {data['dividend_yield']:.2%}")
    
    # Test portfolio analysis
    print("\nTesting portfolio analysis:")
    portfolio = analyzer.portfolio_analysis(["AAPL", "MSFT", "GOOGL"], 10000)
    print(portfolio["summary"])

def test_news_fetcher():
    print("\n=== Testing News Fetcher ===")
    fetcher = NewsFetcher()
    
    # Test market news
    print("\nTesting market news:")
    market_news = fetcher.fetch_market_news()
    print(f"Market news count: {len(market_news)}")
    
    # Test ticker news
    print("\nTesting ticker news:")
    ticker_news = fetcher.fetch_ticker_news(["AAPL", "MSFT"])
    print(f"Ticker news count: {sum(len(news) for news in ticker_news.values())}")

def test_social_media():
    print("\n=== Testing Social Media ===")
    fetcher = SocialMediaFetcher()
    
    # Test fetching from trusted sources
    print("\nTesting trusted sources:")
    instagram_posts = fetcher.fetch_from_trusted("instagram", limit=2)
    twitter_posts = fetcher.fetch_from_trusted("twitter", limit=2)
    print(f"Instagram posts: {len(instagram_posts)}")
    print(f"Twitter posts: {len(twitter_posts)}")
    
    # Test insights aggregation
    print("\nTesting insights aggregation:")
    all_posts = instagram_posts + twitter_posts
    insights = fetcher.aggregate_insights(all_posts)
    print(f"Insights: {insights['summary']}")

def test_user_session():
    print("\n=== Testing User Session ===")
    
    # Create a test session
    session = UserSession(
        profile="Test",
        goal="growth",
        risk="medium",
        tone="professional",
        accessibility=False,
        mode_type="guided"
    )
    
    # Test saving
    print("\nTesting session save:")
    session.save("test_session.json")
    print("Session saved to test_session.json")
    
    # Test loading
    print("\nTesting session load:")
    loaded_session = UserSession.load("test_session.json")
    print(f"Loaded profile: {loaded_session.profile}")
    print(f"Loaded goal: {loaded_session.goal}")

def main():
    print("Starting project tests...")
    
    try:
        test_stock_analysis()
        test_news_fetcher()
        test_social_media()
        test_user_session()
        print("\nAll tests completed!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    main() 