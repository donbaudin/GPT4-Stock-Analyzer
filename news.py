from typing import List, Dict, Any, Optional
import feedparser
from datetime import datetime, timedelta
import yfinance as yf
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsFetcher:
    """
    Fetches and analyzes news articles from trusted sources.
    Complies with Twitter's API terms and privacy requirements.
    """
    def __init__(self, cache_dir: str = "cache", data_retention_days: int = 7):
        """
        Initialize news fetcher with caching.
        
        Args:
            cache_dir: Directory to store cached data
            data_retention_days: Number of days to retain cached data (default: 7 days)
        """
        load_dotenv()
        
        # Create cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data retention settings
        self.data_retention_days = data_retention_days
        
        # Trusted news sources
        self.trusted_sources = [
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "cnbc.com",
            "marketwatch.com",
            "investing.com",
            "seekingalpha.com",
            "yahoo.com/finance",
            "businessinsider.com"
        ]
        
        # Clean up old cache files
        self._cleanup_old_cache()

    def _cleanup_old_cache(self) -> None:
        """
        Remove cache files older than the retention period.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        cache_file.unlink()
                        logger.info(f"Removed old cache file: {cache_file}")
                except Exception as e:
                    logger.error(f"Error processing cache file {cache_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")

    def _get_cache_path(self, key: str) -> Path:
        """
        Get the cache file path for a given key.
        """
        return self.cache_dir / f"{key}.json"

    def _load_from_cache(self, key: str, max_age_hours: int = 1) -> Optional[Dict[str, Any]]:
        """
        Load data from cache if it exists and is not expired.
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                cache_time = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - cache_time > timedelta(hours=max_age_hours):
                    return None
                return data['content']
        except Exception as e:
            logger.error(f"Error loading from cache {key}: {str(e)}")
            return None

    def _save_to_cache(self, key: str, content: Dict[str, Any]) -> None:
        """
        Save data to cache with timestamp.
        """
        try:
            cache_path = self._get_cache_path(key)
            data = {
                'timestamp': datetime.now().isoformat(),
                'content': content
            }
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Successfully cached data for key: {key}")
        except Exception as e:
            logger.error(f"Error saving to cache {key}: {str(e)}")

    def fetch_market_news(self, limit: int = 10, focus_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch recent market news articles from trusted sources.
        """
        # Check cache first
        cache_key = f"market_news_{limit}_{'-'.join(focus_topics or [])}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            # Fetch news from Yahoo Finance RSS feed
            feed = feedparser.parse("https://finance.yahoo.com/news/rss")
            articles = []
            
            for entry in feed.entries[:limit]:
                # Only include articles from trusted sources
                if any(source in entry.link.lower() for source in self.trusted_sources):
                    # Filter by focus topics if specified
                    if focus_topics and not any(topic.lower() in entry.title.lower() 
                                             for topic in focus_topics):
                        continue
                    
                    articles.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.published,
                        "source": entry.link.split('/')[2],
                        "summary": entry.summary
                    })
            
            # Cache the results
            self._save_to_cache(cache_key, articles)
            return articles
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return []

    def fetch_ticker_news(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent news for a specific ticker.
        """
        # Check cache first
        cache_key = f"ticker_news_{ticker}_{limit}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get news from yfinance
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                logger.info(f"No news found for ticker: {ticker}")
                return []
            
            # Filter and format news
            filtered_news = []
            for article in news[:limit]:
                # Only include articles from trusted sources
                if any(source in article.get('link', '').lower() 
                      for source in self.trusted_sources):
                    filtered_news.append({
                        "title": article.get('title', ''),
                        "link": article.get('link', ''),
                        "published": datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                        "source": article.get('publisher', ''),
                        "summary": article.get('summary', '')
                    })
            
            # Cache the results
            self._save_to_cache(cache_key, filtered_news)
            return filtered_news
        except Exception as e:
            logger.error(f"Error fetching ticker news for {ticker}: {str(e)}")
            return []

    def extract_entities_and_events(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract relevant entities and events from news articles.
        Returns only aggregated metrics, not raw content.
        """
        entities = {}
        events = {}
        
        for article in articles:
            # Extract mentioned companies
            title = article['title'].lower()
            for source in self.trusted_sources:
                if source in title:
                    company = source.split('.')[0]
                    entities[company] = entities.get(company, 0) + 1
            
            # Extract event types
            event_keywords = {
                'earnings': ['earnings', 'quarterly results', 'financial results'],
                'acquisition': ['acquisition', 'buyout', 'merger'],
                'partnership': ['partnership', 'deal', 'agreement'],
                'product': ['launch', 'release', 'new product'],
                'regulation': ['regulation', 'compliance', 'legal']
            }
            
            for event_type, keywords in event_keywords.items():
                if any(keyword in title for keyword in keywords):
                    events[event_type] = events.get(event_type, 0) + 1
        
        return {
            "entities": entities,
            "events": events
        }

    def summarize_news(self, articles: List[Dict[str, Any]]) -> str:
        """
        Provide a user-friendly summary of the news articles.
        Only includes aggregated metrics, not raw content.
        """
        if not articles:
            return "No recent news articles found."
        
        # Extract entities and events
        analysis = self.extract_entities_and_events(articles)
        
        # Create summary
        summary = [f"Found {len(articles)} recent news articles."]
        
        if analysis['entities']:
            summary.append("\nMost mentioned companies:")
            for company, count in sorted(analysis['entities'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
                summary.append(f"- {company}: {count} mentions")
        
        if analysis['events']:
            summary.append("\nKey events:")
            for event, count in sorted(analysis['events'].items(), 
                                    key=lambda x: x[1], reverse=True):
                summary.append(f"- {event}: {count} articles")
        
        return "\n".join(summary)
