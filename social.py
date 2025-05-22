from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('social_media.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trusted, verified, and vetted sources for each platform
TRUSTED_SOURCES = {
    "instagram": [
        {"handle": "TheMorningBrew", "verified": True},
        {"handle": "Dividendology_yt", "verified": True},
        {"handle": "quarterchart", "verified": True},
        {"handle": "moby.invest", "verified": True},
        {"handle": "UnusualWhales", "verified": True},
        {"handle": "ceowatchlist", "verified": True},
        {"handle": "dubadvisors", "verified": True},
        {"handle": "dubinvest", "verified": True},
        {"handle": "TheMotelyFool", "verified": True},
    ],
    "twitter": [
        {"handle": "UnusualWhales", "verified": True},
        # Add more as needed
    ],
    "tiktok": [
        # Add trusted TikTok sources as needed
    ]
}

class SocialMediaFetcher:
    """
    Fetches and analyzes social media posts from trusted sources.
    Currently using placeholder implementations for API integrations.
    """
    
    def __init__(self, cache_dir: str = "cache", data_retention_days: int = 7):
        """
        Initialize the social media fetcher.
        
        Args:
            cache_dir: Directory to store cached data
            data_retention_days: Number of days to retain cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_retention_days = data_retention_days
        
        # Clean up old cache files
        self._cleanup_old_cache()
        
        logger.info("SocialMediaFetcher initialized")
    
    def _cleanup_old_cache(self) -> None:
        """Remove cache files older than retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            for cache_file in self.cache_dir.glob("*.json"):
                if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_date:
                    cache_file.unlink()
                    logger.info(f"Removed old cache file: {cache_file}")
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the path for a cache file."""
        return self.cache_dir / f"{key}.json"
    
    def _load_from_cache(self, key: str) -> Optional[Dict]:
        """Load data from cache if available and not expired."""
        try:
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=1):
                return None
            
            return data['data']
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, key: str, data: Dict) -> None:
        """Save data to cache with timestamp."""
        try:
            cache_path = self._get_cache_path(key)
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Saved data to cache: {key}")
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    def fetch_twitter_posts(self, query: str, max_posts: int = 100) -> List[Dict]:
        """
        Placeholder for fetching Twitter posts.
        To be implemented with Twitter API integration.
        
        Args:
            query: Search query
            max_posts: Maximum number of posts to fetch
            
        Returns:
            List of post data dictionaries
        """
        logger.info(f"Twitter API integration pending. Query: {query}")
        return []
    
    def fetch_reddit_posts(self, subreddit: str, max_posts: int = 100) -> List[Dict]:
        """
        Placeholder for fetching Reddit posts.
        To be implemented with Reddit API integration.
        
        Args:
            subreddit: Subreddit name
            max_posts: Maximum number of posts to fetch
            
        Returns:
            List of post data dictionaries
        """
        logger.info(f"Reddit API integration pending. Subreddit: {subreddit}")
        return []
    
    def fetch_ticker_sentiment(self, ticker: str) -> Dict:
        """
        Placeholder for fetching sentiment analysis for a ticker.
        To be implemented with social media API integration.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        logger.info(f"Sentiment analysis pending for ticker: {ticker}")
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'sentiment_score': 0.0,
            'post_count': 0,
            'source_breakdown': {
                'twitter': 0,
                'reddit': 0
            }
        }
    
    def get_trusted_sources(self) -> Dict[str, List[str]]:
        """
        Get list of trusted sources for social media analysis.
        
        Returns:
            Dictionary containing lists of trusted sources
        """
        return {
            'twitter': [
                'WSJ',
                'Bloomberg',
                'CNBC',
                'Reuters',
                'FinancialTimes'
            ],
            'reddit': [
                'wallstreetbets',
                'stocks',
                'investing',
                'stockmarket'
            ]
        }

    def fetch_from_trusted(self, platform: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent posts from all trusted sources on a given platform.
        Only uses accounts marked as verified in TRUSTED_SOURCES.
        Returns mock data for demo/compliance.
        """
        sources = TRUSTED_SOURCES.get(platform, [])
        all_posts = []
        for src in sources:
            if src["verified"]:
                if platform == "instagram":
                    all_posts += self.fetch_instagram_posts(src["handle"], limit)
                elif platform == "twitter":
                    all_posts += self.fetch_twitter_posts(src["handle"], limit)
                elif platform == "tiktok":
                    all_posts += self.fetch_tiktok_posts(src["handle"], limit)
        return all_posts

    def fetch_instagram_posts(self, account: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent posts from a public Instagram account (stub).
        For demo/compliance, returns mock data only.
        """
        # TODO: Integrate with instaloader or similar, handle login, rate limits, and TOS compliance.
        return [
            {"platform": "instagram", "account": account, "content": f"Sample post {i+1} about markets.", "date": "2024-06-01"}
            for i in range(limit)
        ]

    def fetch_tiktok_posts(self, account: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent posts from a public TikTok account (stub).
        For demo/compliance, returns mock data only.
        """
        # TODO: Integrate with TikTok API or similar, handle rate limits, and TOS compliance.
        return [
            {"platform": "tiktok", "account": account, "content": f"Sample TikTok {i+1} about investing.", "date": "2024-06-01"}
            for i in range(limit)
        ]

    def aggregate_insights(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate and summarize insights from a list of social media posts (stub).
        TODO: Use NLP to extract tickers, sentiment, and trends. Filter outliers and flag potential bias or data poisoning.
        """
        # For now, just return a mock summary
        return {"summary": "[Aggregated insights from trusted social media posts]"}

    def filter_outliers(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out outlier or suspicious posts to mitigate bias and data poisoning (stub).
        TODO: Implement NLP/statistical checks for real use.
        """
        return posts  # No filtering in stub

    def list_trusted_sources(self) -> None:
        """
        Print all trusted sources by platform for user review.
        """
        print("Trusted Social Media Sources:")
        for platform, sources in TRUSTED_SOURCES.items():
            print(f"- {platform.capitalize()}:")
            for src in sources:
                print(f"    @{src['handle']} (Verified: {src['verified']})")

    def suggest_trusted_source(self, platform: str, handle: str) -> None:
        """
        Accept a user suggestion for a new trusted source (stub).
        For now, just print/log the suggestion. In production, this would be reviewed by an admin and possibly stored persistently.
        """
        print(f"Suggestion received: Platform={platform}, Handle=@{handle}. This will be reviewed for trust and compliance.")
        # TODO: Store suggestion for admin review, add persistent storage if needed.

    def export_summary(self, data: Dict[str, Any]) -> str:
        """
        Return a summary-only string for quick output.
        Only includes aggregated metrics, not raw content.
        """
        summary = []
        for key, value in data.items():
            if isinstance(value, dict):
                summary.append(f"{key}:")
                for subkey, subvalue in value.items():
                    if subkey not in ["text", "content"]:  # Exclude raw content
                        summary.append(f"  {subkey}: {subvalue}")
            else:
                summary.append(f"{key}: {value}")
        
        return "\n".join(summary)
