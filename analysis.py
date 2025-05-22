from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Suppress GLib warnings
os.environ['G_MESSAGES_DEBUG'] = 'none'

class StockAnalyzer:
    """
    Core analysis engine for the GPT-4o Stock Analyzer Hybrid.
    Handles top picks, comparison, and portfolio analysis.
    """
    def __init__(self, user_profile: Optional[Dict] = None):
        self.user_profile = user_profile or {}
        self.default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        self._ticker_cache = {}

    def get_stock_data(self, ticker: str, retries: int = 2) -> Optional[Dict]:
        """
        Get basic stock data for a given ticker.
        Returns a dictionary with stock information or None if failed.
        """
        for attempt in range(retries + 1):
            try:
                if ticker not in self._ticker_cache:
                    self._ticker_cache[ticker] = yf.Ticker(ticker)
                
                stock = self._ticker_cache[ticker]
                info = stock.info
                hist = stock.history(period="1mo")
                
                if not info or "longName" not in info:
                    print(f"⚠️ {ticker} returned no valid financial data.")
                    return None
                
                return {
                    "name": info.get("longName", "N/A"),
                    "sector": info.get("sector", "Unknown"),
                    "market_cap": info.get("marketCap", "N/A"),
                    "price": info.get("currentPrice", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "beta": info.get("beta", "N/A"),
                    "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                    "month_change": ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100) if not hist.empty else 0
                }
            except Exception as e:
                if attempt < retries:
                    print(f"[Retry {attempt+1}] Error fetching data for {ticker}: {e}")
                    continue
                print(f"[Error] Could not fetch data for {ticker}: {e}")
                return None

    def top_picks(self) -> List[str]:
        """
        Suggest top stock picks based on user profile and market data.
        Returns a list of ticker symbols.
        """
        # For now, return some popular tech stocks
        return self.default_tickers[:5]

    def compare(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Compare a list of tickers and return analysis results.
        """
        results = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1mo")
                
                results[ticker] = {
                    "name": info.get("longName", ticker),
                    "current_price": info.get("currentPrice", 0),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                    "month_change": ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100) if not hist.empty else 0
                }
            except Exception as e:
                results[ticker] = {"error": str(e)}
        
        return results

    def portfolio_analysis(self, portfolio: List[str], amount: float = 0.0) -> Dict:
        """
        Analyze a user's portfolio and suggest improvements.
        """
        if not portfolio:
            return {"summary": "No stocks in portfolio to analyze."}
        
        try:
            # Get basic analysis for each stock
            analysis = self.compare(portfolio)
            
            # Calculate portfolio metrics
            total_value = sum(stock.get("current_price", 0) for stock in analysis.values() if "error" not in stock)
            valid_stocks = [stock for stock in analysis.values() if "error" not in stock]
            avg_pe = sum(stock.get("pe_ratio", 0) for stock in valid_stocks) / len(valid_stocks) if valid_stocks else 0
            avg_dividend = sum(stock.get("dividend_yield", 0) for stock in valid_stocks) / len(valid_stocks) if valid_stocks else 0
            
            return {
                "summary": f"Portfolio Analysis:\nTotal Value: ${total_value:,.2f}\nAvg P/E: {avg_pe:.2f}\nAvg Dividend Yield: {avg_dividend:.2%}",
                "stocks": analysis
            }
        except Exception as e:
            return {"summary": f"Error analyzing portfolio: {str(e)}"}
