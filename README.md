# GPT4-Stock-Analyzer
GPT-4 powered Python tool that generates investment-grade stock analysis using real-time market data.

import yfinance as yf
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Check your .env file.")
client = OpenAI(api_key=api_key)

def get_stock_summary(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    # Error handling: some symbols return empty or bad data
    if not info or "longName" not in info:
        return None

    name = info.get("longName", "N/A")
    sector = info.get("sector", "N/A")
    market_cap = info.get("marketCap", "N/A")
    pe_ratio = info.get("trailingPE", "N/A")
    price = info.get("currentPrice", "N/A")

    return {
        "name": name,
        "sector": sector,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "price": price
    }

def generate_analysis(summary):
    prompt = f"""
    Provide a concise investment analysis for the following stock based on its key data:

    Company: {summary['name']}
    Sector: {summary['sector']}
    Market Cap: {summary['market_cap']}
    Price: {summary['price']}
    P/E Ratio: {summary['pe_ratio']}

    Mention potential strengths, weaknesses, and whether this stock appears undervalued or overvalued.
    Limit the response to 150 words.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    tickers = input("Enter one or more stock ticker symbols (comma-separated): ").upper().split(",")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"gpt4_stock_analysis_{timestamp}.txt"

    with open(output_file, "w") as f:
        for ticker in tickers:
            ticker = ticker.strip()
            print(f"\n🔍 Analyzing {ticker}...")
            summary = get_stock_summary(ticker)

            if summary is None:
                error_msg = f"⚠️ Could not fetch valid data for ticker: {ticker}\n"
                print(error_msg)
                f.write(error_msg)
                continue

            print("📊 Stock Summary:", summary)
            analysis = generate_analysis(summary)

            print("\n🤖 GPT-4 Investment Analysis:\n")
            print(analysis)

            f.write(f"===== {ticker} Analysis =====\n")
            f.write(f"Stock Summary: {summary}\n")
            f.write(f"GPT-4 Analysis:\n{analysis}\n\n")

    print(f"\n📁 All analyses saved to: {output_file}")
