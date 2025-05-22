import os
import sys
sys.stderr = open(os.devnull, 'w')
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import platform
import subprocess
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from colorama import init, Fore, Style
import base64
from weasyprint import HTML
import time
import signal
import threading
import warnings
import traceback
import contextlib
import feedparser
from collections import Counter
from typing import List, Dict, Any
import pytest
from analysis import StockAnalyzer

# After all imports and initializations, restore stderr
# (so Python errors are visible during runtime)
sys.stderr = sys.__stderr__

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")

client = OpenAI(api_key=api_key)

SECTOR_ETFS = {
    'Technology': 'XLK',
    'Financial Services': 'XLF',
    'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
}

# In-memory cache for yfinance Ticker objects and their history
_ticker_cache = {}
_history_cache = {}

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def get_ticker_cached(ticker):
    try:
        with suppress_stderr():
            if ticker not in _ticker_cache:
                _ticker_cache[ticker] = yf.Ticker(ticker)
        return _ticker_cache[ticker]
    except Exception as e:
        print(Fore.RED + f"[Error] Could not cache ticker {ticker}: {e}" + Style.RESET_ALL)
        return None

def get_history_cached(ticker, period="max", show_progress=False, retries=2):
    key = (ticker, period)
    if key not in _history_cache:
        for attempt in range(retries+1):
            try:
                with suppress_stderr():
                    if show_progress:
                        tqdm.write(f"Fetching data for {ticker} ({period})...")
                    _history_cache[key] = get_ticker_cached(ticker).history(period=period)
                break
            except Exception as e:
                if attempt < retries:
                    print(Fore.YELLOW + f"[Retry {attempt+1}] Error fetching history for {ticker}: {e}" + Style.RESET_ALL)
                    time.sleep(1)
                else:
                    print(Fore.RED + f"[Error] Could not fetch history for {ticker}: {e}" + Style.RESET_ALL)
                    _history_cache[key] = pd.DataFrame()
    return _history_cache[key]

def get_sector_etf(sector):
    return SECTOR_ETFS.get(sector, None)

def get_stock_data(ticker, retries=2):
    for attempt in range(retries+1):
        try:
            ticker_obj = get_ticker_cached(ticker)
            if not ticker_obj:
                return None
            with suppress_stderr():
                info = ticker_obj.info
            if not info or "longName" not in info:
                print(Fore.YELLOW + f"⚠️ {ticker} returned no valid financial data." + Style.RESET_ALL)
                return None
            return {
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "price": info.get("currentPrice", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "beta": info.get("beta", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
            }
        except Exception as e:
            if attempt < retries:
                print(Fore.YELLOW + f"[Retry {attempt+1}] Error fetching data for {ticker}: {e}" + Style.RESET_ALL)
                time.sleep(1)
            else:
                print(Fore.RED + f"[Error] Could not fetch data for {ticker}: {e}" + Style.RESET_ALL)
                return None

def get_top_stock_picks(profile=None, prefer_sectors=None, exclude_sectors=None, risk_tolerance=None):
    profile_text = f"for a {profile} investor" if profile else ""
    prefer_text = f" Prefer sectors: {', '.join(prefer_sectors)}." if prefer_sectors else ""
    exclude_text = f" Exclude sectors: {', '.join(exclude_sectors)}." if exclude_sectors else ""
    risk_text = f" Risk tolerance: {risk_tolerance}." if risk_tolerance else ""
    
    prompt = (
        f"Based on recent market conditions, provide a diversified set of investment recommendations {profile_text}. "
        f"Consider the following criteria:\n"
        f"1. Include a mix of:\n"
        f"   - Large, mid, and small-cap stocks\n"
        f"   - ETFs (index, sector, thematic)\n"
        f"   - Options strategies (if risk tolerance allows)\n"
        f"   - Futures contracts (if risk tolerance allows)\n"
        f"   - Crypto assets (if risk tolerance allows)\n"
        f"   - Fixed income (bonds, preferred shares)\n"
        f"2. Diversification requirements:\n"
        f"   - Across different sectors and industries\n"
        f"   - Different market caps\n"
        f"   - Various investment vehicles\n"
        f"3. Risk considerations:\n"
        f"   - Match recommendations to the specified risk tolerance\n"
        f"   - Include defensive positions if risk tolerance is low\n"
        f"   - Consider market conditions and volatility\n"
        f"4. Current market context:\n"
        f"   - Recent market trends\n"
        f"   - Economic indicators\n"
        f"   - Sector rotations\n"
        f"5. Specific preferences:\n"
        f"{prefer_text}\n"
        f"{exclude_text}\n"
        f"{risk_text}\n\n"
        f"For each asset class or category (Large-Cap Stocks, Mid-Cap Stocks, Small-Cap Stocks, ETFs, Options Strategies, Futures Contracts, Crypto Assets, Fixed Income), provide at least three distinct recommendations. For each, include:\n"
        f"1. Symbol and name\n"
        f"2. Asset type (stock/ETF/option/future/crypto/bond)\n"
        f"3. Brief rationale\n"
        f"4. Risk level\n"
        f"5. Time horizon\n"
        f"6. Key metrics or considerations\n\n"
        f"Format the response in clear sections with bullet points for easy reading."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a comprehensive market strategist and financial analyst with expertise in all asset classes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        traceback.print_exc()
        return "Error: Could not fetch top picks."

def generate_comparison(ticker1, data1, ticker2, data2):
    prompt = f"""
    Compare the following two companies based on these financial metrics:

    {ticker1}:
    - Name: {data1['name']}
    - Sector: {data1['sector']}
    - Market Cap: {data1['market_cap']}
    - Price: {data1['price']}
    - P/E Ratio: {data1['pe_ratio']}
    - Beta: {data1['beta']}
    - Dividend Yield: {data1['dividend_yield']}

    {ticker2}:
    - Name: {data2['name']}
    - Sector: {data2['sector']}
    - Market Cap: {data2['market_cap']}
    - Price: {data2['price']}
    - P/E Ratio: {data2['pe_ratio']}
    - Beta: {data2['beta']}
    - Dividend Yield: {data2['dividend_yield']}

    Provide an objective comparison of their valuation, growth potential, and risk profile. Keep the analysis under 200 words.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=350
    )

    return response.choices[0].message.content

def summarize_sentiment(comparison):
    followup = (
        "Please provide a one-sentence summary for the following investment comparison. "
        "Focus on sentiment and clarity for a non-technical investor audience:\n\n"
        + comparison
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You write simplified summaries for investors."},
            {"role": "user", "content": followup}
        ],
        temperature=0.6,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def extract_tickers_from_text(text):
    # Extract possible tickers (2-5 uppercase letters)
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
    # Validate each candidate using get_stock_data
    valid_tickers = []
    for t in candidates:
        if get_stock_data(t):
            valid_tickers.append(t)
    return valid_tickers

def get_backtest_data(ticker1, ticker2):
    t1_hist = get_history_cached(ticker1, period="6mo")
    t2_hist = get_history_cached(ticker2, period="6mo")

    def calc_return(hist, ticker):
        if hist is not None and len(hist) > 1:
            close_prices = hist["Close"].dropna()
            if close_prices.empty or len(close_prices) < 2:
                print(Fore.YELLOW + f"⚠️ {ticker}: Not enough valid price data for backtest period." + Style.RESET_ALL)
                return "N/A", None, None
            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]
            start_date = close_prices.index[0]
            end_date = close_prices.index[-1]
            ret = round(((end_price - start_price) / start_price) * 100, 2)
            return ret, start_date, end_date
        else:
            print(Fore.YELLOW + f"⚠️ {ticker}: Not enough data for backtest period." + Style.RESET_ALL)
            return "N/A", None, None

    t1_return, t1_start, t1_end = calc_return(t1_hist, ticker1)
    t2_return, t2_start, t2_end = calc_return(t2_hist, ticker2)

    # Print the actual dates used
    print(Fore.CYAN + f"\nBacktest period for {ticker1}: {t1_start.date() if t1_start else 'N/A'} to {t1_end.date() if t1_end else 'N/A'}" + Style.RESET_ALL)
    print(Fore.CYAN + f"Backtest period for {ticker2}: {t2_start.date() if t2_start else 'N/A'} to {t2_end.date() if t2_end else 'N/A'}" + Style.RESET_ALL)

    return t1_return, t2_return

def get_period_returns_and_volatility(ticker, periods, custom_start=None, custom_end=None):
    hist = get_history_cached(ticker, period="max")
    hist.index = hist.index.tz_localize(None)  # Make index timezone-naive
    hist = hist[hist.index <= pd.Timestamp.today()]
    results = {}
    for label, days in periods.items():
        if days == 'YTD':
            start = pd.Timestamp(datetime.now().year, 1, 1)
        else:
            start = pd.Timestamp.today() - pd.Timedelta(days=days)
        start = pd.to_datetime(start).tz_localize(None)  # Ensure timezone-naive
        period_hist = hist[hist.index >= start]
        if len(period_hist) < 2:
            results[label] = {'return': None, 'volatility': None, 'drawdown': None}
            continue
        ret = (period_hist['Close'].iloc[-1] - period_hist['Close'].iloc[0]) / period_hist['Close'].iloc[0] * 100
        vol = period_hist['Close'].pct_change().rolling(window=21).std().mean() * (252 ** 0.5) * 100
        # Max drawdown
        roll_max = period_hist['Close'].cummax()
        drawdown = ((period_hist['Close'] - roll_max) / roll_max).min() * 100
        results[label] = {'return': round(ret, 2), 'volatility': round(vol, 2), 'drawdown': round(drawdown, 2)}
    return results

def get_earnings_dates(ticker, start=None, end=None):
    try:
        cal = get_ticker_cached(ticker).earnings_dates
        if start:
            cal = cal[cal.index >= pd.to_datetime(start)]
        if end:
            cal = cal[cal.index <= pd.to_datetime(end)]
        return cal.index.to_pydatetime().tolist()
    except Exception:
        return []

def plot_price_with_earnings(ticker1, ticker2, start=None, end=None, sector1=None, sector2=None, show_progress=True):
    # Parallel fetch for all histories with progress bar
    tickers_needed = [ticker1, ticker2, 'SPY']
    if sector1 and sector1 == sector2:
        etf = get_sector_etf(sector1)
        if etf:
            tickers_needed.append(etf)
    if show_progress:
        for t in tqdm(tickers_needed, desc="Fetching price data"):
            get_history_cached(t, period="max", show_progress=False)
    else:
        with ThreadPoolExecutor() as executor:
            for t in tickers_needed:
                executor.submit(get_history_cached, t, period="max")
    hist1 = get_history_cached(ticker1, period="max")
    hist2 = get_history_cached(ticker2, period="max")
    hist1.index = hist1.index.tz_localize(None)
    hist2.index = hist2.index.tz_localize(None)
    if start:
        start = pd.to_datetime(start).tz_localize(None)
        hist1 = hist1[hist1.index >= start]
        hist2 = hist2[hist2.index >= start]
    if end:
        end = pd.to_datetime(end).tz_localize(None)
        hist1 = hist1[hist1.index <= end]
        hist2 = hist2[hist2.index <= end]
    plt.figure(figsize=(14, 7))
    plt.plot(hist1.index, hist1['Close'], label=f'{ticker1} Close', linewidth=2)
    plt.plot(hist2.index, hist2['Close'], label=f'{ticker2} Close', linewidth=2)
    # Moving averages for ticker1
    if len(hist1) >= 50:
        plt.plot(hist1.index, hist1['Close'].rolling(window=50).mean(), label=f'{ticker1} 50d MA', color='blue', linestyle='-.', linewidth=1)
    if len(hist1) >= 200:
        plt.plot(hist1.index, hist1['Close'].rolling(window=200).mean(), label=f'{ticker1} 200d MA', color='blue', linestyle=':', linewidth=1)
    # Moving averages for ticker2
    if len(hist2) >= 50:
        plt.plot(hist2.index, hist2['Close'].rolling(window=50).mean(), label=f'{ticker2} 50d MA', color='orange', linestyle='-.', linewidth=1)
    if len(hist2) >= 200:
        plt.plot(hist2.index, hist2['Close'].rolling(window=200).mean(), label=f'{ticker2} 200d MA', color='orange', linestyle=':', linewidth=1)
    # Sector ETF overlay (if both stocks are in the same sector)
    etf = None
    if sector1 and sector1 == sector2:
        etf = get_sector_etf(sector1)
        if etf:
            hist_etf = get_history_cached(etf, period="max")
            hist_etf.index = hist_etf.index.tz_localize(None)
            if start:
                hist_etf = hist_etf[hist_etf.index >= start]
            if end:
                hist_etf = hist_etf[hist_etf.index <= end]
            plt.plot(hist_etf.index, hist_etf['Close'], label=f'{etf} (Sector ETF)', color='green', linestyle=':')
    # SPY overlay
    hist_spy = get_history_cached('SPY', period="max")
    hist_spy.index = hist_spy.index.tz_localize(None)
    if start:
        hist_spy = hist_spy[hist_spy.index >= start]
    if end:
        hist_spy = hist_spy[hist_spy.index <= end]
    plt.plot(hist_spy.index, hist_spy['Close'], label='SPY (S&P 500)', color='black', linestyle='--', linewidth=1.5)
    # Earnings dates
    earnings1 = get_earnings_dates(ticker1, start, end)
    earnings2 = get_earnings_dates(ticker2, start, end)
    for edate in earnings1:
        plt.axvline(edate, color='blue', linestyle='--', alpha=0.3, label=f'{ticker1} Earnings' if edate == earnings1[0] else "")
    for edate in earnings2:
        plt.axvline(edate, color='orange', linestyle='--', alpha=0.3, label=f'{ticker2} Earnings' if edate == earnings2[0] else "")
    # Annotate most recent earnings
    if earnings1:
        plt.annotate(f'{ticker1} Last Earnings', xy=(earnings1[-1], hist1['Close'].loc[earnings1[-1]] if earnings1[-1] in hist1.index else hist1['Close'].iloc[-1]),
                     xytext=(0, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->'), color='blue')
    if earnings2:
        plt.annotate(f'{ticker2} Last Earnings', xy=(earnings2[-1], hist2['Close'].loc[earnings2[-1]] if earnings2[-1] in hist2.index else hist2['Close'].iloc[-1]),
                     xytext=(0, -30), textcoords='offset points', arrowprops=dict(arrowstyle='->'), color='orange')
    plt.legend()
    plt.title(f'{ticker1} vs {ticker2} Price History with Earnings Dates' + (f' and {etf}' if etf else ''))
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart_file = f"{ticker1}_{ticker2}_price_chart.png"
    plt.savefig(chart_file)
    plt.close()
    return chart_file

def export_html_to_pdf(html_file, pdf_file):
    HTML(html_file).write_pdf(pdf_file)

def save_outputs(ticker1, data1, ticker2, data2, top_picks, comparison, summary, backtest, timestamp):
    md_file = f"gpt4o_stock_comparison_{timestamp}.md"
    json_file = f"gpt4o_stock_data_{timestamp}.json"
    header = f"===== GPT-4o Stock Comparison =====\nGenerated on: {timestamp}\nCompared: {ticker1} vs {ticker2}\n"
    backtest_note = f"\n📉 Backtest Results (6-Month Return):\n- {ticker1}: {backtest[0]}%\n- {ticker2}: {backtest[1]}%\n"
    markdown = f"""# GPT-4o Stock Comparison\n**Generated:** {timestamp}\n**Tickers Compared:** `{ticker1}` vs `{ticker2}`\n\n## 📈 GPT-4o Daily Top Picks\n{top_picks}\n\n## 🧠 Comparison Result\n{comparison}\n\n## 💬 Summary\n{summary}\n\n## 📉 Backtest (6-Month Performance)\n- `{ticker1}`: **{backtest[0]}%**\n- `{ticker2}`: **{backtest[1]}%**\n    """
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({ticker1: data1, ticker2: data2}, f, indent=4)
    print(f"\n✅ Analysis saved to: {md_file}, {json_file}")
    if platform.system() == "Windows":
        os.startfile(md_file)
    elif platform.system() == "Darwin":
        subprocess.call(["open", md_file])
    elif platform.system() == "Linux":
        subprocess.call(["xdg-open", md_file])

def get_valid_ticker(prompt_text):
    while True:
        user_input = input(prompt_text).upper().strip()
        if "," in user_input or " " in user_input:
            print(Fore.YELLOW + "⚠️ Please enter a single stock ticker (no commas or spaces). Try again." + Style.RESET_ALL)
            continue
        data = get_stock_data(user_input)
        if data:
            return user_input, data
        print(Fore.RED + f"❌ {user_input} is not a valid ticker. Please try again." + Style.RESET_ALL)

def generate_sector_macro_commentary(ticker1, data1, ticker2, data2):
    year = datetime.now().year
    sector1 = data1.get('sector', 'N/A')
    sector2 = data2.get('sector', 'N/A')
    if sector1 == sector2:
        sector_info = f"both {ticker1} and {ticker2} are in the {sector1} sector"
    else:
        sector_info = f"{ticker1} is in the {sector1} sector, while {ticker2} is in the {sector2} sector"
    prompt = (
        f"Provide a brief, up-to-date macroeconomic and sector context for a stock comparison where {sector_info}. "
        f"Discuss any recent sector rotation, macro cycles, or notable events that could impact performance in {year}. "
        f"Keep it under 120 words and focus on what an investor should know about the sector and macro backdrop."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial market strategist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def generate_scenario_risk_commentary(ticker1, data1, ticker2, data2, scenarios=None):
    year = datetime.now().year
    sector1 = data1.get('sector', 'N/A')
    sector2 = data2.get('sector', 'N/A')
    if sector1 == sector2:
        sector_info = f"both {ticker1} and {ticker2} are in the {sector1} sector"
    else:
        sector_info = f"{ticker1} is in the {sector1} sector, while {ticker2} is in the {sector2} sector"
    scenario_text = f"Scenarios to consider: {', '.join(scenarios)}. " if scenarios else ""
    prompt = (
        f"For a stock comparison where {sector_info}, provide:\n"
        f"1. A brief scenario analysis: What could happen to these stocks if interest rates rise or fall in {year}?\n"
        f"2. A risk commentary: What are the main risks for these stocks/sectors in the next year?\n"
        f"{scenario_text}Keep it under 120 words and focus on practical, investor-relevant insights."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial risk analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

def get_outperformance_summary(ticker1, ticker2, stats1, stats2, stats_spy):
    summary_lines = []
    for label in stats1.keys():
        r1 = stats1[label]['return']
        r2 = stats2[label]['return']
        r_spy = stats_spy[label]['return']
        best = None
        if all(isinstance(x, (int, float)) for x in [r1, r2, r_spy]):
            if r1 >= r2 and r1 >= r_spy:
                best = ticker1
            elif r2 >= r1 and r2 >= r_spy:
                best = ticker2
            else:
                best = 'SPY'
            summary_lines.append(f"{label}: {best} outperformed ({ticker1}: {r1}%, {ticker2}: {r2}%, SPY: {r_spy}%)")
        else:
            summary_lines.append(f"{label}: Insufficient data to determine outperformance.")
    return summary_lines

def export_to_excel(ticker1, ticker2, stats1, stats2, stats_spy, backtest, summary, filename, outperf_summary=None):
    # Prepare returns/volatility table
    periods = list(stats1.keys())
    data = []
    for label in periods:
        s1 = stats1.get(label, {'return': None, 'volatility': None, 'drawdown': None})
        s2 = stats2.get(label, {'return': None, 'volatility': None, 'drawdown': None})
        s_spy = stats_spy.get(label, {'return': None, 'volatility': None, 'drawdown': None})
        data.append({
            'Period': label,
            f'{ticker1} Return': s1['return'],
            f'{ticker1} Volatility': s1['volatility'],
            f'{ticker1} DD': s1['drawdown'],
            f'{ticker2} Return': s2['return'],
            f'{ticker2} Volatility': s2['volatility'],
            f'{ticker2} DD': s2['drawdown'],
            'SPY Return': s_spy['return'],
            'SPY Volatility': s_spy['volatility'],
            'SPY DD': s_spy['drawdown'],
        })
    df = pd.DataFrame(data)
    # Backtest and summary
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, index=False, sheet_name='Returns & Volatility')
        pd.DataFrame({'Backtest': [f'{ticker1}: {backtest[0]}%', f'{ticker2}: {backtest[1]}%']}).to_excel(writer, index=False, sheet_name='Backtest')
        pd.DataFrame({'Summary': [summary]}).to_excel(writer, index=False, sheet_name='Summary')
        if outperf_summary:
            pd.DataFrame({'Outperformance': outperf_summary}).to_excel(writer, index=False, sheet_name='Outperformance')
        # Conditional formatting
        wb = writer.book
        ws = writer.sheets['Returns & Volatility']
        from openpyxl.formatting.rule import ColorScaleRule
        # Find columns for returns and volatility
        for col in range(2, 11):  # columns B-K (returns/vol)
            col_letter = chr(65 + col - 1)
            if 'Return' in ws.cell(row=1, column=col).value:
                ws.conditional_formatting.add(f'{col_letter}2:{col_letter}{len(df)+1}',
                    ColorScaleRule(start_type='min', start_color='F8696B',
                                   mid_type='percentile', mid_value=50, mid_color='FFEB84',
                                   end_type='max', end_color='63BE7B'))
            elif 'Vol' in ws.cell(row=1, column=col).value:
                ws.conditional_formatting.add(f'{col_letter}2:{col_letter}{len(df)+1}',
                    ColorScaleRule(start_type='min', start_color='63BE7B',
                                   mid_type='percentile', mid_value=50, mid_color='FFEB84',
                                   end_type='max', end_color='F8696B'))

def analyze_diversification_and_critique(top_picks):
    tickers = extract_tickers_from_text(top_picks)
    sector_counts = {}
    ticker_sectors = {}
    for t in tickers:
        data = get_stock_data(t)
        sector = data['sector'] if data else 'N/A'
        ticker_sectors[t] = sector
        if sector not in sector_counts:
            sector_counts[sector] = 0
        sector_counts[sector] += 1
    # Diversification score: number of unique sectors / number of picks
    unique_sectors = len([s for s in sector_counts if s != 'N/A'])
    diversification_score = unique_sectors / max(1, len(tickers))
    warning = ""
    if unique_sectors < max(2, len(tickers)//2):
        warning = f"⚠️ Diversification Warning: {len(tickers)-unique_sectors} picks share the same sector(s). Sectors: {sector_counts}"
    # GPT critique
    critique_prompt = (
        f"Here is a list of stock picks and their sectors: {ticker_sectors}. "
        f"Please provide a critical analysis of this list, focusing on sector/industry concentration, valuation, and macro headwinds. "
        f"Warn if the list is not diversified or is exposed to specific risks. Keep it under 80 words."
    )
    critique = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a critical financial analyst."},
            {"role": "user", "content": critique_prompt}
        ],
        temperature=0.7,
        max_tokens=150
    ).choices[0].message.content.strip()
    return warning, critique, ticker_sectors

def export_to_html(filename, top_picks, comparison, summary, stats1, stats2, stats_spy, ticker1, ticker2, outperf_summary, macro_comment, scenario_comment, chart_file=None, diversification_warning=None, picks_critique=None, picks_sectors=None):
    def html_table():
        periods = list(stats1.keys())
        rows = []
        for label in periods:
            s1 = stats1.get(label, {'return': None, 'volatility': None, 'drawdown': None})
            s2 = stats2.get(label, {'return': None, 'volatility': None, 'drawdown': None})
            s_spy = stats_spy.get(label, {'return': None, 'volatility': None, 'drawdown': None})
            def fmt_ret(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'orange'
                    return f"<td style='color:{color};font-weight:bold'>{val}</td>"
                return "<td style='color:orange'>N/A</td>"
            def fmt_vol(val):
                if isinstance(val, (int, float)):
                    if val < 20:
                        color = 'green'
                    elif val < 35:
                        color = 'goldenrod'
                    else:
                        color = 'red'
                    return f"<td style='color:{color}'>{val}</td>"
                return "<td style='color:orange'>N/A</td>"
            def fmt_dd(val):
                if isinstance(val, (int, float)):
                    if val < -20:
                        color = 'red'
                    elif val < -10:
                        color = 'orange'
                    else:
                        color = 'green'
                    return f"<td style='color:{color};font-weight:bold'>{val}</td>"
                return "<td style='color:orange'>N/A</td>"
            rows.append(f"<tr><td>{label}</td>" +
                        fmt_ret(s1['return']) + fmt_vol(s1['volatility']) + fmt_dd(s1['drawdown']) +
                        fmt_ret(s2['return']) + fmt_vol(s2['volatility']) + fmt_dd(s2['drawdown']) +
                        fmt_ret(s_spy['return']) + fmt_vol(s_spy['volatility']) + fmt_dd(s_spy['drawdown']) +
                        "</tr>")
        return """
        <table border='1' cellpadding='4' cellspacing='0'>
            <tr><th>Period</th><th>{0} Return</th><th>{0} Vol</th><th>{0} DD</th><th>{1} Return</th><th>{1} Vol</th><th>{1} DD</th><th>SPY Return</th><th>SPY Vol</th><th>SPY DD</th></tr>
            {2}
        </table>
        """.format(ticker1, ticker2, "\n".join(rows))
    def html_outperf():
        return "<ul>" + "".join(f"<li>{line}</li>" for line in outperf_summary) + "</ul>"
    def html_img():
        if chart_file:
            with open(chart_file, "rb") as imgf:
                img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
            return f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%;height:auto;border:1px solid #ccc;margin:10px 0;'>"
        return ""
    def html_sectors():
        if picks_sectors:
            return "<ul>" + "".join(f"<li>{t}: {s}</li>" for t, s in picks_sectors.items()) + "</ul>"
        return ""
    html = f"""
    <html><head><meta charset='utf-8'><title>GPT-4o Stock Comparison</title>
    <style>body{{font-family:sans-serif;max-width:900px;margin:0 auto;padding:2em;background:#f9f9f9;}}h1,h2{{color:#2a4d7a;}}.section{{margin-bottom:2em;}}</style>
    <style>.textblock { white-space: pre-wrap; word-break: break-word; }
    table { max-width: 100%; word-break: break-word; }
    img { max-width: 100%; height: auto; }</style>
    </head><body>
    <h1>GPT-4o Stock Comparison</h1>
    <div class='section'><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    <div class='section'><h2>Top Picks</h2><div class='textblock'>{top_picks}</div></div>
    <div class='section'><h2>Comparison</h2><pre>{comparison}</pre></div>
    <div class='section'><h2>Summary</h2><pre>{summary}</pre></div>
    <div class='section'><h2>Returns & Volatility</h2>{html_table()}</div>
    <div class='section'><h2>Outperformance by Period</h2>{html_outperf()}</div>
    <div class='section'><h2>Macro Context</h2><pre>{macro_comment}</pre></div>
    <div class='section'><h2>Scenario & Risk Commentary</h2><pre>{scenario_comment}</pre></div>
    <div class='section'><h2>Chart</h2>{html_img()}</div>
    </body></html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

def get_recent_news(ticker, n=5):
    try:
        news = get_ticker_cached(ticker).news
        if not news:
            return []
        return [(item['title'], item.get('publisher', ''), item.get('link', '')) for item in news[:n]]
    except Exception:
        return []

# Save/load user settings
SETTINGS_FILE = "gpt4o_user_settings.json"
def save_user_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        print(Fore.GREEN + f"[Settings] Saved to {SETTINGS_FILE}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[Settings] Error saving settings: {e}" + Style.RESET_ALL)
def load_user_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings = json.load(f)
        print(Fore.GREEN + f"[Settings] Loaded from {SETTINGS_FILE}" + Style.RESET_ALL)
        return settings
    except Exception:
        return {}

# Portfolio rebalancing suggestion (simple version)
def suggest_portfolio_rebalance(portfolio_tickers):
    sector_counts = {}
    for t in portfolio_tickers:
        data = get_stock_data(t)
        sector = data['sector'] if data else 'N/A'
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    if len(sector_counts) == 1:
        print(Fore.YELLOW + f"[Rebalance] All portfolio tickers are in {list(sector_counts.keys())[0]}. Consider adding other sectors for diversification." + Style.RESET_ALL)
    elif len(sector_counts) < len(portfolio_tickers)//2:
        print(Fore.YELLOW + f"[Rebalance] Portfolio is concentrated in a few sectors: {sector_counts}. Consider rebalancing for better diversification." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + "[Rebalance] Portfolio is reasonably diversified by sector." + Style.RESET_ALL)

# Sector/industry breakdown pie chart
def plot_sector_breakdown(tickers, filename="sector_breakdown.png"):
    import matplotlib.pyplot as plt
    sector_counts = {}
    for t in tickers:
        data = get_stock_data(t)
        sector = data['sector'] if data else 'N/A'
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    labels = list(sector_counts.keys())
    sizes = list(sector_counts.values())
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Sector/Industry Breakdown')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(Fore.CYAN + f"[Chart] Sector breakdown saved to: {filename}" + Style.RESET_ALL)

# CSV export for summary/returns
def export_to_csv(ticker1, ticker2, stats1, stats2, stats_spy, filename):
    try:
        periods = list(stats1.keys())
        data = []
        for label in periods:
            s1 = stats1.get(label, {'return': None, 'volatility': None, 'drawdown': None})
            s2 = stats2.get(label, {'return': None, 'volatility': None, 'drawdown': None})
            s_spy = stats_spy.get(label, {'return': None, 'volatility': None, 'drawdown': None})
            data.append({
                'Period': label,
                f'{ticker1} Return': s1['return'],
                f'{ticker1} Volatility': s1['volatility'],
                f'{ticker1} DD': s1['drawdown'],
                f'{ticker2} Return': s2['return'],
                f'{ticker2} Volatility': s2['volatility'],
                f'{ticker2} DD': s2['drawdown'],
                'SPY Return': s_spy['return'],
                'SPY Volatility': s_spy['volatility'],
                'SPY DD': s_spy['drawdown'],
            })
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(Fore.CYAN + f"[CSV export] Results saved to: {filename}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[CSV export] Error: {e}" + Style.RESET_ALL)

# Ticker autocomplete (stub, only works in some terminals)
def autocomplete_ticker(prompt_text, valid_tickers=None):
    try:
        import readline
        if valid_tickers:
            def completer(text, state):
                options = [i for i in valid_tickers if i.startswith(text.upper())]
                if state < len(options):
                    return options[state]
                return None
            readline.set_completer(completer)
            readline.parse_and_bind("tab: complete")
        return input(prompt_text)
    except Exception:
        return input(prompt_text)

# Helper: Fetch recent market news headlines (Yahoo Finance, Google News)
def fetch_market_news(n=50, focus_topics=None):
    feeds = [
        'https://finance.yahoo.com/rss/topstories',
        'https://news.google.com/rss/search?q=stock+market',
        'https://news.google.com/rss/search?q=acquisition+OR+merger+OR+AI+OR+OpenAI+OR+geopolitics+OR+sanctions+OR+IPO+OR+earnings',
    ]
    headlines = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for entry in d.entries[:n]:
                headlines.append(entry.title)
        except Exception as e:
            print(Fore.YELLOW + f"[News] Error fetching feed {url}: {e}" + Style.RESET_ALL)
    # Filter by focus topics if provided
    if focus_topics:
        focus_topics = [t.lower() for t in focus_topics]
        headlines = [h for h in headlines if any(t in h.lower() for t in focus_topics)]
    return headlines[:n]

# Helper: Simple entity/event extraction from headlines (stub for now)
def extract_entities_events(headlines):
    # TODO: Use spaCy or similar for robust extraction
    keywords = ['acquisition', 'merger', 'AI', 'OpenAI', 'IPO', 'lawsuit', 'regulation', 'sanction', 'geopolitics', 'partnership', 'deal', 'breakthrough', 'earnings', 'expansion', 'layoff', 'strike', 'scandal', 'hack', 'cyber', 'antitrust', 'approval', 'FDA', 'SEC', 'FTC', 'buyout', 'investment', 'funding', 'bankruptcy', 'collapse', 'AI', 'cloud', 'quantum', 'chip', 'semiconductor', 'defense', 'energy', 'EV', 'battery', 'China', 'Russia', 'Ukraine', 'Middle East', 'Israel', 'Palestine', 'Taiwan', 'India', 'Africa', 'BRICS']
    companies = []
    events = []
    for h in headlines:
        for k in keywords:
            if k.lower() in h.lower():
                events.append((k, h))
        # Simple company extraction: look for uppercase words (tickers or names)
        companies += re.findall(r'\b[A-Z]{2,5}\b', h)
    return companies, events

# Enhanced: News-driven picks logic
def get_news_driven_picks(top_picks, focus_topics=None):
    headlines = fetch_market_news(n=50, focus_topics=focus_topics)
    companies, events = extract_entities_events(headlines)
    # Count company mentions
    company_counts = Counter(companies)
    # Extract tickers from top picks
    top_tickers = extract_tickers_from_text(top_picks)
    # Boost tickers in top picks if mentioned in news
    boosted = [t for t in top_tickers if t in company_counts and company_counts[t] > 0]
    # Add new tickers from news if not in top picks
    extra = [c for c, count in company_counts.items() if c not in top_tickers and count > 1]
    # Compose explanation
    explanations = []
    for t in boosted:
        explanations.append(f"{t}: In top picks and mentioned in recent news headlines.")
    for c in extra:
        explanations.append(f"{c}: Not in top picks but frequently mentioned in recent news.")
    # Highlight major events
    event_lines = [f"- {k}: {h}" for k, h in events]
    return boosted + extra, explanations, event_lines

def filter_outliers(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # TODO: Use NLP/statistics to flag outlier or suspicious posts
    return posts  # For now, no filtering

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

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def analyze_portfolio(portfolio_tickers, amount):
    """Analyze a portfolio of stocks with proper error handling and validation."""
    if not portfolio_tickers or not amount:
        print(Fore.RED + "❌ Please provide both portfolio tickers and amount." + Style.RESET_ALL)
        return

    # Validate tickers and get data
    valid_tickers = []
    ticker_data = {}
    total_value = 0
    
    for ticker in portfolio_tickers:
        data = get_stock_data(ticker)
        if data:
            valid_tickers.append(ticker)
            ticker_data[ticker] = data
            total_value += data['price']
        else:
            print(Fore.YELLOW + f"⚠️ Skipping invalid ticker: {ticker}" + Style.RESET_ALL)
    
    if not valid_tickers:
        print(Fore.RED + "❌ No valid tickers in portfolio." + Style.RESET_ALL)
        return

    # Calculate allocations
    n = len(valid_tickers)
    allocation = amount / n
    allocations = {ticker: allocation for ticker in valid_tickers}
    
    # Calculate portfolio metrics
    pe_ratios = [safe_float(data['pe_ratio']) for data in ticker_data.values()]
    pe_ratios = [x for x in pe_ratios if x is not None]
    div_yields = [safe_float(data['dividend_yield']) for data in ticker_data.values()]
    div_yields = [x for x in div_yields if x is not None]
    
    avg_pe = sum(pe_ratios) / len(pe_ratios) if pe_ratios else None
    avg_div_yield = sum(div_yields) / len(div_yields) if div_yields else None

    # Print portfolio analysis
    print(Fore.CYAN + "\n💼 Portfolio Analysis:" + Style.RESET_ALL)
    print(f"Total Value: ${amount:,.2f}")
    print(f"Number of Positions: {n}")
    print(f"Average Position Size: ${allocation:,.2f}")
    if avg_pe:
        print(f"Average P/E Ratio: {avg_pe:.2f}")
    if avg_div_yield:
        print(f"Average Dividend Yield: {avg_div_yield:.2f}%")

    print(Fore.CYAN + "\nIndividual Positions:" + Style.RESET_ALL)
    print(f"{'Ticker':<8} | {'Allocation':>12} | {'P/E':>8} | {'Div Yield':>10} | {'Sector':<15}")
    print("-" * 65)
    
    def format_float(val, width=8, precision=2, suffix=''):
        try:
            return f"{float(val):>{width}.{precision}f}{suffix}"
        except (TypeError, ValueError):
            return f"{'N/A':>{width}}{suffix}"

    for ticker in valid_tickers:
        data = ticker_data[ticker]
        pe_str = format_float(data['pe_ratio'])
        div_str = format_float(data['dividend_yield'], width=9, precision=2, suffix='%')
        print(f"{ticker:<8} | ${allocation:>10,.2f} | {pe_str} | {div_str} | {data['sector']:<15}")

    # Calculate and display returns
    print(Fore.CYAN + "\nHistorical Returns:" + Style.RESET_ALL)
    periods = {'1mo': 30, '3mo': 90, 'YTD': 'YTD', '1yr': 365}
    
    for ticker in valid_tickers:
        returns = get_period_returns_and_volatility(ticker, periods)
        print(f"\n{ticker}:")
        for period, data in returns.items():
            if data['return'] is not None:
                print(f"  {period}: {data['return']:>6.2f}% (Vol: {data['volatility']:.2f}%)")

    # Generate portfolio critique
    critique_prompt = (
        f"Analyze this portfolio: {valid_tickers}\n"
        f"Total Value: ${amount:,.2f}\n"
        f"Average P/E: {avg_pe:.2f}\n"
        f"Average Dividend Yield: {avg_div_yield:.2f}%\n"
        f"Sectors: {[data['sector'] for data in ticker_data.values()]}\n"
        "Provide a brief critique focusing on diversification, risk, and potential improvements."
    )
    
    critique = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
            {"role": "system", "content": "You are a portfolio analyst."},
            {"role": "user", "content": critique_prompt}
                    ],
        temperature=0.7,
        max_tokens=200
                ).choices[0].message.content.strip()
    
    print(Fore.CYAN + "\n🧐 Portfolio Critique:" + Style.RESET_ALL)
    print(critique)

    # Generate sector breakdown chart
    plot_sector_breakdown(valid_tickers, filename="portfolio_sector_breakdown.png")
    print(Fore.GREEN + f"\n✅ Sector breakdown chart saved to: portfolio_sector_breakdown.png" + Style.RESET_ALL)

def run_all_features(include_social=False, user_profile=None, profile_summary=None):
    """Run all available features of the stock analyzer and generate an HTML report."""
    print(Fore.CYAN + "\n=== Running All Features ===" + Style.RESET_ALL)

    # Get top picks
    print(Fore.CYAN + "\n=== Top Picks ===" + Style.RESET_ALL)
    top_picks = get_top_stock_picks(profile=user_profile['experience'] if user_profile else None, risk_tolerance=user_profile['risk'] if user_profile else None)
    print(top_picks)

    # Compare stocks
    print(Fore.CYAN + "\n=== Compare Stocks ===" + Style.RESET_ALL)
    while True:
        print("Enter tickers to compare (comma-separated) [AAPL,MSFT,GOOGL]: ", end='')
        tickers_input = input().strip() or "AAPL,MSFT,GOOGL"
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        valid_tickers = [t for t in tickers if is_valid_ticker(t)]
        invalid_tickers = [t for t in tickers if not is_valid_ticker(t)]
        if not valid_tickers:
            print(Fore.RED + "❌ No valid tickers entered. Please try again." + Style.RESET_ALL)
            continue
        if invalid_tickers:
            print(Fore.YELLOW + f"⚠️ Skipping invalid tickers: {', '.join(invalid_tickers)}" + Style.RESET_ALL)
        break
    ticker_data = {t: get_stock_data(t) for t in valid_tickers}
    comparisons = []
    summaries = []
    backtests = []
    for i in range(len(valid_tickers)):
        for j in range(i + 1, len(valid_tickers)):
            t1, t2 = valid_tickers[i], valid_tickers[j]
            if t1 in ticker_data and t2 in ticker_data:
                comparison = generate_comparison(t1, ticker_data[t1], t2, ticker_data[t2])
                summary = summarize_sentiment(comparison)
                backtest = get_backtest_data(t1, t2)
                comparisons.append((t1, t2, comparison))
                summaries.append((t1, t2, summary))
                backtests.append((t1, t2, backtest))
                print(Fore.CYAN + f"\n=== {t1} vs {t2} ===" + Style.RESET_ALL)
                print(comparison)
                print(Fore.GREEN + f"\nSummary: {summary}" + Style.RESET_ALL)
                print(Fore.YELLOW + f"\nBacktest (6-month return):" + Style.RESET_ALL)
                print(format_backtest_result(t1, backtest[0]))
                print(format_backtest_result(t2, backtest[1]))

    # Portfolio analysis
    print(Fore.CYAN + "\n=== Portfolio Analysis ===" + Style.RESET_ALL)
    # Loop until user enters at least one valid ticker
    while True:
        print("Enter portfolio tickers (comma-separated): ", end='')
        tickers_input = input().strip()
        if not tickers_input:
            print(Fore.RED + "❌ Please enter at least one ticker." + Style.RESET_ALL)
            continue
        portfolio_tickers = [t.strip().upper() for t in tickers_input.split(",")]
        valid_portfolio_tickers = [t for t in portfolio_tickers if is_valid_ticker(t)]
        invalid_portfolio_tickers = [t for t in portfolio_tickers if not is_valid_ticker(t)]
        if not valid_portfolio_tickers:
            print(Fore.RED + "❌ No valid tickers entered. Please try again." + Style.RESET_ALL)
            continue
        if invalid_portfolio_tickers:
            print(Fore.YELLOW + f"⚠️ Skipping invalid tickers: {', '.join(invalid_portfolio_tickers)}" + Style.RESET_ALL)
        break
    while True:
        print("\nEnter total portfolio value ($): ", end='')
        amount_input = input().strip()
        # Accept commas and dollar signs
        amount_clean = re.sub(r'[^0-9.]', '', amount_input)
        try:
            amount = float(amount_clean)
            break
        except ValueError:
            print(Fore.RED + "❌ Please enter a valid number." + Style.RESET_ALL)
    # Gather portfolio analysis output
    from io import StringIO
    import contextlib
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        analyze_portfolio(valid_portfolio_tickers, amount)
    portfolio_output = buf.getvalue()

    # News (coming soon)
    print(Fore.CYAN + "\n=== News ===" + Style.RESET_ALL)
    news_output = "[News] Feature coming soon!"
    print(news_output)

    # Social media insights
    if include_social:
        print(Fore.CYAN + "\n=== Social Media Insights ===" + Style.RESET_ALL)
        social_output = "[Social] Feature coming soon!"
        print(social_output)
    else:
        print(Fore.CYAN + "\n=== Social Media Insights ===" + Style.RESET_ALL)
        social_output = "Social media insights are disabled. Run with --include-social to enable."
        print(social_output)

    # Generate HTML report
    html = f"""
    <html><head><meta charset='utf-8'><title>GPT-4o Stock Analyzer Report</title>
    <style>
    body {{ font-family: sans-serif; background: #f9f9f9; margin: 0; padding: 0; }}
    .container {{ width: 100vw; overflow-x: auto; white-space: nowrap; }}
    .section {{ display: inline-block; vertical-align: top; width: 900px; margin: 2em 2em 2em 0; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #0001; padding: 2em; }}
    h1, h2 {{ color: #2a4d7a; }}
    pre, code, .textblock {{ white-space: pre-wrap; word-break: break-word; }}
    table {{ max-width: 100%; word-break: break-word; }}
    img {{ max-width: 100%; height: auto; }}
    a {{ color: #2a4d7a; text-decoration: underline; }}
    </style>
    </head><body>
    <div class='container'>
    <div class='section'><h1>GPT-4o Stock Analyzer Report</h1>
    <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
    <b>User Profile:</b> {user_profile if user_profile else ''}<br>
    <b>Profile Summary:</b> {profile_summary if profile_summary else ''}
    </div>
    <div class='section'>
      <h2>Table of Contents</h2>
      <ul>
        <li><a href='#top_picks'>Top Picks</a></li>
        <li><a href='#comparisons'>Comparisons</a></li>
        <li><a href='#summaries'>Summaries</a></li>
        <li><a href='#backtests'>Backtests</a></li>
        <li><a href='#portfolio'>Portfolio Analysis</a></li>
        <li><a href='#news'>News</a></li>
        <li><a href='#social'>Social Media Insights</a></li>
      </ul>
    </div>
    <div class='section' id='top_picks'><h2>Top Picks</h2><div class='textblock'>{top_picks}</div></div>
    <div class='section' id='comparisons'><h2>Comparisons</h2>"""
    for t1, t2, comparison in comparisons:
        html += f"<h3>{t1} vs {t2}</h3><pre>{comparison}</pre>"
    html += "</div>"
    html += "<div class='section' id='summaries'><h2>Summaries</h2>"
    for t1, t2, summary in summaries:
        html += f"<h3>{t1} vs {t2}</h3><pre>{summary}</pre>"
    html += "</div>"
    html += "<div class='section' id='backtests'><h2>Backtests</h2>"
    for t1, t2, backtest in backtests:
        html += f"<h3>{t1} vs {t2}</h3><pre>{t1}: {backtest[0]}%\n{t2}: {backtest[1]}%</pre>"
    html += "</div>"
    html += f"<div class='section' id='portfolio'><h2>Portfolio Analysis</h2><pre>{portfolio_output}</pre></div>"
    html += f"<div class='section' id='news'><h2>News</h2><pre>{news_output}</pre></div>"
    html += f"<div class='section' id='social'><h2>Social Media Insights</h2><pre>{social_output}</pre></div>"
    html += "</div></body></html>"

    html_file = "gpt4o_stock_report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(Fore.GREEN + f"\n✅ HTML report saved to: {html_file}" + Style.RESET_ALL)
    # Open the HTML file automatically
    if platform.system() == "Windows":
        os.startfile(html_file)
    elif platform.system() == "Darwin":
        subprocess.call(["open", html_file])
    elif platform.system() == "Linux":
        subprocess.call(["xdg-open", html_file])

def compare_stocks():
    """Interactive stock comparison feature."""
    print("\nEnter tickers to compare (comma-separated) [AAPL,MSFT,GOOGL]: ", end='')
    tickers_input = input().strip() or "AAPL,MSFT,GOOGL"
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    if len(tickers) < 2:
        print(Fore.RED + "❌ Please enter at least 2 tickers to compare." + Style.RESET_ALL)
        return
        
    # Get data for each ticker
    ticker_data = {}
    for t in tickers:
        data = get_stock_data(t)
        if data:
            ticker_data[t] = data
        else:
            print(Fore.RED + f"❌ Could not get data for {t}" + Style.RESET_ALL)
            return
            
    # Compare each pair of tickers
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            if t1 in ticker_data and t2 in ticker_data:
                comparison = generate_comparison(t1, ticker_data[t1], t2, ticker_data[t2])
                summary = summarize_sentiment(comparison)
                backtest = get_backtest_data(t1, t2)
                
                print(Fore.CYAN + f"\n=== {t1} vs {t2} ===" + Style.RESET_ALL)
                print(comparison)
                print(Fore.GREEN + f"\nSummary: {summary}" + Style.RESET_ALL)
                print(Fore.YELLOW + f"\nBacktest (6-month return):" + Style.RESET_ALL)
                print(format_backtest_result(t1, backtest[0]))
                print(format_backtest_result(t2, backtest[1]))

def format_backtest_result(ticker, result):
    if isinstance(result, (int, float)):
        return f"{ticker}: {result}%"
    elif result == "N/A":
        return f"{ticker}: Data unavailable (check ticker or data source)"
    else:
        return f"{ticker}: {result}"

def analyze_portfolio_interactive():
    """Interactive portfolio analysis feature."""
    print("\nEnter portfolio tickers (comma-separated): ", end='')
    tickers_input = input().strip()
    if not tickers_input:
        print(Fore.RED + "❌ Please enter at least one ticker." + Style.RESET_ALL)
        return
        
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    while True:
        print("\nEnter total portfolio value ($): ", end='')
        amount_input = input().strip()
        try:
            amount = float(amount_input)
            break
        except ValueError:
            print(Fore.RED + "❌ Please enter a valid number." + Style.RESET_ALL)
    
    analyze_portfolio(tickers, amount)

def get_social_insights():
    """Get social media insights for stocks."""
    print("[Social] Feature coming soon!")

def list_trusted_sources():
    """List trusted social media sources."""
    print("\nTrusted Social Media Sources:")
    print("1. Instagram: @verified")
    print("2. Twitter: @verified")
    print("3. LinkedIn: Official Company Pages")
    print("4. Reddit: r/stocks, r/investing")
    print("5. Seeking Alpha: Verified Contributors")

def suggest_new_source():
    """Suggest a new trusted source."""
    print("[Suggest Source] Feature coming soon!")

def prompt_required(prompt_text, valid_options=None):
    while True:
        value = input(prompt_text).strip()
        if not value:
            print(Fore.RED + "This field is required. Please enter a value." + Style.RESET_ALL)
            continue
        if valid_options:
            # Accept case-insensitive input, but store canonical value
            matches = [opt for opt in valid_options if value.lower() == opt.lower()]
            if not matches:
                print(Fore.RED + f"Please enter one of: {', '.join(valid_options)}" + Style.RESET_ALL)
                continue
            return matches[0]
        return value

def prompt_user_profile():
    print(Fore.CYAN + "Let's set up your user profile." + Style.RESET_ALL)
    experience = prompt_required(
        "What is your investing experience? (Novice/Intermediate/Sophisticated): ",
        ["Novice", "Intermediate", "Sophisticated"]
    )
    goal = prompt_required(
        "What is your primary investment goal? (Capital Preservation/Income/Growth/Balanced/All): ",
        ["Capital Preservation", "Income", "Growth", "Balanced", "All"]
    )
    risk = prompt_required(
        "What is your risk profile? (Conservative/Moderate/Aggressive): ",
        ["Conservative", "Moderate", "Aggressive"]
    )
    freeform = input("Briefly describe your investing style, interests, or anything else you want considered (optional): ").strip()
    return {"experience": experience, "goal": goal, "risk": risk, "freeform": freeform}

def get_gpt4_user_profile_summary(profile):
    prompt = (
        f"Summarize this investor profile in one or two sentences, using professional financial language. "
        f"Profile:\n"
        f"- Experience: {profile['experience']}\n"
        f"- Goal: {profile['goal']}\n"
        f"- Risk: {profile['risk']}\n"
        f"- Additional: {profile['freeform']}\n"
        f"Focus on risk appetite, investment objectives, and any unique interests."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional financial advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=120
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(Fore.YELLOW + f"[Warning] Could not generate profile summary: {e}" + Style.RESET_ALL)
        return None

def is_valid_ticker(ticker):
    data = get_stock_data(ticker)
    return data is not None

def main():
    parser = argparse.ArgumentParser(description='GPT-4o Stock Analyzer')
    parser.add_argument('--portfolio', type=str, help='Comma-separated list of portfolio tickers')
    parser.add_argument('--amount', type=float, help='Total portfolio value')
    parser.add_argument('--include-social', action='store_true', help='Include social media analysis')
    args = parser.parse_args()

    print(Fore.CYAN + "\nWelcome to the GPT-4o Financial Analyzer!" + Style.RESET_ALL)
    # User profile onboarding
    user_profile = prompt_user_profile()
    profile_summary = get_gpt4_user_profile_summary(user_profile)
    print(Fore.GREEN + f"\nLoaded user profile: {user_profile['experience']}, goal: {user_profile['goal']}, risk: {user_profile['risk']}" + Style.RESET_ALL)
    if profile_summary:
        print(Fore.CYAN + f"\nProfile Summary: {profile_summary}" + Style.RESET_ALL)

    print(Fore.CYAN + "\n=== Top Picks ===" + Style.RESET_ALL)
    print(get_top_stock_picks(profile=user_profile['experience'], risk_tolerance=user_profile['risk']))
    
    # If portfolio and amount are provided, run portfolio analysis directly
    if args.portfolio and args.amount:
        portfolio_tickers = [t.strip().upper() for t in args.portfolio.split(",") if t.strip()]
        # Validate tickers
        valid_portfolio_tickers = []
        invalid_portfolio_tickers = []
        ticker_data = {}
        total_value = 0
        
        for t in portfolio_tickers:
            data = get_stock_data(t)
            if data:
                valid_portfolio_tickers.append(t)
                ticker_data[t] = data
                total_value += data['price']
            else:
                invalid_portfolio_tickers.append(t)
                
        if invalid_portfolio_tickers:
            print(Fore.RED + f"❌ Invalid tickers removed from portfolio: {', '.join(invalid_portfolio_tickers)}" + Style.RESET_ALL)
            
        portfolio_tickers = valid_portfolio_tickers
        amount = args.amount
        n = len(portfolio_tickers)
        
        if n == 0:
            print(Fore.RED + "❌ No valid tickers provided for portfolio." + Style.RESET_ALL)
            return
            
        print(Fore.CYAN + f"\n💼 Portfolio Simulation for: {', '.join(portfolio_tickers)} (Total: ${amount:,.2f})" + Style.RESET_ALL)
        alloc = {t: amount / n for t in portfolio_tickers}
        print(Fore.CYAN + f"{'Ticker':<8} | {'Allocation ($)':>14} | {'Price':>10} | {'P/E':>8} | {'Div Yld':>8} | {'Sector':<15}" + Style.RESET_ALL)
        print(Fore.CYAN + "-"*65 + Style.RESET_ALL)
        
        for t in portfolio_tickers:
            data = ticker_data[t]
            pe_str = format_float(data['pe_ratio'])
            div_str = format_float(data['dividend_yield'], width=9, precision=2, suffix='%')
            print(f"{t:<8} | {alloc[t]:>14,.2f} | {data['price']:>10.2f} | {pe_str} | {div_str} | {data['sector']:<15}")
        
        # Calculate 1yr return/volatility for each ticker
        periods = {'1yr': 365}
        stats = {t: get_period_returns_and_volatility(t, periods)['1yr'] for t in portfolio_tickers}
        
        print(Fore.CYAN + f"\n{'Ticker':<8} | {'1yr Return':>10} | {'1yr Vol':>8} | {'1yr DD':>8}" + Style.RESET_ALL)
        print(Fore.CYAN + "-"*40 + Style.RESET_ALL)
        
        for t in portfolio_tickers:
            s = stats[t]
            print(f"{t:<8} | {s['return'] if s['return'] is not None else 'N/A':>10} | {s['volatility'] if s['volatility'] is not None else 'N/A':>8} | {s['drawdown'] if s['drawdown'] is not None else 'N/A':>8}")
        
        # Estimate portfolio return/volatility (equal-weighted, no covariance)
        valid_returns = [s['return'] for s in stats.values() if s['return'] is not None]
        valid_vols = [s['volatility'] for s in stats.values() if s['volatility'] is not None]
        port_return = round(sum(valid_returns)/len(valid_returns), 2) if valid_returns else 'N/A'
        port_vol = round(sum(valid_vols)/len(valid_vols), 2) if valid_vols else 'N/A'
        
        print(Fore.CYAN + f"\nEstimated Portfolio 1yr Return: {port_return}%" + Style.RESET_ALL)
        print(Fore.CYAN + f"Estimated Portfolio 1yr Volatility: {port_vol}%" + Style.RESET_ALL)
        
        # GPT critique
        critique_prompt = (
            f"Here is a simulated portfolio: {portfolio_tickers}\n"
            f"Total Value: ${amount:,.2f}\n"
            f"Average Position Size: ${amount/n:,.2f}\n"
            f"1yr Returns: {[stats[t]['return'] for t in portfolio_tickers]}\n"
            f"1yr Volatilities: {[stats[t]['volatility'] for t in portfolio_tickers]}\n"
            f"Estimated Portfolio Return: {port_return}%\n"
            f"Estimated Portfolio Volatility: {port_vol}%\n"
            f"Sectors: {[ticker_data[t]['sector'] for t in portfolio_tickers]}\n"
            f"Please provide a critical analysis of this portfolio, focusing on diversification, risk, and sector concentration. Keep it under 60 words."
        )
        
        critique = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a critical financial analyst."},
                {"role": "user", "content": critique_prompt}
            ],
            temperature=0.7,
            max_tokens=120
        ).choices[0].message.content.strip()
        
        print(Fore.CYAN + "\n🧐 Portfolio Critique:" + Style.RESET_ALL)
        print(critique)

        # Call portfolio analysis functions
        suggest_portfolio_rebalance(portfolio_tickers)
        plot_sector_breakdown(portfolio_tickers)
        return

    # Interactive mode
    while True:
        print("\nWhat would you like to do? top_picks/compare/portfolio/news/social/list_sources/suggest_source/all/exit: ", end='')
        action = input().strip().lower() or 'top_picks'
        
        if action == 'exit':
            break
        elif action == 'all':
            run_all_features(args.include_social, user_profile, profile_summary)
        elif action == 'top_picks':
            print(get_top_stock_picks(profile=user_profile['experience'], risk_tolerance=user_profile['risk']))
        elif action == 'compare':
            compare_stocks()
        elif action == 'portfolio':
            analyze_portfolio_interactive()
        elif action == 'news':
            print("[News] Feature coming soon!")
        elif action == 'social':
            if args.include_social:
                get_social_insights()
            else:
                print("Social media insights are disabled. Run with --include-social to enable.")
        elif action == 'list_sources':
            list_trusted_sources()
        elif action == 'suggest_source':
            suggest_new_source()
        else:
            print(f"Unknown action: {action}")
            print("Available actions: top_picks, compare, portfolio, news, social, list_sources, suggest_source, all, exit")

if __name__ == "__main__":
    main()

# Suppress non-critical warnings (e.g., GLib-GIO, matplotlib, etc.)
if sys.platform.startswith('win'):
    import ctypes
    try:
        # Suppress Windows error dialogs
        ctypes.windll.kernel32.SetErrorMode(1)
    except Exception:
        pass
warnings.filterwarnings("ignore")

# Suppress GLib-GIO and similar warnings on all platforms
os.environ["GIO_USE_VFS"] = "local"
