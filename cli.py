import typer
from user import UserSession
from analysis import StockAnalyzer
from export import Exporter
from news import NewsFetcher
from social import SocialMediaFetcher
from typing import Optional
import time

app = typer.Typer(help="GPT-4o Stock Analyzer Hybrid - A modern stock analysis tool")

VALID_ACTIONS = ["top_picks", "compare", "portfolio", "news", "social", "list_sources", "suggest_source", "all", "exit"]

@app.command()
def start(
    session_file: str = typer.Option("user_session.json", help="Path to user session file."),
    include_social: bool = typer.Option(False, help="Include social media insights from trusted sources.")
):
    """
    Start the GPT-4o Stock Analyzer Hybrid CLI.
    """
    try:
        # Load or create user session
        session = UserSession.load(session_file)
        if not session:
            typer.echo("No saved user session found. Let's set up your profile.")
            # Minimal onboarding for prototype; expand as needed
            profile = typer.prompt("Profile (Beginner/Novice/Intermediate/Advanced/Other)", default="Beginner")
            goal = typer.prompt("Main investing goal (growth/income/safety/excitement/other)", default="growth")
            risk = typer.prompt("Risk tolerance (low/medium/high)", default="medium")
            tone = typer.prompt("Preferred tone (professional/casual/fun)", default="professional")
            accessibility = typer.confirm("Need accessibility features?", default=False)
            mode_type = typer.prompt("Mode (guided/quick)", default="guided")
            session = UserSession(profile, goal, risk, tone, accessibility, mode_type)
            session.save(session_file)
            typer.echo("Profile saved! You can update it anytime.")
        else:
            typer.echo(f"Loaded user profile: {session.profile}, goal: {session.goal}, risk: {session.risk}")

        # Main menu
        while True:
            typer.echo("\nWelcome to the GPT-4o Stock Analyzer Hybrid!")
            action = typer.prompt(
                f"What would you like to do? ({'/'.join(VALID_ACTIONS)})",
                default="top_picks"
            )

            if action == "exit":
                typer.echo("Goodbye!")
                break

            if action not in VALID_ACTIONS:
                typer.echo(f"Unknown action: {action}")
                typer.echo(f"Available actions: {', '.join(VALID_ACTIONS)}")
                continue

            handle_action(action, include_social)

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}")
        raise typer.Exit(1)

def handle_action(action: str, include_social: bool):
    """Handle different CLI actions"""
    try:
        fetcher = SocialMediaFetcher()
        analyzer = StockAnalyzer()
        
        if action == "all":
            typer.echo("\n[Running All Features]")
            actions = ["top_picks", "compare", "portfolio", "news", "social", "list_sources"]
            for act in actions:
                try:
                    typer.echo(f"\n{'='*50}")
                    typer.echo(f"Running: {act}")
                    typer.echo(f"{'='*50}")
                    # Enable social media for the 'all' command
                    handle_action(act, True if act == "social" else include_social)
                    time.sleep(1)  # Small pause between features
                except Exception as e:
                    typer.echo(f"Error running {act}: {str(e)}")
                    continue
            return

        if action == "top_picks":
            typer.echo("\n[Top Stock Picks]")
            picks = analyzer.top_picks()
            typer.echo("Suggested stocks:")
            for pick in picks:
                typer.echo(f"- {pick}")
        elif action == "compare":
            typer.echo("\n[Compare Stocks]")
            tickers = typer.prompt("Enter tickers to compare (comma-separated)", default="AAPL,MSFT,GOOGL").split(",")
            tickers = [t.strip().upper() for t in tickers]
            results = analyzer.compare(tickers)
            
            for ticker, data in results.items():
                if "error" in data:
                    typer.echo(f"\n{ticker}: Error - {data['error']}")
                else:
                    typer.echo(f"\n{ticker} ({data['name']}):")
                    typer.echo(f"  Current Price: ${data['current_price']:,.2f}")
                    typer.echo(f"  Market Cap: ${data['market_cap']:,.2f}")
                    typer.echo(f"  P/E Ratio: {data['pe_ratio']:.2f}")
                    typer.echo(f"  Dividend Yield: {data['dividend_yield']:.2%}")
                    typer.echo(f"  1-Month Change: {data['month_change']:.2f}%")
        elif action == "portfolio":
            typer.echo("\n[Portfolio Analysis]")
            while True:
                try:
                    tickers_input = typer.prompt("Enter your portfolio tickers (comma-separated)", default="AAPL,MSFT,GOOGL")
                    tickers = [t.strip().upper() for t in tickers_input.split(",")]
                    if not tickers or not all(tickers):
                        typer.echo("Error: Please enter at least one valid ticker")
                        continue
                    break
                except Exception:
                    typer.echo("Error: Invalid input format. Please use comma-separated tickers")
                    continue

            while True:
                try:
                    amount_str = typer.prompt("Enter total portfolio value", default="10000")
                    amount = float(amount_str)
                    if amount <= 0:
                        typer.echo("Error: Portfolio value must be greater than 0")
                        continue
                    break
                except ValueError:
                    typer.echo("Error: Please enter a valid number")
                    continue
            
            results = analyzer.portfolio_analysis(tickers, amount)
            typer.echo(f"\n{results['summary']}")
            
            if "stocks" in results:
                typer.echo("\nIndividual Stock Analysis:")
                for ticker, data in results["stocks"].items():
                    if "error" not in data:
                        typer.echo(f"\n{ticker} ({data['name']}):")
                        typer.echo(f"  Current Price: ${data['current_price']:,.2f}")
                        typer.echo(f"  P/E Ratio: {data['pe_ratio']:.2f}")
                        typer.echo(f"  Dividend Yield: {data['dividend_yield']:.2%}")
        elif action == "news":
            typer.echo("[News] Feature coming soon!")
        elif action == "social":
            if not include_social:
                typer.echo("Social media insights are disabled. Run with --include-social to enable.")
            else:
                typer.echo("\n[Social Media Insights]")
                typer.echo("Fetching insights from trusted, verified sources only (Instagram, Twitter/X, TikTok)...")
                typer.echo("(For demo: using mock data. Only public, vetted sources are used. Bias and data poisoning are mitigated by source curation and aggregation.)")
                posts = fetcher.fetch_from_trusted("instagram", limit=2) + fetcher.fetch_from_trusted("twitter", limit=2)
                insights = fetcher.aggregate_insights(posts)
                typer.echo(f"\nAggregated Social Media Insights: {insights['summary']}")
        elif action == "list_sources":
            typer.echo("\n[Trusted Social Media Sources]")
            fetcher.list_trusted_sources()
        elif action == "suggest_source":
            typer.echo("\n[Suggest a Trusted Source]")
            platform = typer.prompt("Platform (instagram/twitter/tiktok)", default="instagram")
            handle = typer.prompt("Account handle (without @)")
            fetcher.suggest_trusted_source(platform, handle)
            typer.echo("Thank you! Your suggestion will be reviewed for trust and compliance.")
    except Exception as e:
        typer.echo(f"Error in handle_action: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
