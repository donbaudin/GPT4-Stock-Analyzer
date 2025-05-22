from typing import List, Optional

# Utility: Validate a stock ticker symbol

def validate_ticker(ticker: str) -> bool:
    """
    Return True if the ticker symbol is valid (stub).
    """
    # TODO: Implement actual validation logic
    return True

# Utility: Format a number as currency

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format a float as a currency string (stub).
    """
    # TODO: Implement actual formatting logic
    return f"${amount:,.2f} {currency}"

# Utility: Prompt user with choices (for CLI)

def prompt_with_choices(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    Prompt the user to select from a list of choices (stub).
    """
    # TODO: Implement actual prompt logic
    return default or choices[0]

# Utility: Color text for accessibility

def color_text(text: str, color: str = "default") -> str:
    """
    Return colored text for CLI output (stub).
    """
    # TODO: Implement actual color logic
    return text
