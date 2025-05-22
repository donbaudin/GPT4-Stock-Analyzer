# GPT4O Stock Analyzer

A comprehensive stock analysis tool that combines financial data analysis, news sentiment analysis, and social media insights to provide detailed stock comparisons and portfolio analysis.

## Features

- **Stock Comparison**: Compare multiple stocks with detailed financial metrics
- **Portfolio Analysis**: Analyze your investment portfolio with sector breakdowns and performance metrics
- **News Analysis**: Track and analyze news sentiment for stocks
- **Social Media Integration**: Monitor social media sentiment from Twitter and Reddit
- **Data Export**: Export analysis results in multiple formats (JSON, MD, PNG)
- **Interactive CLI**: User-friendly command-line interface with rich formatting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt4o_stock_analyzer.git
cd gpt4o_stock_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

## Usage

### Basic Stock Comparison
```bash
python gpt4o_stock_comparator.py compare AAPL MSFT GOOGL
```

### Portfolio Analysis
```bash
python gpt4o_stock_comparator.py portfolio --stocks AAPL MSFT GOOGL --weights 0.4 0.3 0.3
```

### News Analysis
```bash
python gpt4o_stock_comparator.py news AAPL --days 7
```

### Social Media Analysis
```bash
python gpt4o_stock_comparator.py social AAPL --platform twitter
```

## Project Structure

- `gpt4o_stock_comparator.py`: Main script for stock comparison and analysis
- `analysis.py`: Core analysis functions
- `news.py`: News gathering and sentiment analysis
- `social.py`: Social media data collection and analysis
- `export.py`: Data export functionality
- `cli.py`: Command-line interface implementation
- `utils.py`: Utility functions
- `tests/`: Test suite

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 integration
- Yahoo Finance for financial data
- Twitter and Reddit APIs for social media data
- All contributors who have helped improve this project
