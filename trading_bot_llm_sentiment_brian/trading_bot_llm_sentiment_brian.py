import os
import sys
from datetime import datetime, timedelta
from openai import OpenAI
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pathlib import Path
import requests
sys.path.append(str(Path(__file__).parent.parent))
from logger_setup import get_bot_logger

# Bot name (same as directory name)
BOT_NAME = "trading_bot_llm_sentiment_brian"
logger = get_bot_logger(BOT_NAME)

class TradingBotLLMSentiment:
    """
    A trading bot that leverages an LLM for sentiment analysis,
    integrating the sentiment score into a trading decision based on a fixed daily budget.
    Trades the MAG 7 (minus Tesla) stocks.
    """
    
    def __init__(self):
        """Initialize the trading bot with API credentials and settings."""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be provided in environment variables")
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        self.symbols = ['AAPL', 'MSFT', 'META', 'GOOGL', 'AMZN', 'NVDA']
        self.timeframe = TimeFrame.Day
        
        # Trading parameters for fixed daily budget strategy
        self.daily_budget_percent = 0.05  # Use 5% of available cash per day for new trades
        
        # Sentiment thresholds for generating signals
        self.sentiment_buy_threshold = 0.2   # Buy if sentiment is at or above 0.2
        self.sentiment_sell_threshold = -0.2   # Sell if sentiment is at or below -0.2
        
        logger.info(f"Trading bot {BOT_NAME} initialized with symbols: {self.symbols}")
    
    def get_account_info(self):
        """Retrieve and display account information."""
        account = self.trading_client.get_account()
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Cash: ${account.cash}")
        logger.info(f"Portfolio value: ${account.portfolio_value}")
        logger.info(f"Buying power: ${account.buying_power}")
        return account
    
    def get_positions(self):
        """Get current positions."""
        positions = self.trading_client.get_all_positions()
        for position in positions:
            logger.info(f"Position: {position.symbol}, Qty: {position.qty}, Market value: ${position.market_value}")
        return positions
    
    def get_historical_data(self, symbol, days=50):
        """Fetch historical stock data."""
        end = datetime.now()
        start = end - timedelta(days=days)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self.timeframe,
            start=start,
            end=end
        )
        
        bars = self.data_client.get_stock_bars(request_params)
        df = bars.df
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None
            
        df = df.reset_index().sort_values(by=["timestamp"])
        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df

    def get_news_articles(self, symbol):
        """
        Fetch the 3 latest news articles for the given stock symbol using NewsAPI.
        Requires the NEWS_API_KEY environment variable to be set.
        Returns a list of tuples containing (title, url) for each article.
        """
        news_api_key = os.getenv("NEWS_API_KEY")
        if not news_api_key:
            logger.warning(f"NEWS_API_KEY is not set in environment variables for {symbol}.")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,          # Search query for the symbol
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 3         # Limit to 3 articles
        }
        
        try:
            response = requests.get(url, params=params, headers={"Authorization": news_api_key})
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            articles = []
            if data.get("status") == "ok":
                for article in data.get("articles", []):
                    # Store both title and URL
                    articles.append((article["title"], article["url"]))
                return articles
            else:
                logger.warning(f"News API error for {symbol}: {data.get('message')}")
                return []
        except Exception as e:
            logger.error(f"Error fetching news articles for {symbol}: {e}")
            return []
    
    def get_sentiment_signal(self, symbol):
        """
        Use OpenAI API to analyze three news articles and return a sentiment score between -1 and 1.
        Logs the URLs of the articles used for analysis.
        """
        article_data = self.get_news_articles(symbol)
        if not article_data:
            logger.warning(f"No articles found for {symbol}. Defaulting sentiment to 0.")
            return 0.0
        
        # Extract titles and URLs
        titles = [article[0] for article in article_data]
        urls = [article[1] for article in article_data]
        
        # Log the URLs
        logger.info(f"News articles for {symbol}:")
        for i, url in enumerate(urls, 1):
            logger.info(f"  Article {i}: {url}")
        
        combined_news = "\n".join(titles)
        prompt = (
            f"Analyze the sentiment of the following news articles and return ONLY a single number between -1 and 1 "
            f"(where -1 is very negative, 0 is neutral, and 1 is very positive). Do not include any explanation, just the number.\n"
            f"Articles:\n{combined_news}"
        )
        
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.0
            )
            sentiment_text = response.choices[0].message.content.strip()
            try:
                sentiment_score = float(sentiment_text)
            except ValueError:
                logger.warning(f"Could not parse sentiment score '{sentiment_text}' for {symbol}. Defaulting to 0.")
                sentiment_score = 0.0
            logger.info(f"Sentiment score for {symbol}: {sentiment_score}")
            return sentiment_score
        except Exception as e:
            logger.error(f"Error calling OpenAI API for {symbol}: {e}")
            return 0.0
    
    def submit_order(self, symbol, side, qty):
        """Submit a market order."""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            logger.info(f"Order placed: {side} {qty} shares of {symbol}")
            logger.info(f"Order ID: {order.id}")
            return order
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {e}")
            return None
    
    def run_strategy(self):
        """
        Run the trading strategy:
          - Use OpenAI-driven sentiment signals for each stock based on news articles.
          - Apply a fixed daily budget to allocate funds for new buys.
          - Sell positions if negative sentiment is detected.
        """
        logger.info("Running LLM sentiment strategy...")
        
        # Simplified check for market open (example only)
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour >= 16:
            logger.info("Market is closed. Running in simulation mode.")
        
        # Get current positions and account info
        positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
        account = self.get_account_info()
        available_cash = float(account.cash)
        daily_budget = available_cash * self.daily_budget_percent
        logger.info(f"Daily budget for new trades: ${daily_budget:.2f}")
        
        # Collect buy and sell signals based on sentiment analysis
        buy_signals = {}
        sell_signals = {}
        
        # Run sentiment analysis once per symbol
        for symbol in self.symbols:
            sentiment = self.get_sentiment_signal(symbol)
            
            # Determine action based on sentiment and current positions
            if sentiment >= self.sentiment_buy_threshold:
                # Buy signal: positive sentiment 
                buy_signals[symbol] = sentiment
            elif sentiment <= self.sentiment_sell_threshold and symbol in positions:
                # Sell signal: negative sentiment and currently held
                sell_signals[symbol] = sentiment
        
        # Process buy signals
        if buy_signals:
            allocation_per_stock = daily_budget / len(buy_signals)
            logger.info(f"Buy signals: {buy_signals}. Allocation per stock: ${allocation_per_stock:.2f}")
            
            for symbol, sentiment in buy_signals.items():
                # Get latest price to determine share quantity
                df = self.get_historical_data(symbol, days=5)
                if df is None:
                    continue
                current_price = df['close'].iloc[-1]
                qty = max(1, int(allocation_per_stock / current_price))
                logger.info(f"Executing BUY for {symbol}: {qty} shares at ${current_price:.2f} per share.")
                self.submit_order(symbol, OrderSide.BUY, qty)
        else:
            logger.info("No buy signals detected today.")
        
        # Process sell signals
        if sell_signals:
            logger.info(f"Sell signals: {sell_signals}")
            
            for symbol, sentiment in sell_signals.items():
                position = positions[symbol]
                position_qty = float(position.qty)
                logger.info(f"Executing SELL for {symbol}: {position_qty} shares with sentiment {sentiment}.")
                self.submit_order(symbol, OrderSide.SELL, position_qty)
        else:
            logger.info("No sell signals detected today.")
        
        # Log holdings with neutral sentiment (neither buy nor sell)
        for symbol in positions:
            if symbol not in sell_signals:
                logger.info(f"Holding {symbol} with neutral or positive sentiment.")
        
        logger.info("Strategy execution completed.")

def main():
    """Main function to run the trading bot."""
    logger.info(f"Starting {BOT_NAME}")
    try:
        bot = TradingBotLLMSentiment()
        bot.run_strategy()
        logger.info(f"{BOT_NAME} completed successfully")
    except Exception as e:
        logger.error(f"Error running {BOT_NAME}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
