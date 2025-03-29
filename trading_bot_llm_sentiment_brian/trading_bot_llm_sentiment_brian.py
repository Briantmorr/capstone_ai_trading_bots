import os
import sys
from datetime import datetime, timedelta
from openai import OpenAI
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
from datetime import timedelta
from pathlib import Path
import numpy as np
import finnhub
sys.path.append(str(Path(__file__).parent.parent))
from logger_setup import get_bot_logger

# Bot name (same as directory name)
BOT_NAME = "trading_bot_llm_sentiment_brian"
logger = get_bot_logger(BOT_NAME)

class TradingBotLLMSentiment:
    # Constant for the combined historical data filename
    COMBINED_HISTORICAL_FILENAME = f"{BOT_NAME}/data/combined_historical_with_daily_sentiment.csv"
    
    def __init__(self):
        """Initialize the trading bot with API credentials and settings."""
        self.symbols = ['AAPL', 'MSFT', 'META', 'GOOGL', 'AMZN', 'NVDA']
        self.timeframe = TimeFrame.Day
        self.model_name = 'lstm_combined_model_2025-03-27.keras'
        # Trading parameters for fixed daily budget strategy
        self.daily_budget_percent = 0.05  # Use 5% of available cash per day for new trades
        self.trading_threshold = 0.01   # trade when prediction is this % different from actual
        self.time_series_length = 30

        # Initialize clients
        self.api_key = os.environ['ALPACA_API_KEY']
        self.api_secret = os.environ['ALPACA_API_SECRET']
        self.news_api_key = os.environ["FINNHUB_API_KEY"]
        self.openai_api_key = os.environ["OPENAI_API_KEY"]

        if not self.api_key or not self.api_secret or not self.news_api_key or not self.openai_api_key:
            raise ValueError("API key and secret must be provided in environment variables")
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.news_client = finnhub.Client(api_key=self.news_api_key)

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
    
    def get_historical_data(self, symbol, days=30):
        """Fetch historical stock data."""
        end = datetime.now()
        start = end - timedelta(days=int(days))
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

    def get_news_articles(self, symbol, article_count, to_date, lookback_range):
        """
        Fetch the latest news articles for the given stock symbol using Finnhub.
        Returns a list of dictionaries containing the summary and url for each article.
        """
        to_date_str = to_date.strftime('%Y-%m-%d')
        from_date_str = (to_date - lookback_range).strftime('%Y-%m-%d')
        news = self.news_client.company_news(symbol, _from=from_date_str, to=to_date_str)
        top_mapped_news = [{'summary': f"{news['headline']}: {news['summary']}", 'url': news['url']} 
                           for news in news[:article_count]]
        return top_mapped_news
       
    def get_sentiment_signal(self, symbol, article_count=5, to_date=datetime.today().date(), lookback_range=timedelta(days=2)):
        """
        Use OpenAI API to analyze news articles and return a sentiment score between -1 and 1.
        """
        news_data = self.get_news_articles(symbol, article_count, to_date, lookback_range)
        if not news_data:
            logger.warning(f"No articles found for {symbol}. Defaulting sentiment to 0.")
            return 0.0
        
        summary = [news['summary'] for news in news_data]
        urls = [news['url'] for news in news_data]
        
        logger.info(f"News articles for {symbol}:")
        for i, url in enumerate(urls, 1):
            logger.info(f"  Article {i}: {url}")
        
        combined_news = "\n".join(summary)
        prompt = (
            f"Analyze the sentiment of the following news articles and return ONLY a single number between -1 and 1 "
            f"(where -1 is very negative, 0 is neutral, and 1 is very positive). Do not include any explanation, just the number.\n"
            f"We are only evaluating sentiment for symbol {symbol}. Ignore news about other symbols. Articles:\n{combined_news}"
        )
        
        try:
            client = self.openai_client
    
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
    
    def load_and_update_sentiment_data(self, days=30, sentiment_file=None):
        """
        Load combined historical price and sentiment data for a given symbol from file if it exists.
        The function always updates sentiment data before prediction:
          1. It fetches the last `days` of price data from Alpaca.
          2. It identifies which dates are missing in the combined CSV.
          3. For each missing date, it finds the corresponding price data (with high, low, etc.),
             fetches sentiment for that day, and appends a new row.
          4. The updated combined DataFrame is re-saved and only the most recent `days` rows are returned.
        
        Args:
            days (int): Number of recent days to update and return (default 30).
            sentiment_file (str): Path to the CSV file containing combined historical data.
                                    Defaults to the class constant.
        
        Returns:
            DataFrame: Updated combined historical data with sentiment for the most recent `days` days.
        """
      

        if sentiment_file is None:
            sentiment_file = self.COMBINED_HISTORICAL_FILENAME
        
        # Attempt to load existing combined historical data.
        try:
            combined_df = pd.read_csv(sentiment_file, parse_dates=['timestamp'])
        except FileNotFoundError:
            logger.error(f"File {sentiment_file} not found. Collecting combined data for the past {days} days...")
            return pd.DataFrame()

        
        updates = []
        for symbol in self.symbols:
            # Fetch the latest `days` of price data from Alpaca.
            price_df = self.get_historical_data(symbol, days=days)
            if price_df is None:
                logger.info("No price data found to update combined data.")
                return combined_df.tail(days)
            
            price_df = price_df.copy()
            price_df['date'] = pd.to_datetime(price_df['timestamp']).dt.date
            
            # Convert price dates to string for set operations.
            price_dates = set([d.strftime('%Y-%m-%d') for d in price_df['date'].unique().tolist()])
            # Filter combined_df for the given symbol.
            filtered_df = combined_df[combined_df['symbol'] == symbol]
            combined_dates = set(filtered_df['date'].unique().tolist())
            
            # Identify dates in price data that are missing in our combined file.
            missing_dates = sorted(list(price_dates - combined_dates))
            if missing_dates:
                logger.info(f"Missing combined data for dates: {missing_dates}. Fetching updates...")
                for date in missing_dates:
                    # Convert string to date for filtering.
                    # target_date = pd.to_datetime(date).date()
                    price_row_df = price_df[price_df['date'] == date]
                    if price_row_df.empty:
                        continue
                    articles = 5
                    news_date = pd.to_datetime(date)
                    lookback_range = timedelta(days=1)
                    sentiment_value = self.get_sentiment_signal(symbol, articles, news_date, lookback_range)
                    # Add the sentiment info.
                    price_row_df['sentiment'] = sentiment_value
                    # Append this row to the combined DataFrame.
                    combined_df = pd.concat([combined_df, price_row_df], ignore_index=True)
                    updates.append(f"{symbol}:{date}")
            
        if updates:
            combined_df.sort_values(by="timestamp", inplace=True)
            combined_df.to_csv(sentiment_file, index=False)
            logger.info(f"updated: {updates}")
            logger.info(f"Combined historical data updated and saved to {sentiment_file}")
        else:
            logger.info("Combined historical data is up-to-date.")

        # Return only the most recent x `days` of data.
        distinct_dates = combined_df['date'].unique()       
        last_30_dates = distinct_dates[-30:]
        updated_df = combined_df[combined_df['date'].isin(last_30_dates)]
        
        return updated_df
    
    def get_signals(self):
        """
        Generate trading signals by comparing the predicted closing price to the latest quote for each symbol.
        
        Returns:
            dict: A dictionary where each key is a symbol and each value is a dict with:
                - 'signal': "BUY", "SELL", or "HOLD"
                - 'predicted': predicted closing price (float)
                - 'current': current price (float)
        """
        signals = {}
        # Create a multi-symbol request for the latest quotes.
        multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=self.symbols)
        latest_quotes = self.data_client.get_stock_latest_quote(multisymbol_request_params)
        
        # Build positions dictionary (for checking sell conditions).
        positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
        
        for symbol in self.symbols:
            try:
                current_quote = latest_quotes[symbol]
            except KeyError:
                logger.warning(f"No quote available for {symbol}. Skipping signal generation.")
                continue
            
            # Compute current price as the average of ask and bid if ask is valid; otherwise, use bid.
            if current_quote.ask_price > 0:
                current_price = (current_quote.ask_price + current_quote.bid_price) / 2.0
            else:
                current_price = current_quote.bid_price
            
            predicted_price = self.predict_todays_closing_price_enriched(symbol)
            if predicted_price is None:
                continue
            
            # Generate signals if the predicted price deviates by more than 1% from the current price.
            if predicted_price > current_price * 1.01:
                signals[symbol] = {"signal": "BUY", "predicted": predicted_price, "current": current_price}
            elif predicted_price < current_price * 0.99:
                # Only generate a SELL signal if we already hold the asset.
                if symbol in positions:
                    signals[symbol] = {"signal": "SELL", "predicted": predicted_price, "current": current_price}
                else:
                    signals[symbol] = {"signal": "HOLD", "predicted": predicted_price, "current": current_price}
            else:
                signals[symbol] = {"signal": "HOLD", "predicted": predicted_price, "current": current_price}
            
            logger.info(f"{symbol}: current ${current_price:.2f}, predicted ${predicted_price:.2f} => signal {signals[symbol]['signal']}")
        
        return signals


    def execute_trading_strategy(self, signals):
        """
        Execute trading orders based on the provided signals.
        
        For BUY signals:
        - Allocate a portion of the daily budget.
        For SELL signals:
        - Sell the entire position.
        
        Args:
            signals (dict): Dictionary of signals generated by get_signals().
        """
        account = self.get_account_info()
        available_cash = float(account.cash)
        daily_budget = available_cash * self.daily_budget_percent
        logger.info(f"Daily budget for new trades: ${daily_budget:.2f}")
        
        # Build positions and open orders dictionaries.
        positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
        open_orders = {}
        try:
            orders = self.trading_client.list_orders(status="open")
            for order in orders:
                open_orders[order.symbol] = order
        except Exception as e:
            logger.error(f"Error retrieving open orders: {e}")
            open_orders = {}
        
        # Process each signal.
        # Count BUY signals to compute allocation per stock.
        buy_symbols = [s for s, data in signals.items() if data["signal"] == "BUY"]
        if buy_symbols:
            allocation_per_stock = daily_budget / len(buy_symbols)
        else:
            allocation_per_stock = 0
        
        for symbol, signal_data in signals.items():
            sig = signal_data["signal"]
            if sig == "BUY":
                # Get the latest price from the signal data.
                current_price = signal_data["current"]
                qty = max(1, int(allocation_per_stock / current_price))
                logger.info(f"Executing BUY for {symbol}: {qty} shares at ${current_price:.2f} per share.")
                self.submit_order(symbol, OrderSide.BUY, qty)
            elif sig == "SELL":
                # Check if there's an open order.
                if symbol in open_orders:
                    logger.info(f"Skipping SELL for {symbol}: Already has an open order.")
                    continue
                # Sell the entire held position.
                if symbol in positions:
                    position_qty = float(positions[symbol].qty)
                    logger.info(f"Executing SELL for {symbol}: {position_qty} shares at current price ${signal_data['current']:.2f}.")
                    self.submit_order(symbol, OrderSide.SELL, position_qty)
                else:
                    logger.info(f"Skipping SELL for {symbol}: No position held.")
            else:
                logger.info(f"Holding {symbol}: no trading action required.")
        
        logger.info("Strategy execution completed.")


    def predict_todays_closing_price_enriched(self, symbol):
        """
        Predict today's closing price for the given symbol using enriched price data that includes sentiment.
        
        Process:
        1. Load and update historical sentiment data.
        2. Filter the enriched data for the given symbol.
        3. Use the last TIME_SERIES_LENGTH rows (the most recent trading days) as the input sequence.
        4. Scale the sequence and predict today's closing price using the trained model.
        
        Returns:
            float or None: The predicted closing price for today, or None if not enough data.
        """
        # Load the scaler
        with open(f"{BOT_NAME}/data/scaler.pkl", 'rb') as f:
            SCALER = pickle.load(f)
        
        # Load the trained model.
        MODEL = load_model(f"{BOT_NAME}/data/models/{self.model_name}")
        
        # Step 1: Load and update historical sentiment data.
        enriched_df = self.load_and_update_sentiment_data(self.time_series_length)
        if enriched_df is None or enriched_df.empty:
            print("Failed to load sentiment data.")
            return None

        # Step 2: Filter for the specific symbol and sort by timestamp.
        symbol_df = enriched_df[enriched_df['symbol'] == symbol].sort_values(by="timestamp")
        if symbol_df.empty:
            print(f"No data available for {symbol}.")
            return None

        # Step 3: Check if there are at least TIME_SERIES_LENGTH rows.
        if len(symbol_df) < self.time_series_length:
            print("Not enough data to form a prediction sequence.")
            return None
        else:
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'sentiment']
            latest_seq = symbol_df.iloc[-self.time_series_length:][feature_columns].values

        # Step 4: Reshape and scale the sequence.
        latest_seq_2d = latest_seq.reshape(-1, len(feature_columns))
        latest_seq_scaled_2d = SCALER.transform(latest_seq_2d)
        latest_seq_scaled = latest_seq_scaled_2d.reshape(1, self.time_series_length, len(feature_columns))
        
        # Predict using the trained model.
        predicted_price = MODEL.predict(latest_seq_scaled)
        predicted_value = predicted_price[0][0]
        print(f"Predicted closing price for {symbol} is {predicted_value:.2f}")
        return predicted_value
    

def main():
    """Main function to run the trading bot."""
    logger.info(f"Starting {BOT_NAME}")
    try:
        bot = TradingBotLLMSentiment()
        signals = bot.get_signals()
        print(signals)
        bot.execute_trading_strategy(signals)
        logger.info(f"{BOT_NAME} completed successfully")
    except Exception as e:
        logger.error(f"Error running {BOT_NAME}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
