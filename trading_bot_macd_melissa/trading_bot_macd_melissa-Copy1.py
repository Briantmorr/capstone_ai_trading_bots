import datetime as dt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from logger_setup import get_bot_logger
from dotenv import load_dotenv
import os

# Load variables from .env into the environment
load_dotenv()

# Bot name (same as directory name)
BOT_NAME = "trading_bot_macd_melissa"
logger = get_bot_logger(BOT_NAME)

class Macd_trading_bot:

    def __init__(self):
        """Initialize the momentum strategy bot with API credentials and settings."""
        # API Keys from environment variables
        self.api_key = os.environ[f"{BOT_NAME}_ALPACA_API_KEY"]
        self.api_secret = os.environ[f"{BOT_NAME}_ALPACA_API_SECRET"]
        
        self.client = StockHistoricalDataClient(self.api_key, self.api_secret)
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)

        # setting class variables
        self.sequence_length = 50  
        self.symbols = ["AAPL", "GOOGL", "AMZN", "META", "MSFT", "NVDA"]

        logger.info(f"Momentum strategy bot {BOT_NAME} initialized with {len(self.symbols)} symbols")

    def compute_macd(self, data, fast=12, slow=26, signal=9):
        data['ema_fast'] = data['close'].ewm(span=fast, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=slow, adjust=False).mean()
        data['macd'] = data['ema_fast'] - data['ema_slow']
        data['signal'] = data['macd'].ewm(span=signal, adjust=False).mean()
        data['histogram'] = data['macd'] - data['signal']
        return data
    
    def compute_atr(self, data, period=14):
        """Calculates the Average True Range (ATR) for volatility-based trading."""
        data['high-low'] = data['high'] - data['low']
        data['high-close'] = np.abs(data['high'] - data['close'].shift(1))
        data['low-close'] = np.abs(data['low'] - data['close'].shift(1))
        data['true_range'] = data[['high-low', 'high-close', 'low-close']].max(axis=1)
        data['atr'] = data['true_range'].rolling(window=period).mean()
        return data

    def fetch_historical_data(self):
        start_date = dt.datetime.now() - dt.timedelta(days=365)
        end_date = dt.datetime.now() - dt.timedelta(minutes=20)

        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request_params).df
        stock_dic = {}
        scaler = MinMaxScaler(feature_range=(0, 1))

        for symbol in self.symbols:
            df = bars[bars.index.get_level_values(0) == symbol].copy()
            df.reset_index(inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = self.compute_macd(df)
            df = self.compute_atr(df)
            df[['close', 'macd', 'signal']] = scaler.fit_transform(df[['close', 'macd', 'signal']])
            stock_dic[symbol] = df
        return stock_dic
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])  # Predicting closing price
        return np.array(X), np.array(y)
    
    def predict_and_execute(self, model, stock_dic):
        account = self.trading_client.get_account()
        cash_available = float(account.cash)
        risk_per_trade = 0.10 * cash_available  # Risk 10% of available capital

        for symbol in self.symbols:
            stock_dic[symbol] = self.compute_atr(stock_dic[symbol])
            test_data = stock_dic[symbol][['close', 'macd', 'signal']].values
            X_test, y_test = self.create_sequences(test_data)
            pred = model.predict(X_test)

            last_actual_price = stock_dic[symbol]['close'].iloc[-1]
            last_predicted_price = pred[-1][0]
            atr = stock_dic[symbol]['atr'].iloc[-1]

            stop_loss_distance = 2 * atr
            quantity = int(risk_per_trade / stop_loss_distance)

            logger.info(f"Processing {symbol}: ATR: {atr:.2f} | Stop-Loss Distance: {stop_loss_distance:.2f} | Position Size: {quantity} shares")
            logger.info(f"Predicted Price: {last_predicted_price:.2f} | Actual Price: {last_actual_price:.2f}")

            if quantity > 0:
                if last_predicted_price > last_actual_price:
                    order = MarketOrderRequest(symbol=symbol, qty=quantity, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                    self.trading_client.submit_order(order)
                    logger.info(f"BUY {quantity} shares of {symbol} at market price")
                elif last_predicted_price < last_actual_price:
                    order = MarketOrderRequest(symbol=symbol, qty=quantity, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                    self.trading_client.submit_order(order)
                    logger.info(f"SELL {quantity} shares of {symbol} at market price")

        logger.info("Trading bot executed based on ATR, LSTM predictions, and MACD analysis.")

def main():
    logger.info(f"Starting {BOT_NAME}")
    try:
        bot = Macd_trading_bot()
        stock_dic = bot.fetch_historical_data()
        model = bot.build_model(stock_dic)
        bot.predict_and_execute(model, stock_dic)
        logger.info(f"{BOT_NAME} completed successfully")
    except Exception as e:
        logger.error(f"Error running {BOT_NAME}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
