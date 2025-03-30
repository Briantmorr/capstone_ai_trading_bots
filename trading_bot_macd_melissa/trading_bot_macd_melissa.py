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
    
    # Fetch historical stock data for MAG6
    def fetch_historical_data(self):
        start_date = dt.datetime.now() - dt.timedelta(days=365)
        # End needs to be > 15 minutes to prevent api call failing in free tier
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
            df[['close', 'macd', 'signal']] = scaler.fit_transform(df[['close', 'macd', 'signal']])
            stock_dic[symbol] = df
        return stock_dic
    

    def create_sequences(self, data):
            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:i + self.sequence_length])
                y.append(data[i + self.sequence_length, 0])  # Predicting closing price
            return np.array(X), np.array(y)
    
    def build_model(self, stock_dic):
        X_train, y_train = [], []
        for symbol in self.symbols:
            train_data = stock_dic[symbol][['close', 'macd', 'signal']].values
            X, y = self.create_sequences(train_data)
            X_train.append(X)
            y_train.append(y)

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0).reshape(-1, 1)  # Ensure y_train is 2D

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 3)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=16)

        return model

    def predict_and_execute(self, model, stock_dic):
        # Predict future prices and execute trades
        predictions = {}
        for symbol in self.symbols:
            test_data = stock_dic[symbol][['close', 'macd', 'signal']].values
            X_test, y_test = self.create_sequences(test_data)
            pred = model.predict(X_test)
            predictions[symbol] = pred
            
            last_actual_price = stock_dic[symbol]['close'].iloc[-1]
            last_predicted_price = pred[-1][0]
            
            if last_predicted_price > last_actual_price:  # Buy condition
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
                self.trading_client.submit_order(order)
                logger.info(f"BUY Order placed for {symbol}")
            elif last_predicted_price < last_actual_price:  # Sell condition
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
                self.trading_client.submit_order(order)
                logger.info(f"SELL Order placed for {symbol}")

        logger.info("Trading bot executed based on LSTM predictions and MACD analysis.")

def main():
    """Main function to run the momentum strategy bot."""
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