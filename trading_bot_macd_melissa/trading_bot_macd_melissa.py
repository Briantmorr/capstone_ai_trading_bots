#import necessary libraries
import os
import time
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
from alpaca.data.timeframe import TimeFrame,TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from logger_setup import get_bot_logger
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv("trading_bot_macd_melissa.env")

# Bot name (same as directory name)
BOT_NAME = "trading_bot_macd_melissa"
logger = get_bot_logger(BOT_NAME)

class Macd_trading_bot:
    def __init__(self):
        """Initialize the strategy bot with API credentials and settings."""
        self.api_key = os.getenv('BOT_API_KEY_1')
        self.api_secret = os.getenv('BOT_API_SECRET_1')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing API credentials. Please set BOT_API_KEY_1 and BOT_API_SECRET_1.")
        
        self.client = StockHistoricalDataClient(self.api_key, self.api_secret)
        # paper=True for simulated live trading; if market is open and you have live creds, use paper=False
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        
        # Set class variables
        self.sequence_length = 50  
        self.symbols = ["AAPL", "GOOGL", "AMZN", "META", "MSFT", "NVDA"]
        
        logger.info(f"Momentum strategy bot {BOT_NAME} initialized with {len(self.symbols)} symbols")
    
    def is_market_open(self):
        """Return True if the market is open, using Alpaca's market clock."""
        clock = self.trading_client.get_clock()
        return clock.is_open
    
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
        start_date = dt.datetime.now() - dt.timedelta(days=45)
        end_date = dt.datetime.now() - dt.timedelta(minutes=16)

        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbols,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),  # 15-minute bars
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
    
    def build_model(self, stock_dic):
        """
        Build and train an LSTM model on historical data from all symbols.
        The model uses past sequences of [close, macd, signal] to predict the future closing price.
        """
        X_train_list = []
        y_train_list = []
        for symbol in self.symbols:
            data = stock_dic[symbol][['close', 'macd', 'signal']].values
            X, y = self.create_sequences(data)
            X_train_list.append(X)
            y_train_list.append(y)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 3)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logger.info("Starting model training...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        logger.info("Model training completed.")
        
        return model

    def predict_and_execute(self, model, stock_dic):
        """Run strategy using live Alpaca data or simulation mode if market is closed."""
        logger.info("Running MACD strategy...")
        market_open = self.is_market_open()
        simulation_mode = not market_open
        
        if simulation_mode:
            logger.info("Market is closed according to Alpaca. Running in simulation mode.")
        else:
            logger.info("Market is open. Executing live trades.")
            
        account = self.trading_client.get_account()
        cash_available = float(account.cash)
        risk_per_trade = 0.10 * cash_available  # Risk 10% of available capital

        # Retrieve open positions and all orders, then filter for open orders
        open_positions = self.trading_client.get_all_positions()
        all_orders = self.trading_client.get_orders()
        open_orders = [o for o in all_orders if o.status == "open"]

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

            if quantity <= 0:
                logger.info(f"Quantity is 0 or less for {symbol}, skipping trade.")
                continue

            # Cancel any existing open orders for this symbol
            symbol_orders = [o for o in open_orders if o.symbol == symbol]
            if symbol_orders:
                logger.info(f"Open order(s) exist for {symbol}: {symbol_orders}. Canceling them.")
                for order in symbol_orders:
                    try:
                        if simulation_mode:
                            logger.info(f"Simulated cancel of order {order.id} for {symbol}")
                        else:
                            self.trading_client.cancel_order_by_id(order.id)
                            logger.info(f"Canceled order {order.id} for {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order.id} for {symbol}: {e}")

            # Check if there's an existing position in the symbol
            current_position = next((pos for pos in open_positions if pos.symbol == symbol), None)
            qty_held = float(current_position.qty) if current_position else 0.0
            side_held = "long" if qty_held > 0 else ("short" if qty_held < 0 else "none")

            # SIMPLE EXIT LOGIC: If holding a long position but the signal is bearish, close it before placing a new order.
            if side_held == "long" and last_predicted_price < last_actual_price:
                logger.info(f"Closing existing long position in {symbol} before new SELL order.")
                # Cancel orders again (if any still exist)
                symbol_orders = [o for o in open_orders if o.symbol == symbol]
                for order in symbol_orders:
                    try:
                        if simulation_mode:
                            logger.info(f"Simulated cancel of order {order.id} for {symbol} before closing position.")
                        else:
                            self.trading_client.cancel_order_by_id(order.id)
                            logger.info(f"Canceled order {order.id} for {symbol} before closing position.")
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order.id} for {symbol}: {e}")
                # Attempt to close the position, with retries
                closed = False
                for attempt in range(3):
                    try:
                        if simulation_mode:
                            logger.info(f"Simulated closing of long position in {symbol} on attempt {attempt+1}.")
                        else:
                            self.trading_client.close_position(symbol)
                            logger.info(f"Successfully closed long position in {symbol} on attempt {attempt+1}.")
                        closed = True
                        break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} to close position in {symbol} failed: {e}")
                        time.sleep(2)
                if not closed:
                    logger.warning(f"Could not close position in {symbol} after retries, skipping trade for {symbol}.")
                    continue

            # After handling exit logic, decide whether to place a new order:
            if last_predicted_price > last_actual_price:
                # BUY signal (go long)
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
                if simulation_mode:
                    logger.info(f"Simulated BUY {quantity} shares of {symbol} at market price")
                else:
                    try:
                        self.trading_client.submit_order(order)
                        logger.info(f"BUY {quantity} shares of {symbol} at market price")
                    except Exception as e:
                        logger.warning(f"Error placing BUY order for {symbol}: {e}")
            elif last_predicted_price < last_actual_price:
                # SELL signal (could be used to open a short, if that fits your strategy)
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
                if simulation_mode:
                    logger.info(f"Simulated SELL {quantity} shares of {symbol} at market price")
                else:
                    try:
                        self.trading_client.submit_order(order)
                        logger.info(f"SELL {quantity} shares of {symbol} at market price")
                    except Exception as e:
                        logger.warning(f"Error placing SELL order for {symbol}: {e}")
        
        logger.info("Trading bot executed with simple exit logic for position reversals.")

def main():
    logger.info(f"Starting {BOT_NAME} loop (running every 16 minutes)")
    while True:
        try:
            bot = Macd_trading_bot()
            stock_dic = bot.fetch_historical_data()
            model = bot.build_model(stock_dic)
            bot.predict_and_execute(model, stock_dic)
            logger.info(f"{BOT_NAME} cycle completed successfully")
        except Exception as e:
            logger.error(f"Error running {BOT_NAME} cycle: {e}", exc_info=True)
        logger.info("Waiting for 16 minutes until next cycle...\n")
        time.sleep(960)  # 16 minutes in seconds

if __name__ == "__main__":
    main()