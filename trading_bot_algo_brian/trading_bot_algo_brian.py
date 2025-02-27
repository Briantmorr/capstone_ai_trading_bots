import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pathlib import Path
from logger_setup import get_bot_logger


# Bot name (same as directory name)
BOT_NAME = "trading_bot_algo_brian"
logger = get_bot_logger(BOT_NAME, f"{Path.cwd()}/{BOT_NAME}")

class TradingBotBrian:
    """
    A simple trading bot using the Alpaca API.
    Implements a bollinger band strategy.
    """
    
    def __init__(self):
        """Initialize the trading bot with API credentials and settings."""
        # API Keys from environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be provided in environment variables")
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Trading parameters
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']  # Stocks to trade
        self.timeframe = TimeFrame.Day
        self.bb_window = 20           # Bollinger Bands window
        self.bb_std_dev = 2           # Number of standard deviations
        self.qty_per_trade = 1        # Number of shares per trade
        
        logger.info(f"Trading bot {BOT_NAME} initialized with {len(self.symbols)} symbols")
        
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
            
        # Reset index to make timestamp a column and sort
        df = df.reset_index()
        df = df.sort_values(by=["timestamp"])
        
        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df
    
    def calculate_bollinger_signals(self, df):
        """Calculate trading signals using Bollinger Bands."""
        if df is None or len(df) < self.bb_window:
            return None
            
        # Calculate 20-day rolling mean and standard deviation
        df['sma'] = df['close'].rolling(window=self.bb_window).mean()
        df['std'] = df['close'].rolling(window=self.bb_window).std()
        
        # Calculate upper and lower Bollinger Bands
        df['upper_band'] = df['sma'] + (df['std'] * self.bb_std_dev)
        df['lower_band'] = df['sma'] - (df['std'] * self.bb_std_dev)
        
        # Calculate signal: 1 for buy (price below lower band), -1 for sell (price above upper band)
        df['signal'] = 0
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1
        
        # Get the current signal (last row)
        current_price = df['close'].iloc[-1]
        current_signal = df['signal'].iloc[-1]
        upper_band = df['upper_band'].iloc[-1]
        lower_band = df['lower_band'].iloc[-1]
        
        logger.info(f"Current price: ${current_price:.2f}, Upper band: ${upper_band:.2f}, Lower band: ${lower_band:.2f}")
        
        return current_signal
    
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
            logger.error(f"Error submitting order: {e}")
            return None
    
    def run_strategy(self):
        """Run the Bollinger Bands trading strategy."""
        logger.info("Running Bollinger Bands strategy...")
        
        # Check if market is open (simplified check for example)
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour >= 16:
            logger.info("Market is closed. Running in simulation mode.")
            
        # Get current positions
        positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
        
        # Show account info
        self.get_account_info()
        
        for symbol in self.symbols:
            logger.info(f"Analyzing {symbol}...")
            
            # Get historical data
            df = self.get_historical_data(symbol)
            if df is None:
                continue
                
            # Calculate signal using Bollinger Bands
            signal = self.calculate_bollinger_signals(df)
            
            if signal is None:
                logger.warning(f"Insufficient data to generate signal for {symbol}")
                continue
                
            # Execute trades based on signals
            if signal == 1 and symbol not in positions:
                # Buy signal (price below lower band) and we don't have a position
                logger.info(f"BUY signal for {symbol} (price below lower Bollinger Band)")
                self.submit_order(symbol, OrderSide.BUY, self.qty_per_trade)
                
            elif signal == -1 and symbol in positions:
                # Sell signal (price above upper band) and we have a position
                logger.info(f"SELL signal for {symbol} (price above upper Bollinger Band)")
                position_qty = float(positions[symbol].qty)
                self.submit_order(symbol, OrderSide.SELL, position_qty)
            
            else:
                logger.info(f"No action needed for {symbol}")
        
        logger.info("Strategy execution completed")

def main():
    """Main function to run the trading bot."""
    logger.info(f"Starting {BOT_NAME}")
    
    try:
        # Initialize and run bot
        bot = TradingBotBrian()
        bot.run_strategy()
        logger.info(f"{BOT_NAME} completed successfully")
    except Exception as e:
        logger.error(f"Error running {BOT_NAME}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
