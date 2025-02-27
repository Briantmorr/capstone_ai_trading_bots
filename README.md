# Multi-Bot Alpaca Trading Framework

A simplified framework for running multiple algorithmic trading bots, each with its own Alpaca paper trading account and strategy. Bots are configured in a `.env` file and executed from their own directories.

## Features

- Run multiple trading bots with different strategies
- Each bot uses its own Alpaca paper trading account
- Simple configuration via .env file
- Bot-specific logging in each bot's directory
- Flexible architecture that allows each bot to have its own implementation

## Project Structure

```
alpaca-trading-bots/
├── bot_manager.py         # Main bot management system
├── requirements.txt       # Project dependencies
├── .env                   # Bot configurations
├── trading_bot_algo_brian/  # Example bot directory
│   ├── trading_bot_algo_brian.py  # Bot implementation
│   └── trading_bot_algo_brian.log # Bot-specific logs
└── trading_bot_algo_sarah/   # Another bot directory
    ├── trading_bot_algo_sarah.py  # Bot implementation
    └── trading_bot_algo_sarah.log # Bot-specific logs
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/alpaca-trading-bots.git
cd alpaca-trading-bots
```

### 2. Install dependencies

```bash
pipenv install
pipenv shell
```

### 3. Configure your bots in .env file

Create a `.env` file in the project root directory with your bot configurations:

```
# Bot 1
BOT_NAME_1=trading_bot_algo_brian
BOT_API_KEY_1=your_alpaca_api_key_here
BOT_API_SECRET_1=your_alpaca_api_secret_here
BOT_SCHEDULE_1=09:30

# Bot 2
BOT_NAME_2=trading_bot_algo_sarah
BOT_API_KEY_2=another_alpaca_api_key_here
BOT_API_SECRET_2=another_alpaca_api_secret_here
BOT_SCHEDULE_2=10:00
```

### 4. Create your bot implementations

For each bot specified in the `.env` file, create a directory with the same name and a Python script with the same name inside it:

```
mkdir -p trading_bot_algo_brian
touch trading_bot_algo_brian/trading_bot_algo_brian.py
```

## Bot Implementation

Each bot should be implemented in its own directory and should have a `main()` function that will be called by the bot manager. The bot can access its API credentials from environment variables:

```python
import os

# Get API credentials
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_API_SECRET')
```

## Usage

### List registered bots

```bash
python bot_manager.py --list
```

### Run a specific bot

```bash
python bot_manager.py --run trading_bot_algo_brian
```

### Run all bots once

```bash
python bot_manager.py --run-all
```

## Creating a New Bot

1. Add the bot entry to the `.env` file with a new index number
2. Create a directory with the same name as the bot
3. Create a Python script inside the directory with the same name as the bot
4. Implement your trading strategy in the script
5. Make sure your script has a `main()` function that will be called by the bot manager

## Bot Examples

The repository includes an example bot implementation:

- `trading_bot_algo_brian`: Implements a Bollinger Bands trading strategy

You can use this as a template for creating your own trading bots with different strategies.

## Logging

Each bot logs to both the console and a log file in its own directory. The bot manager also logs to a `bot_manager.log` file in the root directory.

## Disclaimer

This trading bot framework is for educational purposes only. Trading involves risk, and algorithmic trading adds additional technical risks. Always use paper trading to test strategies before risking real money.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
