# Multi-Bot Alpaca Trading Framework

A simplified framework for running multiple algorithmic trading bots, each with its own Alpaca paper trading account and strategy. Bots are configured via environment variables and executed from their own directories using a centralized bot manager.

## Features

- **Multiple Bot Execution:**  
  Run multiple trading bots with different strategies. Each bot has its own Alpaca paper trading account and implementation.

- **Centralized Bot Manager:**  
  The `bot_manager.py` script calls each bot’s strategy method and coordinates their execution.

- **Scheduled Runs (Weekdays Only):**  
  Bots are automatically executed at **8:00 AM PST** (weekdays only) via GitHub Actions.  
  *Note:* The scheduled workflow also updates a trading leaderboard.

- **Logging and Version Control:**  
  - Each bot logs detailed run information to its own log file.
  - The bot manager logs its output to `bot_manager.log` in the project root.
  - After each run, updated log files are automatically pushed to the `bot-execution` branch.

- **Leaderboard Update:**  
  A scheduled run calls an external API endpoint to update the trading leaderboard with the latest execution data.


## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Briantmorr/capstone_ai_trading_bots.git
cd capstone_ai_trading_bots
```

### 1. Install Dependencies
```bash
pipenv install
pipenv shell
```

### 3. Configure Your Bots
#### Bot 1
```
BOT_NAME_1=trading_bot_llm_sentiment_brian
BOT_API_KEY_1=your_alpaca_api_key_here
BOT_API_SECRET_1=your_alpaca_api_secret_here

#### Bot 2
BOT_NAME_2=momentum_bot_carlo
BOT_API_KEY_2=another_alpaca_api_key_here
BOT_API_SECRET_2=another_alpaca_api_secret_here

#### Additional API Keys
FINNHUB_API_KEY=your_finnhub_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Create Your Bot Implementations
For each bot specified in the configuration, create a directory with the bot's name and add a Python script with the same name. For example:
```bash
mkdir -p trading_bot_llm_sentiment_brian
touch trading_bot_llm_sentiment_brian/trading_bot_llm_sentiment_brian.py
```

Implement your trading strategy in the script. Ensure that each script has a main() function that will be invoked by the bot manager.


each bot will can retrieve credentials with:
```python
api_key = os.getenv(f"{BOT_NAME}_API_KEY_1")
api_secret = os.getenv(f"{BOT_NAME}_API_SECRET_1")
```

### 5. Run the Bot Manager
List registered bots
```bash
python bot_manager.py --list
```
currently returns 
```
2025-04-12 13:33:53,541 - bot_manager - INFO - Logger initialized and header written.
2025-04-12 13:33:53,541 - bot_manager - INFO - Discovered bot: trading_bot_llm_sentiment_brian
2025-04-12 13:33:53,541 - bot_manager - INFO - Discovered bot: momentum_bot_carlo
2025-04-12 13:33:53,541 - bot_manager - INFO - Discovered bot: trading_bot_macd_melissa
2025-04-12 13:33:53,541 - bot_manager - INFO - Discovered bot: momentum_ml_carlo
2025-04-12 13:33:53,541 - bot_manager - INFO - Bot Manager initialized with 4 bots
Registered bots (4):
  trading_bot_llm_sentiment_brian
    Path: trading_bot_llm_sentiment_brian/trading_bot_llm_sentiment_brian.py
  momentum_bot_carlo
    Path: momentum_bot_carlo/momentum_bot_carlo.py
  trading_bot_macd_melissa
    Path: trading_bot_macd_melissa/trading_bot_macd_melissa.py
  momentum_ml_carlo
    Path: momentum_ml_carlo/momentum_ml_carlo.py
(capstone_ai_trading_bots) 
```
Run a specific bot
```bash
python bot_manager.py --run trading_bot_llm_sentiment_brian
```
Run all bots
```bash
python bot_manager.py --run-all
```

### 6. Scheduled Execution & Leaderboard Update

The scheduled GitHub Actions workflow (.github/workflows/scheduled_bot_run.yml) performs the following each weekday at 8:00 AM PST:

- ### Executes All Bots:

Calls the bot_manager.py script to run all registered bots.

- ### Logs Run Details:

Captures run details in bot_manager.log and pushes updates to the bot-execution branch.

- ### Updates the Leaderboard:
Calls the external API at https://trading-leaderboard-three.vercel.app/api/update-leaderboard with the necessary payload to update the trading leaderboard.

## Logging
Bot-Specific Logs:
Each bot writes logs to a file in its respective directory.

Bot Manager Log:
bot_manager.log in the project root records overall execution details.

Automated Git Push:
After each scheduled run, the workflow automatically commits and pushes updated log files to the bot-execution branch. Logs are ephemeral (lasting 1 day)

---

### Disclaimer
This trading bot framework is provided for educational purposes only. Trading involves significant risk, and algorithmic trading introduces additional technical complexities. Always test with paper trading accounts before risking real capital.

License
This project is licensed under the MIT License. See the LICENSE file for details.