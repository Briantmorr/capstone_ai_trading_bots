import os
import sys
import logging
import importlib.util
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from logger_setup import get_bot_logger  # Ensure this module is available in your project

# Load environment variables
load_dotenv()

class BotManager:
    """
    Simple manager class to handle multiple trading bots.
    Each bot runs from its own directory with its own Alpaca paper trading account.
    Bots are specified in the .env file with names, API keys, and secrets.
    """
    
    def __init__(self):
        """Initialize the bot manager."""
        # Use the abstracted logger for BotManager
        self.logger = get_bot_logger("BotManager", log_dir=Path(__file__).parent)
        
        # Discover bots from environment variables
        self.bots = self._discover_bots()
        
        self.logger.info(f"Bot Manager initialized with {len(self.bots)} bots")
    
    def _discover_bots(self):
        """
        Discover bots from environment variables.
        Each bot should have variables in the format:
            BOT_NAME_1=trading_bot_algo_brian
            BOT_API_KEY_1=your_api_key
            BOT_API_SECRET_1=your_api_secret
        """
        bots = {}
        bot_index = 1
        
        while True:
            bot_name_var = f"BOT_NAME_{bot_index}"
            if bot_name_var not in os.environ:
                break
                
            bot_name = os.environ[bot_name_var]
            api_key = os.environ.get(f"BOT_API_KEY_{bot_index}", "")
            api_secret = os.environ.get(f"BOT_API_SECRET_{bot_index}", "")
            
            if not bot_name or not api_key or not api_secret:
                self.logger.warning(f"Bot at index {bot_index} is missing required fields")
                bot_index += 1
                continue
                
            bots[bot_name] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "path": f"{bot_name}/{bot_name}.py"
            }
            
            self.logger.info(f"Discovered bot: {bot_name}")
            bot_index += 1
            
        return bots
    
    def run_bot(self, bot_name):
        """
        Run a specific bot by name.
        
        Args:
            bot_name (str): Name of the bot to run.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if bot_name not in self.bots:
            self.logger.warning(f"Bot '{bot_name}' not found")
            return False
            
        bot_config = self.bots[bot_name]
        bot_path = Path(bot_config["path"])
        
        if not bot_path.exists():
            self.logger.error(f"Bot file not found: {bot_path}")
            return False
            
        # Set up bot-specific logging using get_bot_logger.
        # This writes the log in a subdirectory named after the bot.
        bot_log_dir = Path(bot_name)
        bot_log_dir.mkdir(exist_ok=True)
        bot_logger = get_bot_logger(bot_name, log_dir=bot_log_dir)
        
        self.logger.info(f"Running bot: {bot_name}")
        
        try:
            # Load the bot module dynamically.
            spec = importlib.util.spec_from_file_location(bot_name, bot_path)
            module = importlib.util.module_from_spec(spec)
            
            # Set API credentials in environment for the bot to use.
            os.environ["ALPACA_API_KEY"] = bot_config["api_key"]
            os.environ["ALPACA_API_SECRET"] = bot_config["api_secret"]
            
            # Execute the bot module.
            spec.loader.exec_module(module)
            
            # If the module has a main() function, call it.
            if hasattr(module, "main"):
                module.main()
                
            self.logger.info(f"Bot {bot_name} completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error running bot {bot_name}: {e}", exc_info=True)
            return False
    
    def run_all_bots(self):
        """Run all registered bots once and log a summary."""
        self.logger.info(f"Running all bots ({len(self.bots)} total)")
        summary_results = []
        success_count = 0
        
        for bot_name in self.bots:
            result = self.run_bot(bot_name)
            summary_results.append((bot_name, result))
            if result:
                success_count += 1
                
        summary_message = f"Bot Execution Summary: {success_count}/{len(self.bots)} bots executed successfully\n"
        for bot, result in summary_results:
            summary_message += f"  - {bot}: {'Success' if result else 'Failed'}\n"
        
        self.logger.info(summary_message)
        return success_count
    
    def list_bots(self):
        """Return a list of all registered bots and their file paths."""
        bot_list = []
        for bot_name, config in self.bots.items():
            bot_info = {
                "name": bot_name,
                "path": config["path"]
            }
            bot_list.append(bot_info)
        return bot_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot Manager")
    parser.add_argument("--run-all", action="store_true", help="Run all bots once and exit")
    parser.add_argument("--run", type=str, help="Run a specific bot by name")
    parser.add_argument("--list", action="store_true", help="List all registered bots")
    
    args = parser.parse_args()
    bot_manager = BotManager()
    
    if args.list:
        bots = bot_manager.list_bots()
        print(f"Registered bots ({len(bots)}):")
        for bot in bots:
            print(f"  {bot['name']}")
            print(f"    Path: {bot['path']}")
    elif args.run:
        success = bot_manager.run_bot(args.run)
        print(f"Bot {args.run} executed: {'Success' if success else 'Failed'}")
    elif args.run_all:
        count = bot_manager.run_all_bots()
        print(f"Ran {count} bots successfully")
    else:
        parser.print_help()
