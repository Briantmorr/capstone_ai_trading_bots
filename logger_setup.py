# logger_setup.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def get_bot_logger(bot_name: str) -> logging.Logger:
    """
    Returns a logger for the given bot. Always resets any existing logger handlers 
    so that a new header is appended at the start of each run.
    
    Args:
        bot_name (str): The name of the bot (log file will be bot_name.log)
        log_dir (Path, optional): Directory where logs are stored. Defaults to current working directory.
        
    Returns:
        logging.Logger: Configured logger for the bot.
    """
    # Use current working directory if no log_dir provided
    log_dir = Path.cwd()
    if bot_name == 'bot_manager':
        log_file = f"{log_dir}/{bot_name}.log"
    else: 
        log_file = f"{log_dir}/{bot_name}/{bot_name}.log"
        
    logger = logging.getLogger(bot_name)
    
    # Reset any existing handlers so we get a fresh config each run.
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # Append a header to mark a new run.
    header = "\n" + "=" * 80 + "\n"
    header += f"=== Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    header += "=" * 80 + "\n"
    try:
        with open(log_file, "a") as f:
            f.write(header)
    except Exception as e:
        print(f"Error writing header to log file: {e}", file=sys.stderr)
    
    # File handler (append mode)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (ensure logs print to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Test message to confirm logger is working
    logger.info("Logger initialized and header written.")
    
    return logger
