# utils/logger.py

import logging
import sys
import os

def setup_logger(name: str = "shabda", level: int = logging.INFO) -> logging.Logger:
    # Force UTF-8 encoding for stdout (Windows fix)
    if os.name == 'nt':
        sys.stdout.reconfigure(encoding='utf-8')
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] : %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
    file_handler = logging.FileHandler("logs/shabda.log")
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(handler)

    return logger
