"""
Centralized logging configuration for the RAG application.

This module creates a separate log file for each application run,
organized inside a folder for the current date.
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"

date_folder = datetime.now().strftime("%Y%m%d")
daily_log_dir = os.path.join(LOG_DIR, date_folder)
os.makedirs(daily_log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(daily_log_dir, f"rag_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)