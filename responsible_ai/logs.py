import logging
from logging import getLogger
from concurrent_log_handler import ConcurrentTimedRotatingFileHandler
import os
import sys
from typing import Dict, Any, Optional, Union


def setup_logger(logger_name=__name__, log_folder="", log_filename="responsible_ai.log", level="INFO", log_to_console=True, logs_dir="logs"):
    """
    Configures and returns a logger with time-based rotation at midnight and safe
    concurrent access for multi-process environments using ConcurrentTimedRotatingFileHandler.

    Args:
        logger_name (str): Name of the logger
        log_folder (str): Optional subfolder within base logs directory
        log_filename (str): Name of the log file
        level (str): Log level for all handlers (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console (bool): Whether to log to console
        logs_dir (str): Base logs directory from config

    Returns:
        logging.Logger: Configured logger instance
    """
    # Set log level for all handlers
    log_level = getattr(logging, level.upper())

    # Define the log folder and ensure it exists
    project_root = os.path.dirname(os.path.abspath(__file__))

    # First set the base log directory from LOGS_DIR config
    base_log_folder = os.path.join(project_root, logs_dir)

    # Then add any additional subfolder if specified
    if log_folder:
        base_log_folder = os.path.join(base_log_folder, log_folder)

    # Create or get the logger
    logger = getLogger(logger_name)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Disable propagation to avoid duplicate logs from parent loggers
    logger.propagate = False

    # Set overall logger level
    logger.setLevel(log_level)

    # Create a console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Always create a file handler for logging to file
    try:
        # Ensure directory exists
        os.makedirs(base_log_folder, exist_ok=True)

        # Complete path for the log file
        log_file_path = os.path.join(base_log_folder, log_filename)

        # Create a ConcurrentTimedRotatingFileHandler with time-based rotation at midnight
        file_handler = ConcurrentTimedRotatingFileHandler(
            filename=log_file_path,
            when="midnight",  # Rotate logs at midnight
            backupCount=100,  # Retain logs for the last 100 days
        )
        file_handler.setLevel(log_level)

        # Define the log format for file (more detailed than console)
        file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    except Exception as e:
        print(f"Error setting up file logging: {str(e)}")
        # If file logging fails and console logging is disabled, enable it as fallback
        if not log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            print("Falling back to console logging only due to file logging error")

    return logger


def get_logger(name=__name__, config_path=None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name (str): Name for the logger
        config_path (str): Path to configuration file (optional)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Default values
    log_level = "INFO"
    log_folder = ""  # Default to no additional subfolder
    log_filename = "responsible_ai.log"
    log_to_console = True
    logs_dir = "logs"  # Default base logs directory

    # Only try to load config if we're not in the ConfigManager class
    # (to avoid circular dependency)
    if not name.endswith("ConfigManager"):
        try:
            from utils.config_manager import ConfigManager

            config = ConfigManager().get_config("app_config")
            log_level = config.get("LOG_LEVEL", log_level)
            logs_dir = config.get("LOGS_DIR", logs_dir)  # Get base logs directory from config
            log_filename = config.get("LOG_FILE", log_filename)
            log_to_console = config.get("LOG_TO_CONSOLE", log_to_console)

        except Exception as e:
            # If config loading fails, use defaults
            if name == "__main__":
                print(f"Failed to load logging config: {str(e)}")
            pass

    return setup_logger(
        logger_name=name,
        log_folder=log_folder,  # Additional subfolder (empty by default)
        log_filename=log_filename,
        level=log_level,
        log_to_console=log_to_console,
        logs_dir=logs_dir,  # Base logs directory from config
    )


def format_log_message_with_context(message: str, context: Dict[str, Any]) -> str:
    """
    Format a log message with additional context data.

    Args:
        message: Base log message
        context: Dictionary of context values to append

    Returns:
        Formatted message with context
    """
    if not context:
        return message

    context_str = " | ".join(f"{k}={v}" for k, v in context.items())
    return f"{message} [{context_str}]"
