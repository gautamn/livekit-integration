import logging
import os
import sys
from typing import Optional

def setup_logger(
    name: str, 
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up a logger with the specified name and configuration.
    
    Args:
        name: The name of the logger
        level: The logging level (default: INFO)
        log_file: Optional file path to write logs to
        log_format: The format string for log messages
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default logger for the application
app_logger = setup_logger('livekit_integration')

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: The name of the module
        
    Returns:
        A logger instance for the module
    """
    return logging.getLogger(f'livekit_integration.{module_name}')
