"""
Centralized error handling utilities for the application.
Provides consistent error handling and reporting across the codebase.
"""

import sys
import traceback
from typing import Optional, Type, Dict, Any, Callable, TypeVar, Union
from functools import wraps
import asyncio

from utils.logging_utils import get_logger

logger = get_logger("error_handling")

T = TypeVar('T')

class ApplicationError(Exception):
    """Base exception class for all application errors."""
    
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR", details: Optional[Dict[str, Any]] = None):
        """
        Initialize an application error.
        
        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)
        
    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.code}: {self.message} - {self.details}"
        return f"{self.code}: {self.message}"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary for serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(ApplicationError):
    """Error raised when there's a configuration issue."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class APIError(ApplicationError):
    """Error raised when there's an issue with an API call."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if status_code is not None:
            details["status_code"] = status_code
        super().__init__(message, "API_ERROR", details)


class ValidationError(ApplicationError):
    """Error raised when there's a validation issue."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if field is not None:
            details["field"] = field
        super().__init__(message, "VALIDATION_ERROR", details)


def handle_errors(
    error_map: Optional[Dict[Type[Exception], Callable[[Exception], Any]]] = None,
    default_handler: Optional[Callable[[Exception], Any]] = None,
    log_traceback: bool = True
):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_map: Mapping of exception types to handler functions
        default_handler: Default handler function for unhandled exceptions
        log_traceback: Whether to log the traceback for unhandled exceptions
        
    Returns:
        Decorated function
    """
    error_map = error_map or {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we have a specific handler for this exception type
                for exc_type, handler in error_map.items():
                    if isinstance(e, exc_type):
                        return handler(e)
                
                # Log the error
                if log_traceback:
                    logger.exception(f"Error in {func.__name__}: {str(e)}")
                else:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                # Use the default handler if provided
                if default_handler:
                    return default_handler(e)
                
                # Re-raise the exception
                raise
                
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if we have a specific handler for this exception type
                for exc_type, handler in error_map.items():
                    if isinstance(e, exc_type):
                        if asyncio.iscoroutinefunction(handler):
                            return await handler(e)
                        else:
                            return handler(e)
                
                # Log the error
                if log_traceback:
                    logger.exception(f"Error in {func.__name__}: {str(e)}")
                else:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                # Use the default handler if provided
                if default_handler:
                    if asyncio.iscoroutinefunction(default_handler):
                        return await default_handler(e)
                    else:
                        return default_handler(e)
                
                # Re-raise the exception
                raise
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def format_exception(exc: Exception) -> str:
    """
    Format an exception for display.
    
    Args:
        exc: The exception to format
        
    Returns:
        Formatted exception string
    """
    if isinstance(exc, ApplicationError):
        return str(exc)
    
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(tb)


def safe_execute(func: Callable[..., T], *args, **kwargs) -> Union[T, None]:
    """
    Execute a function safely, catching and logging any exceptions.
    
    Args:
        func: The function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The function result or None if an exception occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Error executing {func.__name__}: {str(e)}")
        return None


async def safe_execute_async(func: Callable[..., T], *args, **kwargs) -> Union[T, None]:
    """
    Execute an async function safely, catching and logging any exceptions.
    
    Args:
        func: The async function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The function result or None if an exception occurred
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Error executing {func.__name__}: {str(e)}")
        return None
