import aiohttp
import asyncio
import json
import time
from typing import Dict, Any, Optional, AsyncIterator, Callable, Union
import backoff

from utils.logging_utils import get_logger
from utils.config import config

logger = get_logger("http_client")

class HTTPClientError(Exception):
    """Base exception for HTTP client errors"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP error {status_code}: {message}")

class RetryableHTTPError(HTTPClientError):
    """Exception for HTTP errors that can be retried"""
    pass

class NonRetryableHTTPError(HTTPClientError):
    """Exception for HTTP errors that should not be retried"""
    pass

def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception is retryable, False otherwise
    """
    if isinstance(exception, RetryableHTTPError):
        return True
    if isinstance(exception, aiohttp.ClientError):
        # Network errors, connection errors, etc. are retryable
        return True
    if isinstance(exception, asyncio.TimeoutError):
        return True
    return False

class HTTPClient:
    """
    HTTP client with retry logic, error handling, and streaming support.
    """
    
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Dict[str, float]] = None,
        max_retries: int = config.HTTP_MAX_RETRIES
    ):
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout or config.get_http_timeouts()
        self.max_retries = max_retries
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp client session.
        
        Returns:
            An aiohttp client session
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.timeout.get("connect", 15.0),
                total=self.timeout.get("total", None),
                sock_read=self.timeout.get("read", 30.0),
                sock_connect=self.timeout.get("connect", 15.0)
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the underlying session if it exists"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @backoff.on_exception(
        backoff.expo,
        (RetryableHTTPError, aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=config.HTTP_MAX_RETRIES + 1,  # +1 because first try counts
        giveup=lambda e: not is_retryable_error(e),
        jitter=backoff.full_jitter
    )
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to base_url)
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.request
            
        Returns:
            Response data as a dictionary
            
        Raises:
            HTTPClientError: If the request fails
        """
        url = f"{self.base_url}"
        session = await self._get_session()
        
        # Merge default headers with request-specific headers
        headers = {**self.headers}
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        
        try:
            start_time = time.time()
            logger.debug(f"Making {method} request to {url}")
            
            async with session.request(method, url, headers=headers, **kwargs) as response:
                elapsed_time = time.time() - start_time
                logger.debug(f"Received response from {url} in {elapsed_time:.2f}s with status {response.status}")
                
                # Handle error responses
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"Error response from {url}: {response.status} - {error_text}")
                    
                    if response.status in (429, 502, 503, 504):
                        # These are retryable status codes
                        raise RetryableHTTPError(response.status, error_text)
                    else:
                        # Other errors are not retryable
                        raise NonRetryableHTTPError(response.status, error_text)
                
                # Parse JSON response
                try:
                    data = await response.json()
                    return data
                except json.JSONDecodeError:
                    # If response is not JSON, return the raw text
                    text = await response.text()
                    return {"text": text}
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error for {url}: {str(e)}")
            raise RetryableHTTPError(0, str(e))
        except asyncio.TimeoutError:
            logger.error(f"Timeout error for {url}")
            raise RetryableHTTPError(0, "Request timed out")
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a GET request"""
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a POST request"""
        return await self.request("POST", endpoint, **kwargs)
    
    async def stream(
        self,
        method: str,
        endpoint: str,
        parser: Callable[[str], Any],
        **kwargs
    ) -> AsyncIterator[Any]:
        """
        Make a streaming HTTP request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to base_url)
            parser: Function to parse each line of the response
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.request
            
        Yields:
            Parsed chunks from the streaming response
            
        Raises:
            HTTPClientError: If the request fails
        """
        url = f"{self.base_url}"
        session = await self._get_session()
        
        # Merge default headers with request-specific headers
        headers = {**self.headers}
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        
        try:
            logger.debug(f"Making streaming {method} request to {url}")
            
            async with session.request(method, url, headers=headers, **kwargs) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"Error response from {url}: {response.status} - {error_text}")
                    
                    if response.status in (429, 502, 503, 504):
                        raise RetryableHTTPError(response.status, error_text)
                    else:
                        raise NonRetryableHTTPError(response.status, error_text)
                
                # Process the streaming response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    
                    try:
                        parsed_chunk = parser(line)
                        if parsed_chunk is not None:
                            yield parsed_chunk
                    except Exception as e:
                        logger.error(f"Error parsing chunk: {str(e)}")
                        continue
                        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error for {url}: {str(e)}")
            raise RetryableHTTPError(0, str(e))
        except asyncio.TimeoutError:
            logger.error(f"Timeout error for {url}")
            raise RetryableHTTPError(0, "Request timed out")
