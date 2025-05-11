import aiohttp
import asyncio
from typing import AsyncIterator, Dict, Any, Optional


class AgentCompletionChunk:
    """Represents a chunk of completion from your agent API"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @property
    def choices(self):
        return self.data.get("choices", [])

    @property
    def usage(self):
        return self.data.get("usage")


class AgentStreamWrapper(AsyncIterator[AgentCompletionChunk]):
    """Wrapper for HTTP streaming response from agent API"""

    def __init__(self, response):
        self.response = response
        self.usage = None
        self._is_closed = False
        self._response_iter = None

    def __aiter__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    async def close(self):
        """Close the underlying response if it's not already closed"""
        if not self._is_closed:
            if hasattr(self.response, 'close'):
                # Check if close is a coroutine function
                if asyncio.iscoroutinefunction(self.response.close):
                    await self.response.close()
                else:
                    self.response.close()
            self._is_closed = True

    async def __anext__(self):
        """Get the next chunk from the stream"""
        if self._is_closed:
            raise StopAsyncIteration

        try:
            # Different HTTP clients have different ways to read lines
            # Try various methods that might be available
            line = None

            # Method 1: If response has a readLine method
            if hasattr(self.response, 'readline') and callable(self.response.readline):
                if asyncio.iscoroutinefunction(self.response.readline):
                    line = await self.response.readline()
                else:
                    line = self.response.readline()

            # Method 2: If response has a content attribute with a readline method
            elif hasattr(self.response, 'content') and hasattr(self.response.content, 'readline'):
                if asyncio.iscoroutinefunction(self.response.content.readline):
                    line = await self.response.content.readline()
                else:
                    line = self.response.content.readline()

            # Method 3: If response has an __aiter__ method (async iterator)
            # This is the most common case for aiohttp streaming responses
            elif hasattr(self.response, '__aiter__'):
                # Initialize the iterator if we haven't already
                if self._response_iter is None:
                    self._response_iter = self.response.__aiter__()
                try:
                    # For aiohttp, this will typically return a chunk of bytes
                    line = await self._response_iter.__anext__()
                    print(f"Received streaming chunk: {line[:100]}...") if len(line) > 100 else print(f"Received streaming chunk: {line}")
                except StopAsyncIteration:
                    line = None

            # Method 4: For aiohttp streaming responses with content.read() method
            elif hasattr(self.response, 'content') and hasattr(self.response.content, 'read'):
                if not hasattr(self, '_read_size'):
                    self._read_size = 1024  # Read 1KB at a time
                if asyncio.iscoroutinefunction(self.response.content.read):
                    line = await self.response.content.read(self._read_size)
                    if not line:  # Empty response means we're done
                        line = None
                else:
                    line = self.response.content.read(self._read_size)
                    if not line:  # Empty response means we're done
                        line = None

            # For requests/httpx Response objects where content is already loaded
            elif hasattr(self.response, 'text') or hasattr(self.response, 'json'):
                # If we haven't processed the response yet
                if not hasattr(self, '_processed'):
                    self._processed = True
                    self._lines = []

                    # Try to get JSON content
                    if hasattr(self.response, 'json'):
                        try:
                            data = self.response.json()
                            # Create a fake line with the JSON data
                            line = json.dumps(data).encode('utf-8')
                            self._lines.append(line)
                        except Exception as e:
                            print(f"Error parsing JSON: {e}")
                            # If JSON parsing fails, try to use the raw text
                            if hasattr(self.response, 'text'):
                                text = self.response.text
                                self._lines = [line.encode('utf-8') for line in text.split('\n') if line.strip()]

                    # Fall back to text content
                    elif hasattr(self.response, 'text'):
                        text = self.response.text
                        # Split by lines and encode
                        self._lines = [line.encode('utf-8') for line in text.split('\n') if line.strip()]

                # Return the next line if we have any
                if hasattr(self, '_lines') and self._lines:
                    line = self._lines.pop(0)
                else:
                    line = None

            # If we couldn't get a line, close and stop iteration
            if line is None or line == b'':
                await self.close()
                raise StopAsyncIteration

            # Process the line (assuming it's bytes)
            if isinstance(line, bytes):
                data_str = line.decode('utf-8').strip()
            else:
                data_str = line.strip()

            # Handle server-sent events format
            if data_str.startswith('data: '):
                data_str = data_str[6:]

            # Handle end of stream marker
            if data_str == "[DONE]":
                await self.close()
                raise StopAsyncIteration

            # Parse the JSON data
            import json
            try:
                data = json.loads(data_str)
                chunk = AgentCompletionChunk(data)

                # Store usage info if present
                if chunk.usage:
                    self.usage = chunk.usage

                return chunk
            except json.JSONDecodeError:
                # If we can't parse the data, skip this chunk
                return await self.__anext__()

        except Exception as e:
            # If there's any error, close the response and re-raise
            await self.close()
            raise