"""
Caching utilities for the application.
Provides in-memory and persistent caching mechanisms.
"""

import time
import json
import os
import pickle
from typing import Any, Dict, Optional, Callable, TypeVar, Generic, Union, List
from functools import wraps
import asyncio
import hashlib

from utils.logging_utils import get_logger

logger = get_logger("caching")

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CacheItem(Generic[T]):
    """A cache item with expiration."""
    
    def __init__(self, value: T, ttl: Optional[float] = None):
        """
        Initialize a cache item.
        
        Args:
            value: The value to cache
            ttl: Time to live in seconds, or None for no expiration
        """
        self.value = value
        self.expiration = time.time() + ttl if ttl is not None else None
        self.created_at = time.time()
        self.last_accessed = self.created_at
        
    def is_expired(self) -> bool:
        """
        Check if the cache item is expired.
        
        Returns:
            True if the item is expired, False otherwise
        """
        if self.expiration is None:
            return False
        return time.time() > self.expiration
        
    def access(self) -> None:
        """Update the last accessed time."""
        self.last_accessed = time.time()


class MemoryCache(Generic[K, V]):
    """In-memory cache with expiration."""
    
    def __init__(self, default_ttl: Optional[float] = 300, max_size: Optional[int] = 1000):
        """
        Initialize an in-memory cache.
        
        Args:
            default_ttl: Default time to live in seconds, or None for no expiration
            max_size: Maximum number of items in the cache, or None for unlimited
        """
        self._cache: Dict[K, CacheItem[V]] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        
    def get(self, key: K) -> Optional[V]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        if key not in self._cache:
            return None
            
        item = self._cache[key]
        if item.is_expired():
            self._cache.pop(key)
            return None
            
        item.access()
        return item.value
        
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time to live in seconds, or None to use the default
        """
        # Use the default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl
            
        # Enforce max size if specified
        if self.max_size is not None and len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_one()
            
        self._cache[key] = CacheItem(value, ttl)
        
    def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was deleted, False if it wasn't in the cache
        """
        if key in self._cache:
            self._cache.pop(key)
            return True
        return False
        
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        
    def _evict_one(self) -> None:
        """Evict one item from the cache based on LRU policy."""
        if not self._cache:
            return
            
        # Find the least recently used item
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        self._cache.pop(lru_key)
        
    def cleanup(self) -> int:
        """
        Remove expired items from the cache.
        
        Returns:
            Number of items removed
        """
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            self._cache.pop(key)
        return len(expired_keys)


class DiskCache(Generic[K, V]):
    """Persistent disk-based cache with expiration."""
    
    def __init__(
        self, 
        cache_dir: str, 
        default_ttl: Optional[float] = 3600,
        max_size: Optional[int] = 1000,
        serializer: str = "pickle"
    ):
        """
        Initialize a disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time to live in seconds, or None for no expiration
            max_size: Maximum number of items in the cache, or None for unlimited
            serializer: Serialization format ("pickle" or "json")
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.serializer = serializer
        
        # Create the cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create an index file to track cache items
        self.index_file = os.path.join(cache_dir, "index.json")
        self.index: Dict[str, Dict[str, Union[float, str]]] = {}
        self._load_index()
        
    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r") as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache index: {str(e)}")
                self.index = {}
                
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")
            
    def _key_to_filename(self, key: K) -> str:
        """
        Convert a cache key to a filename.
        
        Args:
            key: The cache key
            
        Returns:
            The filename for the cache item
        """
        # Convert the key to a string and hash it
        key_str = str(key)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        # Use the serializer as the file extension
        ext = "pkl" if self.serializer == "pickle" else "json"
        return os.path.join(self.cache_dir, f"{key_hash}.{ext}")
        
    def _serialize(self, value: V) -> bytes:
        """
        Serialize a value.
        
        Args:
            value: The value to serialize
            
        Returns:
            The serialized value
            
        Raises:
            ValueError: If the serializer is not supported
        """
        if self.serializer == "pickle":
            return pickle.dumps(value)
        elif self.serializer == "json":
            return json.dumps(value).encode()
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")
            
    def _deserialize(self, data: bytes) -> V:
        """
        Deserialize a value.
        
        Args:
            data: The serialized data
            
        Returns:
            The deserialized value
            
        Raises:
            ValueError: If the serializer is not supported
        """
        if self.serializer == "pickle":
            return pickle.loads(data)
        elif self.serializer == "json":
            return json.loads(data.decode())
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")
            
    def get(self, key: K) -> Optional[V]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        key_str = str(key)
        if key_str not in self.index:
            return None
            
        # Check if the item is expired
        item_info = self.index[key_str]
        if "expiration" in item_info and item_info["expiration"] is not None:
            if time.time() > item_info["expiration"]:
                # Item is expired, remove it
                self.delete(key)
                return None
                
        # Load the item from disk
        filename = self._key_to_filename(key)
        if not os.path.exists(filename):
            # File doesn't exist, remove from index
            self.index.pop(key_str, None)
            self._save_index()
            return None
            
        try:
            with open(filename, "rb") as f:
                data = f.read()
            
            # Update the last accessed time
            item_info["last_accessed"] = time.time()
            self._save_index()
            
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Error loading cache item {key}: {str(e)}")
            return None
            
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time to live in seconds, or None to use the default
        """
        # Use the default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl
            
        # Enforce max size if specified
        if self.max_size is not None and len(self.index) >= self.max_size and str(key) not in self.index:
            self._evict_one()
            
        # Serialize the value and save it to disk
        filename = self._key_to_filename(key)
        try:
            data = self._serialize(value)
            with open(filename, "wb") as f:
                f.write(data)
                
            # Update the index
            now = time.time()
            self.index[str(key)] = {
                "filename": os.path.basename(filename),
                "created_at": now,
                "last_accessed": now,
                "expiration": now + ttl if ttl is not None else None
            }
            self._save_index()
        except Exception as e:
            logger.error(f"Error saving cache item {key}: {str(e)}")
            
    def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was deleted, False if it wasn't in the cache
        """
        key_str = str(key)
        if key_str not in self.index:
            return False
            
        # Remove the file
        filename = self._key_to_filename(key)
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as e:
                logger.error(f"Error deleting cache file {filename}: {str(e)}")
                
        # Remove from the index
        self.index.pop(key_str)
        self._save_index()
        return True
        
    def clear(self) -> None:
        """Clear the cache."""
        # Remove all cache files
        for key_str in list(self.index.keys()):
            item_info = self.index[key_str]
            filename = os.path.join(self.cache_dir, item_info["filename"])
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except Exception as e:
                    logger.error(f"Error deleting cache file {filename}: {str(e)}")
                    
        # Clear the index
        self.index = {}
        self._save_index()
        
    def _evict_one(self) -> None:
        """Evict one item from the cache based on LRU policy."""
        if not self.index:
            return
            
        # Find the least recently used item
        lru_key = min(self.index.keys(), key=lambda k: self.index[k]["last_accessed"])
        self.delete(lru_key)
        
    def cleanup(self) -> int:
        """
        Remove expired items from the cache.
        
        Returns:
            Number of items removed
        """
        now = time.time()
        expired_keys = [
            k for k, v in self.index.items() 
            if "expiration" in v and v["expiration"] is not None and now > v["expiration"]
        ]
        
        for key in expired_keys:
            self.delete(key)
            
        return len(expired_keys)


def cache(cache_instance: Union[MemoryCache, DiskCache], key_fn: Optional[Callable[..., Any]] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_instance: The cache instance to use
        key_fn: Function to generate the cache key from the function arguments
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate the cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # Default key is the function name and arguments
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
            # Check if the result is cached
            cached_result = cache_instance.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
                
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache_instance.set(key, result)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
            
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate the cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # Default key is the function name and arguments
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
            # Check if the result is cached
            cached_result = cache_instance.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
                
            # Call the function and cache the result
            result = await func(*args, **kwargs)
            cache_instance.set(key, result)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


# Create global cache instances
memory_cache = MemoryCache()
disk_cache = DiskCache(os.path.join(os.path.dirname(__file__), "..", "cache"))
