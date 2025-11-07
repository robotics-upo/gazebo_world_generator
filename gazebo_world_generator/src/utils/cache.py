"""
Multi-level Caching System for Gazebo World Generator

Provides fast in-memory caching with optional persistent disk storage.
Supports TTL, LRU eviction, and cache statistics.

Usage:
    # Memory-only cache
    cache = Cache(name="llm_responses", max_size=1000)
    cache.set("key", value, ttl=3600)
    result = cache.get("key")

    # Persistent cache (survives restarts)
    cache = PersistentCache(name="model_database", cache_dir=Path("/tmp/cache"))
    cache.set("model_info", data)
"""

import json
import pickle
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional, Dict, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Entry in cache with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict:
        """Export stats as dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'evictions': self.evictions,
            'expirations': self.expirations,
            'hit_rate': self.hit_rate(),
            'total_requests': self.hits + self.misses
        }


class Cache:
    """
    Thread-safe in-memory LRU cache with TTL support.

    Features:
    - LRU (Least Recently Used) eviction policy
    - Optional TTL (Time To Live) per entry
    - Thread-safe operations
    - Cache statistics tracking
    """

    def __init__(
        self,
        name: str = "default",
        max_size: int = 1000,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize cache.

        Args:
            name: Cache name for logging/monitoring
            max_size: Maximum number of entries (LRU eviction when exceeded)
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        if max_size < 1:
            raise ValueError("max_size must be >= 1")

        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self.stats = CacheStats()

        logger.debug(
            f"Cache '{name}' initialized: max_size={max_size}, "
            f"default_ttl={default_ttl}"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Value to return if key not found

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self.stats.misses += 1
                return default

            # Check expiration
            if entry.is_expired():
                self.stats.expirations += 1
                self.stats.misses += 1
                del self._cache[key]
                logger.debug(f"Cache '{self.name}': expired key '{key}'")
                return default

            # Update access metadata and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)
            self.stats.hits += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides default_ttl)
        """
        with self._lock:
            # Use provided TTL or fall back to default
            actual_ttl = ttl if ttl is not None else self.default_ttl

            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=actual_ttl
            )

            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self.stats.sets += 1

            # Evict oldest entries if cache is full
            while len(self._cache) > self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self.stats.evictions += 1
                logger.debug(
                    f"Cache '{self.name}': evicted key '{evicted_key}' "
                    f"(size={len(self._cache)}/{self.max_size})"
                )

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            logger.info(f"Cache '{self.name}' cleared")

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]
                self.stats.expirations += 1

            if expired_keys:
                logger.debug(
                    f"Cache '{self.name}': cleaned up {len(expired_keys)} "
                    f"expired entries"
                )

            return len(expired_keys)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (without updating access time)."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                self.stats.expirations += 1
                return False
            return True

    def __len__(self) -> int:
        """Get cache size."""
        return self.size()


class PersistentCache(Cache):
    """
    Cache with persistent disk storage.

    Extends in-memory cache with disk persistence for durability across restarts.
    """

    def __init__(
        self,
        name: str = "default",
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        cache_dir: Optional[Path] = None,
        auto_save: bool = True,
        save_interval: int = 100  # Save after N operations
    ):
        """
        Initialize persistent cache.

        Args:
            name: Cache name
            max_size: Maximum entries
            default_ttl: Default TTL in seconds
            cache_dir: Directory for cache files
            auto_save: Automatically save to disk
            save_interval: Save after N set operations
        """
        super().__init__(name, max_size, default_ttl)

        self.cache_dir = cache_dir or Path.home() / ".gazebo_world_generator" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save = auto_save
        self.save_interval = save_interval
        self._operations_since_save = 0

        # Load existing cache from disk
        self._load()

        logger.debug(
            f"Persistent cache '{name}' initialized at {self.cache_dir}"
        )

    @property
    def cache_file(self) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{self.name}.cache"

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value with optional auto-save."""
        super().set(key, value, ttl)

        self._operations_since_save += 1
        if self.auto_save and self._operations_since_save >= self.save_interval:
            self.save()
            self._operations_since_save = 0

    def save(self):
        """Save cache to disk."""
        try:
            with self._lock:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(dict(self._cache), f)
                logger.debug(
                    f"Saved cache '{self.name}' to disk ({len(self._cache)} entries)"
                )
        except Exception as e:
            logger.error(f"Failed to save cache '{self.name}': {e}")

    def _load(self):
        """Load cache from disk."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)

            # Filter expired entries
            valid_entries = {
                key: entry for key, entry in loaded_cache.items()
                if not entry.is_expired()
            }

            self._cache = OrderedDict(valid_entries)
            logger.info(
                f"Loaded cache '{self.name}' from disk "
                f"({len(self._cache)}/{len(loaded_cache)} entries valid)"
            )
        except Exception as e:
            logger.error(f"Failed to load cache '{self.name}': {e}")

    def clear(self):
        """Clear cache and delete disk file."""
        super().clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info(f"Deleted cache file: {self.cache_file}")


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Hash-based cache key
    """
    # Create deterministic string representation
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(key_parts)

    # Hash for consistent length
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def cached(
    cache: Cache,
    ttl: Optional[float] = None,
    key_prefix: str = ""
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        cache: Cache instance to use
        ttl: Time to live for cached results
        key_prefix: Prefix for cache keys

    Usage:
        my_cache = Cache(name="function_results")

        @cached(my_cache, ttl=3600)
        def expensive_function(x, y):
            return x + y
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            key = f"{key_prefix}{func.__name__}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Call function and cache result
            logger.debug(f"Cache miss for {func.__name__}, executing")
            result = func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)
            return result

        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_stats = lambda: cache.get_stats()

        return wrapper
    return decorator


# Global cache registry
_cache_registry: Dict[str, Cache] = {}
_registry_lock = RLock()


def get_cache(
    name: str,
    max_size: int = 1000,
    default_ttl: Optional[float] = None,
    persistent: bool = False
) -> Cache:
    """
    Get or create a cache from global registry.

    Args:
        name: Cache name
        max_size: Maximum entries
        default_ttl: Default TTL
        persistent: Use persistent disk storage

    Returns:
        Cache instance
    """
    with _registry_lock:
        if name in _cache_registry:
            return _cache_registry[name]

        if persistent:
            cache = PersistentCache(
                name=name,
                max_size=max_size,
                default_ttl=default_ttl
            )
        else:
            cache = Cache(
                name=name,
                max_size=max_size,
                default_ttl=default_ttl
            )

        _cache_registry[name] = cache
        return cache


def get_all_cache_stats() -> Dict[str, dict]:
    """Get statistics for all registered caches."""
    with _registry_lock:
        return {
            name: cache.get_stats().to_dict()
            for name, cache in _cache_registry.items()
        }
