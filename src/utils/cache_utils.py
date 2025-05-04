"""
Cache utilities for storing and retrieving data with expiry.

This module provides functions to cache API responses and other data
to improve performance and reduce API calls.
"""

import os
import json
import logging
import hashlib
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def _get_cache_path(cache_dir: Union[str, Path], key: str) -> Path:
    """
    Generate a cache file path from a cache key.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        
    Returns:
        Path to the cache file
    """
    # Create a hash of the key to use as filename
    filename = hashlib.md5(key.encode('utf-8')).hexdigest() + '.json'
    return Path(cache_dir) / filename

def save_to_cache(cache_dir: Union[str, Path], key: str, data: Any) -> None:
    """
    Save data to cache with timestamp.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        data: Data to cache (must be JSON serializable)
    """
    try:
        cache_path = _get_cache_path(cache_dir, key)
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Add timestamp to cached data
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Write to cache file
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
            
        logger.debug(f"Data saved to cache: {cache_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save data to cache: {e}")

def load_from_cache(cache_dir: Union[str, Path], key: str, expiry_hours: int = 24) -> Optional[Any]:
    """
    Load data from cache if it exists and is not expired.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        expiry_hours: Cache expiry time in hours
        
    Returns:
        Cached data if found and not expired, None otherwise
    """
    try:
        cache_path = _get_cache_path(cache_dir, key)
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            return None
            
        # Read cache file
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            
        # Parse cache timestamp
        timestamp = datetime.fromisoformat(cache_data['timestamp'])
        
        # Check if cache is expired
        if datetime.now() - timestamp > timedelta(hours=expiry_hours):
            logger.debug(f"Cache expired: {cache_path}")
            return None
            
        logger.debug(f"Using cached data from: {cache_path}")
        return cache_data['data']
        
    except Exception as e:
        logger.warning(f"Failed to load data from cache: {e}")
        return None

def clear_cache(cache_dir: Union[str, Path], older_than_hours: Optional[int] = None) -> int:
    """
    Clear all cache files or only those older than a specified time.
    
    Args:
        cache_dir: Directory for cache files
        older_than_hours: Only clear files older than this many hours (None for all files)
        
    Returns:
        Number of files deleted
    """
    try:
        cache_dir_path = Path(cache_dir)
        
        if not cache_dir_path.exists():
            return 0
            
        count = 0
        for cache_file in cache_dir_path.glob('*.json'):
            try:
                if older_than_hours is not None:
                    # Read cache file to check timestamp
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        
                    # Parse cache timestamp
                    timestamp = datetime.fromisoformat(cache_data['timestamp'])
                    
                    # Skip if not old enough
                    if datetime.now() - timestamp <= timedelta(hours=older_than_hours):
                        continue
                
                # Delete the file
                os.remove(cache_file)
                count += 1
                
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                
        logger.info(f"Cleared {count} cache files from {cache_dir}")
        return count
        
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")
        return 0

def get_cache_size(cache_dir: Union[str, Path]) -> int:
    """
    Get the total size of all cache files in bytes.
    
    Args:
        cache_dir: Directory for cache files
        
    Returns:
        Total size in bytes
    """
    try:
        cache_dir_path = Path(cache_dir)
        
        if not cache_dir_path.exists():
            return 0
            
        total_size = 0
        for cache_file in cache_dir_path.glob('*.json'):
            total_size += os.path.getsize(cache_file)
                
        return total_size
        
    except Exception as e:
        logger.warning(f"Failed to get cache size: {e}")
        return 0 