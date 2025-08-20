#!/usr/bin/env python3
"""
Global Thread Pool Manager for the Financial Data Processing Application

This module provides a centralized thread pool management system that can be shared
across all parallel processing components (reconciliation, data cleaning, date processing).

Key Benefits:
- Prevents thread pool proliferation 
- Better resource management
- Consistent threading behavior
- Improved health check performance by reducing thread contention
"""

import logging
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PoolType(Enum):
    """Types of thread pools available"""
    GENERAL = "general"           # General purpose work
    DATA_PROCESSING = "data"      # Data cleaning, transformation
    RECONCILIATION = "recon"      # Reconciliation operations
    DATE_PROCESSING = "date"      # Date normalization
    IO_BOUND = "io"              # File I/O, S3 operations


@dataclass
class PoolConfig:
    """Configuration for a thread pool"""
    max_workers: int
    pool_type: PoolType
    timeout: float = 300.0  # 5 minutes default timeout
    queue_size: Optional[int] = None


class GlobalThreadPoolManager:
    """
    Centralized thread pool manager that provides shared pools across the application.
    
    Features:
    - Lazy initialization of pools
    - Automatic cleanup on shutdown
    - Resource monitoring
    - Pool sharing and reuse
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GlobalThreadPoolManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._pools: Dict[PoolType, ThreadPoolExecutor] = {}
        self._pool_configs: Dict[PoolType, PoolConfig] = {}
        self._pool_stats: Dict[PoolType, Dict[str, Any]] = {}
        self._shutdown = False
        self._active_futures: List[weakref.ref] = []
        
        # Initialize with environment-based configuration
        self._setup_default_configs()
        
        logger.info("GlobalThreadPoolManager initialized")
    
    def _setup_default_configs(self):
        """Setup default pool configurations based on system resources"""
        # Get system configuration
        env_cores = os.getenv('FTT_ML_CORES')
        max_workers = int(env_cores) if env_cores and env_cores.isdigit() else 4
        
        # Conservative approach: limit total threads to prevent resource exhaustion
        total_cores = max_workers
        
        # Distribute threads across different workload types
        self._pool_configs = {
            PoolType.GENERAL: PoolConfig(
                max_workers=max(2, total_cores // 4),
                pool_type=PoolType.GENERAL,
                timeout=60.0
            ),
            PoolType.DATA_PROCESSING: PoolConfig(
                max_workers=max(2, total_cores // 2),
                pool_type=PoolType.DATA_PROCESSING, 
                timeout=300.0
            ),
            PoolType.RECONCILIATION: PoolConfig(
                max_workers=max(2, total_cores // 2),
                pool_type=PoolType.RECONCILIATION,
                timeout=600.0
            ),
            PoolType.DATE_PROCESSING: PoolConfig(
                max_workers=max(2, total_cores // 3),
                pool_type=PoolType.DATE_PROCESSING,
                timeout=300.0
            ),
            PoolType.IO_BOUND: PoolConfig(
                max_workers=max(4, total_cores),  # I/O can handle more threads
                pool_type=PoolType.IO_BOUND,
                timeout=120.0
            )
        }
        
        logger.info(f"Thread pool configuration setup with {total_cores} total cores")
        for pool_type, config in self._pool_configs.items():
            logger.info(f"  {pool_type.value}: {config.max_workers} workers")
    
    def get_pool(self, pool_type: PoolType) -> ThreadPoolExecutor:
        """Get or create a thread pool of the specified type"""
        if self._shutdown:
            raise RuntimeError("Thread pool manager has been shut down")
            
        if pool_type not in self._pools:
            with self._lock:
                # Double-check locking pattern
                if pool_type not in self._pools:
                    config = self._pool_configs[pool_type]
                    
                    self._pools[pool_type] = ThreadPoolExecutor(
                        max_workers=config.max_workers,
                        thread_name_prefix=f"ftt-{pool_type.value}"
                    )
                    
                    self._pool_stats[pool_type] = {
                        'created_at': time.time(),
                        'tasks_submitted': 0,
                        'tasks_completed': 0,
                        'max_workers': config.max_workers
                    }
                    
                    logger.info(f"Created thread pool '{pool_type.value}' with {config.max_workers} workers")
        
        return self._pools[pool_type]
    
    @contextmanager
    def get_executor(self, pool_type: PoolType):
        """Context manager for getting a thread pool executor"""
        executor = self.get_pool(pool_type)
        try:
            yield executor
        finally:
            # Pool is reused, no cleanup needed here
            pass
    
    def submit_task(self, pool_type: PoolType, fn: Callable, *args, **kwargs) -> Future:
        """Submit a task to the specified pool type"""
        executor = self.get_pool(pool_type)
        
        # Track task submission
        if pool_type in self._pool_stats:
            self._pool_stats[pool_type]['tasks_submitted'] += 1
        
        future = executor.submit(fn, *args, **kwargs)
        
        # Track completion (weakref to avoid circular references)
        def on_done(fut):
            if pool_type in self._pool_stats:
                self._pool_stats[pool_type]['tasks_completed'] += 1
        
        future.add_done_callback(on_done)
        
        # Keep weak reference to active futures for monitoring
        self._active_futures.append(weakref.ref(future))
        
        return future
    
    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools"""
        stats = {}
        
        for pool_type, pool_stats in self._pool_stats.items():
            active_futures = sum(1 for ref in self._active_futures if ref() is not None and not ref().done())
            
            stats[pool_type.value] = {
                **pool_stats,
                'active_tasks': active_futures,
                'pool_created': pool_type in self._pools
            }
        
        # Clean up dead references
        self._active_futures = [ref for ref in self._active_futures if ref() is not None]
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the thread pool manager"""
        total_active = 0
        total_submitted = 0
        total_completed = 0
        
        for stats in self._pool_stats.values():
            total_submitted += stats.get('tasks_submitted', 0)
            total_completed += stats.get('tasks_completed', 0)
        
        # Count active futures
        active_futures = [ref() for ref in self._active_futures if ref() is not None]
        total_active = sum(1 for fut in active_futures if not fut.done())
        
        return {
            'status': 'healthy' if not self._shutdown else 'shutdown',
            'total_pools_created': len(self._pools),
            'total_tasks_submitted': total_submitted,
            'total_tasks_completed': total_completed,
            'active_tasks': total_active,
            'pools': self.get_pool_stats()
        }
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """Shutdown all thread pools"""
        if self._shutdown:
            return
            
        logger.info("Shutting down GlobalThreadPoolManager...")
        self._shutdown = True
        
        shutdown_start = time.time()
        
        for pool_type, executor in self._pools.items():
            try:
                logger.info(f"Shutting down {pool_type.value} pool...")
                executor.shutdown(wait=wait)
                logger.info(f"✓ {pool_type.value} pool shut down")
            except Exception as e:
                logger.error(f"Error shutting down {pool_type.value} pool: {e}")
        
        shutdown_time = time.time() - shutdown_start
        logger.info(f"GlobalThreadPoolManager shutdown completed in {shutdown_time:.2f}s")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, '_pools') and not self._shutdown:
            self.shutdown(wait=False, timeout=5.0)


# Global instance
_global_pool_manager = None
_manager_lock = threading.Lock()


def get_global_thread_pool() -> GlobalThreadPoolManager:
    """Get the global thread pool manager instance"""
    global _global_pool_manager
    
    if _global_pool_manager is None:
        with _manager_lock:
            if _global_pool_manager is None:
                _global_pool_manager = GlobalThreadPoolManager()
    
    return _global_pool_manager


# Convenience functions for common operations
def submit_data_processing_task(fn: Callable, *args, **kwargs) -> Future:
    """Submit a data processing task"""
    manager = get_global_thread_pool()
    return manager.submit_task(PoolType.DATA_PROCESSING, fn, *args, **kwargs)


def submit_reconciliation_task(fn: Callable, *args, **kwargs) -> Future:
    """Submit a reconciliation task"""
    manager = get_global_thread_pool()
    return manager.submit_task(PoolType.RECONCILIATION, fn, *args, **kwargs)


def submit_date_processing_task(fn: Callable, *args, **kwargs) -> Future:
    """Submit a date processing task"""
    manager = get_global_thread_pool()
    return manager.submit_task(PoolType.DATE_PROCESSING, fn, *args, **kwargs)


def submit_io_task(fn: Callable, *args, **kwargs) -> Future:
    """Submit an I/O bound task"""
    manager = get_global_thread_pool()
    return manager.submit_task(PoolType.IO_BOUND, fn, *args, **kwargs)


@contextmanager
def get_data_processing_executor():
    """Get data processing executor as context manager"""
    manager = get_global_thread_pool()
    with manager.get_executor(PoolType.DATA_PROCESSING) as executor:
        yield executor


@contextmanager
def get_reconciliation_executor():
    """Get reconciliation executor as context manager"""
    manager = get_global_thread_pool()
    with manager.get_executor(PoolType.RECONCILIATION) as executor:
        yield executor


@contextmanager 
def get_date_processing_executor():
    """Get date processing executor as context manager"""
    manager = get_global_thread_pool()
    with manager.get_executor(PoolType.DATE_PROCESSING) as executor:
        yield executor


@contextmanager
def get_io_executor():
    """Get I/O executor as context manager"""
    manager = get_global_thread_pool()
    with manager.get_executor(PoolType.IO_BOUND) as executor:
        yield executor


# Cleanup function for application shutdown
def shutdown_global_pools(wait: bool = True, timeout: float = 30.0):
    """Shutdown global thread pools - call this during application shutdown"""
    global _global_pool_manager
    
    if _global_pool_manager is not None:
        _global_pool_manager.shutdown(wait=wait, timeout=timeout)
        _global_pool_manager = None


if __name__ == "__main__":
    # Test the global thread pool
    import asyncio
    
    def test_task(task_id: int, duration: float = 0.1):
        time.sleep(duration)
        return f"Task {task_id} completed"
    
    # Test the thread pool manager
    manager = get_global_thread_pool()
    
    print("Testing global thread pool manager...")
    
    # Submit some test tasks
    futures = []
    for i in range(10):
        future = submit_data_processing_task(test_task, i, 0.1)
        futures.append(future)
    
    # Wait for completion
    for future in futures:
        print(future.result())
    
    # Print statistics
    print("\nPool Statistics:")
    stats = manager.get_health_status()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Shutdown
    shutdown_global_pools()
    print("Test completed successfully!")