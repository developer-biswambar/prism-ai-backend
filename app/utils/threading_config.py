#!/usr/bin/env python3
"""
Centralized Hardware-Aware Threading Configuration

This module provides intelligent thread allocation optimized for different server classes,
particularly high-end servers like Intel Xeon Platinum 8260 with 48 cores.

Usage:
    from app.utils.threading_config import get_threading_config
    
    config = get_threading_config('reconciliation')  # Get config for reconciliation workload
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Use config.batch_size for optimal batching
"""

import logging
import os
from multiprocessing import cpu_count
from typing import NamedTuple, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class ThreadingConfig:
    """Configuration for hardware-aware threading"""
    max_workers: int
    batch_size: int
    server_class: str
    available_cores: int
    optimization_level: str
    timeout_multiplier: float = 1.0
    memory_efficiency_mode: bool = False

class WorkloadType:
    """Predefined workload types with specific optimization characteristics"""
    RECONCILIATION = "reconciliation"      # CPU-intensive matching operations
    DATA_CLEANING = "data_cleaning"       # Mixed I/O and CPU operations  
    DATE_PROCESSING = "date_processing"   # String parsing and conversion
    FILE_PROCESSING = "file_processing"   # I/O heavy operations
    AI_PROCESSING = "ai_processing"       # LLM/AI operations
    GENERAL = "general"                   # General purpose parallel processing

@lru_cache(maxsize=None)
def _detect_hardware_class() -> tuple:
    """
    Detect hardware class and return (cores, class_name, optimization_level)
    
    Returns:
        Tuple of (available_cores, server_class, optimization_level)
    """
    available_cores = cpu_count()
    
    # Allow override via environment variable for testing/configuration
    env_cores = os.getenv('FTT_ML_CORES')
    if env_cores and env_cores.isdigit():
        available_cores = int(env_cores)
        logger.info(f"🔧 Using environment override: {available_cores} cores")
    
    if available_cores >= 40:
        return available_cores, "high_end_server", "maximum_performance"
    elif available_cores >= 20:
        return available_cores, "mid_range_server", "high_performance"  
    elif available_cores >= 6:   # Include 6-core systems for parallel processing
        return available_cores, "standard_workstation", "balanced"
    else:
        return available_cores, "limited_system", "conservative"

def _calculate_optimal_threading(available_cores: int, server_class: str, workload_type: str) -> Dict:
    """
    Calculate optimal threading configuration based on hardware and workload
    
    Args:
        available_cores: Number of available CPU cores
        server_class: Detected server class
        workload_type: Type of workload being optimized
        
    Returns:
        Dictionary with threading parameters
    """
    
    # Base configurations by server class
    base_configs = {
        "high_end_server": {
            "reserve_cores": 4,
            "base_batch_size": 5000,
            "timeout_multiplier": 1.5,
            "memory_efficiency": False
        },
        "mid_range_server": {
            "reserve_cores": 2, 
            "base_batch_size": 3000,
            "timeout_multiplier": 1.2,
            "memory_efficiency": False
        },
        "standard_workstation": {
            "reserve_cores": 1,
            "base_batch_size": 2000,  # Set to 2000 as requested
            "timeout_multiplier": 1.0,
            "memory_efficiency": True
        },
        "limited_system": {
            "reserve_cores": 1,
            "base_batch_size": 1000,
            "timeout_multiplier": 0.8,
            "memory_efficiency": True
        }
    }
    
    base_config = base_configs[server_class]
    usable_cores = available_cores - base_config["reserve_cores"]
    
    # Workload-specific adjustments
    workload_adjustments = {
        WorkloadType.RECONCILIATION: {
            "max_worker_multiplier": 0.8,  # Increased for better parallelization on 6-core systems
            "batch_size_multiplier": 1.0,  # Use base batch size directly (2000 for 6-core systems)
            "max_workers_cap": 32
        },
        WorkloadType.DATA_CLEANING: {
            "max_worker_multiplier": 0.8,  # Mixed workload
            "batch_size_multiplier": 1.2,
            "max_workers_cap": 28
        },
        WorkloadType.DATE_PROCESSING: {
            "max_worker_multiplier": 0.7,  # String processing intensive
            "batch_size_multiplier": 1.0,
            "max_workers_cap": 24
        },
        WorkloadType.FILE_PROCESSING: {
            "max_worker_multiplier": 0.6,  # I/O bound
            "batch_size_multiplier": 0.8,
            "max_workers_cap": 20
        },
        WorkloadType.AI_PROCESSING: {
            "max_worker_multiplier": 0.4,  # Memory and API rate limited
            "batch_size_multiplier": 0.5,
            "max_workers_cap": 8
        },
        WorkloadType.GENERAL: {
            "max_worker_multiplier": 0.6,  # Conservative general purpose
            "batch_size_multiplier": 1.0,
            "max_workers_cap": 20
        }
    }
    
    adjustment = workload_adjustments.get(workload_type, workload_adjustments[WorkloadType.GENERAL])
    
    # Calculate final values with minimum guarantees for parallel processing
    max_workers = min(
        int(usable_cores * adjustment["max_worker_multiplier"]),
        adjustment["max_workers_cap"]
    )
    # Ensure at least 2 workers for parallel processing on systems with 6+ cores
    if available_cores >= 6:
        max_workers = max(max_workers, 2)  # Force at least 2 workers for 6+ core systems
    else:
        max_workers = max(max_workers, 1)  # Single worker for limited systems
    
    batch_size = int(base_config["base_batch_size"] * adjustment["batch_size_multiplier"])
    batch_size = max(batch_size, 100)  # Minimum reasonable batch size
    
    return {
        "max_workers": max_workers,
        "batch_size": batch_size,
        "timeout_multiplier": base_config["timeout_multiplier"],
        "memory_efficiency_mode": base_config["memory_efficiency"]
    }

@lru_cache(maxsize=32)  # Cache configurations for different workload types
def get_threading_config(workload_type: str = WorkloadType.GENERAL, 
                        max_workers_override: Optional[int] = None,
                        batch_size_override: Optional[int] = None) -> ThreadingConfig:
    """
    Get optimized threading configuration for specific workload type
    
    Args:
        workload_type: Type of workload (use WorkloadType constants)
        max_workers_override: Override max_workers if specified
        batch_size_override: Override batch_size if specified
        
    Returns:
        ThreadingConfig with optimal settings for the workload and hardware
        
    Examples:
        >>> config = get_threading_config(WorkloadType.RECONCILIATION)
        >>> print(f"Using {config.max_workers} workers with batch size {config.batch_size}")
        
        >>> # Override for testing
        >>> config = get_threading_config(WorkloadType.DATA_CLEANING, max_workers_override=4)
    """
    available_cores, server_class, optimization_level = _detect_hardware_class()
    
    # Calculate optimal configuration
    threading_params = _calculate_optimal_threading(available_cores, server_class, workload_type)
    
    # Apply overrides if provided
    max_workers = max_workers_override or threading_params["max_workers"]
    batch_size = batch_size_override or threading_params["batch_size"]
    
    config = ThreadingConfig(
        max_workers=max_workers,
        batch_size=batch_size,
        server_class=server_class,
        available_cores=available_cores,
        optimization_level=optimization_level,
        timeout_multiplier=threading_params["timeout_multiplier"],
        memory_efficiency_mode=threading_params["memory_efficiency_mode"]
    )
    
    # Log configuration on first use for each workload type
    logger.info(f"🚀 Threading Config [{workload_type}]: {max_workers} workers, "
               f"{batch_size} batch size on {server_class} ({available_cores} cores)")
    
    return config

def get_timeout_for_operation(base_timeout_seconds: float, config: ThreadingConfig) -> float:
    """
    Calculate adaptive timeout based on hardware configuration
    
    Args:
        base_timeout_seconds: Base timeout for the operation
        config: Threading configuration
        
    Returns:
        Adjusted timeout in seconds
    """
    return base_timeout_seconds * config.timeout_multiplier

def log_threading_summary():
    """Log a summary of available threading configurations"""
    available_cores, server_class, optimization_level = _detect_hardware_class()
    
    logger.info("=" * 60)
    logger.info(f"🖥️  Hardware Detection Summary")
    logger.info(f"   Available Cores: {available_cores}")
    logger.info(f"   Server Class: {server_class}")
    logger.info(f"   Optimization Level: {optimization_level}")
    logger.info("=" * 60)
    
    # Show configurations for all workload types
    workload_types = [
        WorkloadType.RECONCILIATION,
        WorkloadType.DATA_CLEANING, 
        WorkloadType.DATE_PROCESSING,
        WorkloadType.FILE_PROCESSING,
        WorkloadType.AI_PROCESSING,
        WorkloadType.GENERAL
    ]
    
    for workload in workload_types:
        config = get_threading_config(workload)
        logger.info(f"📊 {workload.upper()}: {config.max_workers} workers, {config.batch_size} batch size")
    
    logger.info("=" * 60)

# Convenience functions for common workload types
def get_reconciliation_config(max_workers_override: Optional[int] = None) -> ThreadingConfig:
    """Get optimized configuration for reconciliation workloads"""
    config = get_threading_config(WorkloadType.RECONCILIATION, max_workers_override)
    
    # Special handling for 6-core systems to ensure proper parallel processing
    if config.available_cores == 6 and config.batch_size != 2000:
        logger.info(f"🔧 Adjusting 6-core system: Setting batch size to 2000 (was {config.batch_size})")
        config.batch_size = 2000
        
    return config

def get_cleaning_config(max_workers_override: Optional[int] = None) -> ThreadingConfig:
    """Get optimized configuration for data cleaning workloads"""
    return get_threading_config(WorkloadType.DATA_CLEANING, max_workers_override)

def get_date_processing_config(max_workers_override: Optional[int] = None) -> ThreadingConfig:
    """Get optimized configuration for date processing workloads"""
    return get_threading_config(WorkloadType.DATE_PROCESSING, max_workers_override)

def get_ai_processing_config(max_workers_override: Optional[int] = None) -> ThreadingConfig:
    """Get optimized configuration for AI/LLM processing workloads"""
    return get_threading_config(WorkloadType.AI_PROCESSING, max_workers_override)

# Environment variable configuration
def set_cores_override(cores: int):
    """Set cores override via environment variable (useful for testing)"""
    os.environ['FTT_ML_CORES'] = str(cores)
    # Clear cache to force recalculation
    _detect_hardware_class.cache_clear()
    get_threading_config.cache_clear()
    logger.info(f"🔧 Cores override set to {cores} - configurations will be recalculated")

def force_config_refresh():
    """Force refresh of threading configuration (clears cache)"""
    _detect_hardware_class.cache_clear()
    get_threading_config.cache_clear()
    logger.info("🔄 Threading configuration cache cleared - will recalculate on next use")