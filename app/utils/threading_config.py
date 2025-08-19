#!/usr/bin/env python3
"""
Simplified Threading Configuration

This module provides simple threading configuration based on provided thread count and batch size.

Usage:
    from app.utils.threading_config import get_threading_config
    
    config = get_threading_config(max_workers=4, batch_size=1000)
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Use config.batch_size for batching
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ThreadingConfig:
    """Simple threading configuration"""
    max_workers: int
    batch_size: int
    timeout_multiplier: float = 1.0
    # Backward compatibility attributes
    server_class: str = "standard"
    available_cores: int = 4
    optimization_level: str = "balanced"
    memory_efficiency_mode: bool = False


class WorkloadType:
    """Predefined workload types for backward compatibility"""
    RECONCILIATION = "reconciliation"
    DATA_CLEANING = "data_cleaning"
    DATE_PROCESSING = "date_processing"
    FILE_PROCESSING = "file_processing"
    AI_PROCESSING = "ai_processing"
    GENERAL = "general"


def get_threading_config(workload_type: str = WorkloadType.GENERAL,
                         max_workers_override: Optional[int] = None,
                         batch_size_override: Optional[int] = None,
                         max_workers: Optional[int] = None,
                         batch_size: Optional[int] = None) -> ThreadingConfig:
    """
    Get threading configuration
    
    Args:
        workload_type: Type of workload (for backward compatibility)
        max_workers_override: Override max_workers if specified (backward compatibility)
        batch_size_override: Override batch_size if specified (backward compatibility)
        max_workers: Number of worker threads to use
        batch_size: Batch size for processing
        
    Returns:
        ThreadingConfig with the specified settings
    """
    # Check environment variables
    env_cores = os.getenv('FTT_ML_CORES')
    env_max_workers = int(env_cores) if env_cores and env_cores.isdigit() else None

    env_batch_size_str = os.getenv('BATCH_SIZE')
    env_batch_size = int(env_batch_size_str) if env_batch_size_str and env_batch_size_str.isdigit() else None

    # Use parameters if provided, otherwise fall back to overrides, then env variables, then defaults
    final_max_workers = max_workers or max_workers_override or env_max_workers or 4
    final_batch_size = batch_size or batch_size_override or env_batch_size or 3000

    config = ThreadingConfig(
        max_workers=final_max_workers,
        batch_size=final_batch_size,
        timeout_multiplier=1.0,
        server_class="standard",
        available_cores=final_max_workers,  # Use max_workers as a proxy for available cores
        optimization_level="balanced",
        memory_efficiency_mode=False
    )

    logger.info(f"Threading Config [{workload_type}]: {final_max_workers} workers, {final_batch_size} batch size")

    return config


def get_timeout_for_operation(base_timeout_seconds: float, config: ThreadingConfig) -> float:
    """
    Calculate timeout based on configuration
    
    Args:
        base_timeout_seconds: Base timeout for the operation
        config: Threading configuration
        
    Returns:
        Adjusted timeout in seconds
    """
    return base_timeout_seconds * config.timeout_multiplier


# Convenience functions for backward compatibility
def get_reconciliation_config(max_workers_override: Optional[int] = None,
                              batch_size_override: Optional[int] = None) -> ThreadingConfig:
    """Get configuration for reconciliation workloads"""
    return get_threading_config(WorkloadType.RECONCILIATION, max_workers_override, batch_size_override)


def get_cleaning_config(max_workers_override: Optional[int] = None,
                        batch_size_override: Optional[int] = None) -> ThreadingConfig:
    """Get configuration for data cleaning workloads"""
    return get_threading_config(WorkloadType.DATA_CLEANING, max_workers_override, batch_size_override)


def get_date_processing_config(max_workers_override: Optional[int] = None,
                               batch_size_override: Optional[int] = None) -> ThreadingConfig:
    """Get configuration for date processing workloads"""
    return get_threading_config(WorkloadType.DATE_PROCESSING, max_workers_override, batch_size_override)


def get_ai_processing_config(max_workers_override: Optional[int] = None,
                             batch_size_override: Optional[int] = None) -> ThreadingConfig:
    """Get configuration for AI/LLM processing workloads"""
    return get_threading_config(WorkloadType.AI_PROCESSING, max_workers_override, batch_size_override)
