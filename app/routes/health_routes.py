import logging
import os
from datetime import datetime

from fastapi import APIRouter

from app.utils.global_thread_pool import get_global_thread_pool

# Get configuration from environment
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()


# API Endpoints
@router.get("/actuator/health")
def health_check():
    """Simple, fast health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "financial-data-processing",
        "uptime": "running"
    }


@router.get("/config")
def get_config():
    """Get basic configuration information"""
    return {
        "success": True,
        "data": {
            "batch_size": BATCH_SIZE,
            "multi_column_support": True,
            "service": "financial-data-processing",
            "version": "2.0.0"
        }
    }


@router.get("/thread-pools/status")
def get_thread_pool_status():
    """Get detailed thread pool status and statistics"""
    thread_pool_manager = get_global_thread_pool()
    pool_health = thread_pool_manager.get_health_status()
    
    return {
        "success": True,
        "data": {
            "global_status": pool_health["status"],
            "summary": {
                "total_pools_created": pool_health["total_pools_created"],
                "total_tasks_submitted": pool_health["total_tasks_submitted"],
                "total_tasks_completed": pool_health["total_tasks_completed"],
                "active_tasks": pool_health["active_tasks"]
            },
            "pools": pool_health["pools"],
            "recommendations": {
                "status": "healthy" if pool_health["active_tasks"] < 50 else "consider_scaling",
                "message": "Thread pools operating normally" if pool_health["active_tasks"] < 50 else "High task load detected"
            }
        }
    }
