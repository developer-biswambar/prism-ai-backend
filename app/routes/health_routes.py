import logging
import os
from datetime import datetime

from fastapi import APIRouter

from app.services.storage_service import uploaded_files, extractions
from app.services.llm_service import get_llm_service

# Get configuration from environment
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()


# API Endpoints
@router.get("/health")
async def health_check():
    # Get LLM service status
    llm_service = get_llm_service()
    llm_available = llm_service.is_available()
    llm_provider = llm_service.get_provider_name()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "llm_configured": llm_available,
        "llm_provider": llm_provider,
        "multi_column_support": True,
        "batch_processing_enabled": True,
        "current_batch_size": BATCH_SIZE,
        "uploaded_files_count": len(uploaded_files),
        "extractions_count": len(extractions)
    }


@router.get("/config")
async def get_config():
    # Get LLM service configuration
    llm_service = get_llm_service()
    llm_available = llm_service.is_available()
    llm_provider = llm_service.get_provider_name()
    
    config_data = {
        "llm_configured": llm_available,
        "llm_provider": llm_provider,
        "batch_size": BATCH_SIZE,
        "multi_column_support": True,
    }
    
    # Add provider-specific information
    if llm_provider == "OpenAI":
        config_data["api_key_set"] = llm_available
        # Don't expose API key details for security
        config_data["api_key_preview"] = "sk-****" if llm_available else "Not set"
    elif llm_provider == "JPMC LLM":
        config_data["internal_service"] = True
        config_data["service_available"] = llm_available
    
    return {
        "success": True,
        "data": config_data
    }
