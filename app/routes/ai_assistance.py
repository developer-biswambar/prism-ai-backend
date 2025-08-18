# backend/app/api/routes/ai_assistance.py
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.openai_service import openai_service

router = APIRouter(prefix="/ai-assistance", tags=["AI Assistance"])


class GenericAIRequest(BaseModel):
    """Generic AI request for flexible prompting"""
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4000)
    response_format: Optional[Dict[str, str]] = None


class TransformationSuggestionRequest(BaseModel):
    """Request for transformation rule suggestions"""
    source_columns: Dict[str, List[str]] = Field(
        description="Source data columns grouped by file alias"
    )
    output_schema: Dict[str, Any] = Field(
        description="Target output schema definition"
    )
    transformation_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about the transformation"
    )


class AIResponse(BaseModel):
    """Standard AI response format"""
    success: bool
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None


@router.post("/generic-call", response_model=AIResponse)
async def generic_ai_call(
        request: GenericAIRequest
):
    """
    Make a generic AI call without predefined prompts
    Useful for custom AI assistance tasks
    """
    try:
        start_time = time.time()

        response_content = await openai_service.call_llm_generic(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format=request.response_format
        )

        processing_time = time.time() - start_time

        return AIResponse(
            success=True,
            content=response_content,
            processing_time=processing_time,
            model_used=openai_service.llm_service.get_provider_name()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI call failed: {str(e)}"
        )


@router.post("/suggest-transformations", response_model=AIResponse)
async def suggest_transformation_rules(
        request: TransformationSuggestionRequest
):
    """
    Get AI-powered suggestions for transformation rules
    Analyzes source data and output schema to suggest optimal transformations
    """
    try:
        result = await openai_service.suggest_transformation_rules(
            source_columns=request.source_columns,
            output_schema=request.output_schema,
            transformation_context=request.transformation_context
        )

        if result["success"]:
            return AIResponse(
                success=True,
                data=result["suggestions"],
                processing_time=result["processing_time"],
                model_used=result["model_used"]
            )
        else:
            return AIResponse(
                success=False,
                error=result["error"],
                data=result["suggestions"]  # Return empty suggestions even on error
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transformation suggestion failed: {str(e)}"
        )


@router.post("/analyze-data-patterns", response_model=AIResponse)
async def analyze_data_patterns(
        request: Dict[str, Any]
):
    """
    Analyze data patterns and provide insights
    Generic endpoint for data analysis tasks
    """
    try:
        system_prompt = """
You are a data analysis expert. Analyze the provided data structure and identify:
1. Data quality issues
2. Business logic patterns
3. Potential optimization opportunities
4. Data transformation recommendations

Provide insights in a structured, actionable format.
"""

        user_prompt = f"""
Please analyze this data structure and provide insights:

{request}

Focus on practical recommendations for data processing and transformation.
"""

        response_content = await openai_service.call_llm_generic(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3
        )

        return AIResponse(
            success=True,
            content=response_content,
            model_used=openai_service.llm_service.get_provider_name()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data pattern analysis failed: {str(e)}"
        )


@router.get("/test-connection")
async def test_ai_connection():
    """
    Test AI service connection
    """
    try:
        is_connected = await openai_service.test_connection()

        return {
            "success": is_connected,
            "message": "AI service is available" if is_connected else "AI service unavailable",
            "model": openai_service.llm_service.get_provider_name()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Connection test failed: {str(e)}"
        )


# Add import for time
import time
