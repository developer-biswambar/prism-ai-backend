"""
Template Management API Routes
Provides REST endpoints for template CRUD operations, search, and analytics.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field, validator

from app.services.dynamodb_templates_service import dynamodb_templates_service
from app.services.template_application_service import template_application_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/templates", tags=["Templates"])


# Request/Response Models
class TemplateConfigModel(BaseModel):
    """Template configuration schema"""
    prompt_template: str = Field(..., description="Parameterized natural language prompt")
    required_columns: List[str] = Field(default_factory=list, description="Required column mappings")
    optional_columns: List[str] = Field(default_factory=list, description="Optional column mappings") 
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="User-configurable parameters")
    validation_rules: List[str] = Field(default_factory=list, description="Data validation requirements")
    output_format: Dict[str, Any] = Field(default_factory=dict, description="Expected output structure")
    sample_data: Dict[str, Any] = Field(default_factory=dict, description="Example input/output data")


class CreateTemplateRequest(BaseModel):
    """Request model for creating a new template"""
    name: str = Field(..., min_length=1, max_length=200, description="Template display name")
    description: str = Field(..., min_length=1, max_length=1000, description="Detailed description")
    template_type: str = Field(..., description="Template type")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    template_config: TemplateConfigModel = Field(..., description="Template configuration")
    is_public: bool = Field(default=False, description="Whether template is publicly shared")
    created_by: Optional[str] = Field(None, description="Creator user identifier")
    
    @validator('template_type')
    def validate_template_type(cls, v):
        valid_types = ['data_processing', 'reconciliation', 'analysis', 'transformation', 'reporting']
        if v not in valid_types:
            raise ValueError(f'template_type must be one of: {valid_types}')
        return v


class UpdateTemplateRequest(BaseModel):
    """Request model for updating an existing template"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    category: Optional[str] = Field(None)
    industry: Optional[str] = Field(None)
    tags: Optional[List[str]] = Field(None)
    template_config: Optional[TemplateConfigModel] = Field(None)
    is_public: Optional[bool] = Field(None)


class TemplateResponse(BaseModel):
    """Response model for template data"""
    id: str
    name: str
    description: str
    template_type: str
    category: str
    tags: List[str]
    template_config: Dict[str, Any]
    version: str
    is_public: bool
    created_by: Optional[str]
    created_at: str
    updated_at: str
    usage_count: int
    last_used_at: Optional[str]
    rating: float
    rating_count: int


class TemplateListResponse(BaseModel):
    """Response model for template list"""
    templates: List[TemplateResponse]
    total_count: int
    offset: int
    limit: int


class RateTemplateRequest(BaseModel):
    """Request model for rating a template"""
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating from 1.0 to 5.0")


# Template CRUD Routes

@router.post("/", response_model=TemplateResponse, status_code=201)
async def create_template(request: CreateTemplateRequest):
    """Create a new template"""
    try:
        # Generate unique template ID
        template_id = str(uuid.uuid4())
        
        # Prepare template data
        template_data = {
            "id": template_id,
            "name": request.name,
            "description": request.description,
            "template_type": request.template_type,
            "category": request.category,
            "industry": request.industry,
            "tags": request.tags,
            "template_config": request.template_config.dict(),
            "is_public": request.is_public,
            "created_by": request.created_by,
            "version": "1.0",
            "usage_count": 0,
            "rating": 0.0,
            "rating_count": 0
        }
        
        # Save to DynamoDB
        success = dynamodb_templates_service.save_template(template_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save template")
        
        logger.info(f"Created template {template_id}: {request.name}")
        return TemplateResponse(**template_data)
        
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str = Path(..., description="Template ID"),
    template_type: Optional[str] = Query(None, description="Template type for faster lookup")
):
    """Get a specific template by ID"""
    try:
        template = dynamodb_templates_service.get_template(template_id, template_type)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return TemplateResponse(**template)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str = Path(..., description="Template ID"),
    request: UpdateTemplateRequest = Body(...),
    template_type: Optional[str] = Query(None, description="Template type for faster lookup")
):
    """Update an existing template"""
    try:
        # Prepare updates (only include non-None values)
        updates = {}
        if request.name is not None:
            updates['name'] = request.name
        if request.description is not None:
            updates['description'] = request.description
        if request.category is not None:
            updates['category'] = request.category
        if request.industry is not None:
            updates['industry'] = request.industry
        if request.tags is not None:
            updates['tags'] = request.tags
        if request.template_config is not None:
            updates['template_config'] = request.template_config.dict()
        if request.is_public is not None:
            updates['is_public'] = request.is_public
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        # Update template
        success = dynamodb_templates_service.update_template(template_id, updates, template_type)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found or update failed")
        
        # Return updated template
        updated_template = dynamodb_templates_service.get_template(template_id, template_type)
        logger.info(f"Updated template {template_id}")
        return TemplateResponse(**updated_template)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: str = Path(..., description="Template ID"),
    template_type: Optional[str] = Query(None, description="Template type for faster lookup")
):
    """Delete a template"""
    try:
        success = dynamodb_templates_service.delete_template(template_id, template_type)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        logger.info(f"Deleted template {template_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")


# Template Discovery Routes

@router.get("/", response_model=TemplateListResponse)
async def list_templates(
    template_type: Optional[str] = Query(None, description="Filter by template type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_public: Optional[bool] = Query(None, description="Filter by public/private"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    limit: int = Query(50, ge=1, le=200, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """List templates with optional filtering"""
    try:
        templates = dynamodb_templates_service.list_templates(
            template_type=template_type,
            category=category,
            is_public=is_public,
            created_by=created_by,
            limit=limit,
            offset=offset
        )
        
        template_responses = [TemplateResponse(**template) for template in templates]
        
        return TemplateListResponse(
            templates=template_responses,
            total_count=len(template_responses),  # Note: This is just the current page count
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@router.get("/search/query", response_model=TemplateListResponse)
async def search_templates(
    q: str = Query(..., min_length=1, description="Search query"),
    template_type: Optional[str] = Query(None, description="Filter by template type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags")
):
    """Search templates by name, description, or tags"""
    try:
        templates = dynamodb_templates_service.search_templates(
            search_query=q,
            template_type=template_type,
            category=category,
            tags=tags
        )
        
        template_responses = [TemplateResponse(**template) for template in templates]
        
        return TemplateListResponse(
            templates=template_responses,
            total_count=len(template_responses),
            offset=0,
            limit=len(template_responses)
        )
        
    except Exception as e:
        logger.error(f"Error searching templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search templates: {str(e)}")


@router.get("/popular/list", response_model=TemplateListResponse)
async def get_popular_templates(
    limit: int = Query(10, ge=1, le=50, description="Number of popular templates to return"),
    template_type: Optional[str] = Query(None, description="Filter by template type")
):
    """Get most popular templates by usage and rating"""
    try:
        templates = dynamodb_templates_service.get_popular_templates(
            limit=limit,
            template_type=template_type
        )
        
        template_responses = [TemplateResponse(**template) for template in templates]
        
        return TemplateListResponse(
            templates=template_responses,
            total_count=len(template_responses),
            offset=0,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error getting popular templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get popular templates: {str(e)}")


# Template Analytics Routes

@router.post("/{template_id}/usage", status_code=204)
async def mark_template_usage(
    template_id: str = Path(..., description="Template ID"),
    template_type: Optional[str] = Query(None, description="Template type for faster lookup")
):
    """Mark template as used (increment usage count)"""
    try:
        success = dynamodb_templates_service.mark_template_as_used(template_id, template_type)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        logger.info(f"Marked template {template_id} as used")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking template usage {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark template usage: {str(e)}")


@router.post("/{template_id}/rating", status_code=204)
async def rate_template(
    template_id: str = Path(..., description="Template ID"),
    request: RateTemplateRequest = Body(...),
    template_type: Optional[str] = Query(None, description="Template type for faster lookup")
):
    """Rate a template (1-5 stars)"""
    try:
        success = dynamodb_templates_service.rate_template(
            template_id, 
            request.rating, 
            template_type
        )
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        logger.info(f"Rated template {template_id} with {request.rating} stars")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rating template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rate template: {str(e)}")


# Utility Routes

@router.get("/categories/list", response_model=List[str])
async def get_template_categories(
    template_type: Optional[str] = Query(None, description="Filter by template type")
):
    """Get all available template categories"""
    try:
        categories = dynamodb_templates_service.get_categories(template_type)
        return categories
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@router.get("/types/list", response_model=List[str])
async def get_template_types():
    """Get all available template types"""
    return ['data_processing', 'reconciliation', 'analysis', 'transformation', 'reporting']


@router.get("/health/check")
async def health_check():
    """Health check endpoint for template service"""
    try:
        health_status = dynamodb_templates_service.get_health_status()
        
        if health_status['status'] == 'healthy':
            return {
                "status": "healthy",
                "service": "template_service",
                "database": health_status
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "service": "template_service",
                    "database": health_status
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "service": "template_service",
                "error": str(e)
            }
        )


# Template Application Routes

class TemplateSuggestionRequest(BaseModel):
    """Request model for template suggestions"""
    user_prompt: str = Field(..., min_length=1, description="User's natural language query")
    file_schemas: List[Dict[str, Any]] = Field(..., description="File schema information")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of suggestions")


class TemplateApplicationRequest(BaseModel):
    """Request model for applying a template"""
    template_id: str = Field(..., description="Template ID to apply")
    files: List[Dict[str, Any]] = Field(..., description="User's file data")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Template parameters")


class CreateTemplateFromQueryRequest(BaseModel):
    """Request model for creating template from successful query"""
    query_data: Dict[str, Any] = Field(..., description="Successful query execution data")
    template_name: str = Field(..., min_length=1, max_length=200, description="Template name")
    template_description: str = Field(..., min_length=1, max_length=1000, description="Template description")
    template_type: str = Field(..., description="Template type")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    is_public: bool = Field(default=False, description="Make template public")
    created_by: Optional[str] = Field(None, description="Creator identifier")


@router.post("/suggest", response_model=TemplateListResponse)
async def suggest_templates(request: TemplateSuggestionRequest):
    """Suggest templates based on user query and data structure"""
    try:
        suggestions = template_application_service.suggest_templates(
            user_prompt=request.user_prompt,
            file_schemas=request.file_schemas,
            limit=request.limit
        )
        
        # Convert to response format
        template_responses = []
        for suggestion in suggestions:
            # Add suggestion-specific fields
            template_data = {**suggestion}
            template_data.pop('match_score', None)  # Remove from main data
            template_data.pop('match_reasons', None)
            
            template_resp = TemplateResponse(**template_data)
            # Add suggestion metadata (these would need to be added to the model)
            # template_resp.match_score = suggestion.get('match_score', 0.0)
            # template_resp.match_reasons = suggestion.get('match_reasons', [])
            template_responses.append(template_resp)
        
        return TemplateListResponse(
            templates=template_responses,
            total_count=len(template_responses),
            offset=0,
            limit=request.limit
        )
        
    except Exception as e:
        logger.error(f"Error suggesting templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to suggest templates: {str(e)}")


@router.post("/apply", response_model=Dict[str, Any])
async def apply_template(request: TemplateApplicationRequest):
    """Apply a template to user data"""
    try:
        # Prepare user data context
        user_data = {
            'files': request.files
        }
        
        result = template_application_service.apply_template(
            template_id=request.template_id,
            user_data=user_data,
            user_params=request.parameters
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=400, 
                detail=f"Template application failed: {result.get('error')}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply template: {str(e)}")


@router.post("/create-from-query", response_model=TemplateResponse, status_code=201)
async def create_template_from_query(request: CreateTemplateFromQueryRequest):
    """Create a new template from a successful query execution"""
    try:
        # Prepare template metadata
        template_metadata = {
            'name': request.template_name,
            'description': request.template_description,
            'template_type': request.template_type,
            'category': request.category,
            'tags': request.tags,
            'is_public': request.is_public,
            'created_by': request.created_by
        }
        
        result = template_application_service.create_template_from_successful_query(
            query_data=request.query_data,
            template_metadata=template_metadata
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Template creation failed: {result.get('error')}"
            )
        
        return TemplateResponse(**result['template'])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating template from query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")