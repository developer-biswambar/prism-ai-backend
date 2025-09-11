"""
Use Case Management API Routes
Provides REST endpoints for use case CRUD operations, search, and analytics.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field, validator

from app.services.dynamodb_templates_service import dynamodb_templates_service
from app.services.template_application_service import template_application_service
from app.services.smart_template_execution_service import smart_template_execution_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/saved-use-cases", tags=["Saved Use Cases"])


def _map_template_to_use_case_response(template_data: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to map template data to UseCaseResponse format"""
    return {
        **template_data,
        'use_case_type': template_data.get('template_type', 'data_processing'),
        'use_case_config': template_data.get('template_config', {}),
        'use_case_content': template_data.get('template_content'),
        'use_case_metadata': template_data.get('template_metadata')
    }


# Request/Response Models
class UseCaseConfigModel(BaseModel):
    """Use case configuration schema"""
    prompt_template: str = Field(..., description="Parameterized natural language prompt")
    required_columns: List[str] = Field(default_factory=list, description="Required column mappings")
    optional_columns: List[str] = Field(default_factory=list, description="Optional column mappings") 
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="User-configurable parameters")
    validation_rules: List[str] = Field(default_factory=list, description="Data validation requirements")
    output_format: Dict[str, Any] = Field(default_factory=dict, description="Expected output structure")
    sample_data: Dict[str, Any] = Field(default_factory=dict, description="Example input/output data")


class CreateUseCaseRequest(BaseModel):
    """Request model for creating a new use case"""
    name: str = Field(..., min_length=1, max_length=200, description="Use case display name")
    description: str = Field(..., min_length=1, max_length=5000, description="Detailed description")
    use_case_type: str = Field(..., description="Use case type")
    category: str = Field(..., description="Use case category")
    tags: List[str] = Field(default_factory=list, description="Use case tags")
    use_case_config: UseCaseConfigModel = Field(..., description="Use case configuration")
    created_by: Optional[str] = Field(None, description="Creator user identifier")
    
    @validator('use_case_type')
    def validate_use_case_type(cls, v):
        valid_types = ['data_processing', 'reconciliation', 'analysis', 'transformation', 'reporting']
        if v not in valid_types:
            raise ValueError(f'use_case_type must be one of: {valid_types}')
        return v


class UpdateUseCaseRequest(BaseModel):
    """Request model for updating an existing use case"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=5000)
    category: Optional[str] = Field(None)
    industry: Optional[str] = Field(None)
    tags: Optional[List[str]] = Field(None)
    use_case_config: Optional[UseCaseConfigModel] = Field(None)
    use_case_content: Optional[str] = Field(None, description="Rich use case content")
    use_case_metadata: Optional[Dict[str, Any]] = Field(None, description="Enhanced use case metadata")


class UseCaseResponse(BaseModel):
    """Response model for use case data with enhanced fields"""
    id: str
    name: str
    description: str
    use_case_type: str
    category: str
    tags: List[str]
    use_case_config: Dict[str, Any]
    version: str
    created_by: Optional[str]
    created_at: str
    updated_at: str
    usage_count: int
    last_used_at: Optional[str]
    rating: float
    rating_count: int
    
    # Enhanced use case fields
    use_case_content: Optional[str] = None
    use_case_metadata: Optional[Dict[str, Any]] = None


class UseCaseListResponse(BaseModel):
    """Response model for use case list"""
    use_cases: List[UseCaseResponse]
    total_count: int
    offset: int
    limit: int


class RateUseCaseRequest(BaseModel):
    """Request model for rating a use case"""
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating from 1.0 to 5.0")


# Smart Execution Models
class SmartExecutionRequest(BaseModel):
    """Request model for smart template execution"""
    template_id: str = Field(..., description="Template ID to execute")
    files: List[Dict[str, Any]] = Field(..., description="Files to process")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Runtime parameters")


class ColumnMappingRequest(BaseModel):
    """Request model for applying user column mapping"""
    template_id: str = Field(..., description="Template ID")
    column_mapping: Dict[str, str] = Field(..., description="User-provided column mapping")
    files: List[Dict[str, Any]] = Field(..., description="Files to process")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Runtime parameters")


class SmartExecutionResponse(BaseModel):
    """Response model for smart execution results - same as MiscellaneousResponse with additional fields"""
    # Base fields from MiscellaneousResponse
    success: bool
    message: str
    process_id: Optional[str] = Field(None, description="Process ID for result retrieval")
    generated_sql: Optional[str] = None
    row_count: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    errors: Optional[List[str]] = []
    warnings: Optional[List[str]] = []
    
    # Data field for results
    data: Optional[Any] = Field(None, description="Execution results if successful")
    
    # Smart execution specific fields
    execution_method: Optional[str] = Field(None, description="Method used: exact, mapped, ai_assisted")
    applied_mapping: Optional[Dict[str, str]] = Field(None, description="Applied column mapping")
    ai_adaptations: Optional[str] = Field(None, description="Description of AI adaptations made")
    
    # Error handling fields (only for needs_user_intervention cases)
    template_id: Optional[str] = Field(None, description="Template ID for user intervention")
    execution_error: Optional[str] = Field(None, description="Raw execution error")
    error_analysis: Optional[Dict[str, Any]] = Field(None, description="Detailed error analysis")
    available_options: Optional[List[Dict[str, str]]] = Field(None, description="Available user options")
    file_schemas: Optional[List[Dict[str, Any]]] = Field(None, description="File schema information")


# Use Case CRUD Routes

@router.post("/", response_model=UseCaseResponse, status_code=201)
async def create_template(request: CreateUseCaseRequest):
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
            "template_config": {
                **request.template_config.dict(),
                # Smart execution configuration
                "column_mapping": {},  # Will be populated as templates are used
                "fallback_strategy": "fuzzy_match",
                "primary_sql": None,  # Will be set when template generates SQL
                "last_mapping_update": None
            },
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
        return UseCaseResponse(**_map_template_to_use_case_response(template_data))
        
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@router.get("/{template_id}", response_model=UseCaseResponse)
async def get_template(
    template_id: str = Path(..., description="Template ID"),
    template_type: Optional[str] = Query(None, description="Template type for faster lookup")
):
    """Get a specific template by ID"""
    try:
        template = dynamodb_templates_service.get_template(template_id, template_type)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return UseCaseResponse(**_map_template_to_use_case_response(template))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@router.put("/{template_id}", response_model=UseCaseResponse)
async def update_template(
    template_id: str = Path(..., description="Template ID"),
    request: UpdateUseCaseRequest = Body(...),
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
        if request.template_content is not None:
            updates['template_content'] = request.template_content
        if request.template_metadata is not None:
            updates['template_metadata'] = request.template_metadata
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        # Update template
        success = dynamodb_templates_service.update_template(template_id, updates, template_type)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found or update failed")
        
        # Return updated template
        updated_template = dynamodb_templates_service.get_template(template_id, template_type)
        logger.info(f"Updated template {template_id}")
        return UseCaseResponse(**_map_template_to_use_case_response(updated_template))
        
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

@router.get("/", response_model=UseCaseListResponse)
async def list_templates(
    template_type: Optional[str] = Query(None, description="Filter by template type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    limit: int = Query(50, ge=1, le=200, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """List templates with optional filtering"""
    try:
        templates = dynamodb_templates_service.list_templates(
            template_type=template_type,
            category=category,
            created_by=created_by,
            limit=limit,
            offset=offset
        )
        
        template_responses = [UseCaseResponse(**_map_template_to_use_case_response(template)) for template in templates]
        
        return UseCaseListResponse(
            use_cases=template_responses,
            total_count=len(template_responses),  # Note: This is just the current page count
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@router.get("/search/query", response_model=UseCaseListResponse)
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
        
        template_responses = [UseCaseResponse(**_map_template_to_use_case_response(template)) for template in templates]
        
        return UseCaseListResponse(
            use_cases=template_responses,
            total_count=len(template_responses),
            offset=0,
            limit=len(template_responses)
        )
        
    except Exception as e:
        logger.error(f"Error searching templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search templates: {str(e)}")


@router.get("/popular/list", response_model=UseCaseListResponse)
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
        
        template_responses = [UseCaseResponse(**_map_template_to_use_case_response(template)) for template in templates]
        
        return UseCaseListResponse(
            use_cases=template_responses,
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
    request: RateUseCaseRequest = Body(...),
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


# Smart Execution Routes

@router.post("/execute", response_model=SmartExecutionResponse)
async def smart_execute_template(request: SmartExecutionRequest):
    """Execute a template with smart fallback strategies"""
    try:
        logger.info(f"Smart execution request for template {request.template_id}")
        
        result = smart_template_execution_service.execute_template(
            template_id=request.template_id,
            files=request.files,
            parameters=request.parameters
        )
        
        logger.info(f"Smart execution result: {'success' if result.get('success') else 'failed'}")
        return SmartExecutionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in smart template execution: {e}")
        raise HTTPException(status_code=500, detail=f"Smart execution failed: {str(e)}")


@router.post("/execute/with-mapping", response_model=SmartExecutionResponse)
async def execute_with_user_mapping(request: ColumnMappingRequest):
    """Execute template with user-provided column mapping"""
    try:
        logger.info(f"Executing template {request.template_id} with user mapping")
        
        result = smart_template_execution_service.apply_user_column_mapping(
            template_id=request.template_id,
            user_mapping=request.column_mapping,
            files=request.files,
            parameters=request.parameters
        )
        
        logger.info(f"Mapped execution result: {'success' if result.get('success') else 'failed'}")
        return SmartExecutionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in mapped template execution: {e}")
        raise HTTPException(status_code=500, detail=f"Mapped execution failed: {str(e)}")


@router.post("/execute/ai-assisted", response_model=SmartExecutionResponse)
async def ai_assisted_execution(request: SmartExecutionRequest):
    """Execute template with AI assistance - only called with user consent"""
    try:
        logger.info(f"AI-assisted execution request for template {request.template_id} (user consented)")
        
        # This will use AI to adapt the query to the user's data
        result = smart_template_execution_service.execute_with_ai_assistance(
            template_id=request.template_id,
            files=request.files,
            parameters=request.parameters
        )
        
        logger.info(f"AI-assisted execution result: {'success' if result.get('success') else 'failed'}")
        return SmartExecutionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in AI-assisted template execution: {e}")
        raise HTTPException(status_code=500, detail=f"AI-assisted execution failed: {str(e)}")


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
    use_case_id: str = Field(..., description="Use case ID to apply")
    files: List[Dict[str, Any]] = Field(..., description="User's file data")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Template parameters")


class CreateTemplateFromQueryRequest(BaseModel):
    """Request model for creating template from successful query"""
    query_data: Dict[str, Any] = Field(..., description="Successful query execution data")
    template_name: str = Field(..., min_length=1, max_length=200, description="Template name")
    template_description: str = Field(..., min_length=1, max_length=5000, description="Template description")
    template_type: str = Field(..., description="Template type")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    created_by: Optional[str] = Field(None, description="Creator identifier")
    
    # Enhanced template data
    template_content: Optional[str] = Field(None, description="Rich template content (ideal prompt)")
    template_metadata: Optional[Dict[str, Any]] = Field(None, description="Enhanced template metadata including file patterns, transformation patterns, etc.")


@router.post("/suggest", response_model=UseCaseListResponse)
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
            
            template_resp = UseCaseResponse(**_map_template_to_use_case_response(template_data))
            # Add suggestion metadata (these would need to be added to the model)
            # template_resp.match_score = suggestion.get('match_score', 0.0)
            # template_resp.match_reasons = suggestion.get('match_reasons', [])
            template_responses.append(template_resp)
        
        return UseCaseListResponse(
            use_cases=template_responses,
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
            template_id=request.use_case_id,
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


@router.post("/create-from-query", response_model=UseCaseResponse, status_code=201)
async def create_template_from_query(request: CreateTemplateFromQueryRequest):
    """Create a new template from a successful query execution"""
    try:
        # Debug logging - what data are we receiving from frontend?
        logger.info(f"Template creation request received:")
        logger.info(f"  Template name: {request.template_name}")
        logger.info(f"  Template description: {request.template_description[:50]}...")
        logger.info(f"  Template type: {request.template_type}")
        logger.info(f"  Category: {request.category}")
        logger.info(f"  Tags: {request.tags}")
        logger.info(f"  Enhanced content present: {bool(request.template_content)}")
        logger.info(f"  Enhanced metadata present: {bool(request.template_metadata)}")
        if request.template_content:
            logger.info(f"  Content length: {len(request.template_content)}")
        if request.template_metadata:
            logger.info(f"  Metadata keys: {list(request.template_metadata.keys())}")
        if hasattr(request, 'query_data') and request.query_data:
            logger.info(f"  Query data present: {bool(request.query_data)}")
            if 'user_prompt' in request.query_data:
                logger.info(f"  User prompt: {request.query_data['user_prompt'][:50]}...")
        
        # Prepare enhanced template metadata
        template_metadata = {
            'name': request.template_name,
            'description': request.template_description,
            'template_type': request.template_type,
            'category': request.category,
            'tags': request.tags,
            'created_by': request.created_by,
            'template_content': request.template_content,
            'template_metadata': request.template_metadata
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
        
        # Map template fields to use case response format
        use_case_data = _map_template_to_use_case_response(result['template'])
        return UseCaseResponse(**use_case_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating template from query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")