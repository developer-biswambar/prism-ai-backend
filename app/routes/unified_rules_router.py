# Unified Rules Router - Combines Delta, Reconciliation, and Transformation Rules
# Maintains backward compatibility while providing a unified API

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.utils.uuid_generator import generate_uuid
from app.services.dynamodb_rules_service import dynamodb_rules_service

# Unified router for all rule types
unified_rules_router = APIRouter(prefix="/rules", tags=["rules"])

# Pydantic Models for all rule types

# Common base models
class RuleMetadata(BaseModel):
    name: str
    description: Optional[str] = ""
    category: Optional[str] = "general"
    tags: Optional[List[str]] = []
    template_id: Optional[str] = None
    template_name: Optional[str] = None

# Delta Rule specific models
class DeltaRuleConfig(BaseModel):
    Files: List[Dict[str, Any]] = []
    KeyRules: List[Dict[str, Any]] = []
    ComparisonRules: List[Dict[str, Any]] = []
    selected_columns_file_a: List[str] = []
    selected_columns_file_b: List[str] = []
    user_requirements: str = "Generate delta between older and newer files using configured key and comparison rules"

class DeltaRuleCreate(BaseModel):
    metadata: RuleMetadata
    rule_config: DeltaRuleConfig

class DeltaRuleResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    category: str
    tags: List[str] = []
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    rule_type: str
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    rule_config: DeltaRuleConfig

# Reconciliation/Transformation Rule models
class ReconciliationRuleCreate(BaseModel):
    metadata: RuleMetadata
    rule_config: Dict[str, Any]

class TransformationRuleCreate(BaseModel):
    metadata: RuleMetadata
    rule_config: Dict[str, Any]

class SavedReconciliationRule(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    category: str = "general"
    tags: List[str] = []
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    created_at: str
    updated_at: str
    version: str = "1.0"
    rule_config: Dict[str, Any]
    usage_count: int = 0
    last_used_at: Optional[str] = None

class SavedTransformationRule(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    category: str = "general"
    tags: List[str] = []
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    created_at: str
    updated_at: str
    version: str = "1.0"
    rule_config: Dict[str, Any]
    usage_count: int = 0
    last_used_at: Optional[str] = None

class RuleUpdateRequest(BaseModel):
    metadata: Optional[RuleMetadata] = None
    rule_config: Optional[Dict[str, Any]] = None

# Helper functions
def create_rule_record(rule_type: str, metadata: RuleMetadata, rule_config: Dict[str, Any], rule_id: str = None) -> Dict:
    """Create a rule record for DynamoDB storage"""
    if rule_id is None:
        rule_id = generate_uuid(f'{rule_type}_rule')

    return {
        "id": rule_id,
        "name": metadata.name,
        "description": metadata.description or "",
        "category": metadata.category or rule_type,
        "tags": metadata.tags or [],
        "template_id": metadata.template_id,
        "template_name": metadata.template_name,
        "rule_type": rule_type,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "version": "1.0",
        "usage_count": 0,
        "last_used_at": None,
        "rule_config": rule_config
    }

def validate_rule_type(rule_type: str):
    """Validate rule type"""
    valid_types = ["delta", "reconciliation", "transformation"]
    if rule_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid rule type '{rule_type}'. Must be one of: {', '.join(valid_types)}"
        )

# CORS Options Handler
@unified_rules_router.options("/{path:path}")
async def options_handler():
    """Handle preflight OPTIONS requests"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# =====================================================
# UNIFIED ENDPOINTS - New Structure /rules/{rule_type}/...
# =====================================================

@unified_rules_router.post("/{rule_type}/save")
async def save_rule(rule_type: str, request: Dict[str, Any]):
    """Save a rule of any type"""
    validate_rule_type(rule_type)
    
    try:
        metadata = RuleMetadata(**request.get("metadata", {}))
        rule_config = request.get("rule_config", {})
        
        # Create rule record
        rule_record = create_rule_record(rule_type, metadata, rule_config)
        
        # Save to DynamoDB
        success = dynamodb_rules_service.save_rule(rule_type, rule_record)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save rule to database")
        
        # Return appropriate response based on rule type
        if rule_type == "delta":
            delta_config = DeltaRuleConfig(**rule_config)
            rule_record["rule_config"] = delta_config
            rule_record["created_at"] = datetime.fromisoformat(rule_record["created_at"])
            rule_record["updated_at"] = datetime.fromisoformat(rule_record["updated_at"])
            return DeltaRuleResponse(**rule_record)
        else:
            return SavedReconciliationRule(**rule_record) if rule_type == "reconciliation" else SavedTransformationRule(**rule_record)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save {rule_type} rule: {str(e)}")

@unified_rules_router.get("/{rule_type}/list")
async def list_rules(
    rule_type: str,
    category: Optional[str] = Query(None, description="Filter by category"),
    template_id: Optional[str] = Query(None, description="Filter by template ID"),
    limit: int = Query(50, description="Maximum number of rules to return"),
    offset: int = Query(0, description="Number of rules to skip")
):
    """List rules of a specific type"""
    validate_rule_type(rule_type)
    
    try:
        rules = dynamodb_rules_service.list_rules(
            rule_type,
            category=category,
            template_id=template_id,
            limit=limit,
            offset=offset
        )

        # Convert to appropriate response models
        if rule_type == "delta":
            response_rules = []
            for rule in rules:
                rule_copy = rule.copy()
                rule_copy["created_at"] = datetime.fromisoformat(rule["created_at"])
                rule_copy["updated_at"] = datetime.fromisoformat(rule["updated_at"])
                if rule.get("last_used_at"):
                    rule_copy["last_used_at"] = datetime.fromisoformat(rule["last_used_at"])
                
                delta_config = DeltaRuleConfig(**rule["rule_config"])
                rule_copy["rule_config"] = delta_config
                response_rules.append(DeltaRuleResponse(**rule_copy))
            return response_rules
        else:
            return [SavedReconciliationRule(**rule) if rule_type == "reconciliation" else SavedTransformationRule(**rule) for rule in rules]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list {rule_type} rules: {str(e)}")

@unified_rules_router.get("/{rule_type}/{rule_id}")
async def get_rule(rule_type: str, rule_id: str):
    """Get a specific rule by ID"""
    validate_rule_type(rule_type)
    
    try:
        rule = dynamodb_rules_service.get_rule(rule_type, rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        # Convert to appropriate response model
        if rule_type == "delta":
            rule["created_at"] = datetime.fromisoformat(rule["created_at"])
            rule["updated_at"] = datetime.fromisoformat(rule["updated_at"])
            if rule.get("last_used_at"):
                rule["last_used_at"] = datetime.fromisoformat(rule["last_used_at"])
            
            delta_config = DeltaRuleConfig(**rule["rule_config"])
            rule["rule_config"] = delta_config
            return DeltaRuleResponse(**rule)
        else:
            return SavedReconciliationRule(**rule) if rule_type == "reconciliation" else SavedTransformationRule(**rule)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get {rule_type} rule: {str(e)}")

@unified_rules_router.put("/{rule_type}/{rule_id}")
async def update_rule(rule_type: str, rule_id: str, updates: RuleUpdateRequest):
    """Update an existing rule"""
    validate_rule_type(rule_type)
    
    try:
        update_data = {}
        if updates.metadata:
            update_data.update(updates.metadata.dict(exclude_unset=True))
        if updates.rule_config is not None:
            update_data["rule_config"] = updates.rule_config
        
        success = dynamodb_rules_service.update_rule(rule_type, rule_id, update_data)
        if not success:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        # Get updated rule
        updated_rule = dynamodb_rules_service.get_rule(rule_type, rule_id)
        
        # Convert to appropriate response model
        if rule_type == "delta":
            updated_rule["created_at"] = datetime.fromisoformat(updated_rule["created_at"])
            updated_rule["updated_at"] = datetime.fromisoformat(updated_rule["updated_at"])
            if updated_rule.get("last_used_at"):
                updated_rule["last_used_at"] = datetime.fromisoformat(updated_rule["last_used_at"])
            
            delta_config = DeltaRuleConfig(**updated_rule["rule_config"])
            updated_rule["rule_config"] = delta_config
            return DeltaRuleResponse(**updated_rule)
        else:
            return SavedReconciliationRule(**updated_rule) if rule_type == "reconciliation" else SavedTransformationRule(**updated_rule)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update {rule_type} rule: {str(e)}")

@unified_rules_router.delete("/{rule_type}/{rule_id}")
async def delete_rule(rule_type: str, rule_id: str):
    """Delete a rule"""
    validate_rule_type(rule_type)
    
    try:
        success = dynamodb_rules_service.delete_rule(rule_type, rule_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        return {"message": f"{rule_type.title()} rule '{rule_id}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete {rule_type} rule: {str(e)}")

@unified_rules_router.post("/{rule_type}/{rule_id}/use")
async def mark_rule_as_used(rule_type: str, rule_id: str):
    """Mark a rule as used (increment usage count)"""
    validate_rule_type(rule_type)
    
    try:
        success = dynamodb_rules_service.mark_rule_as_used(rule_type, rule_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        # Get updated usage count
        updated_rule = dynamodb_rules_service.get_rule(rule_type, rule_id)
        return {
            "message": f"{rule_type.title()} rule marked as used",
            "usage_count": updated_rule.get("usage_count", 0)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark {rule_type} rule as used: {str(e)}")

@unified_rules_router.post("/{rule_type}/search")
async def search_rules(rule_type: str, filters: Dict[str, Any]):
    """Search rules with filters"""
    validate_rule_type(rule_type)
    
    try:
        rules = dynamodb_rules_service.search_rules(rule_type, filters)
        
        # Convert to appropriate response models
        if rule_type == "delta":
            response_rules = []
            for rule in rules:
                rule_copy = rule.copy()
                rule_copy["created_at"] = datetime.fromisoformat(rule["created_at"])
                rule_copy["updated_at"] = datetime.fromisoformat(rule["updated_at"])
                if rule.get("last_used_at"):
                    rule_copy["last_used_at"] = datetime.fromisoformat(rule["last_used_at"])
                
                delta_config = DeltaRuleConfig(**rule["rule_config"])
                rule_copy["rule_config"] = delta_config
                response_rules.append(DeltaRuleResponse(**rule_copy))
            return response_rules
        else:
            return [SavedReconciliationRule(**rule) if rule_type == "reconciliation" else SavedTransformationRule(**rule) for rule in rules]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search {rule_type} rules: {str(e)}")

@unified_rules_router.get("/{rule_type}/categories/list")
async def get_categories(rule_type: str):
    """Get available categories for a rule type"""
    validate_rule_type(rule_type)
    
    try:
        categories = dynamodb_rules_service.get_categories(rule_type)
        return {"categories": categories}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get {rule_type} categories: {str(e)}")

@unified_rules_router.post("/{rule_type}/bulk-delete")
async def bulk_delete_rules(rule_type: str, rule_ids: List[str]):
    """Delete multiple rules at once"""
    validate_rule_type(rule_type)
    
    try:
        result = dynamodb_rules_service.bulk_delete_rules(rule_type, rule_ids)
        return {
            "message": f"Bulk delete completed for {rule_type} rules",
            "deleted_count": result["deleted_count"],
            "not_found_ids": result["not_found_ids"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bulk delete {rule_type} rules: {str(e)}")

# =====================================================
# HEALTH CHECK ENDPOINT
# =====================================================

@unified_rules_router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test DynamoDB connection
        health = dynamodb_rules_service.get_health_status()
        return {
            "status": "healthy" if health["status"] == "healthy" else "unhealthy",
            "service": "unified-rules-router",
            "database": health,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "unified-rules-router", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }