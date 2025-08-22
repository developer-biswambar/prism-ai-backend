# Backend API endpoints for Delta Rule Management with DynamoDB Storage
# Preserves exact same API contract while using DynamoDB backend

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.utils.uuid_generator import generate_uuid
from app.services.dynamodb_rules_service import dynamodb_rules_service

# Router for delta rule management
delta_rules_router = APIRouter(prefix="/delta-rules", tags=["delta-rules"])


# Pydantic models for Delta Rules
class DeltaRuleMetadata(BaseModel):
    name: str
    description: Optional[str] = ""
    category: str = "delta"
    tags: List[str] = []
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    rule_type: str = "delta_generation"


class DeltaRuleConfig(BaseModel):
    Files: List[Dict[str, Any]] = []
    KeyRules: List[Dict[str, Any]] = []
    ComparisonRules: List[Dict[str, Any]] = []
    selected_columns_file_a: List[str] = []
    selected_columns_file_b: List[str] = []
    user_requirements: str = "Generate delta between older and newer files using configured key and comparison rules"


class DeltaRuleCreate(BaseModel):
    metadata: DeltaRuleMetadata
    rule_config: DeltaRuleConfig


class DeltaRuleUpdate(BaseModel):
    metadata: Optional[DeltaRuleMetadata] = None
    rule_config: Optional[DeltaRuleConfig] = None


class DeltaRuleResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    category: str
    tags: List[str]
    template_id: Optional[str]
    template_name: Optional[str]
    rule_type: str
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    rule_config: DeltaRuleConfig


# Helper functions for DynamoDB storage
def create_delta_rule_record(rule_data: DeltaRuleCreate, rule_id: str = None) -> Dict:
    """Create a delta rule record for DynamoDB storage"""
    if rule_id is None:
        rule_id = generate_uuid('delta_rule')

    return {
        "id": rule_id,
        "name": rule_data.metadata.name,
        "description": rule_data.metadata.description,
        "category": rule_data.metadata.category,
        "tags": rule_data.metadata.tags,
        "template_id": rule_data.metadata.template_id,
        "template_name": rule_data.metadata.template_name,
        "rule_type": rule_data.metadata.rule_type,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "version": "1.0",
        "usage_count": 0,
        "last_used_at": None,
        "rule_config": rule_data.rule_config.dict()
    }


# API Endpoints

@delta_rules_router.post("/save", response_model=DeltaRuleResponse)
async def save_delta_rule(rule_data: DeltaRuleCreate):
    """Save a new delta generation rule"""
    try:
        # Validate the rule data
        if not rule_data.metadata.name.strip():
            raise HTTPException(status_code=400, detail="Rule name is required")

        # Check for duplicate names by searching existing rules
        existing_rules = dynamodb_rules_service.search_rules('delta', {'name_contains': rule_data.metadata.name})
        for existing_rule in existing_rules:
            if existing_rule["name"] == rule_data.metadata.name:
                raise HTTPException(status_code=400, detail="Rule name already exists")

        # Create the rule record
        rule_record = create_delta_rule_record(rule_data)

        # Save to DynamoDB
        success = dynamodb_rules_service.save_rule('delta', rule_record)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save rule to database")

        # Convert datetime strings back to datetime objects for response
        response_record = rule_record.copy()
        response_record["created_at"] = datetime.fromisoformat(rule_record["created_at"])
        response_record["updated_at"] = datetime.fromisoformat(rule_record["updated_at"])

        return DeltaRuleResponse(**response_record)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save delta rule: {str(e)}")


@delta_rules_router.get("/list", response_model=List[DeltaRuleResponse])
async def list_delta_rules(
        category: Optional[str] = None,
        template_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
):
    """List delta generation rules with optional filtering"""
    try:
        # Get rules from DynamoDB
        rules = dynamodb_rules_service.list_rules(
            'delta',
            category=category,
            template_id=template_id,
            limit=limit,
            offset=offset
        )

        # Convert datetime strings to datetime objects for response
        response_rules = []
        for rule in rules:
            response_rule = rule.copy()
            response_rule["created_at"] = datetime.fromisoformat(rule["created_at"])
            response_rule["updated_at"] = datetime.fromisoformat(rule["updated_at"])
            if rule.get("last_used_at"):
                response_rule["last_used_at"] = datetime.fromisoformat(rule["last_used_at"])
            response_rules.append(DeltaRuleResponse(**response_rule))

        return response_rules

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list delta rules: {str(e)}")


@delta_rules_router.get("/template/{template_id}", response_model=List[DeltaRuleResponse])
async def get_delta_rules_by_template(template_id: str):
    """Get delta rules for a specific template"""
    try:
        # Get rules from DynamoDB using template filter
        rules = dynamodb_rules_service.list_rules(
            'delta',
            template_id=template_id,
            limit=1000  # Large limit for template queries
        )

        # Sort by usage count (most used first) - DynamoDB returns sorted by recency
        rules.sort(key=lambda x: x.get("usage_count", 0), reverse=True)

        # Convert datetime strings to datetime objects for response
        response_rules = []
        for rule in rules:
            response_rule = rule.copy()
            response_rule["created_at"] = datetime.fromisoformat(rule["created_at"])
            response_rule["updated_at"] = datetime.fromisoformat(rule["updated_at"])
            if rule.get("last_used_at"):
                response_rule["last_used_at"] = datetime.fromisoformat(rule["last_used_at"])
            response_rules.append(DeltaRuleResponse(**response_rule))

        return response_rules

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get delta rules by template: {str(e)}")


@delta_rules_router.get("/{rule_id}", response_model=DeltaRuleResponse)
async def get_delta_rule(rule_id: str):
    """Get a specific delta rule by ID"""
    try:
        rule = dynamodb_rules_service.get_rule('delta', rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        # Convert datetime strings to datetime objects for response
        response_rule = rule.copy()
        response_rule["created_at"] = datetime.fromisoformat(rule["created_at"])
        response_rule["updated_at"] = datetime.fromisoformat(rule["updated_at"])
        if rule.get("last_used_at"):
            response_rule["last_used_at"] = datetime.fromisoformat(rule["last_used_at"])

        return DeltaRuleResponse(**response_rule)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get delta rule: {str(e)}")


@delta_rules_router.put("/{rule_id}", response_model=DeltaRuleResponse)
async def update_delta_rule(rule_id: str, updates: DeltaRuleUpdate):
    """Update an existing delta rule"""
    try:
        # Check if rule exists
        existing_rule = dynamodb_rules_service.get_rule('delta', rule_id)
        if not existing_rule:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        # Prepare updates
        update_data = {}

        # Update metadata if provided
        if updates.metadata:
            # Check for duplicate names (excluding current rule)
            if updates.metadata.name and updates.metadata.name != existing_rule["name"]:
                existing_rules = dynamodb_rules_service.search_rules('delta', {'name_contains': updates.metadata.name})
                for other_rule in existing_rules:
                    if other_rule["id"] != rule_id and other_rule["name"] == updates.metadata.name:
                        raise HTTPException(status_code=400, detail="Rule name already exists")

            if updates.metadata.name:
                update_data["name"] = updates.metadata.name
            if updates.metadata.description is not None:
                update_data["description"] = updates.metadata.description
            if updates.metadata.category:
                update_data["category"] = updates.metadata.category
            if updates.metadata.tags is not None:
                update_data["tags"] = updates.metadata.tags
            if updates.metadata.template_id is not None:
                update_data["template_id"] = updates.metadata.template_id
            if updates.metadata.template_name is not None:
                update_data["template_name"] = updates.metadata.template_name

        # Update rule config if provided
        if updates.rule_config:
            update_data["rule_config"] = updates.rule_config.dict()

        # Update in DynamoDB
        success = dynamodb_rules_service.update_rule('delta', rule_id, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update rule in database")

        # Get updated rule
        updated_rule = dynamodb_rules_service.get_rule('delta', rule_id)
        
        # Convert datetime strings to datetime objects for response
        response_rule = updated_rule.copy()
        response_rule["created_at"] = datetime.fromisoformat(updated_rule["created_at"])
        response_rule["updated_at"] = datetime.fromisoformat(updated_rule["updated_at"])
        if updated_rule.get("last_used_at"):
            response_rule["last_used_at"] = datetime.fromisoformat(updated_rule["last_used_at"])

        return DeltaRuleResponse(**response_rule)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update delta rule: {str(e)}")


@delta_rules_router.delete("/{rule_id}")
async def delete_delta_rule(rule_id: str):
    """Delete a delta rule"""
    try:
        # Check if rule exists
        existing_rule = dynamodb_rules_service.get_rule('delta', rule_id)
        if not existing_rule:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        # Delete from DynamoDB
        success = dynamodb_rules_service.delete_rule('delta', rule_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete rule from database")

        return {"message": "Delta rule deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete delta rule: {str(e)}")


@delta_rules_router.post("/{rule_id}/use")
async def mark_delta_rule_as_used(rule_id: str):
    """Mark a delta rule as used (increment usage count)"""
    try:
        # Check if rule exists and mark as used
        success = dynamodb_rules_service.mark_rule_as_used('delta', rule_id)
        if not success:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        # Get updated rule to return usage count
        updated_rule = dynamodb_rules_service.get_rule('delta', rule_id)
        
        return {"message": "Delta rule usage updated", "usage_count": updated_rule["usage_count"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update delta rule usage: {str(e)}")


@delta_rules_router.post("/search", response_model=List[DeltaRuleResponse])
async def search_delta_rules(search_filters: Dict[str, Any]):
    """Search delta rules with complex filters"""
    try:
        # Use DynamoDB search functionality
        rules = dynamodb_rules_service.search_rules('delta', search_filters)

        # Convert datetime strings to datetime objects for response
        response_rules = []
        for rule in rules:
            response_rule = rule.copy()
            response_rule["created_at"] = datetime.fromisoformat(rule["created_at"])
            response_rule["updated_at"] = datetime.fromisoformat(rule["updated_at"])
            if rule.get("last_used_at"):
                response_rule["last_used_at"] = datetime.fromisoformat(rule["last_used_at"])
            response_rules.append(DeltaRuleResponse(**response_rule))

        return response_rules

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search delta rules: {str(e)}")


@delta_rules_router.get("/categories/list")
async def get_delta_rule_categories():
    """Get list of available delta rule categories"""
    try:
        # Get categories from existing rules in DynamoDB
        used_categories = set(dynamodb_rules_service.get_categories('delta'))

        # Default categories
        default_categories = [
            "delta",
            "financial",
            "trading",
            "data-comparison",
            "validation",
            "general",
            "custom"
        ]

        # Combine and sort
        all_categories = sorted(list(set(default_categories + list(used_categories))))

        return {
            "categories": all_categories,
            "default_categories": default_categories,
            "used_categories": list(used_categories)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


# Additional utility endpoints

@delta_rules_router.post("/bulk-delete")
async def bulk_delete_delta_rules(rule_ids: List[str]):
    """Delete multiple delta rules at once"""
    try:
        result = dynamodb_rules_service.bulk_delete_rules('delta', rule_ids)
        deleted_count = result['deleted_count']
        not_found_ids = result['not_found_ids']

        return {
            "message": f"Deleted {deleted_count} delta rules",
            "deleted_count": deleted_count,
            "not_found_ids": not_found_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bulk delete delta rules: {str(e)}")


@delta_rules_router.get("/export")
async def export_delta_rules(category: Optional[str] = None):
    """Export delta rules as JSON"""
    try:
        # Get rules from DynamoDB
        if category:
            rules_to_export = dynamodb_rules_service.list_rules('delta', category=category, limit=10000)
        else:
            rules_to_export = dynamodb_rules_service.list_rules('delta', limit=10000)

        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "rule_type": "delta_generation",
            "total_rules": len(rules_to_export),
            "rules": rules_to_export
        }

        return export_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export delta rules: {str(e)}")


@delta_rules_router.post("/import")
async def import_delta_rules(import_data: Dict[str, Any]):
    """Import delta rules from JSON"""
    try:
        if "rules" not in import_data or not isinstance(import_data["rules"], list):
            raise HTTPException(status_code=400, detail="Invalid import format: 'rules' array required")

        imported_count = 0
        failed_count = 0
        errors = []

        for rule_data in import_data["rules"]:
            try:
                # Generate new ID to avoid conflicts
                new_id = generate_uuid('delta_rule')

                # Modify name to indicate import
                if "name" in rule_data:
                    rule_data["name"] = f"{rule_data['name']} (Imported)"

                # Reset usage stats and timestamps
                rule_data["usage_count"] = 0
                rule_data["last_used_at"] = None
                rule_data["created_at"] = datetime.utcnow().isoformat()
                rule_data["updated_at"] = datetime.utcnow().isoformat()
                rule_data["id"] = new_id
                rule_data["version"] = "1.0"

                # Save to DynamoDB
                success = dynamodb_rules_service.save_rule('delta', rule_data)
                if success:
                    imported_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Rule '{rule_data.get('name', 'Unknown')}': Failed to save to database")

            except Exception as rule_error:
                failed_count += 1
                errors.append(f"Rule '{rule_data.get('name', 'Unknown')}': {str(rule_error)}")

        return {
            "message": f"Import completed: {imported_count} successful, {failed_count} failed",
            "imported_count": imported_count,
            "failed_count": failed_count,
            "errors": errors
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import delta rules: {str(e)}")


@delta_rules_router.post("/clear")
async def clear_delta_rules(confirm: bool = False):
    """Clear all delta rules (use with caution)"""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must set confirm=true to clear all delta rules"
            )

        # Get all delta rules
        all_rules = dynamodb_rules_service.list_rules('delta', limit=10000)
        rule_ids = [rule['id'] for rule in all_rules]
        
        # Bulk delete all rules
        result = dynamodb_rules_service.bulk_delete_rules('delta', rule_ids)
        cleared_count = result['deleted_count']

        return {
            "message": f"Cleared {cleared_count} delta rules",
            "cleared_count": cleared_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear delta rules: {str(e)}")


@delta_rules_router.get("/health")
async def get_delta_rule_management_health():
    """Health check for delta rule management system"""
    try:
        # Get health status from DynamoDB service
        db_health = dynamodb_rules_service.get_health_status()
        
        # Get rule count
        rules = dynamodb_rules_service.list_rules('delta', limit=10000)
        rule_count = len(rules)

        return {
            "status": "healthy" if db_health['status'] == 'healthy' else "unhealthy",
            "total_rules": rule_count,
            "database_status": db_health,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }
