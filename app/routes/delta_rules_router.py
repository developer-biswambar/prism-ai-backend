# Backend API endpoints for Delta Rule Management with In-Memory Storage
# Add these endpoints to your FastAPI backend

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.utils.uuid_generator import generate_uuid

# Router for delta rule management
delta_rules_router = APIRouter(prefix="/delta-rules", tags=["delta-rules"])

# In-Memory Storage
delta_rules_storage = {}


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


# Helper functions for in-memory storage
def create_delta_rule_record(rule_data: DeltaRuleCreate, rule_id: str = None) -> Dict:
    """Create a delta rule record for in-memory storage"""
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
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "usage_count": 0,
        "last_used_at": None,
        "rule_config": rule_data.rule_config.dict()
    }


def filter_delta_rules(
        rules: Dict[str, Dict],
        category: Optional[str] = None,
        template_id: Optional[str] = None,
        search_term: Optional[str] = None
) -> List[Dict]:
    """Filter delta rules based on criteria"""
    filtered_rules = []

    for rule in rules.values():
        # Category filter
        if category and rule["category"] != category:
            continue

        # Template filter
        if template_id and rule["template_id"] != template_id:
            continue

        # Search filter
        if search_term:
            search_lower = search_term.lower()
            if (search_lower not in rule["name"].lower() and
                    search_lower not in (rule["description"] or "").lower()):
                continue

        filtered_rules.append(rule)

    return filtered_rules


# API Endpoints

@delta_rules_router.post("/save", response_model=DeltaRuleResponse)
async def save_delta_rule(rule_data: DeltaRuleCreate):
    """Save a new delta generation rule"""
    try:
        # Validate the rule data
        if not rule_data.metadata.name.strip():
            raise HTTPException(status_code=400, detail="Rule name is required")

        # Check for duplicate names
        for existing_rule in delta_rules_storage.values():
            if existing_rule["name"] == rule_data.metadata.name:
                raise HTTPException(status_code=400, detail="Rule name already exists")

        # Create the rule record
        rule_record = create_delta_rule_record(rule_data)

        # Save to in-memory storage
        delta_rules_storage[rule_record["id"]] = rule_record

        return DeltaRuleResponse(**rule_record)

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
        # Filter rules
        filtered_rules = filter_delta_rules(
            delta_rules_storage,
            category=category,
            template_id=template_id
        )

        # Sort by updated_at (most recent first)
        filtered_rules.sort(key=lambda x: x["updated_at"], reverse=True)

        # Apply pagination
        paginated_rules = filtered_rules[offset:offset + limit]

        return [DeltaRuleResponse(**rule) for rule in paginated_rules]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list delta rules: {str(e)}")


@delta_rules_router.get("/template/{template_id}", response_model=List[DeltaRuleResponse])
async def get_delta_rules_by_template(template_id: str):
    """Get delta rules for a specific template"""
    try:
        filtered_rules = filter_delta_rules(
            delta_rules_storage,
            template_id=template_id
        )

        # Sort by usage count (most used first)
        filtered_rules.sort(key=lambda x: x["usage_count"], reverse=True)

        return [DeltaRuleResponse(**rule) for rule in filtered_rules]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get delta rules by template: {str(e)}")


@delta_rules_router.get("/{rule_id}", response_model=DeltaRuleResponse)
async def get_delta_rule(rule_id: str):
    """Get a specific delta rule by ID"""
    try:
        if rule_id not in delta_rules_storage:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        rule = delta_rules_storage[rule_id]
        return DeltaRuleResponse(**rule)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get delta rule: {str(e)}")


@delta_rules_router.put("/{rule_id}", response_model=DeltaRuleResponse)
async def update_delta_rule(rule_id: str, updates: DeltaRuleUpdate):
    """Update an existing delta rule"""
    try:
        if rule_id not in delta_rules_storage:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        rule = delta_rules_storage[rule_id]

        # Update metadata if provided
        if updates.metadata:
            # Check for duplicate names (excluding current rule)
            if updates.metadata.name and updates.metadata.name != rule["name"]:
                for other_id, other_rule in delta_rules_storage.items():
                    if other_id != rule_id and other_rule["name"] == updates.metadata.name:
                        raise HTTPException(status_code=400, detail="Rule name already exists")

            rule["name"] = updates.metadata.name or rule["name"]
            rule["description"] = updates.metadata.description if updates.metadata.description is not None else rule[
                "description"]
            rule["category"] = updates.metadata.category or rule["category"]
            rule["tags"] = updates.metadata.tags if updates.metadata.tags is not None else rule["tags"]
            rule["template_id"] = updates.metadata.template_id if updates.metadata.template_id is not None else rule[
                "template_id"]
            rule["template_name"] = updates.metadata.template_name if updates.metadata.template_name is not None else \
                rule["template_name"]

        # Update rule config if provided
        if updates.rule_config:
            rule["rule_config"] = updates.rule_config.dict()

        # Update timestamp
        rule["updated_at"] = datetime.utcnow()

        # Save back to storage
        delta_rules_storage[rule_id] = rule

        return DeltaRuleResponse(**rule)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update delta rule: {str(e)}")


@delta_rules_router.delete("/{rule_id}")
async def delete_delta_rule(rule_id: str):
    """Delete a delta rule"""
    try:
        if rule_id not in delta_rules_storage:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        # Remove from storage
        del delta_rules_storage[rule_id]

        return {"message": "Delta rule deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete delta rule: {str(e)}")


@delta_rules_router.post("/{rule_id}/use")
async def mark_delta_rule_as_used(rule_id: str):
    """Mark a delta rule as used (increment usage count)"""
    try:
        if rule_id not in delta_rules_storage:
            raise HTTPException(status_code=404, detail="Delta rule not found")

        rule = delta_rules_storage[rule_id]
        rule["usage_count"] = rule["usage_count"] + 1
        rule["last_used_at"] = datetime.utcnow()

        # Save back to storage
        delta_rules_storage[rule_id] = rule

        return {"message": "Delta rule usage updated", "usage_count": rule["usage_count"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update delta rule usage: {str(e)}")


@delta_rules_router.post("/search", response_model=List[DeltaRuleResponse])
async def search_delta_rules(search_filters: Dict[str, Any]):
    """Search delta rules with complex filters"""
    try:
        filtered_rules = []

        for rule in delta_rules_storage.values():
            include_rule = True

            # Name search
            if search_filters.get("name"):
                if search_filters["name"].lower() not in rule["name"].lower():
                    include_rule = False

            # Description search
            if search_filters.get("description") and include_rule:
                description = rule["description"] or ""
                if search_filters["description"].lower() not in description.lower():
                    include_rule = False

            # Category filter
            if search_filters.get("category") and include_rule:
                if rule["category"] != search_filters["category"]:
                    include_rule = False

            # Tags filter
            if search_filters.get("tags") and include_rule:
                rule_tags = rule["tags"] or []
                search_tags = search_filters["tags"]
                if not any(tag in rule_tags for tag in search_tags):
                    include_rule = False

            # Template filter
            if search_filters.get("template_id") and include_rule:
                if rule["template_id"] != search_filters["template_id"]:
                    include_rule = False

            # Usage count filter
            if search_filters.get("min_usage_count") and include_rule:
                if rule["usage_count"] < search_filters["min_usage_count"]:
                    include_rule = False

            if include_rule:
                filtered_rules.append(rule)

        # Sort by relevance (usage count + recency)
        filtered_rules.sort(
            key=lambda x: (x["usage_count"], x["updated_at"]),
            reverse=True
        )

        return [DeltaRuleResponse(**rule) for rule in filtered_rules]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search delta rules: {str(e)}")


@delta_rules_router.get("/categories/list")
async def get_delta_rule_categories():
    """Get list of available delta rule categories"""
    try:
        # Get categories from existing rules
        used_categories = set()
        for rule in delta_rules_storage.values():
            used_categories.add(rule["category"])

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
        deleted_count = 0
        not_found_ids = []

        for rule_id in rule_ids:
            if rule_id in delta_rules_storage:
                del delta_rules_storage[rule_id]
                deleted_count += 1
            else:
                not_found_ids.append(rule_id)

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
        rules_to_export = []

        for rule in delta_rules_storage.values():
            if category is None or rule["category"] == category:
                rules_to_export.append(rule)

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

                # Reset usage stats
                rule_data["usage_count"] = 0
                rule_data["last_used_at"] = None
                rule_data["created_at"] = datetime.utcnow()
                rule_data["updated_at"] = datetime.utcnow()
                rule_data["id"] = new_id

                # Save to storage
                delta_rules_storage[new_id] = rule_data
                imported_count += 1

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

        rule_count = len(delta_rules_storage)
        delta_rules_storage.clear()

        return {
            "message": f"Cleared {rule_count} delta rules",
            "cleared_count": rule_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear delta rules: {str(e)}")


@delta_rules_router.get("/health")
async def get_delta_rule_management_health():
    """Health check for delta rule management system"""
    try:
        rule_count = len(delta_rules_storage.values())

        return {
            "status": "healthy",
            "total_rules": rule_count,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }
