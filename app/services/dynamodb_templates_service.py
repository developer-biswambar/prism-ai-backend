# DynamoDB Templates Service - Template Management for Reusable Data Processing Patterns
"""
DynamoDB integration for storing templates and reusable data processing patterns.

Table Structure:
- Single table design with primary keys (PK and SK)
- Uses full table scans for queries (no GSI indexes)
- Optimized for cost and schema flexibility over query performance
- Supports various template types for different data operations

Table Name: Templates (configurable via environment variable)

Key Design:
- PK: Template type identifier (DATA_PROCESSING_TEMPLATE#, RECONCILIATION_TEMPLATE#, ANALYSIS_TEMPLATE#)  
- SK: Template ID (unique identifier for each template)

Core Attributes:
- id: Template unique identifier
- template_type: "data_processing", "reconciliation", "analysis", "transformation", "reporting"
- name: Template display name
- description: Detailed description of what the template does
- category: Template category (Finance, Operations, Sales, HR, etc.)
- industry: Industry vertical (optional)
- tags: List of tags (as StringSet)
- created_at: ISO timestamp
- updated_at: ISO timestamp
- version: Template version (default "1.0")
- created_by: User identifier who created the template
- is_public: Boolean indicating if template is shared publicly
- usage_count: Number of times template has been used
- last_used_at: ISO timestamp of last usage (optional)
- rating: Average user rating (1-5 stars)
- rating_count: Number of ratings received

Template Configuration:
- template_config: JSON blob containing:
  - prompt_template: Parameterized natural language prompt
  - required_columns: List of required column mappings
  - optional_columns: List of optional column mappings
  - parameters: List of user-configurable parameters
  - validation_rules: Data validation requirements
  - output_format: Expected output structure
  - sample_data: Example input/output data

Example Items:
1. Reconciliation Template:
   PK: "RECONCILIATION_TEMPLATE#bank_recon_001"
   SK: "bank_recon_001"
   template_type: "reconciliation"
   name: "Bank Transaction Reconciliation"
   category: "Finance"

2. Analysis Template:
   PK: "ANALYSIS_TEMPLATE#sales_analysis_001"
   SK: "sales_analysis_001"
   template_type: "analysis"
   name: "Monthly Sales Performance Analysis"
   category: "Sales"

Query Strategy: All operations use table scans with filters for maximum flexibility.
Performance: Optimized for small to medium datasets (<10k templates) with cost efficiency.
"""

import json
import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

logger = logging.getLogger(__name__)


class DynamoDBTemplatesService:
    """Service for managing templates in DynamoDB with single table design"""
    
    def __init__(self):
        self.table_name = os.getenv('DYNAMODB_TEMPLATES_TABLE', 'Templates')
        self.region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
        
        # Check for local development environment
        use_local_dynamodb = os.getenv('USE_LOCAL_DYNAMODB', 'false').lower() == 'true'
        
        if use_local_dynamodb:
            # Configure for local DynamoDB (LocalStack)
            local_endpoint = os.getenv('LOCAL_DYNAMODB_ENDPOINT', 'http://localhost:4566')
            logger.info(f"Using local DynamoDB endpoint: {local_endpoint}")
            
            # Create session with explicit credentials for LocalStack
            session = boto3.Session(
                aws_access_key_id='test',
                aws_secret_access_key='test',
                region_name=self.region
            )
            
            self.dynamodb = session.resource(
                'dynamodb',
                endpoint_url=local_endpoint,
                use_ssl=False,
                verify=False,
                config=Config(
                    signature_version='v4',
                    s3={'addressing_style': 'path'}
                )
            )
        else:
            # Production AWS DynamoDB
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        
        self.table = self.dynamodb.Table(self.table_name)
        
        # Try to create table for local development
        if use_local_dynamodb:
            self._ensure_table_exists()
        
        logger.info(f"DynamoDBTemplatesService initialized: table={self.table_name}, region={self.region}")
    
    def _ensure_table_exists(self):
        """Create the DynamoDB table if it doesn't exist (for local development)"""
        try:
            # Check if table exists
            self.table.load()
            logger.info(f"DynamoDB table '{self.table_name}' already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                # Table doesn't exist, create it
                logger.info(f"Creating DynamoDB table '{self.table_name}'")
                
                table = self.dynamodb.create_table(
                    TableName=self.table_name,
                    KeySchema=[
                        {
                            'AttributeName': 'PK',
                            'KeyType': 'HASH'  # Partition key
                        },
                        {
                            'AttributeName': 'SK',
                            'KeyType': 'RANGE'  # Sort key
                        }
                    ],
                    AttributeDefinitions=[
                        {
                            'AttributeName': 'PK',
                            'AttributeType': 'S'
                        },
                        {
                            'AttributeName': 'SK',
                            'AttributeType': 'S'
                        }
                    ],
                    BillingMode='PAY_PER_REQUEST'  # On-demand pricing for local dev
                )
                
                # Wait for table to be created
                table.wait_until_exists()
                logger.info(f"DynamoDB table '{self.table_name}' created successfully")
            else:
                logger.error(f"Error checking table existence: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error ensuring table exists: {e}")
            raise
    
    def _get_partition_key(self, template_type: str) -> str:
        """Generate partition key based on template type"""
        type_mapping = {
            'data_processing': 'DATA_PROCESSING_TEMPLATE',
            'reconciliation': 'RECONCILIATION_TEMPLATE', 
            'analysis': 'ANALYSIS_TEMPLATE',
            'transformation': 'TRANSFORMATION_TEMPLATE',
            'reporting': 'REPORTING_TEMPLATE'
        }
        template_type_key = type_mapping.get(template_type, 'UNKNOWN_TEMPLATE')
        return f"{template_type_key}#"
    
    def _convert_floats_to_decimal(self, obj: Any) -> Any:
        """Convert float values to Decimal for DynamoDB compatibility"""
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(item) for item in obj]
        return obj
    
    def _convert_decimal_to_float(self, obj: Any) -> Any:
        """Convert Decimal values back to float for JSON serialization"""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimal_to_float(item) for item in obj]
        return obj
    
    def _ensure_required_fields(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure template has all required fields with default values"""
        # Set defaults for missing fields
        template_data.setdefault('tags', [])
        template_data.setdefault('version', '1.0')
        template_data.setdefault('usage_count', 0)
        template_data.setdefault('rating', 0.0)
        template_data.setdefault('rating_count', 0)
        template_data.setdefault('is_public', False)
        template_data.setdefault('last_used_at', None)
        template_data.setdefault('created_by', None)
        template_data.setdefault('template_config', {})
        
        # Ensure timestamps exist
        if 'created_at' not in template_data:
            template_data['created_at'] = datetime.utcnow().isoformat()
        if 'updated_at' not in template_data:
            template_data['updated_at'] = datetime.utcnow().isoformat()
            
        return template_data
    
    def save_template(self, template_data: Dict[str, Any]) -> bool:
        """Save a template to DynamoDB"""
        try:
            template_id = template_data['id']
            template_type = template_data['template_type']
            
            # Add timestamps if not present
            if 'created_at' not in template_data:
                template_data['created_at'] = datetime.utcnow().isoformat()
            template_data['updated_at'] = datetime.utcnow().isoformat()
            
            # Ensure required fields have defaults
            template_data.setdefault('version', '1.0')
            template_data.setdefault('usage_count', 0)
            template_data.setdefault('rating', 0.0)
            template_data.setdefault('rating_count', 0)
            template_data.setdefault('is_public', False)
            
            # Prepare item for DynamoDB
            item = {
                'PK': f"{self._get_partition_key(template_type)}{template_id}",
                'SK': template_id,
                **self._convert_floats_to_decimal(template_data)
            }
            
            # Convert tags to StringSet if present
            if 'tags' in item and isinstance(item['tags'], list):
                if item['tags']:  # Only set StringSet if list is not empty
                    item['tags'] = set(item['tags'])
                else:
                    item.pop('tags')  # Remove empty tags list
            
            self.table.put_item(Item=item)
            
            logger.info(f"Saved {template_type} template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template {template_data.get('id', 'unknown')}: {e}")
            return False
    
    def get_template(self, template_id: str, template_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a specific template by ID. If template_type is not provided, search all types."""
        if template_type:
            return self._get_template_by_type(template_type, template_id)
        
        # Search all template types
        template_types = ['data_processing', 'reconciliation', 'analysis', 'transformation', 'reporting']
        for t_type in template_types:
            template = self._get_template_by_type(t_type, template_id)
            if template:
                return template
        
        return None
    
    def _get_template_by_type(self, template_type: str, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template of a specific type"""
        try:
            response = self.table.get_item(
                Key={
                    'PK': f"{self._get_partition_key(template_type)}{template_id}",
                    'SK': template_id
                }
            )
            
            if 'Item' not in response:
                return None
            
            item = response['Item']
            
            # Remove DynamoDB-specific keys
            for key in ['PK', 'SK']:
                item.pop(key, None)
            
            # Convert tags from StringSet to list
            if 'tags' in item and hasattr(item['tags'], '__iter__'):
                item['tags'] = list(item['tags'])
            
            # Ensure all required fields have default values
            item = self._ensure_required_fields(item)
            
            # Convert Decimal back to float
            item = self._convert_decimal_to_float(item)
            
            return item
            
        except Exception as e:
            logger.error(f"Error getting {template_type} template {template_id}: {e}")
            return None
    
    def list_templates(self, template_type: Optional[str] = None, category: Optional[str] = None, 
                      is_public: Optional[bool] = None, created_by: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List templates with optional filtering"""
        try:
            filter_expressions = []
            expression_values = {}
            
            # Filter by template type if specified
            if template_type:
                filter_expressions.append('template_type = :template_type')
                expression_values[':template_type'] = template_type
            
            # Add additional filters
            if category:
                filter_expressions.append('category = :category')
                expression_values[':category'] = category
            
            if is_public is not None:
                filter_expressions.append('is_public = :is_public')
                expression_values[':is_public'] = is_public
            
            if created_by:
                filter_expressions.append('created_by = :created_by')
                expression_values[':created_by'] = created_by
            
            # Build filter expression
            filter_expression = None
            if filter_expressions:
                filter_expression = ' AND '.join(filter_expressions)
            
            # Perform scan
            scan_kwargs = {}
            if filter_expression:
                scan_kwargs['FilterExpression'] = filter_expression
                scan_kwargs['ExpressionAttributeValues'] = expression_values
            
            response = self.table.scan(**scan_kwargs)
            templates = response.get('Items', [])
                
            # Sort by usage count and updated_at
            templates.sort(key=lambda x: (x.get('usage_count', 0), x.get('updated_at', '')), reverse=True)
            
            # Process results
            processed_templates = []
            for item in templates:
                # Remove DynamoDB-specific keys
                for key in ['PK', 'SK']:
                    item.pop(key, None)
                
                # Convert tags from StringSet to list
                if 'tags' in item and hasattr(item['tags'], '__iter__'):
                    item['tags'] = list(item['tags'])
                
                # Ensure all required fields have default values
                item = self._ensure_required_fields(item)
                
                # Convert Decimal back to float
                item = self._convert_decimal_to_float(item)
                processed_templates.append(item)
            
            # Apply pagination
            return processed_templates[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
    
    def search_templates(self, search_query: str, template_type: Optional[str] = None,
                        category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search templates by name, description, or tags"""
        try:
            # Start with all templates or filtered by type
            filter_expressions = []
            expression_values = {}
            
            if template_type:
                filter_expressions.append('template_type = :template_type')
                expression_values[':template_type'] = template_type
            
            if category:
                filter_expressions.append('category = :category')
                expression_values[':category'] = category
            
            # Build filter expression
            filter_expression = None
            if filter_expressions:
                filter_expression = ' AND '.join(filter_expressions)
            
            scan_kwargs = {}
            if filter_expression:
                scan_kwargs['FilterExpression'] = filter_expression
                scan_kwargs['ExpressionAttributeValues'] = expression_values
            
            response = self.table.scan(**scan_kwargs)
            templates = response.get('Items', [])
            
            # Apply text search and tag filtering
            search_results = []
            search_terms = search_query.lower().split() if search_query else []
            
            for item in templates:
                # Convert for filtering
                if 'tags' in item and hasattr(item['tags'], '__iter__'):
                    item['tags'] = list(item['tags'])
                
                # Ensure all required fields have default values
                item = self._ensure_required_fields(item)
                
                item = self._convert_decimal_to_float(item)
                
                # Text search in name and description
                name = item.get('name', '').lower()
                description = item.get('description', '').lower()
                template_tags = [tag.lower() for tag in item.get('tags', [])]
                
                # Check if search terms match
                matches_search = True
                if search_terms:
                    matches_search = any(
                        term in name or term in description or any(term in tag for tag in template_tags)
                        for term in search_terms
                    )
                
                # Check tag filter
                matches_tags = True
                if tags:
                    matches_tags = any(tag.lower() in template_tags for tag in tags)
                
                if matches_search and matches_tags:
                    # Remove DynamoDB-specific keys
                    for key in ['PK', 'SK']:
                        item.pop(key, None)
                    search_results.append(item)
            
            # Sort by relevance (usage count + rating)
            search_results.sort(
                key=lambda x: (x.get('usage_count', 0) * 0.7 + x.get('rating', 0) * 0.3),
                reverse=True
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching templates: {e}")
            return []
    
    def update_template(self, template_id: str, updates: Dict[str, Any], template_type: Optional[str] = None) -> bool:
        """Update an existing template"""
        try:
            # Get current template
            current_template = self.get_template(template_id, template_type)
            if not current_template:
                logger.warning(f"Template {template_id} not found for update")
                return False
            
            # Merge updates
            updated_template = {**current_template, **updates}
            updated_template['updated_at'] = datetime.utcnow().isoformat()
            
            # Save updated template
            return self.save_template(updated_template)
            
        except Exception as e:
            logger.error(f"Error updating template {template_id}: {e}")
            return False
    
    def delete_template(self, template_id: str, template_type: Optional[str] = None) -> bool:
        """Delete a template"""
        try:
            if template_type:
                return self._delete_template_by_type(template_type, template_id)
            
            # Search all template types and delete
            template_types = ['data_processing', 'reconciliation', 'analysis', 'transformation', 'reporting']
            for t_type in template_types:
                if self._delete_template_by_type(t_type, template_id):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting template {template_id}: {e}")
            return False
    
    def _delete_template_by_type(self, template_type: str, template_id: str) -> bool:
        """Delete a template of specific type"""
        try:
            # Check if template exists first
            if not self._get_template_by_type(template_type, template_id):
                return False
            
            self.table.delete_item(
                Key={
                    'PK': f"{self._get_partition_key(template_type)}{template_id}",
                    'SK': template_id
                }
            )
            
            logger.info(f"Deleted {template_type} template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting {template_type} template {template_id}: {e}")
            return False
    
    def mark_template_as_used(self, template_id: str, template_type: Optional[str] = None) -> bool:
        """Increment usage count and update last_used_at"""
        try:
            # Get current template
            current_template = self.get_template(template_id, template_type)
            if not current_template:
                return False
            
            # Update usage statistics
            current_template['usage_count'] = current_template.get('usage_count', 0) + 1
            current_template['last_used_at'] = datetime.utcnow().isoformat()
            
            # Save updated template
            return self.save_template(current_template)
            
        except Exception as e:
            logger.error(f"Error marking template {template_id} as used: {e}")
            return False
    
    def rate_template(self, template_id: str, rating: float, template_type: Optional[str] = None) -> bool:
        """Add a rating to a template (1-5 stars)"""
        try:
            if not (1.0 <= rating <= 5.0):
                logger.error(f"Invalid rating {rating}. Must be between 1.0 and 5.0")
                return False
            
            # Get current template
            current_template = self.get_template(template_id, template_type)
            if not current_template:
                return False
            
            # Update rating statistics
            current_rating = current_template.get('rating', 0.0)
            rating_count = current_template.get('rating_count', 0)
            
            # Calculate new average rating
            total_rating_score = current_rating * rating_count
            new_total_rating_score = total_rating_score + rating
            new_rating_count = rating_count + 1
            new_average_rating = new_total_rating_score / new_rating_count
            
            current_template['rating'] = round(new_average_rating, 2)
            current_template['rating_count'] = new_rating_count
            
            # Save updated template
            return self.save_template(current_template)
            
        except Exception as e:
            logger.error(f"Error rating template {template_id}: {e}")
            return False
    
    def get_categories(self, template_type: Optional[str] = None) -> List[str]:
        """Get all categories for templates"""
        try:
            scan_kwargs = {
                'ProjectionExpression': 'category'
            }
            
            if template_type:
                scan_kwargs['FilterExpression'] = 'template_type = :template_type'
                scan_kwargs['ExpressionAttributeValues'] = {':template_type': template_type}
            
            response = self.table.scan(**scan_kwargs)
            
            categories = set()
            for item in response.get('Items', []):
                if 'category' in item:
                    categories.add(item['category'])
            
            return sorted(list(categories))
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    def get_popular_templates(self, limit: int = 10, template_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get most popular templates by usage count and rating"""
        try:
            templates = self.list_templates(template_type=template_type, limit=1000)  # Get more for sorting
            
            # Sort by popularity score (usage_count * 0.7 + rating * rating_count * 0.3)
            templates.sort(
                key=lambda x: (
                    x.get('usage_count', 0) * 0.7 + 
                    x.get('rating', 0) * x.get('rating_count', 0) * 0.3
                ),
                reverse=True
            )
            
            return templates[:limit]
            
        except Exception as e:
            logger.error(f"Error getting popular templates: {e}")
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the DynamoDB templates service"""
        try:
            # Try to describe the table
            table_description = self.table.meta.client.describe_table(
                TableName=self.table_name
            )
            
            table_status = table_description['Table']['TableStatus']
            item_count = table_description['Table'].get('ItemCount', 0)
            
            return {
                'status': 'healthy' if table_status == 'ACTIVE' else 'unhealthy',
                'table_name': self.table_name,
                'table_status': table_status,
                'item_count': item_count,
                'region': self.region
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'table_name': self.table_name,
                'region': self.region
            }


# Global instance
dynamodb_templates_service = DynamoDBTemplatesService()