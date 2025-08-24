# DynamoDB Rules Service - Simplified Schema for Delta Rules and Rule Management
"""
DynamoDB integration for storing delta rules and reconciliation rules using simplified schema.

Table Structure:
- Simplified single table design with only primary keys (PK and SK)
- Uses full table scans for queries (no GSI indexes)
- Optimized for cost and schema flexibility over query performance
- Supports both delta rules and reconciliation/transformation rules

Table Name: Rules (configurable via environment variable)

Key Design (Simplified):
- PK: Rule type identifier (DELTA_RULE#, RECONCILIATION_RULE#, TRANSFORMATION_RULE#)  
- SK: Rule ID (unique identifier for each rule)

Core Attributes:
- id: Rule unique identifier
- rule_type: "delta", "reconciliation", "transformation"
- name: Rule display name
- description: Optional description
- category: Rule category
- tags: List of tags (as StringSet)
- template_id: Template identifier (optional)
- template_name: Template display name (optional)
- created_at: ISO timestamp
- updated_at: ISO timestamp
- version: Rule version (default "1.0")
- rule_config: JSON blob containing rule configuration
- usage_count: Number of times rule has been used
- last_used_at: ISO timestamp of last usage (optional)

Example Items:
1. Delta Rule:
   PK: "DELTA_RULE#delta_rule_123"
   SK: "delta_rule_123"
   rule_type: "delta"
   name: "Monthly Delta Analysis"
   category: "financial"

2. Reconciliation Rule:
   PK: "RECONCILIATION_RULE#rule_456"
   SK: "rule_456"
   rule_type: "reconciliation"
   name: "Transaction Matching"
   category: "financial"

Query Strategy: All operations use table scans with filters for maximum flexibility.
Performance: Optimized for small to medium datasets (<10k rules) with cost efficiency.
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


class DynamoDBRulesService:
    """Service for managing rules in DynamoDB with single table design"""
    
    def __init__(self):
        self.table_name = os.getenv('DYNAMODB_RULES_TABLE', 'Rules')
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
        
        logger.info(f"DynamoDBRulesService initialized: table={self.table_name}, region={self.region}")
    
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
    
    def _get_partition_key(self, rule_type: str) -> str:
        """Generate partition key based on rule type"""
        type_mapping = {
            'delta': 'DELTA_RULE',
            'reconciliation': 'RECONCILIATION_RULE', 
            'transformation': 'TRANSFORMATION_RULE'
        }
        rule_type_key = type_mapping.get(rule_type, 'UNKNOWN_RULE')
        return f"{rule_type_key}#"
    
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
    
    def save_rule(self, rule_type: str, rule_data: Dict[str, Any]) -> bool:
        """Save a rule to DynamoDB - simplified with only primary keys"""
        try:
            rule_id = rule_data['id']
            
            # Prepare minimal item for DynamoDB with only primary keys
            item = {
                'PK': f"{self._get_partition_key(rule_type)}{rule_id}",
                'SK': rule_id,
                **self._convert_floats_to_decimal(rule_data)
            }
            
            # Convert tags to StringSet if present
            if 'tags' in item and isinstance(item['tags'], list):
                if item['tags']:  # Only set StringSet if list is not empty
                    item['tags'] = set(item['tags'])
                else:
                    item.pop('tags')  # Remove empty tags list
            
            self.table.put_item(Item=item)
            
            logger.info(f"Saved {rule_type} rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {rule_type} rule {rule_data.get('id', 'unknown')}: {e}")
            return False
    
    def get_rule(self, rule_type: str, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific rule by ID"""
        try:
            response = self.table.get_item(
                Key={
                    'PK': f"{self._get_partition_key(rule_type)}{rule_id}",
                    'SK': rule_id
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
            
            # Convert Decimal back to float
            item = self._convert_decimal_to_float(item)
            
            return item
            
        except Exception as e:
            logger.error(f"Error getting {rule_type} rule {rule_id}: {e}")
            return None
    
    def list_rules(self, rule_type: str, category: Optional[str] = None, 
                  template_id: Optional[str] = None, limit: int = 50, 
                  offset: int = 0) -> List[Dict[str, Any]]:
        """List rules with optional filtering - simplified with full scan"""
        try:
            # Always scan for rules of this type
            filter_expression = 'rule_type = :rule_type'
            expression_values = {':rule_type': rule_type}
            
            # Add additional filters if specified
            if category:
                filter_expression += ' AND category = :category'
                expression_values[':category'] = category
            
            if template_id:
                filter_expression += ' AND template_id = :template_id'
                expression_values[':template_id'] = template_id
            
            response = self.table.scan(
                FilterExpression=filter_expression,
                ExpressionAttributeValues=expression_values
            )
            rules = response.get('Items', [])
                
            # Sort by updated_at desc
            rules.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            
            # Process results
            processed_rules = []
            for item in rules:
                # Remove DynamoDB-specific keys
                for key in ['PK', 'SK']:
                    item.pop(key, None)
                
                # Convert tags from StringSet to list
                if 'tags' in item and hasattr(item['tags'], '__iter__'):
                    item['tags'] = list(item['tags'])
                
                # Convert Decimal back to float
                item = self._convert_decimal_to_float(item)
                processed_rules.append(item)
            
            # Apply pagination
            return processed_rules[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Error listing {rule_type} rules: {e}")
            return []
    
    def update_rule(self, rule_type: str, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        try:
            # Get current rule first
            current_rule = self.get_rule(rule_type, rule_id)
            if not current_rule:
                return False
            
            # Merge updates
            updated_rule = {**current_rule, **updates}
            updated_rule['updated_at'] = datetime.utcnow().isoformat()
            
            # Save updated rule
            return self.save_rule(rule_type, updated_rule)
            
        except Exception as e:
            logger.error(f"Error updating {rule_type} rule {rule_id}: {e}")
            return False
    
    def delete_rule(self, rule_type: str, rule_id: str) -> bool:
        """Delete a rule"""
        try:
            self.table.delete_item(
                Key={
                    'PK': f"{self._get_partition_key(rule_type)}{rule_id}",
                    'SK': rule_id
                }
            )
            
            logger.info(f"Deleted {rule_type} rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting {rule_type} rule {rule_id}: {e}")
            return False
    
    def search_rules(self, rule_type: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search rules with complex filters"""
        try:
            # Start with all rules of this type
            response = self.table.scan(
                FilterExpression='rule_type = :rule_type',
                ExpressionAttributeValues={
                    ':rule_type': rule_type
                }
            )
            
            rules = response.get('Items', [])
            
            # Apply filters
            filtered_rules = []
            for item in rules:
                include_rule = True
                
                # Convert for filtering
                # Convert tags from StringSet to list
                if 'tags' in item and hasattr(item['tags'], '__iter__'):
                    item['tags'] = list(item['tags'])
                
                # Convert Decimal back to float
                item = self._convert_decimal_to_float(item)
                
                # Apply filters
                if filters.get('category') and item.get('category') != filters['category']:
                    include_rule = False
                
                if filters.get('template_id') and item.get('template_id') != filters['template_id']:
                    include_rule = False
                
                if filters.get('tags') and include_rule:
                    rule_tags = item.get('tags', [])
                    if not any(tag in rule_tags for tag in filters['tags']):
                        include_rule = False
                
                if filters.get('name_contains') and include_rule:
                    search_term = filters['name_contains'].lower()
                    name = item.get('name', '').lower()
                    description = item.get('description', '').lower()
                    if search_term not in name and search_term not in description:
                        include_rule = False
                
                if filters.get('min_usage_count') and include_rule:
                    usage_count = item.get('usage_count', 0)
                    if usage_count < filters['min_usage_count']:
                        include_rule = False
                
                if include_rule:
                    # Remove DynamoDB-specific keys
                    for key in ['PK', 'SK']:
                        item.pop(key, None)
                    filtered_rules.append(item)
            
            # Sort by relevance (usage count + recency)
            filtered_rules.sort(
                key=lambda x: (x.get('usage_count', 0), x.get('updated_at', '')),
                reverse=True
            )
            
            return filtered_rules
            
        except Exception as e:
            logger.error(f"Error searching {rule_type} rules: {e}")
            return []
    
    def get_categories(self, rule_type: str) -> List[str]:
        """Get all categories for a rule type"""
        try:
            response = self.table.scan(
                FilterExpression='rule_type = :rule_type',
                ProjectionExpression='category',
                ExpressionAttributeValues={
                    ':rule_type': rule_type
                }
            )
            
            categories = set()
            for item in response.get('Items', []):
                if 'category' in item:
                    categories.add(item['category'])
            
            return sorted(list(categories))
            
        except Exception as e:
            logger.error(f"Error getting categories for {rule_type} rules: {e}")
            return []
    
    def bulk_delete_rules(self, rule_type: str, rule_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple rules at once"""
        try:
            deleted_count = 0
            not_found_ids = []
            
            for rule_id in rule_ids:
                success = self.delete_rule(rule_type, rule_id)
                if success:
                    deleted_count += 1
                else:
                    not_found_ids.append(rule_id)
            
            return {
                'deleted_count': deleted_count,
                'not_found_ids': not_found_ids
            }
            
        except Exception as e:
            logger.error(f"Error bulk deleting {rule_type} rules: {e}")
            return {'deleted_count': 0, 'not_found_ids': rule_ids}
    
    def mark_rule_as_used(self, rule_type: str, rule_id: str) -> bool:
        """Increment usage count and update last_used_at"""
        try:
            # Get current rule
            current_rule = self.get_rule(rule_type, rule_id)
            if not current_rule:
                return False
            
            # Update usage statistics
            current_rule['usage_count'] = current_rule.get('usage_count', 0) + 1
            current_rule['last_used_at'] = datetime.utcnow().isoformat()
            
            # Save updated rule
            return self.save_rule(rule_type, current_rule)
            
        except Exception as e:
            logger.error(f"Error marking {rule_type} rule {rule_id} as used: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the DynamoDB service"""
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
dynamodb_rules_service = DynamoDBRulesService()