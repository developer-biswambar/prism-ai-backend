# Process Analytics Service - Track and store process execution analytics
"""
DynamoDB integration for storing process execution analytics and token usage.

Table Structure:
- Single table design with primary keys (PK and SK)
- Tracks all user process executions with detailed analytics
- Supports token usage tracking, performance metrics, and cost analysis

Table Name: ProcessAnalytics (configurable via environment variable)

Key Design:
- PK: USER#{user_id} (future user tracking)
- SK: PROCESS#{process_id}#{timestamp}

Core Attributes:
- process_id: Unique process identifier
- process_type: "miscellaneous", "reconciliation", "transformation", etc.
- process_name: User-defined process name
- user_prompt: Original user prompt
- generated_sql: AI-generated SQL query
- status: "success", "failed", "partial"
- confidence_score: AI confidence in result (0-100)

Execution Metrics:
- input_row_count: Total rows in input files
- output_row_count: Number of rows in result
- processing_time_seconds: Total execution time
- created_at: ISO timestamp
- completed_at: ISO timestamp

Token Usage (LLM Analytics):
- prompt_tokens: Input tokens used
- completion_tokens: Output tokens generated
- total_tokens: Total tokens consumed
- estimated_cost_usd: Estimated API cost
- model_used: AI model identifier

File Analytics:
- input_files: List of file info and row counts
- file_schemas: Column structures and data types
- data_quality_issues: Detected data issues

Error Analytics (for failed processes):
- error_type: Classification of error
- error_message: Detailed error description
- error_context: Additional debugging information
- retry_suggestions: AI-generated retry recommendations
"""

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

logger = logging.getLogger(__name__)


class ProcessAnalyticsService:
    """Service for managing process execution analytics in DynamoDB"""
    
    def __init__(self):
        self.table_name = os.getenv('DYNAMODB_PROCESS_ANALYTICS_TABLE', 'ProcessAnalytics')
        self.region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
        
        # Check for local development environment
        use_local_dynamodb = os.getenv('USE_LOCAL_DYNAMODB', 'false').lower() == 'true'
        
        if use_local_dynamodb:
            # Configure for local DynamoDB (LocalStack)
            self.dynamodb = boto3.resource(
                'dynamodb',
                endpoint_url=os.getenv('DYNAMODB_ENDPOINT', 'http://localhost:4566'),
                region_name=self.region,
                aws_access_key_id='test',
                aws_secret_access_key='test'
            )
            # self._ensure_table_exists()

        else:
            # Configure for AWS DynamoDB
            config = Config(
                region_name=self.region,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
            self.dynamodb = boto3.resource('dynamodb', config=config)
        
        self.table = self.dynamodb.Table(self.table_name)
        logger.info(f"ProcessAnalyticsService initialized with table: {self.table_name}")

    def _ensure_table_exists(self):
        """Create the DynamoDB table if it doesn't exist (for local development)"""

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


    def _decimal_to_number(self, obj):
        """Convert Decimal objects to float/int for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._decimal_to_number(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._decimal_to_number(v) for v in obj]
        elif isinstance(obj, Decimal):
            return float(obj) if obj % 1 else int(obj)
        return obj

    def _number_to_decimal(self, obj):
        """Convert numbers to Decimal for DynamoDB storage"""
        if isinstance(obj, dict):
            return {k: self._number_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._number_to_decimal(v) for v in obj]
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return Decimal(str(obj))
        return obj

    def record_process_execution(
        self,
        process_id: str,
        process_type: str,
        process_name: str,
        user_prompt: str,
        input_files: List[Dict[str, Any]],
        status: str = "success",
        generated_sql: Optional[str] = None,
        confidence_score: Optional[float] = None,
        processing_time_seconds: Optional[float] = None,
        output_row_count: Optional[int] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None,
        user_id: str = "default_user"
    ) -> bool:
        """
        Record a process execution with comprehensive analytics
        
        Args:
            process_id: Unique identifier for the process
            process_type: Type of process (miscellaneous, reconciliation, etc.)
            process_name: User-defined name for the process
            user_prompt: Original user prompt
            input_files: List of input file information
            status: Execution status (success, failed, partial)
            generated_sql: AI-generated SQL query
            confidence_score: AI confidence in result (0-100)
            processing_time_seconds: Total execution time
            output_row_count: Number of rows in result
            token_usage: LLM token usage information
            error_info: Error details if process failed
            user_id: User identifier (for future multi-user support)
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Calculate total input rows
            input_row_count = sum(file_info.get('row_count', 0) for file_info in input_files)
            
            # Prepare token usage with cost estimation
            processed_token_usage = {}
            if token_usage:
                processed_token_usage = {
                    'prompt_tokens': token_usage.get('prompt_tokens', 0),
                    'completion_tokens': token_usage.get('completion_tokens', 0),
                    'total_tokens': token_usage.get('total_tokens', 0),
                    'model_used': token_usage.get('model', 'unknown'),
                    'estimated_cost_usd': self._estimate_cost(token_usage)
                }
            
            # Prepare the item for DynamoDB
            item = {
                'PK': f"USER#{user_id}",
                'SK': f"PROCESS#{process_id}#{current_time}",
                'process_id': process_id,
                'process_type': process_type,
                'process_name': process_name,
                'user_prompt': user_prompt,
                'status': status,
                'input_row_count': input_row_count,
                'output_row_count': output_row_count or 0,
                'processing_time_seconds': processing_time_seconds or 0,
                'created_at': current_time,
                'completed_at': current_time,
                'input_files': [
                    {
                        'filename': f.get('filename', ''),
                        'file_id': f.get('file_id', ''),
                        'row_count': f.get('row_count', 0),
                        'column_count': len(f.get('columns', [])),
                        'columns': f.get('columns', [])
                    }
                    for f in input_files
                ]
            }
            
            # Add optional fields
            if generated_sql:
                item['generated_sql'] = generated_sql
            
            if confidence_score is not None:
                item['confidence_score'] = confidence_score
                
            if processed_token_usage:
                item['token_usage'] = processed_token_usage
                
            if error_info:
                item['error_info'] = error_info
            
            # Convert numbers to Decimal for DynamoDB
            item = self._number_to_decimal(item)
            
            # Store in DynamoDB
            self.table.put_item(Item=item)
            
            logger.info(f"Process analytics recorded for {process_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record process analytics for {process_id}: {str(e)}")
            return False

    def get_user_processes(
        self,
        user_id: str = "default_user",
        limit: int = 50,
        last_evaluated_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get recent processes for a user"""
        try:
            query_params = {
                'KeyConditionExpression': boto3.dynamodb.conditions.Key('PK').eq(f"USER#{user_id}"),
                'ScanIndexForward': False,  # Most recent first
                'Limit': limit
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = json.loads(last_evaluated_key)
            
            response = self.table.query(**query_params)
            
            processes = [self._decimal_to_number(item) for item in response.get('Items', [])]
            
            return {
                'processes': processes,
                'last_evaluated_key': json.dumps(response.get('LastEvaluatedKey')) if response.get('LastEvaluatedKey') else None,
                'total_count': len(processes)
            }
            
        except Exception as e:
            logger.error(f"Failed to get user processes: {str(e)}")
            return {'processes': [], 'last_evaluated_key': None, 'total_count': 0}

    def get_process_analytics_summary(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Get analytics summary for a user"""
        try:
            # Get all processes for the user (could be optimized with time ranges)
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('PK').eq(f"USER#{user_id}"),
                Limit=1000  # Reasonable limit for analytics
            )
            
            processes = [self._decimal_to_number(item) for item in response.get('Items', [])]
            
            if not processes:
                return self._empty_analytics_summary()
            
            # Calculate summary statistics
            total_processes = len(processes)
            successful_processes = len([p for p in processes if p.get('status') == 'success'])
            failed_processes = len([p for p in processes if p.get('status') == 'failed'])
            
            total_input_rows = sum(p.get('input_row_count', 0) for p in processes)
            total_output_rows = sum(p.get('output_row_count', 0) for p in processes)
            total_processing_time = sum(p.get('processing_time_seconds', 0) for p in processes)
            
            # Token usage analytics
            total_tokens = sum(p.get('token_usage', {}).get('total_tokens', 0) for p in processes)
            total_cost = sum(p.get('token_usage', {}).get('estimated_cost_usd', 0) for p in processes)
            
            # Process type breakdown
            process_types = {}
            for process in processes:
                ptype = process.get('process_type', 'unknown')
                process_types[ptype] = process_types.get(ptype, 0) + 1
            
            return {
                'total_processes': total_processes,
                'successful_processes': successful_processes,
                'failed_processes': failed_processes,
                'success_rate': round((successful_processes / total_processes) * 100, 2) if total_processes > 0 else 0,
                'total_input_rows': total_input_rows,
                'total_output_rows': total_output_rows,
                'total_processing_time_seconds': round(total_processing_time, 2),
                'avg_processing_time_seconds': round(total_processing_time / total_processes, 2) if total_processes > 0 else 0,
                'total_tokens_used': total_tokens,
                'total_estimated_cost_usd': round(total_cost, 4),
                'avg_cost_per_process': round(total_cost / total_processes, 4) if total_processes > 0 else 0,
                'process_type_breakdown': process_types,
                'recent_processes': processes[:10]  # Last 10 processes
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {str(e)}")
            return self._empty_analytics_summary()

    def _estimate_cost(self, token_usage: Dict[str, Any]) -> float:
        """Estimate cost based on token usage and model"""
        model = token_usage.get('model', '').lower()
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        
        # Cost per 1000 tokens (approximate pricing as of 2024)
        cost_rates = {
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
            'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
            'claude-3-opus': {'prompt': 0.015, 'completion': 0.075},
            'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
            'claude-3-haiku': {'prompt': 0.00025, 'completion': 0.00125},
            # New ones:
            'gpt-4.1-nano': {'prompt': 0.10, 'completion': 0.40},
            'o1': {'prompt': 15.0, 'completion': 60.0},
            'o3': {'prompt': 2.0, 'completion': 8.0},
        }

        # Default to GPT-4 pricing if model not recognized
        rates = cost_rates.get(model, cost_rates['gpt-4'])
        
        prompt_cost = (prompt_tokens / 1000) * rates['prompt']
        completion_cost = (completion_tokens / 1000) * rates['completion']
        
        return round(prompt_cost + completion_cost, 6)

    def _empty_analytics_summary(self) -> Dict[str, Any]:
        """Return empty analytics summary"""
        return {
            'total_processes': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'success_rate': 0,
            'total_input_rows': 0,
            'total_output_rows': 0,
            'total_processing_time_seconds': 0,
            'avg_processing_time_seconds': 0,
            'total_tokens_used': 0,
            'total_estimated_cost_usd': 0,
            'avg_cost_per_process': 0,
            'process_type_breakdown': {},
            'recent_processes': []
        }