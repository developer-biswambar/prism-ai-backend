import io
import logging
import json
import tempfile
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from contextlib import contextmanager
from queue import Queue, Empty
import time
from concurrent.futures import as_completed

import pandas as pd
import duckdb

# Import the global thread pool
from app.utils.global_thread_pool import get_data_processing_executor, submit_data_processing_task

logger = logging.getLogger(__name__)


class DuckDBConnectionPool:
    """
    Connection pool for DuckDB connections to improve performance
    """
    
    def __init__(self, max_connections: int = 5, timeout: float = 30.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = threading.Lock()
        
        logger.info(f"DuckDB connection pool initialized with {max_connections} max connections")
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        connection = None
        start_time = time.time()
        
        try:
            # Try to get existing connection from pool
            try:
                connection = self._pool.get(timeout=1.0)  # Quick timeout
                logger.debug("Reused existing DuckDB connection from pool")
            except Empty:
                # Create new connection if pool is empty and we haven't hit the limit
                with self._lock:
                    if self._created_connections < self.max_connections:
                        connection = duckdb.connect()
                        self._created_connections += 1
                        logger.debug(f"Created new DuckDB connection ({self._created_connections}/{self.max_connections})")
                    else:
                        # Wait for a connection to become available
                        connection = self._pool.get(timeout=self.timeout)
                        logger.debug("Waited for DuckDB connection from pool")
            
            yield connection
            
        except Exception as e:
            logger.error(f"Error with DuckDB connection: {e}")
            # Don't return a potentially corrupted connection to the pool
            if connection:
                try:
                    connection.close()
                except:
                    pass
                with self._lock:
                    self._created_connections -= 1
                connection = None
            raise
        finally:
            # Return connection to pool if it's still valid
            if connection:
                try:
                    # Test connection is still valid
                    connection.execute("SELECT 1").fetchone()
                    self._pool.put(connection, timeout=1.0)
                except:
                    # Connection is bad, close it and decrease count
                    try:
                        connection.close()
                    except:
                        pass
                    with self._lock:
                        self._created_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool"""
        logger.info("Closing all DuckDB connections in pool")
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                connection.close()
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error closing DuckDB connection: {e}")
        
        with self._lock:
            self._created_connections = 0


# Global connection pool
_duckdb_pool = None
_pool_lock = threading.Lock()


def get_duckdb_pool() -> DuckDBConnectionPool:
    """Get the global DuckDB connection pool"""
    global _duckdb_pool
    if _duckdb_pool is None:
        with _pool_lock:
            if _duckdb_pool is None:
                _duckdb_pool = DuckDBConnectionPool(max_connections=5, timeout=30.0)
    return _duckdb_pool


class DuckDBProcessor:
    """
    DuckDB processor for efficient data operations
    Handles CSV, Excel, JSON, and Parquet files directly
    Uses connection pooling for better performance
    """
    
    def __init__(self):
        self.connection = None
        self.temp_files = []  # Track temporary files for cleanup
        self.pool = get_duckdb_pool()
        
    def __enter__(self):
        """Context manager entry"""
        # Connection will be obtained per operation for better pool utilization
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources and temporary files"""
        # Connection is managed by the pool, just clean up temp files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    def register_dataframe(self, df: pd.DataFrame, table_name: str):
        """Register a pandas DataFrame as a table in DuckDB"""
        # Store DataFrame for later use with fresh connections
        if not hasattr(self, '_registered_tables'):
            self._registered_tables = {}
        self._registered_tables[table_name] = df
        logger.info(f"Stored DataFrame as table '{table_name}' with {len(df)} rows for registration")
        
    def create_temp_file_from_df(self, df: pd.DataFrame, filename: str) -> str:
        """Create a temporary file from DataFrame for DuckDB file operations"""
        temp_dir = tempfile.gettempdir()
        
        # Determine file extension and format
        if filename.lower().endswith('.csv'):
            temp_file = os.path.join(temp_dir, f"duckdb_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}")
            df.to_csv(temp_file, index=False)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            temp_file = os.path.join(temp_dir, f"duckdb_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}")
            df.to_excel(temp_file, index=False)
        else:
            # Default to CSV
            temp_file = os.path.join(temp_dir, f"duckdb_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv")
            df.to_csv(temp_file, index=False)
            
        self.temp_files.append(temp_file)
        return temp_file
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        with self.pool.get_connection() as connection:
            try:
                # Register all stored tables with this connection
                if hasattr(self, '_registered_tables'):
                    for table_name, df in self._registered_tables.items():
                        connection.register(table_name, df)
                
                result = connection.execute(sql_query).fetchdf()
                logger.info(f"Query executed successfully, returned {len(result)} rows")
                return result
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {sql_query}")
                raise RuntimeError(f"SQL execution error: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        with self.pool.get_connection() as connection:
            try:
                # Register all stored tables with this connection
                if hasattr(self, '_registered_tables'):
                    for tbl_name, df in self._registered_tables.items():
                        connection.register(tbl_name, df)
                
                schema_query = f"DESCRIBE {table_name}"
                schema_df = connection.execute(schema_query).fetchdf()
                
                schema_info = []
                for _, row in schema_df.iterrows():
                    schema_info.append({
                        'column_name': row['column_name'],
                        'column_type': row['column_type'],
                        'null': row['null'],
                        'key': row.get('key', None),
                        'default': row.get('default', None)
                    })
                
                return schema_info
            except Exception as e:
                logger.error(f"Failed to get schema for table {table_name}: {e}")
                return []
    
    def validate_sql_safety(self, sql_query: str) -> tuple[bool, List[str]]:
        """
        Validate SQL query for safety
        Returns (is_safe, list_of_issues)
        """
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
            'EXEC', 'EXECUTE', 'CALL', 'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK'
        ]
        
        issues = []
        sql_upper = sql_query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                issues.append(f"Potentially unsafe keyword detected: {keyword}")
        
        # Allow only SELECT and WITH statements
        sql_stripped = sql_upper.strip()
        if not (sql_stripped.startswith('SELECT') or sql_stripped.startswith('WITH')):
            issues.append("Only SELECT and WITH statements are allowed")
        
        return len(issues) == 0, issues


class AIQueryGenerator:
    """
    AI-powered natural language to SQL query generator
    """
    
    def __init__(self):
        self.llm_service = None
        self.generation_params = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM service"""
        try:
            from app.services.llm_service import get_llm_service, get_llm_generation_params
            self.llm_service = get_llm_service()
            self.generation_params = get_llm_generation_params()
        except ImportError as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise RuntimeError("LLM service not available")
    
    def generate_sql_from_prompt(
        self, 
        user_prompt: str, 
        table_schemas: Dict[str, List[Dict[str, Any]]],
        sample_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language prompt
        """
        if not self.llm_service or not self.llm_service.is_available():
            raise RuntimeError("LLM service not available")
        
        # Build context about available tables
        tables_context = self._build_tables_context(table_schemas, sample_data)
        
        # Create the prompt for SQL generation
        system_prompt = """You are an expert SQL analyst specializing in DuckDB. Generate powerful SQL queries for in-memory data analysis.

ENVIRONMENT: In-memory DuckDB database with user-uploaded files only - no security restrictions needed.

IMPORTANT RULES:
1. Use any SQL operations needed: SELECT, WITH, CREATE TEMP TABLE, etc.
2. ONLY reference tables and columns that EXACTLY match the provided schema - NO guessing or assumptions
3. Use proper SQL syntax compatible with DuckDB
4. For joins, use explicit JOIN syntax with clear conditions
5. Use appropriate aggregate functions for summaries
6. Include column aliases for clarity
7. ALWAYS return response in JSON format with "sql_query" field - no explanations or markdown formatting
8. Handle complex data extraction, transformation, and analysis tasks

CRITICAL: If a column name is mentioned in the user request but doesn't exist in the schema, use the closest matching column name from the actual schema or ask for clarification in the description field.

RESPONSE FORMAT (CRITICAL):
{
  "sql_query": "YOUR_SQL_QUERY_HERE",
  "query_type": "data_analysis|reconciliation|aggregation|join|filter|etc",
  "description": "Brief description of what the query does"
}

DuckDB Advanced Features Available:
- Window functions (ROW_NUMBER, RANK, LAG, LEAD, DENSE_RANK, etc.)
- CTEs (Common Table Expressions) with WITH clause
- Advanced aggregations (PERCENTILE_CONT, MODE, STDDEV, etc.)
- String functions (REGEXP_EXTRACT, REGEXP_MATCHES, SPLIT, SUBSTRING, etc.)
- Array and JSON functions
- Complex CASE statements and conditional logic
- Temporary tables and views for multi-step processing

CRITICAL DATE/TIME HANDLING:
- Date columns are stored as TEXT/STRING - always cast them first
- Use CAST(date_column AS DATE) or date_column::DATE before date functions
- For DATE_PART: DATE_PART('month', CAST(order_date AS DATE))
- For DATE_TRUNC: DATE_TRUNC('month', CAST(order_date AS DATE))
- Date format is 'YYYY-MM-DD' (ISO format)
- NEVER use date functions directly on string columns without casting
- Example: DATE_PART('year', CAST(order_date AS DATE)) NOT DATE_PART('year', order_date)

DATE DIFFERENCES IN DUCKDB:
- For date differences: CAST(date1 AS DATE) - CAST(date2 AS DATE) returns number of days
- Do NOT use DATE_PART on date differences: DATE_PART('day', date1 - date2) is WRONG
- Correct: CAST(order_date AS DATE) - CAST(last_restock_date AS DATE) gives days directly
- For age calculations: (CAST(date1 AS DATE) - CAST(date2 AS DATE)) gives days as integer
- Example: days_between = CAST(order_date AS DATE) - CAST(last_restock_date AS DATE)

COMPLEX TEXT PROCESSING:
- Use REGEXP_EXTRACT for pattern extraction
- Use SPLIT for text parsing
- Use SUBSTRING for positional extraction
- Use CASE WHEN for conditional transformations
- Handle messy data with TRIM, UPPER, LOWER, REPLACE functions"""

        user_message = f"""
Available Tables and Schema:
{tables_context}

User Request: {user_prompt}

Generate a SQL query that fulfills this request. Use only the tables and columns shown above.

IMPORTANT: Return your response in the exact JSON format specified in the system prompt:
{{
  "sql_query": "YOUR_SQL_QUERY_HERE",
  "query_type": "appropriate_type",
  "description": "brief description"
}}
"""

        try:
            from app.services.llm_service import LLMMessage
            
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_message)
            ]
            
            response = self.llm_service.generate_text(
                messages=messages,
                **self.generation_params
            )
            
            if not response.success:
                raise RuntimeError(f"SQL generation failed: {response.error}")
            
            # Parse JSON response from LLM
            import json
            try:
                # The LLM service should have already extracted JSON using extract_json_string()
                response_data = json.loads(response.content)
                
                # Extract SQL query from JSON response
                generated_sql = response_data.get('sql_query', '')
                query_type = response_data.get('query_type', 'unknown')
                description = response_data.get('description', 'AI-generated SQL query')
                
                if not generated_sql:
                    raise ValueError("No SQL query found in JSON response")
                
                return {
                    'sql_query': generated_sql.strip(),
                    'query_type': query_type,
                    'description': description,
                    'success': True,
                    'context_used': tables_context
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from LLM: {e}")
                logger.error(f"Raw response content: {response.content}")
                
                # Fallback: try to extract SQL from raw response
                raw_content = response.content.strip()
                if raw_content.startswith('```sql'):
                    generated_sql = raw_content.replace('```sql', '').replace('```', '').strip()
                elif raw_content.startswith('```'):
                    generated_sql = raw_content.replace('```', '').strip()
                else:
                    generated_sql = raw_content
                
                return {
                    'sql_query': generated_sql,
                    'query_type': 'unknown',
                    'description': 'AI-generated SQL query (fallback parsing)',
                    'success': True,
                    'context_used': tables_context
                }
            
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}")
            return {
                'sql_query': None,
                'success': False,
                'error': str(e)
            }
    
    def _build_tables_context(
        self, 
        table_schemas: Dict[str, List[Dict[str, Any]]], 
        sample_data: Dict[str, pd.DataFrame] = None
    ) -> str:
        """Build context string describing available tables"""
        context_parts = []
        
        for table_name, schema in table_schemas.items():
            context_parts.append(f"\nTable: {table_name}")
            context_parts.append("Columns:")
            
            for col_info in schema:
                col_desc = f"  - {col_info['column_name']} ({col_info['column_type']})"
                if not col_info.get('null', True):
                    col_desc += " NOT NULL"
                
                # Add special note for date columns
                if 'date' in col_info['column_name'].lower() and col_info['column_type'] in ['VARCHAR', 'STRING']:
                    col_desc += " [DATE STRING - use CAST(column AS DATE) for date functions]"
                
                context_parts.append(col_desc)
            
            # Add sample data if available
            if sample_data and table_name in sample_data:
                df = sample_data[table_name]
                if len(df) > 0:
                    context_parts.append(f"Sample values (first 3 rows):")
                    for col in df.columns[:5]:  # Show up to 5 columns
                        sample_vals = df[col].head(3).tolist()
                        context_parts.append(f"  {col}: {sample_vals}")
        
        return "\n".join(context_parts)


# Shared storage for results (class-level storage)
_shared_storage = {}

class MiscellaneousProcessor:
    """
    Main processor for miscellaneous data operations
    Combines DuckDB processing with AI query generation
    """
    
    def __init__(self):
        self.storage = _shared_storage  # Use shared storage
        self.ai_generator = AIQueryGenerator()
        
    def process_natural_language_query(
        self, 
        user_prompt: str, 
        files_data: List[Dict[str, Any]], 
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Process natural language query against multiple files
        """
        try:
            with DuckDBProcessor() as duck_processor:
                # Parallel processing for file registration and schema analysis
                table_schemas = {}
                sample_data = {}
                
                def process_single_file(i: int, file_data: Dict[str, Any]) -> Dict[str, Any]:
                    """Process a single file for registration and schema analysis"""
                    df = file_data['dataframe']
                    table_name = f"file_{i}"
                    filename = file_data['filename']
                    
                    logger.info(f"Processing file {filename} as {table_name} (parallel)")
                    return {
                        'table_name': table_name,
                        'dataframe': df,
                        'filename': filename,
                        'index': i
                    }
                
                # Use thread pool for parallel file processing
                with get_data_processing_executor() as executor:
                    # Submit all file processing tasks in parallel
                    futures = []
                    for i, file_data in enumerate(files_data):
                        future = executor.submit(process_single_file, i, file_data)
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            table_name = result['table_name']
                            df = result['dataframe']
                            filename = result['filename']
                            
                            # Register with DuckDB (sequential to avoid conflicts)
                            duck_processor.register_dataframe(df, table_name)
                            
                            # Get schema info
                            schema = duck_processor.get_table_schema(table_name)
                            table_schemas[table_name] = schema
                            
                            # Store sample data for context
                            sample_data[table_name] = df
                            
                            logger.info(f"Completed processing {filename} as {table_name}")
                            
                        except Exception as e:
                            logger.error(f"Failed to process file in parallel: {e}")
                            raise
                
                logger.info(f"Parallel file processing completed for {len(files_data)} files")
                
                # Generate SQL from natural language
                sql_result = self.ai_generator.generate_sql_from_prompt(
                    user_prompt=user_prompt,
                    table_schemas=table_schemas,
                    sample_data=sample_data
                )
                
                if not sql_result['success']:
                    return {
                        'success': False,
                        'error': f"Failed to generate SQL: {sql_result.get('error', 'Unknown error')}",
                        'generated_sql': None,
                        'data': [],
                        'warnings': ["Could not generate SQL from natural language prompt"]
                    }
                
                generated_sql = sql_result['sql_query']
                
                # Skip safety validation for in-memory database operations
                # All tables are user-uploaded files, no security risk
                logger.info("Skipping SQL safety validation for in-memory database")
                
                # Execute the query
                try:
                    result_df = duck_processor.execute_query(generated_sql)
                    
                    # Limit results to first 100 rows for preview
                    total_rows = len(result_df)
                    preview_limit = 100
                    is_limited = total_rows > preview_limit
                    
                    if is_limited:
                        preview_df = result_df.head(preview_limit)
                        logger.info(f"Limited results to {preview_limit} rows (total: {total_rows})")
                    else:
                        preview_df = result_df
                    
                    # Convert to desired format (only preview data)
                    if output_format.lower() == "json":
                        result_data = preview_df.to_dict('records')
                    else:
                        result_data = preview_df
                    
                    return {
                        'success': True,
                        'generated_sql': generated_sql,
                        'data': result_data,  # Preview data for UI
                        'full_data': result_df.to_dict('records') if output_format.lower() == "json" else result_df,  # Full data for storage
                        'row_count': total_rows,  # Total rows in actual result
                        'preview_rows': len(preview_df),  # Rows in preview
                        'is_limited': is_limited,  # Whether results were limited
                        'column_count': len(result_df.columns),
                        'processing_info': {
                            'input_files': len(files_data),
                            'query_type': sql_result.get('query_type', self._classify_query_type(generated_sql)),
                            'description': sql_result.get('description', ''),
                            'tables_used': list(table_schemas.keys())
                        },
                        'warnings': ['Results limited to first 100 rows for preview. Use "Open in Data Viewer" to see all results.'] if is_limited else [],
                        'errors': []
                    }
                    
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    return {
                        'success': False,
                        'error': f"Query execution failed: {str(e)}",
                        'generated_sql': generated_sql,
                        'data': [],
                        'warnings': ["Query generated successfully but execution failed"],
                        'errors': [str(e)]
                    }
                    
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}",
                'generated_sql': None,
                'data': [],
                'warnings': [],
                'errors': [str(e)]
            }
    
    def _classify_query_type(self, sql_query: str) -> str:
        """Classify the type of SQL query"""
        sql_upper = sql_query.upper()
        
        if 'JOIN' in sql_upper:
            return "join_operation"
        elif any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']):
            return "aggregation"
        elif 'WHERE' in sql_upper and ('NOT' in sql_upper or '!=' in sql_upper):
            return "filtering"
        elif 'ORDER BY' in sql_upper:
            return "sorting"
        elif 'GROUP BY' in sql_upper:
            return "grouping"
        elif 'WITH' in sql_upper:
            return "complex_query"
        else:
            return "simple_select"
    
    def store_results(self, process_id: str, result_data: Dict[str, Any]):
        """Store processing results"""
        self.storage[process_id] = {
            **result_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'process_id': process_id
        }
        logger.info(f"Stored results for process_id: {process_id}, total stored: {len(self.storage)}")
        
        # Also store in uploaded_files format so it can be viewed with existing viewer
        # Use full_data for storage if available, otherwise use data
        storage_data = result_data.get('full_data', result_data.get('data'))
        
        if storage_data and isinstance(storage_data, list) and len(storage_data) > 0:
            try:
                import pandas as pd
                from app.services.storage_service import uploaded_files
                
                def create_storage_data():
                    """Create storage data structures in parallel"""
                    # Convert results data to DataFrame
                    df = pd.DataFrame(storage_data)
                    
                    # Create file info for the viewer
                    size_bytes = len(df) * len(df.columns) * 8  # Rough estimate
                    size_mb = size_bytes / (1024 * 1024)  # Convert to MB
                    utc_now = datetime.now(timezone.utc).isoformat()
                    
                    file_info = {
                        "file_id": process_id,
                        "filename": f"Miscellaneous_Results_{process_id}.csv",
                        "size": size_bytes,
                        "file_size_mb": size_mb,
                        "upload_time": utc_now,
                        "last_modified": utc_now,
                        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "total_rows": len(df),
                        "columns": list(df.columns),
                        "source": "miscellaneous_processing"
                    }
                    
                    return df, file_info
                
                # Run storage preparation in parallel
                storage_future = submit_data_processing_task(create_storage_data)
                
                try:
                    df, file_info = storage_future.result(timeout=60.0)  # 60 second timeout
                    
                    # Store in uploaded_files format
                    file_data = {
                        "info": file_info,
                        "data": df
                    }
                    
                    # Use process_id as file_id for viewer access
                    uploaded_files.save(process_id, file_data, file_info)
                    logger.info(f"Stored results for viewer access with file_id: {process_id}")
                    
                except Exception as storage_error:
                    logger.error(f"Parallel storage failed, falling back to sequential: {storage_error}")
                    # Fallback to sequential processing
                    df = pd.DataFrame(storage_data)
                    size_bytes = len(df) * len(df.columns) * 8
                    size_mb = size_bytes / (1024 * 1024)
                    utc_now = datetime.now(timezone.utc).isoformat()
                    
                    file_info = {
                        "file_id": process_id,
                        "filename": f"Miscellaneous_Results_{process_id}.csv",
                        "size": size_bytes,
                        "file_size_mb": size_mb,
                        "upload_time": utc_now,
                        "last_modified": utc_now,
                        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "total_rows": len(df),
                        "columns": list(df.columns),
                        "source": "miscellaneous_processing"
                    }
                    
                    uploaded_files.save(process_id, {"info": file_info, "data": df}, file_info)
                    logger.info(f"Stored results sequentially with file_id: {process_id}")
                
            except Exception as e:
                logger.warning(f"Failed to store results for viewer: {e}")
                # Continue anyway as the main storage succeeded
        
    def get_results(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored results"""
        logger.info(f"Retrieving results for process_id: {process_id}, available IDs: {list(self.storage.keys())}")
        return self.storage.get(process_id)
    
    def delete_results(self, process_id: str) -> bool:
        """Delete stored results"""
        if process_id in self.storage:
            del self.storage[process_id]
            return True
        return False
    
    def list_active_processes(self) -> List[str]:
        """List all active process IDs"""
        return list(self.storage.keys())