import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator

from app.utils.uuid_generator import generate_uuid

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/miscellaneous", tags=["miscellaneous"])

# In-memory storage for saved prompts (in production, use database)
saved_prompts_storage = {}


class FileReference(BaseModel):
    """File reference for miscellaneous data processing"""
    file_id: str
    role: str  # file_0, file_1, file_2, file_3, file_4
    label: str


class MiscellaneousRequest(BaseModel):
    """Request model for miscellaneous data processing"""
    process_type: str = "data_analysis"
    process_name: str
    user_prompt: str  # Natural language request
    files: List[FileReference]
    output_format: Optional[str] = "json"  # json, csv, excel
    
    @validator('files')
    def validate_files_count(cls, v):
        if len(v) < 1 or len(v) > 5:
            raise ValueError("Must provide between 1 and 5 files")
        return v


class MiscellaneousResponse(BaseModel):
    """Response model for miscellaneous data processing"""
    success: bool
    message: str
    process_id: str
    generated_sql: Optional[str] = None
    row_count: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    errors: Optional[List[str]] = []
    warnings: Optional[List[str]] = []


def get_file_by_id(file_id: str):
    """
    Retrieve file by ID from storage service
    Reuses the same logic as reconciliation routes
    """
    from app.services.storage_service import uploaded_files

    if not uploaded_files.exists(file_id):
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")

    try:
        file_data = uploaded_files.get(file_id)
        file_info = file_data["info"]
        df = file_data["data"]
        filename = file_info["filename"]

        return {
            'dataframe': df,
            'filename': filename,
            'file_id': file_id,
            'info': file_info
        }

    except Exception as e:
        logger.error(f"Error retrieving file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")


@router.post("/process/", response_model=MiscellaneousResponse)
def process_miscellaneous_data(request: MiscellaneousRequest):
    """
    Process miscellaneous data operations using natural language prompts
    Supports up to 5 files with DuckDB backend
    """
    start_time = datetime.now()
    
    try:
        # Import services
        from app.services.miscellaneous_service import MiscellaneousProcessor
        
        # Initialize processor
        processor = MiscellaneousProcessor()
        
        # Retrieve all files
        retrieved_files = []
        for file_ref in request.files:
            file_data = get_file_by_id(file_ref.file_id)
            file_data['role'] = file_ref.role
            file_data['label'] = file_ref.label
            retrieved_files.append(file_data)
            
        logger.info(f"Processing {len(retrieved_files)} files with prompt: {request.user_prompt[:100]}...")
        
        # Process using the miscellaneous processor
        result = processor.process_natural_language_query(
            user_prompt=request.user_prompt,
            files_data=retrieved_files,
            output_format=request.output_format
        )
        
        # Generate process ID
        process_id = generate_uuid('misc')
        
        # Store results for later retrieval
        processor.store_results(process_id, result)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return MiscellaneousResponse(
            success=True,
            message=f"Successfully processed {len(retrieved_files)} files",
            process_id=process_id,
            generated_sql=result.get('generated_sql'),
            row_count=result.get('row_count'),
            processing_time_seconds=round(processing_time, 3),
            errors=result.get('errors', []),
            warnings=result.get('warnings', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Miscellaneous processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/results/{process_id}")
def get_miscellaneous_results(
    process_id: str,
    page: Optional[int] = 1,
    page_size: Optional[int] = 1000,
    format: Optional[str] = "json"
):
    """Get results from miscellaneous data processing with pagination"""
    
    try:
        from app.services.miscellaneous_service import MiscellaneousProcessor
        
        processor = MiscellaneousProcessor()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Process ID not found")
        
        # Handle pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        result_data = results.get('data', [])
        paginated_data = result_data[start_idx:end_idx]
        import math

        def sanitize_data(data):
            if isinstance(data, dict):
                return {k: sanitize_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [sanitize_data(v) for v in data]
            elif isinstance(data, float):
                if math.isnan(data) or math.isinf(data):
                    return None  # or str(data) if you want "NaN"/"Infinity"
                return data
            return data

        response = {
            'process_id': process_id,
            'data': sanitize_data(paginated_data),
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_rows': len(result_data),
                'total_pages': (len(result_data) + page_size - 1) // page_size,
                'has_next': end_idx < len(result_data),
                'has_prev': start_idx > 0
            },
            'metadata': {
                'generated_sql': results.get('generated_sql'),
                'processing_info': results.get('processing_info', {}),
                'timestamp': results.get('timestamp')
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving results for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")


@router.get("/download/{process_id}")
def download_miscellaneous_results(
    process_id: str,
    format: str = "csv"
):
    """Download results from miscellaneous data processing"""
    
    try:
        from app.services.miscellaneous_service import MiscellaneousProcessor
        
        processor = MiscellaneousProcessor()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Process ID not found")
        
        result_data = results.get('data', [])
        df = pd.DataFrame(result_data)
        
        if format.lower() == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            # Convert to bytes
            output = io.BytesIO(output.getvalue().encode('utf-8'))
            filename = f"miscellaneous_{process_id}.csv"
            media_type = "text/csv"
            
        elif format.lower() == "excel":
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Add metadata sheet
                metadata_df = pd.DataFrame([
                    ['Process ID', process_id],
                    ['Generated SQL', results.get('generated_sql', 'N/A')],
                    ['Row Count', len(result_data)],
                    ['Timestamp', results.get('timestamp', 'N/A')]
                ], columns=['Field', 'Value'])
                
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            output.seek(0)
            filename = f"miscellaneous_{process_id}.xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'excel'")
        
        return StreamingResponse(
            output,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading results for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")


@router.get("/results/{process_id}/summary")
def get_miscellaneous_summary(process_id: str):
    """Get summary of miscellaneous processing results"""
    
    try:
        from app.services.miscellaneous_service import MiscellaneousProcessor
        
        processor = MiscellaneousProcessor()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Process ID not found")
        
        result_data = results.get('data', [])
        
        summary = {
            'process_id': process_id,
            'timestamp': results.get('timestamp'),
            'row_count': len(result_data),
            'column_count': len(result_data[0].keys()) if result_data else 0,
            'generated_sql': results.get('generated_sql'),
            'processing_info': results.get('processing_info', {}),
            'file_count': len(results.get('input_files', [])),
            'success': True
        }
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary error: {str(e)}")


@router.delete("/results/{process_id}")
def delete_miscellaneous_results(process_id: str):
    """Delete miscellaneous processing results to free up storage"""
    
    try:
        from app.services.miscellaneous_service import MiscellaneousProcessor
        
        processor = MiscellaneousProcessor()
        success = processor.delete_results(process_id)
        
        if success:
            return {"success": True, "message": f"Results {process_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Process ID not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting results for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@router.post("/explain-sql/")
def explain_generated_sql(request: dict):
    """Explain the generated SQL query in natural language"""
    
    try:
        sql_query = request.get('sql_query', '')
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage
        
        llm_service = get_llm_service()
        if not llm_service.is_available():
            raise HTTPException(status_code=500, 
                              detail=f"LLM service not available")
        
        generation_params = get_llm_generation_params()
        
        prompt = f"""
Explain this SQL query in simple, non-technical language:

{sql_query}

Provide a clear explanation of:
1. What data sources are being used
2. What operations are being performed
3. What the results will show
4. Any important business logic

Keep the explanation accessible to non-technical users.
"""
        
        messages = [
            LLMMessage(role="system", content="You are a helpful data analyst who explains SQL queries in simple terms."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = llm_service.generate_text(
            messages=messages,
            **generation_params
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=f"Explanation generation failed: {response.error}")
        
        return {
            "success": True,
            "sql_query": sql_query,
            "explanation": response.content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


class ExecuteQueryRequest(BaseModel):
    """Request model for executing custom SQL queries"""
    sql_query: str
    process_id: str  # Reference to existing process data
    limit: Optional[int] = 100  # Limit results to prevent large responses


class SavePromptRequest(BaseModel):
    """Request model for saving an ideal prompt"""
    original_prompt: str
    generated_sql: str
    files_info: List[Dict[str, Any]]
    results_summary: Dict[str, Any]
    process_id: str
    ai_description: Optional[str] = None  # AI-generated description of what the query does


class SavedPrompt(BaseModel):
    """Model for saved prompt"""
    id: str
    name: str
    ideal_prompt: str
    original_prompt: str
    file_pattern: str  # Description of file types/structure this works with
    created_at: str
    category: str
    description: str


@router.post("/execute-query")
def execute_custom_query(request: ExecuteQueryRequest):
    """Execute a custom SQL query on existing processed data"""
    
    try:
        from app.services.miscellaneous_service import MiscellaneousProcessor
        
        processor = MiscellaneousProcessor()
        
        # Get the stored results to access the same data and schemas
        stored_result = processor.get_results(request.process_id)
        if not stored_result:
            raise HTTPException(status_code=404, detail=f"Process ID {request.process_id} not found")
        
        # Debug: Log what we got from storage
        logger.info(f"DEBUG: stored_result keys: {list(stored_result.keys())}")
        logger.info(f"DEBUG: stored_result structure: {type(stored_result)}")
        
        # Extract the file data and table schemas from stored results
        files_data_raw = stored_result.get('files_data', [])
        table_schemas = stored_result.get('table_schemas', {})
        processing_info = stored_result.get('processing_info', {})
        
        logger.info(f"files_data_raw type: {type(files_data_raw)}")
        logger.info(f"table_schemas type: {type(table_schemas)}, content: {list(table_schemas.keys()) if isinstance(table_schemas, dict) else 'not a dict'}")
        
        # If files_data is missing or empty, try to reconstruct from processing info
        if not files_data_raw and processing_info.get('input_files'):
            logger.warning("files_data missing from stored results, attempting to reconstruct from original files")
            
            # Get the original request data that was used for processing 
            # This is a fallback approach - we'll retrieve the original files
            input_files_count = processing_info.get('input_files', 0)
            
            # For now, return an error asking user to re-process
            # In future, we could store file IDs and reconstruct the data
            return {
                'success': False,
                'error': 'File data no longer available for this process. Please re-run the original query to enable execute functionality.',
                'suggestions': [
                    'Click "Process Data with AI" again to regenerate the query',
                    'The execute feature works on freshly processed data',
                    'File data is not persisted long-term for execute queries'
                ]
            }
        
        if not files_data_raw:
            return {
                'success': False,
                'error': 'No file data found for this process',
                'suggestions': [
                    'Please re-run the original data processing',
                    'Execute query only works on recently processed data'
                ]
            }
        
        # Convert list of files_data to dictionary mapping table names to dataframes
        files_data = {}
        for i, file_data in enumerate(files_data_raw):
            table_name = f"file_{i + 1}"  # Changed to start from file_1
            files_data[table_name] = file_data
            logger.info(f"Prepared {table_name} from file: {file_data.get('filename', 'unknown')}")
        
        # Skip column validation for execute queries - let DuckDB handle SQL validation
        # The validation logic is too strict and prevents valid queries with aliases/static values
        logger.info("Skipping column validation for execute query - letting DuckDB handle SQL validation")
        
        # Execute the custom SQL query
        logger.info(f"Executing custom SQL query for process {request.process_id}")
        logger.info(f"Query: {request.sql_query[:200]}...")
        
        # Create a temporary DuckDB connection and load the same data
        import duckdb
        conn = duckdb.connect(':memory:')
        
        try:
            # Simplified helper function - avoid complex pandas operations
            def sanitize_dataframe_for_duckdb(df):
                """Simple cleaning to avoid DuckDB registration issues"""
                import numpy as np
                
                try:
                    logger.info(f"Sanitizing DataFrame with shape {df.shape}, columns: {list(df.columns)}")
                    
                    # Skip sanitization for now - just return the original DataFrame
                    # The issue might be in our sanitization logic itself
                    logger.info("Skipping DataFrame sanitization - using original DataFrame")
                    return df
                    
                except Exception as e:
                    logger.error(f"Error in sanitization function: {e}")
                    return df

            # Handle both dict and list formats for files_data
            if isinstance(files_data, dict):
                # Dictionary format: {table_name: file_data}
                for table_name, file_data in files_data.items():
                    df = file_data.get('dataframe') if isinstance(file_data, dict) else file_data
                    if df is not None:
                        try:
                            clean_df = sanitize_dataframe_for_duckdb(df)
                            conn.register(table_name, clean_df)
                            logger.info(f"Registered table {table_name} with {len(clean_df)} rows")
                        except Exception as e:
                            logger.error(f"Failed to register table {table_name}: {e}")
                            # Try registering without sanitization as last resort
                            try:
                                conn.register(table_name, df)
                                logger.warning(f"Registered table {table_name} without sanitization")
                            except Exception as e2:
                                logger.error(f"Failed to register {table_name} even without sanitization: {e2}")
                                raise HTTPException(status_code=500, 
                                                  detail=f"Could not register table {table_name}: {e2}")
            elif isinstance(files_data, list):
                # List format: [file_data1, file_data2, ...]
                for i, file_data in enumerate(files_data):
                    table_name = f"file_{i + 1}"  # Changed to start from file_1  
                    df = file_data.get('dataframe') if isinstance(file_data, dict) else file_data
                    if df is not None:
                        try:
                            clean_df = sanitize_dataframe_for_duckdb(df)
                            conn.register(table_name, clean_df)
                            logger.info(f"Registered table {table_name} with {len(clean_df)} rows")
                        except Exception as e:
                            logger.error(f"Failed to register table {table_name}: {e}")
                            # Try registering without sanitization as last resort
                            try:
                                conn.register(table_name, df)
                                logger.warning(f"Registered table {table_name} without sanitization")
                            except Exception as e2:
                                logger.error(f"Failed to register {table_name} even without sanitization: {e2}")
                                raise HTTPException(status_code=500, 
                                                  detail=f"Could not register table {table_name}: {e2}")
            else:
                raise HTTPException(status_code=500, detail=f"Unexpected files_data format: {type(files_data)}")
            
            # Get total count before applying limit
            total_count = None
            if request.limit:
                try:
                    # Execute count query to get total records
                    count_query = f"SELECT COUNT(*) as total FROM ({request.sql_query.strip()})"
                    count_result = conn.execute(count_query).fetchone()
                    total_count = count_result[0] if count_result else 0
                except Exception as e:
                    logger.warning(f"Failed to get total count: {e}")
                    total_count = None
            
            # Apply limit to the query if specified
            limited_query = request.sql_query.strip()
            if request.limit and not limited_query.upper().endswith(';'):
                # Check if query already has LIMIT
                if 'LIMIT' not in limited_query.upper():
                    limited_query = f"SELECT * FROM ({limited_query}) LIMIT {request.limit}"
            
            # Execute the query
            result_df = conn.execute(limited_query).df()
            
            # Sanitize the DataFrame to handle JSON non-compliant values
            if len(result_df) > 0:
                import numpy as np
                import pandas as pd
                
                # More aggressive cleaning for JSON compliance
                try:
                    # Replace all problematic float values
                    result_df = result_df.replace([np.inf, -np.inf, np.nan], None)
                    
                    # Double-check: use where() to catch any remaining NaN/null values
                    result_df = result_df.where(pd.notnull(result_df), None)
                    
                    # Handle very large numbers that might cause JSON issues
                    for col in result_df.select_dtypes(include=[np.number]).columns:
                        try:
                            # Replace any remaining NaN values in numeric columns
                            mask = pd.isna(result_df[col])
                            if mask.any():
                                result_df.loc[mask, col] = None
                                logger.info(f"Replaced {mask.sum()} NaN values in column '{col}' with None")
                            
                            # Check for very large numbers
                            finite_mask = pd.notna(result_df[col]) & np.isfinite(result_df[col].astype(float, errors='ignore'))
                            large_mask = finite_mask & (np.abs(result_df[col]) > 1e15)
                            if large_mask.any():
                                result_df.loc[large_mask, col] = None
                                logger.warning(f"Replaced {large_mask.sum()} very large numbers in column '{col}' with None")
                        except Exception as e:
                            logger.warning(f"Error processing numeric column '{col}': {e}")
                            # If there's still an issue, convert the entire column to string then None
                            result_df[col] = result_df[col].astype(str).replace(['nan', 'NaN', 'None', '<NA>'], None)
                
                except Exception as e:
                    logger.error(f"Error in DataFrame sanitization: {e}")
            
            # Convert result to records with additional safety check
            if len(result_df) > 0:
                try:
                    result_records = result_df.to_dict('records')
                    # Final safety check - manually clean any remaining problematic values
                    import math
                    for record in result_records:
                        for key, value in record.items():
                            if isinstance(value, float):
                                if math.isnan(value) or math.isinf(value):
                                    record[key] = None
                except Exception as e:
                    logger.error(f"Error converting DataFrame to records: {e}")
                    result_records = []
            else:
                result_records = []
            
            # Calculate basic statistics
            row_count = len(result_records)
            column_count = len(result_df.columns) if len(result_df) > 0 else 0
            
            logger.info(f"Custom query executed successfully: {row_count} rows, {column_count} columns")
            
            return {
                'success': True,
                'data': result_records,
                'row_count': row_count,
                'total_count': total_count,  # Total records before limiting
                'column_count': column_count,
                'columns': list(result_df.columns) if len(result_df) > 0 else [],
                'limited': request.limit is not None and row_count >= request.limit,
                'query_executed': limited_query if 'limited_query' in locals() else request.sql_query,
                'process_id': request.process_id,
                'tables_available': list(files_data.keys())
            }
            
        finally:
            conn.close()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing custom SQL query: {str(e)}")
        
        # Provide more specific error messages based on common issues
        error_message = str(e)
        suggestions = []
        
        if "file 0" in error_message.lower():
            suggestions.append("Table name should be 'file_0' (with underscore), not 'file 0' (with space)")
        if "does not exist" in error_message.lower() and "table" in error_message.lower():
            suggestions.append("Use table names: file_1, file_2, file_3, etc. (with underscores)")
        if "column" in error_message.lower() and "does not exist" in error_message.lower():
            suggestions.append("Check actual column names - remember column names are case-sensitive")
            suggestions.append("Use double quotes around column names if they contain spaces: \"Column Name\"")
        
        # Default suggestions
        if not suggestions:
            suggestions = [
                "Check that your SQL syntax is correct",
                "Verify that all table references use underscores (file_1, file_2, etc.)",
                "Column names with spaces need double quotes: \"Column Name\"",
                "Try a simpler query first like: SELECT * FROM file_1 LIMIT 5"
            ]
        
        return {
            'success': False,
            'error': f"Query execution failed: {error_message}",
            'suggestions': suggestions
        }


@router.post("/generate-ideal-prompt")
def generate_ideal_prompt(request: SavePromptRequest):
    """Generate an ideal prompt based on successful query results"""
    try:
        from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage
        
        llm_service = get_llm_service()
        if not llm_service.is_available():
            raise HTTPException(status_code=500, detail="LLM service not available")
        
        generation_params = get_llm_generation_params()
        
        # Build context about the files
        files_context = []
        for i, file_info in enumerate(request.files_info, 1):
            files_context.append(f"file_{i}: {file_info.get('filename', 'unknown')} ({file_info.get('rows', 0)} rows)")
        
        files_description = "\n".join(files_context)
        
        # Build enhanced context with AI description
        ai_description_context = ""
        if request.ai_description:
            ai_description_context = f"""
AI'S UNDERSTANDING: "{request.ai_description}"
"""

        prompt = f"""
You are an expert at reverse-engineering successful data operations to create optimal, reusable prompts.

REVERSE ENGINEERING TASK:
Analyze this successful data processing scenario and create an IDEAL PROMPT that would consistently generate the same SQL query and achieve identical results.

ORIGINAL USER PROMPT: "{request.original_prompt}"
{ai_description_context}
GENERATED SQL QUERY:
{request.generated_sql}

FILES USED:
{files_description}

RESULTS ACHIEVED:
- Row count: {request.results_summary.get('row_count', 'N/A')}
- Columns: {request.results_summary.get('column_count', 'N/A')}
- Processing type: {request.results_summary.get('query_type', 'analysis')}

REVERSE ENGINEERING GUIDELINES:
1. **Analyze the SQL** to understand exactly what operations were performed
2. **Extract key patterns** from the generated query (JOINs, WHERE clauses, aggregations, etc.)
3. **Identify column references** and data relationships used
4. **Understand the business logic** from both the original prompt and AI description
5. **Create a more precise prompt** that would generate similar SQL reliably

The IDEAL PROMPT should:
✓ Be more specific than the original user prompt
✓ Explicitly mention key columns, conditions, and operations from the SQL
✓ Include precise business requirements that led to the SQL structure
✓ Be detailed enough to generate consistent results on similar data
✓ Use clear, unambiguous language about the desired outcome

EXAMPLE IMPROVEMENT:
- Original: "Compare file_1 and file_2"
- Ideal: "Compare file_1 and file_2 by matching on customer_id column, show records from file_1 that don't exist in file_2, and include the customer_name, transaction_amount, and transaction_date columns in the results"

Respond in JSON format:
{{
    "ideal_prompt": "the optimized, specific prompt that would generate similar SQL",
    "name": "Short descriptive name (max 50 chars)",
    "description": "Brief description of what this prompt accomplishes",
    "category": "reconciliation|transformation|analysis|delta|reporting|aggregation|filtering",
    "file_pattern": "Description of required file types and structure",
    "improvements_made": "Brief explanation of how this prompt improves on the original"
}}
"""
        
        messages = [
            LLMMessage(role="system", content="You are an expert at analyzing data processing patterns and creating optimized, reusable prompts."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = llm_service.generate_text(
            messages=messages,
            **generation_params
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=f"Failed to generate ideal prompt: {response.error}")
        
        # Parse the JSON response
        try:
            import json
            ideal_prompt_data = json.loads(response.content)
            
            return {
                "success": True,
                "ideal_prompt": ideal_prompt_data.get("ideal_prompt", ""),
                "name": ideal_prompt_data.get("name", "Generated Prompt"),
                "description": ideal_prompt_data.get("description", ""),
                "category": ideal_prompt_data.get("category", "analysis"),
                "file_pattern": ideal_prompt_data.get("file_pattern", "General data files"),
                "improvements_made": ideal_prompt_data.get("improvements_made", ""),
                "original_prompt": request.original_prompt
            }
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "success": True,
                "ideal_prompt": response.content.strip(),
                "name": "Generated Prompt",
                "description": "AI-generated optimized prompt",
                "category": "analysis",
                "file_pattern": "General data files",
                "original_prompt": request.original_prompt
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating ideal prompt: {str(e)}")
        
        # Check for specific OpenAI errors
        error_message = str(e).lower()
        if "insufficient_quota" in error_message or "quota" in error_message:
            return {
                "success": False,
                "error": "AI service quota exceeded. You can still save your original prompt manually.",
                "error_type": "quota_exceeded",
                "fallback_available": True,
                "suggestion": "Consider upgrading your OpenAI plan or use manual prompt saving."
            }
        elif "429" in error_message or "too many requests" in error_message or "rate" in error_message:
            return {
                "success": False, 
                "error": "AI service is currently busy. Please try again in a moment or save your prompt manually.",
                "error_type": "rate_limited",
                "fallback_available": True,
                "suggestion": "Wait a few minutes before trying AI optimization again."
            }
        else:
            return {
                "success": False,
                "error": f"AI optimization temporarily unavailable. You can still save your original prompt.",
                "error_type": "general_error",
                "fallback_available": True,
                "suggestion": "Use manual prompt saving or try AI optimization later."
            }


@router.post("/save-prompt")
def save_prompt(prompt_data: dict):
    """Save an ideal prompt for reuse"""
    try:
        prompt_id = generate_uuid('prompt')
        
        saved_prompt = {
            "id": prompt_id,
            "name": prompt_data.get("name", "Untitled Prompt"),
            "ideal_prompt": prompt_data.get("ideal_prompt", ""),
            "original_prompt": prompt_data.get("original_prompt", ""),
            "description": prompt_data.get("description", ""),
            "category": prompt_data.get("category", "analysis"),
            "file_pattern": prompt_data.get("file_pattern", "General data files"),
            "created_at": datetime.now().isoformat(),
        }
        
        saved_prompts_storage[prompt_id] = saved_prompt
        logger.info(f"Saved prompt {prompt_id}: {saved_prompt['name']}")
        
        return {
            "success": True,
            "message": "Prompt saved successfully",
            "prompt_id": prompt_id,
            "prompt": saved_prompt
        }
        
    except Exception as e:
        logger.error(f"Error saving prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save prompt: {str(e)}")


@router.get("/saved-prompts")
def get_saved_prompts():
    """Get all saved prompts"""
    try:
        prompts = list(saved_prompts_storage.values())
        # Sort by creation date, newest first
        prompts.sort(key=lambda x: x['created_at'], reverse=True)
        
        return {
            "success": True,
            "prompts": prompts,
            "count": len(prompts)
        }
        
    except Exception as e:
        logger.error(f"Error fetching saved prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch prompts: {str(e)}")


@router.delete("/saved-prompts/{prompt_id}")
def delete_saved_prompt(prompt_id: str):
    """Delete a saved prompt"""
    try:
        if prompt_id in saved_prompts_storage:
            deleted_prompt = saved_prompts_storage.pop(prompt_id)
            logger.info(f"Deleted prompt {prompt_id}: {deleted_prompt['name']}")
            return {
                "success": True,
                "message": "Prompt deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Prompt not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt {prompt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {str(e)}")


@router.put("/saved-prompts/{prompt_id}")
def update_saved_prompt(prompt_id: str, prompt_data: dict):
    """Update an existing saved prompt"""
    try:
        if prompt_id not in saved_prompts_storage:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        # Get existing prompt
        existing_prompt = saved_prompts_storage[prompt_id]
        
        # Update the prompt with new data
        updated_prompt = {
            "id": prompt_id,
            "name": prompt_data.get("name", existing_prompt["name"]),
            "ideal_prompt": prompt_data.get("ideal_prompt", existing_prompt["ideal_prompt"]),
            "original_prompt": prompt_data.get("original_prompt", existing_prompt["original_prompt"]),
            "description": prompt_data.get("description", existing_prompt["description"]),
            "category": prompt_data.get("category", existing_prompt["category"]),
            "file_pattern": prompt_data.get("file_pattern", existing_prompt["file_pattern"]),
            "created_at": existing_prompt["created_at"],  # Keep original creation date
            "updated_at": datetime.now().isoformat(),  # Add update timestamp
        }
        
        # Save updated prompt
        saved_prompts_storage[prompt_id] = updated_prompt
        
        logger.info(f"Updated prompt {prompt_id}: {updated_prompt['name']}")
        return {
            "success": True,
            "message": "Prompt updated successfully",
            "prompt": updated_prompt
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update prompt: {str(e)}")


class PromptSuggestionsRequest(BaseModel):
    """Request model for prompt suggestions"""
    text_before_cursor: str
    text_after_cursor: str = ""
    files_context: List[Dict[str, Any]]
    max_suggestions: Optional[int] = 4


class IntentVerificationRequest(BaseModel):
    """Request model for intent verification"""
    user_prompt: str
    files: List[FileReference]


# Simple in-memory cache for common suggestion patterns
suggestions_cache = {}
CACHE_EXPIRY_SECONDS = 600  # 10 minutes - more aggressive caching


@router.post("/prompt-suggestions/")
def get_prompt_suggestions(request: PromptSuggestionsRequest):
    """
    Get AI-powered autocomplete suggestions for natural language data queries
    Optimized for fast response times with caching
    """
    try:
        from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage
        
        # Create cache key from request context
        cache_key = hash(f"{request.text_before_cursor.lower()[:50]}_{len(request.files_context)}")
        current_time = datetime.now().timestamp()
        
        # Check cache first
        if cache_key in suggestions_cache:
            cached_data = suggestions_cache[cache_key]
            if current_time - cached_data['timestamp'] < CACHE_EXPIRY_SECONDS:
                logger.info(f"Returning cached suggestions for key: {cache_key}")
                return {
                    "success": True,
                    "suggestions": cached_data['suggestions'],
                    "cached": True,
                    "processing_time_ms": 5  # Cached response time
                }
        
        start_time = datetime.now()
        
        llm_service = get_llm_service()
        if not llm_service.is_available():
            return {
                "success": False,
                "error": "AI service unavailable",
                "suggestions": []
            }
        
        # Build optimized file context
        files_summary = []
        for i, file_info in enumerate(request.files_context, 1):
            files_summary.append(
                f"file_{i}: {file_info.get('name', 'unknown')} "
                f"({file_info.get('totalRows', 0)} rows) - "
                f"Columns: {', '.join(file_info.get('columns', [])[:8])}"
            )
        
        files_context_str = "\n".join(files_summary)
        
        # Ultra-optimized system prompt for speed
        system_prompt = """Fast autocomplete for data queries. Return 3-4 suggestions max.

JSON format only:
{
  "suggestions": [
    {
      "text": "short text",
      "description": "brief desc", 
      "completion": "completion text",
      "type": "operation|column|file_reference|phrase",
      "confidence": 0.8
    }
  ]
}"""

        # Ultra-short user prompt for speed
        user_prompt = f"""Text: "{request.text_before_cursor}"

Files: {files_context_str}

Generate 3 completions. Focus: operations, file refs, columns. JSON only."""

        generation_params = get_llm_generation_params()
        # Override for faster autocomplete
        generation_params.update({
            'temperature': 0.1,   # Slightly more variety but still fast
            'max_tokens': 250,    # Even shorter for speed
        })
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        response = llm_service.generate_text(
            messages=messages,
            **generation_params
        )
        
        if not response.success:
            return {
                "success": False,
                "error": f"AI generation failed: {response.error}",
                "suggestions": []
            }
        
        # Parse AI response
        try:
            import json
            suggestions_data = json.loads(response.content)
            suggestions = suggestions_data.get('suggestions', [])
            
            # Validate and clean suggestions
            valid_suggestions = []
            for suggestion in suggestions[:request.max_suggestions]:
                if suggestion.get('text') and suggestion.get('completion'):
                    valid_suggestions.append({
                        'text': suggestion.get('text', ''),
                        'description': suggestion.get('description', 'AI suggestion'),
                        'completion': suggestion.get('completion', suggestion.get('text', '')),
                        'type': suggestion.get('type', 'phrase'),
                        'confidence': min(max(suggestion.get('confidence', 0.5), 0.0), 1.0)
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Cache successful results
            if valid_suggestions and len(request.text_before_cursor) > 5:
                suggestions_cache[cache_key] = {
                    'suggestions': valid_suggestions,
                    'timestamp': current_time
                }
                # Clean old cache entries
                if len(suggestions_cache) > 100:
                    oldest_key = min(suggestions_cache.keys(), 
                                   key=lambda k: suggestions_cache[k]['timestamp'])
                    del suggestions_cache[oldest_key]
            
            logger.info(f"Generated {len(valid_suggestions)} suggestions in {processing_time:.0f}ms")
            
            return {
                "success": True,
                "suggestions": valid_suggestions,
                "cached": False,
                "processing_time_ms": round(processing_time, 1),
                "files_analyzed": len(request.files_context)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI suggestions JSON: {e}")
            return {
                "success": False,
                "error": "Invalid AI response format",
                "suggestions": []
            }
        
    except Exception as e:
        logger.error(f"Error generating prompt suggestions: {str(e)}")
        return {
            "success": False,
            "error": f"Suggestion generation failed: {str(e)}",
            "suggestions": []
        }


@router.post("/verify-intent")
def verify_query_intent(request: IntentVerificationRequest):
    """Verify and analyze query intent before execution"""
    try:
        from app.services.intent_service import QueryIntentExtractor
        
        # Get selected file data from file IDs
        file_data_list = []
        for file_ref in request.files:
            try:
                file_data = get_file_by_id(file_ref.file_id)
                file_data_list.append(file_data)
                
            except Exception as e:
                logger.error(f"Error preparing file {file_ref.file_id} for intent analysis: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Could not analyze file {file_ref.file_id}: {str(e)}"
                )
        
        if not file_data_list:
            raise HTTPException(status_code=400, detail="No valid files provided for analysis")
        
        # Extract intent using the new service with file data objects
        intent_extractor = QueryIntentExtractor()
        intent_summary = intent_extractor.extract_intent(request.user_prompt, file_data_list)
        
        logger.info(f"Intent verification completed for prompt: {request.user_prompt[:50]}...")
        
        return {
            "success": True,
            "intent_summary": intent_summary,
            "original_prompt": request.user_prompt,
            "files_count": len(request.files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying query intent: {str(e)}")
        
        # Clean up any temporary files in case of error
        if 'selected_files' in locals():
            for temp_file in selected_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        raise HTTPException(status_code=500, detail=f"Intent verification failed: {str(e)}")


@router.get("/health")
def health_check():
    """Health check endpoint for miscellaneous service"""
    return {
        "status": "healthy",
        "service": "miscellaneous_data_processor",
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "max_files": 5,
            "supported_formats": ["csv", "excel", "json"],
            "features": ["natural_language_queries", "sql_generation", "data_analysis", "prompt_management", "ai_suggestions", "intent_verification"]
        }
    }