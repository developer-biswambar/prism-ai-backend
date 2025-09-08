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
            table_name = f"file_{i}"
            files_data[table_name] = file_data
            logger.info(f"Prepared {table_name} from file: {file_data.get('filename', 'unknown')}")
        
        # Validate the SQL query contains only existing tables and columns
        validation_result = processor._validate_column_references(request.sql_query, table_schemas)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error'],
                'suggestions': validation_result['suggestions']
            }
        
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
                    return df)

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
                    table_name = f"file_{i}"
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
                
                # Replace problematic float values
                result_df = result_df.replace([np.inf, -np.inf], None)  # Replace infinity with None
                
                # Use where() instead of fillna() to avoid the parameter error
                result_df = result_df.where(pd.notnull(result_df), None)
                
                # Handle very large numbers that might cause JSON issues
                for col in result_df.select_dtypes(include=[np.number]).columns:
                    # Check for very large numbers that might cause JSON serialization issues
                    try:
                        mask = np.abs(result_df[col]) > 1e15
                        if mask.any():
                            result_df.loc[mask, col] = None
                            logger.warning(f"Replaced {mask.sum()} very large numbers in column '{col}' with None")
                    except Exception as e:
                        logger.warning(f"Error processing large numbers in column '{col}': {e}")
            
            # Convert result to records
            result_records = result_df.to_dict('records') if len(result_df) > 0 else []
            
            # Calculate basic statistics
            row_count = len(result_records)
            column_count = len(result_df.columns) if len(result_df) > 0 else 0
            
            logger.info(f"Custom query executed successfully: {row_count} rows, {column_count} columns")
            
            return {
                'success': True,
                'data': result_records,
                'row_count': row_count,
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
        return {
            'success': False,
            'error': f"Query execution failed: {str(e)}",
            'suggestions': [
                "Check that your SQL syntax is correct",
                "Ensure you're using the exact column names from the schema",
                "Verify that all table references (file_0, file_1, etc.) are correct",
                "Try a simpler query first to test the connection"
            ]
        }


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
            "features": ["natural_language_queries", "sql_generation", "data_analysis"]
        }
    }