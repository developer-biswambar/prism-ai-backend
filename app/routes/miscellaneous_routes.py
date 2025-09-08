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
router = APIRouter(prefix="/miscellaneous", tags=["miscellaneous"])


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
        
        response = {
            'process_id': process_id,
            'data': paginated_data,
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