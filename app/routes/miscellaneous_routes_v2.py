"""
Miscellaneous Data Processing Routes V2 - LangChain SQL Agent Implementation

This version provides the same API contract as v1 but uses LangChain SQL Agents
for improved query generation, self-correction, and agentic capabilities.

Key improvements over v1:
- LangChain SQL Agent with self-correction
- Automatic schema introspection
- Multi-step reasoning for complex queries
- Better error handling and recovery
- Maintains full backward compatibility
"""

import io
import logging
import math
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator

from app.utils.uuid_generator import generate_uuid

# Import existing models for backward compatibility
from app.routes.miscellaneous_routes import (
    FileReference, 
    MiscellaneousRequest, 
    ProcessingInfo, 
    MiscellaneousResponse,
    get_file_by_id
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router with v2 prefix
router = APIRouter(prefix="/api/miscellaneous/v2", tags=["miscellaneous-v2"])


@router.post("/process/", response_model=MiscellaneousResponse)
def process_miscellaneous_data_v2(request: MiscellaneousRequest):
    """
    Process miscellaneous data operations using LangChain SQL Agent
    
    V2 Features:
    - LangChain SQL Agent with self-correction
    - Automatic schema introspection  
    - Multi-step reasoning for complex queries
    - Better error handling and recovery
    - Full backward compatibility with v1 API contract
    """
    start_time = datetime.now(timezone.utc)
    
    try:
        # Import services
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        # Initialize v2 processor with LangChain SQL Agent
        processor = MiscellaneousProcessorV2()
        
        # Retrieve all files (same as v1)
        retrieved_files = []
        for file_ref in request.files:
            file_data = get_file_by_id(file_ref.file_id)
            file_data['role'] = file_ref.role
            file_data['label'] = file_ref.label
            retrieved_files.append(file_data)
            
        logger.info(f"V2 Processing {len(retrieved_files)} files with prompt: {request.user_prompt[:100]}...")
        
        # Process using the LangChain SQL Agent
        result = processor.process_core_request(
            user_prompt=request.user_prompt,
            files_data=retrieved_files,
            output_format=request.output_format
        )
        
        # Generate process ID
        process_id = generate_uuid('data_analysis_v2')
        
        # Handle processing failures (same logic as v1)
        if not result.get('success', True):
            logger.info(f"V2 Processing failed, storing raw file data for manual SQL exploration - Process ID: {process_id}")
            result['files_data'] = retrieved_files
            result['processing_info'] = {
                'process_type': request.process_type,
                'process_name': request.process_name, 
                'user_prompt': request.user_prompt,
                'input_files': len(retrieved_files),
                'processing_failed': True,
                'failure_reason': result.get('message', 'Unknown error'),
                'version': 'v2_langchain_agent'
            }
        
        # Store results for later retrieval
        processor.store_results(process_id, result)
        
        # Calculate processing time
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Return same response format as v1 for backward compatibility
        return MiscellaneousResponse(
            success=result.get('success', True),
            message=f"V2: Successfully processed {len(retrieved_files)} files with LangChain SQL Agent" if result.get('success', True) else "V2: Processing failed",
            process_id=process_id,
            generated_sql=result.get('generated_sql'),
            row_count=result.get('row_count'),
            processing_time_seconds=round(processing_time, 3),
            errors=result.get('errors', []),
            warnings=result.get('warnings', []),
            error_analysis=result.get('error_analysis')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Miscellaneous processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Processing error: {str(e)}")


@router.get("/results/{process_id}")
def get_miscellaneous_results_v2(
    process_id: str,
    page: Optional[int] = 1,
    page_size: Optional[int] = 1000,
    format: Optional[str] = "json"
):
    """Get results from v2 miscellaneous data processing with pagination"""
    
    try:
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="V2 Process ID not found")
        
        # Handle pagination (same logic as v1)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        result_data = results.get('data', [])
        paginated_data = result_data[start_idx:end_idx]
        
        def sanitize_data(data):
            if isinstance(data, dict):
                return {k: sanitize_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [sanitize_data(v) for v in data]
            elif isinstance(data, float):
                if math.isnan(data) or math.isinf(data):
                    return None
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
                'timestamp': results.get('timestamp'),
                'version': 'v2_langchain_agent',
                'agent_steps': results.get('agent_steps', []),  # V2 specific: show agent reasoning steps
                'self_corrections': results.get('self_corrections', [])  # V2 specific: show any self-corrections made
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving V2 results for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve V2 results: {str(e)}")


@router.get("/download/{process_id}")
def download_miscellaneous_results_v2(
    process_id: str,
    format: str = "csv"
):
    """Download results from v2 miscellaneous data processing"""
    
    try:
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="V2 Process ID not found")
        
        # Get the data
        result_data = results.get('data', [])
        
        if not result_data:
            raise HTTPException(status_code=404, detail="No data found for V2 process")
        
        # Convert to DataFrame
        df = pd.DataFrame(result_data)
        
        # Generate filename with v2 indicator
        filename = f"v2_miscellaneous_results_{process_id}.{format.lower()}"
        
        # Create file stream
        if format.lower() == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            response = StreamingResponse(
                io.StringIO(output.getvalue()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        elif format.lower() == "excel":
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            
            response = StreamingResponse(
                io.BytesIO(output.getvalue()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'excel'")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading V2 results for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download V2 results: {str(e)}")


@router.get("/agent-info/{process_id}")
def get_agent_execution_info(process_id: str):
    """
    V2 Specific Endpoint: Get detailed information about LangChain SQL Agent execution
    
    This endpoint provides insights into:
    - Agent reasoning steps
    - Self-corrections made
    - Schema introspection results
    - Tool usage statistics
    """
    try:
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="V2 Process ID not found")
        
        agent_info = {
            'process_id': process_id,
            'agent_version': 'langchain_sql_agent_v2',
            'execution_steps': results.get('agent_steps', []),
            'self_corrections': results.get('self_corrections', []),
            'schema_introspection': results.get('schema_introspection', {}),
            'tool_usage': results.get('tool_usage', {}),
            'reasoning_chain': results.get('reasoning_chain', []),
            'final_sql': results.get('generated_sql'),
            'confidence_score': results.get('confidence_score'),
            'alternative_queries': results.get('alternative_queries', [])
        }
        
        return {
            'success': True,
            'data': agent_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving V2 agent info for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve V2 agent info: {str(e)}")


@router.get("/results/{process_id}/summary")
def get_miscellaneous_results_summary_v2(process_id: str):
    """Get summary of v2 miscellaneous data processing results"""
    
    try:
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="V2 Process ID not found")
        
        # Enhanced summary with V2 agent information
        summary = {
            'process_id': process_id,
            'success': results.get('success', False),
            'total_rows': results.get('row_count', 0),
            'generated_sql': results.get('generated_sql'),
            'processing_info': results.get('processing_info', {}),
            'timestamp': results.get('timestamp'),
            'version': 'v2_langchain_agent',
            # V2 specific summary info
            'agent_summary': {
                'execution_steps': len(results.get('agent_steps', [])),
                'self_corrections': len(results.get('self_corrections', [])),
                'confidence_score': results.get('confidence_score'),
                'tools_used': list(results.get('tool_usage', {}).keys()),
                'reasoning_depth': len(results.get('reasoning_chain', []))
            },
            'data_preview': results.get('data', [])[:5] if results.get('data') else []
        }
        
        return {
            'success': True,
            'data': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting V2 summary for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get V2 summary: {str(e)}")


@router.delete("/results/{process_id}")
def delete_miscellaneous_results_v2(process_id: str):
    """Delete v2 miscellaneous data processing results"""
    
    try:
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        
        if process_id not in processor.results_storage:
            raise HTTPException(status_code=404, detail="V2 Process ID not found")
        
        # Delete the results
        del processor.results_storage[process_id]
        
        return {
            'success': True,
            'message': f'V2 Results for process {process_id} deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting V2 results for {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete V2 results: {str(e)}")


@router.post("/explain-sql/")
def explain_sql_v2(request: dict):
    """
    V2: Explain SQL query with enhanced LangChain agent analysis
    
    Enhanced features:
    - LangChain agent provides detailed explanations
    - Query optimization suggestions
    - Performance analysis
    """
    
    try:
        sql_query = request.get('sql_query', '').strip()
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        
        # Enhanced SQL explanation using LangChain if available
        if processor.sql_agent_processor.agent_available:
            explanation = processor._explain_sql_with_agent(sql_query)
        else:
            # Fallback to V1 method
            from app.services.miscellaneous_service import MiscellaneousProcessor
            v1_processor = MiscellaneousProcessor()
            explanation = v1_processor._explain_sql_query(sql_query)
        
        return {
            'success': True,
            'explanation': explanation,
            'version': 'v2_enhanced',
            'sql_query': sql_query
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 SQL explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 SQL explanation failed: {str(e)}")


@router.post("/execute-query")
def execute_direct_query_v2(request: dict):
    """
    V2: Execute direct SQL query with enhanced error handling and agent assistance
    
    Enhanced features:
    - LangChain agent assistance for query optimization
    - Enhanced error messages and suggestions
    - Query validation and safety checks
    """
    
    try:
        sql_query = request.get('sql_query', '').strip()
        process_id = request.get('process_id', '')
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        if not process_id:
            raise HTTPException(status_code=400, detail="Process ID is required")
        
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        results = processor.get_results(process_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="V2 Process ID not found")
        
        # Execute query with enhanced error handling
        execution_result = processor.execute_direct_sql_v2(sql_query, results)
        
        return {
            'success': execution_result.get('success', True),
            'data': execution_result.get('data', []),
            'row_count': execution_result.get('row_count', 0),
            'execution_time': execution_result.get('execution_time', 0),
            'enhanced_analysis': execution_result.get('enhanced_analysis', {}),
            'agent_suggestions': execution_result.get('agent_suggestions', []),
            'version': 'v2_enhanced'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Direct query execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Query execution failed: {str(e)}")


@router.post("/generate-ideal-prompt")
def generate_ideal_prompt_v2(request: dict):
    """
    V2: Generate ideal prompt using LangChain agent for better prompt engineering
    
    Enhanced features:
    - LangChain agent analyzes data structure
    - Generates more specific and effective prompts
    - Provides multiple prompt variations
    """
    
    try:
        basic_intent = request.get('basic_intent', '').strip()
        file_ids = request.get('file_ids', [])
        
        if not basic_intent:
            raise HTTPException(status_code=400, detail="Basic intent is required")
        
        if not file_ids:
            raise HTTPException(status_code=400, detail="At least one file ID is required")
        
        # Get file data
        files_data = []
        for file_id in file_ids:
            file_data = get_file_by_id(file_id)
            files_data.append(file_data)
        
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        
        # Enhanced prompt generation with LangChain agent
        if processor.sql_agent_processor.agent_available:
            prompt_result = processor._generate_ideal_prompt_with_agent(basic_intent, files_data)
        else:
            # Fallback to V1 method
            from app.services.miscellaneous_service import MiscellaneousProcessor
            v1_processor = MiscellaneousProcessor()
            prompt_result = v1_processor._generate_ideal_prompt(basic_intent, files_data)
        
        return {
            'success': True,
            'ideal_prompt': prompt_result.get('ideal_prompt'),
            'alternative_prompts': prompt_result.get('alternative_prompts', []),
            'analysis': prompt_result.get('analysis', {}),
            'version': 'v2_enhanced'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Ideal prompt generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Prompt generation failed: {str(e)}")


@router.post("/save-prompt")
def save_prompt_v2(request: dict):
    """Save prompt for reuse in V2 (enhanced with agent analysis)"""
    
    try:
        prompt_name = request.get('prompt_name', '').strip()
        prompt_text = request.get('prompt_text', '').strip()
        description = request.get('description', '').strip()
        
        if not prompt_name or not prompt_text:
            raise HTTPException(status_code=400, detail="Prompt name and text are required")
        
        # Enhanced prompt analysis with V2 capabilities
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        
        # Analyze prompt quality and provide suggestions
        prompt_analysis = processor._analyze_prompt_quality(prompt_text)
        
        # Use existing storage (same as V1 but enhanced)
        from app.routes.miscellaneous_routes import saved_prompts_storage
        
        prompt_id = generate_uuid('prompt_v2')
        saved_prompts_storage[prompt_id] = {
            'id': prompt_id,
            'name': prompt_name,
            'prompt': prompt_text,
            'description': description,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'version': 'v2',
            'quality_analysis': prompt_analysis,
            'usage_count': 0
        }
        
        return {
            'success': True,
            'prompt_id': prompt_id,
            'message': 'V2 Prompt saved successfully',
            'quality_score': prompt_analysis.get('quality_score', 0),
            'suggestions': prompt_analysis.get('suggestions', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Save prompt error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Save prompt failed: {str(e)}")


@router.get("/saved-prompts")
def get_saved_prompts_v2():
    """Get all saved prompts with V2 enhancements"""
    
    try:
        from app.routes.miscellaneous_routes import saved_prompts_storage
        
        # Enhanced response with V2 metadata
        prompts = []
        for prompt_id, prompt_data in saved_prompts_storage.items():
            enhanced_prompt = dict(prompt_data)
            enhanced_prompt['is_v2'] = prompt_data.get('version') == 'v2'
            enhanced_prompt['quality_score'] = prompt_data.get('quality_analysis', {}).get('quality_score', 0)
            prompts.append(enhanced_prompt)
        
        # Sort by creation date (newest first) and quality score
        prompts.sort(key=lambda x: (x.get('quality_score', 0), x.get('created_at', '')), reverse=True)
        
        return {
            'success': True,
            'prompts': prompts,
            'total': len(prompts),
            'v2_prompts': len([p for p in prompts if p.get('is_v2')]),
            'version': 'v2_enhanced'
        }
        
    except Exception as e:
        logger.error(f"V2 Get saved prompts error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Get prompts failed: {str(e)}")


@router.delete("/saved-prompts/{prompt_id}")
def delete_saved_prompt_v2(prompt_id: str):
    """Delete saved prompt in V2"""
    
    try:
        from app.routes.miscellaneous_routes import saved_prompts_storage
        
        if prompt_id not in saved_prompts_storage:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        prompt_data = saved_prompts_storage[prompt_id]
        del saved_prompts_storage[prompt_id]
        
        return {
            'success': True,
            'message': f'V2 Prompt "{prompt_data.get("name", "Unknown")}" deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Delete prompt error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Delete prompt failed: {str(e)}")


@router.put("/saved-prompts/{prompt_id}")
def update_saved_prompt_v2(prompt_id: str, request: dict):
    """Update saved prompt in V2 with enhanced analysis"""
    
    try:
        from app.routes.miscellaneous_routes import saved_prompts_storage
        
        if prompt_id not in saved_prompts_storage:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        prompt_name = request.get('prompt_name', '').strip()
        prompt_text = request.get('prompt_text', '').strip()
        description = request.get('description', '').strip()
        
        if not prompt_name or not prompt_text:
            raise HTTPException(status_code=400, detail="Prompt name and text are required")
        
        # Enhanced prompt analysis with V2 capabilities
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        prompt_analysis = processor._analyze_prompt_quality(prompt_text)
        
        # Update the prompt
        saved_prompts_storage[prompt_id].update({
            'name': prompt_name,
            'prompt': prompt_text,
            'description': description,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'version': 'v2',
            'quality_analysis': prompt_analysis
        })
        
        return {
            'success': True,
            'message': 'V2 Prompt updated successfully',
            'quality_score': prompt_analysis.get('quality_score', 0),
            'improvements': prompt_analysis.get('improvements', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Update prompt error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Update prompt failed: {str(e)}")


@router.post("/prompt-suggestions/")
def get_prompt_suggestions_v2(request: dict):
    """
    V2: Get intelligent prompt suggestions using LangChain agent analysis
    
    Enhanced features:
    - LangChain agent analyzes data patterns
    - Generates context-aware suggestions
    - Provides complexity-based recommendations
    """
    
    try:
        file_ids = request.get('file_ids', [])
        current_prompt = request.get('current_prompt', '').strip()
        
        if not file_ids:
            raise HTTPException(status_code=400, detail="At least one file ID is required")
        
        # Get file data
        files_data = []
        for file_id in file_ids:
            file_data = get_file_by_id(file_id)
            files_data.append(file_data)
        
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        
        # Enhanced suggestions with LangChain agent
        if processor.sql_agent_processor.agent_available:
            suggestions_result = processor._generate_intelligent_suggestions_with_agent(files_data, current_prompt)
        else:
            # Fallback to V1 method
            from app.services.miscellaneous_service import MiscellaneousProcessor
            v1_processor = MiscellaneousProcessor()
            suggestions_result = v1_processor._generate_prompt_suggestions(files_data, current_prompt)
        
        return {
            'success': True,
            'suggestions': suggestions_result.get('suggestions', []),
            'categories': suggestions_result.get('categories', {}),
            'data_insights': suggestions_result.get('data_insights', {}),
            'complexity_analysis': suggestions_result.get('complexity_analysis', {}),
            'version': 'v2_enhanced'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Prompt suggestions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Suggestions failed: {str(e)}")


@router.post("/verify-intent")
def verify_intent_v2(request: dict):
    """
    V2: Verify user intent with enhanced LangChain agent analysis
    
    Enhanced features:
    - LangChain agent provides deeper intent analysis
    - Confidence scoring for intent understanding
    - Alternative interpretations and suggestions
    """
    
    try:
        user_prompt = request.get('user_prompt', '').strip()
        file_ids = request.get('file_ids', [])
        
        if not user_prompt:
            raise HTTPException(status_code=400, detail="User prompt is required")
        
        if not file_ids:
            raise HTTPException(status_code=400, detail="At least one file ID is required")
        
        # Get file data
        files_data = []
        for file_id in file_ids:
            file_data = get_file_by_id(file_id)
            files_data.append(file_data)
        
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        
        # Enhanced intent verification with LangChain agent
        if processor.sql_agent_processor.agent_available:
            verification_result = processor._verify_intent_with_agent(user_prompt, files_data)
        else:
            # Fallback to V1 method
            from app.services.miscellaneous_service import MiscellaneousProcessor
            v1_processor = MiscellaneousProcessor()
            verification_result = v1_processor._verify_user_intent(user_prompt, files_data)
        
        return {
            'success': True,
            'intent_understood': verification_result.get('intent_understood', False),
            'confidence_score': verification_result.get('confidence_score', 0),
            'interpretation': verification_result.get('interpretation', ''),
            'alternative_interpretations': verification_result.get('alternative_interpretations', []),
            'required_clarifications': verification_result.get('required_clarifications', []),
            'suggested_improvements': verification_result.get('suggested_improvements', []),
            'version': 'v2_enhanced'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Intent verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Intent verification failed: {str(e)}")


@router.post("/improve-prompt")
def improve_prompt_v2(request: dict):
    """
    V2: Improve user prompt with enhanced LangChain agent suggestions
    
    Enhanced features:
    - LangChain agent analyzes prompt effectiveness
    - Provides specific improvement recommendations
    - Context-aware optimization suggestions
    """
    
    try:
        original_prompt = request.get('original_prompt', '').strip()
        file_ids = request.get('file_ids', [])
        
        if not original_prompt:
            raise HTTPException(status_code=400, detail="Original prompt is required")
        
        if not file_ids:
            raise HTTPException(status_code=400, detail="At least one file ID is required")
        
        # Get file data
        files_data = []
        for file_id in file_ids:
            file_data = get_file_by_id(file_id)
            files_data.append(file_data)
        
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2
        
        processor = MiscellaneousProcessorV2()
        
        # Enhanced prompt improvement with LangChain agent
        if processor.sql_agent_processor.agent_available:
            improvement_result = processor._improve_prompt_with_agent(original_prompt, files_data)
        else:
            # Fallback to V1 method
            from app.services.miscellaneous_service import MiscellaneousProcessor
            v1_processor = MiscellaneousProcessor()
            improvement_result = v1_processor._improve_user_prompt(original_prompt, files_data)
        
        return {
            'success': True,
            'improved_prompt': improvement_result.get('improved_prompt', ''),
            'improvement_score': improvement_result.get('improvement_score', 0),
            'changes_made': improvement_result.get('changes_made', []),
            'reasoning': improvement_result.get('reasoning', ''),
            'alternative_versions': improvement_result.get('alternative_versions', []),
            'version': 'v2_enhanced'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2 Prompt improvement error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Prompt improvement failed: {str(e)}")


@router.get("/health")
def health_check_v2():
    """Health check for V2 API with LangChain SQL Agent"""
    try:
        # Test LangChain imports
        from app.services.miscellaneous_service_v2 import MiscellaneousProcessorV2, check_langchain_dependencies
        
        dependencies = check_langchain_dependencies()
        
        return {
            "status": "healthy",
            "version": "v2",
            "features": [
                "langchain_sql_agent",
                "self_correction",
                "schema_introspection", 
                "multi_step_reasoning",
                "backward_compatibility",
                "enhanced_prompt_analysis",
                "intelligent_suggestions",
                "intent_verification"
            ],
            "dependencies": dependencies,
            "langchain_available": dependencies.get('langchain_available', False),
            "all_apis_available": True,
            "total_endpoints": 16,  # All V1 endpoints + V2 specific ones
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"V2 Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"V2 Service unhealthy: {str(e)}")