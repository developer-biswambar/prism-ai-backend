import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.recon_models import ReconciliationResponse, ReconciliationSummary, OptimizedRulesConfig
from app.services.reconciliation_service import OptimizedFileProcessor, optimized_reconciliation_storage
from app.utils.uuid_generator import generate_uuid

# Closest Match Configuration Model
class ClosestMatchConfig(BaseModel):
    """Configuration for closest match functionality"""
    enabled: bool = False
    specific_columns: Optional[Dict[str, str]] = None  # {"file_a_column": "file_b_column"} for specific column comparison
    min_score_threshold: Optional[float] = 30.0  # Minimum similarity score to consider
    perfect_match_threshold: Optional[float] = 99.5  # Early termination threshold
    max_comparisons: Optional[int] = None  # Limit number of comparisons for performance
    use_sampling: Optional[bool] = None  # Force enable/disable sampling for large datasets

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/reconciliation", tags=["reconciliation"])


class ReconciliationRequest(BaseModel):
    """Enhanced reconciliation request with column selection"""
    selected_columns_file_a: Optional[List[str]] = None
    selected_columns_file_b: Optional[List[str]] = None
    output_format: Optional[str] = "standard"  # standard, summary, detailed


class FileReference(BaseModel):
    """File reference for JSON-based reconciliation"""
    file_id: str
    role: str  # file_0, file_1
    label: str


class ReconciliationConfig(BaseModel):
    """Reconciliation configuration from JSON input"""
    Files: List[Dict[str, Any]]
    ReconciliationRules: List[Dict[str, Any]]
    selected_columns_file_a: Optional[List[str]] = None
    selected_columns_file_b: Optional[List[str]] = None
    user_requirements: Optional[str] = None
    files: Optional[List[FileReference]] = None


class JSONReconciliationRequest(BaseModel):
    """JSON-based reconciliation request"""
    process_type: str
    process_name: str
    user_requirements: str
    files: List[FileReference]
    reconciliation_config: ReconciliationConfig
    closest_match_config: Optional[ClosestMatchConfig] = None  # Comprehensive closest match configuration


async def get_file_by_id(file_id: str) -> UploadFile:
    """
    Retrieve file by ID from your file storage service
    Uses the existing uploaded_files storage from storage_service
    """
    # Import here to avoid circular imports
    from app.services.storage_service import uploaded_files

    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")

    try:
        file_data = uploaded_files[file_id]
        file_info = file_data["info"]
        df = file_data["data"]

        # Convert DataFrame back to file-like object
        filename = file_info["filename"]

        if filename.lower().endswith('.csv'):
            # Convert DataFrame to CSV bytes
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue().encode('utf-8')
            file_content = io.BytesIO(csv_content)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            # Convert DataFrame to Excel bytes
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_buffer.seek(0)
            file_content = excel_buffer
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {filename}")

        # Create UploadFile-like object
        class FileFromStorage:
            def __init__(self, content: io.BytesIO, filename: str):
                self.file = content
                self.filename = filename
                self.content_type = self._get_content_type(filename)

            def _get_content_type(self, filename: str) -> str:
                if filename.lower().endswith('.csv'):
                    return 'text/csv'
                elif filename.lower().endswith('.xlsx'):
                    return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                elif filename.lower().endswith('.xls'):
                    return 'application/vnd.ms-excel'
                return 'application/octet-stream'

            async def read(self):
                return self.file.read()

            def seek(self, position: int):
                return self.file.seek(position)

        return FileFromStorage(file_content, filename)

    except Exception as e:
        logger.error(f"Error retrieving file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")


@router.post("/process/", response_model=ReconciliationResponse)
async def process_reconciliation_json(
        request: JSONReconciliationRequest
):
    """Process file reconciliation with JSON input - File ID Version"""
    start_time = datetime.now()
    processor = OptimizedFileProcessor()

    try:

        if len(request.files) != 2:
            raise HTTPException(status_code=400, detail="Exactly 2 files are required for reconciliation")

        # Sort files by role to ensure consistent mapping
        files_sorted = sorted(request.files, key=lambda x: x.role)
        file_0 = next((f for f in files_sorted if f.role == "file_0"), None)
        file_1 = next((f for f in files_sorted if f.role == "file_1"), None)

        if not file_0 or not file_1:
            raise HTTPException(status_code=400, detail="Files must have roles 'file_0' and 'file_1'")

        # Retrieve files by ID
        try:
            fileA = await get_file_by_id(file_0.file_id)
            fileB = await get_file_by_id(file_1.file_id)
        except NotImplementedError:
            raise HTTPException(status_code=501,
                                detail="File retrieval service not implemented. Please implement get_file_by_id function.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to retrieve files: {str(e)}")

        # Convert reconciliation config to OptimizedRulesConfig
        rules_dict = {
            "Files": request.reconciliation_config.Files,
            "ReconciliationRules": request.reconciliation_config.ReconciliationRules
        }
        rules_config = OptimizedRulesConfig.parse_obj(rules_dict)

        # Extract column selections
        columns_a = request.reconciliation_config.selected_columns_file_a
        columns_b = request.reconciliation_config.selected_columns_file_b

        return await _process_reconciliation_core(
            processor, rules_config, fileA, fileB, columns_a, columns_b, "standard", start_time,
            closest_match_config=request.closest_match_config
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"JSON Reconciliation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


async def _process_reconciliation_core(
        processor: OptimizedFileProcessor,
        rules_config: OptimizedRulesConfig,
        fileA: UploadFile,
        fileB: UploadFile,
        columns_a: Optional[List[str]],
        columns_b: Optional[List[str]],
        output_format: str,
        start_time: datetime,
        closest_match_config: Optional[ClosestMatchConfig] = None
) -> ReconciliationResponse:
    """Core reconciliation processing logic shared between endpoints"""

    # Validate we have rules for both files
    if len(rules_config.Files) != 2:
        raise HTTPException(status_code=400, detail="Rules must contain exactly 2 file configurations")

    # Read files with optimized settings
    file_rule_a = next((f for f in rules_config.Files if f.Name == "FileA"), None)
    file_rule_b = next((f for f in rules_config.Files if f.Name == "FileB"), None)

    if not file_rule_a or not file_rule_b:
        raise HTTPException(status_code=400, detail="Rules must contain configurations for 'FileA' and 'FileB'")

    # Read files
    df_a = processor.read_file(fileA, getattr(file_rule_a, 'SheetName', None))
    df_b = processor.read_file(fileB, getattr(file_rule_b, 'SheetName', None))

    print(f"Read files: FileA {len(df_a)} rows, FileB {len(df_b)} rows")

    # Validate rules against columns
    errors_a = processor.validate_rules_against_columns(df_a, file_rule_a)
    errors_b = processor.validate_rules_against_columns(df_b, file_rule_b)

    if errors_a or errors_b:
        processor.errors.extend(errors_a + errors_b)
        raise HTTPException(status_code=400, detail={"errors": processor.errors})

    # Process FileA with optimized extraction - Handle optional Extract
    if hasattr(file_rule_a, 'Extract') and file_rule_a.Extract:
        print("Processing FileA extractions...")
        for extract_rule in file_rule_a.Extract:
            df_a[extract_rule.ResultColumnName] = processor.extract_patterns_vectorized(df_a, extract_rule)

    # Apply FileA filters - Handle optional Filter
    if hasattr(file_rule_a, 'Filter') and file_rule_a.Filter:
        print("Applying FileA filters...")
        df_a = processor.apply_filters_optimized(df_a, file_rule_a.Filter)

    # Process FileB with optimized extraction - Handle optional Extract
    if hasattr(file_rule_b, 'Extract') and file_rule_b.Extract:
        print("Processing FileB extractions...")
        for extract_rule in file_rule_b.Extract:
            df_b[extract_rule.ResultColumnName] = processor.extract_patterns_vectorized(df_b, extract_rule)

    # Apply FileB filters - Handle optional Filter
    if hasattr(file_rule_b, 'Filter') and file_rule_b.Filter:
        print("Applying FileB filters...")
        df_b = processor.apply_filters_optimized(df_b, file_rule_b.Filter)

    print(f"After processing: FileA {len(df_a)} rows, FileB {len(df_b)} rows")

    # Validate reconciliation columns exist
    recon_errors = []
    for rule in rules_config.ReconciliationRules:
        if rule.LeftFileColumn not in df_a.columns:
            recon_errors.append(
                f"Reconciliation column '{rule.LeftFileColumn}' not found in FileA after extraction")
        if rule.RightFileColumn not in df_b.columns:
            recon_errors.append(
                f"Reconciliation column '{rule.RightFileColumn}' not found in FileB after extraction")

    if recon_errors:
        processor.errors.extend(recon_errors)
        raise HTTPException(status_code=400, detail={"errors": processor.errors})

    # Perform optimized reconciliation
    print("Starting optimized reconciliation...")
    reconciliation_results = processor.reconcile_files_optimized(
        df_a, df_b, rules_config.ReconciliationRules,
        columns_a, columns_b, closest_match_config=closest_match_config
    )

    # Generate reconciliation ID
    recon_id = generate_uuid('recon')

    # Store results with optimized storage
    storage_success = optimized_reconciliation_storage.store_results(recon_id, reconciliation_results)
    if not storage_success:
        processor.warnings.append("Failed to store results in optimized storage, using fallback")

    # Calculate summary
    processing_time = (datetime.now() - start_time).total_seconds()
    total_a = len(df_a)
    total_b = len(df_b)
    matched = len(reconciliation_results['matched'])

    # Handle division by zero for match percentage
    if max(total_a, total_b) > 0:
        match_percentage = round((matched / max(total_a, total_b)) * 100, 2)
    else:
        match_percentage = 0.0

    summary = ReconciliationSummary(
        total_records_file_a=total_a,
        total_records_file_b=total_b,
        matched_records=matched,
        unmatched_file_a=len(reconciliation_results['unmatched_file_a']),
        unmatched_file_b=len(reconciliation_results['unmatched_file_b']),
        match_percentage=match_percentage,
        processing_time_seconds=round(processing_time, 3)
    )

    print(f"Reconciliation completed in {processing_time:.2f}s - {matched} matches found")

    # Only save results if there's data to save
    from app.routes.save_results_routes import SaveResultsRequest
    from app.routes.save_results_routes import save_results_to_server
    
    # Check if we have meaningful data to save
    has_matched = len(reconciliation_results['matched']) > 0
    
    # Only save results if we have matches - no point saving when everything is unmatched
    if has_matched:
        # Save "all" results (matched + unmatched) when we have matches
        # Save "matched" results 
        try:
            save_request_matched = SaveResultsRequest(
                result_id=recon_id,
                file_id=recon_id+"_matched",
                result_type="matched",
                process_type="reconciliation",
                file_format="csv",
                description="Matched records from reconciliation"
            )
            save_result_res_matched = await save_results_to_server(save_request_matched)
            print(f"✓ Saved matched results: {save_result_res_matched}")
        except Exception as e:
            print(f"⚠️ Could not save matched results: {str(e)}")
            # Continue execution - saving is optional
        try:
            save_request_unmatched_a = SaveResultsRequest(
                result_id=recon_id,
                file_id=recon_id+"_unmatched_a",
                result_type="unmatched_a",
                process_type="reconciliation",
                file_format="csv",
                description="unmatched_a records from reconciliation"
            )
            save_result_res_unmatched_a = await save_results_to_server(save_request_unmatched_a)
            print(f"✓ Saved matched results: {save_result_res_unmatched_a}")
        except Exception as e:
            print(f"⚠️ Could not save unmatched_a results: {str(e)}")
            # Continue execution - saving is optional
        try:
            save_request_unmatched_b = SaveResultsRequest(
                result_id=recon_id,
                file_id=recon_id+'_unmatched_b',
                result_type="unmatched_b",
                process_type="reconciliation",
                file_format="csv",
                description="unmatched_b records from reconciliation"
            )
            save_result_res_unmatched_b = await save_results_to_server(save_request_unmatched_b)
            print(f"✓ Saved matched results: {save_result_res_unmatched_b}")
        except Exception as e:
            print(f"⚠️ Could not save matched results: {str(e)}")
            # Continue execution - saving is optional
    else:
        print("ℹ️ No matches found - not saving any results files. Use 'View Unmatched Records' to see why records didn't match.")

    return ReconciliationResponse(
        success=True,
        summary=summary,
        reconciliation_id=recon_id,
        errors=processor.errors,
        warnings=processor.warnings
    )


@router.get("/results/{reconciliation_id}")
async def get_reconciliation_results_optimized(
        reconciliation_id: str,
        result_type: Optional[str] = "all",  # all, matched, unmatched_a, unmatched_b
        page: Optional[int] = 1,
        page_size: Optional[int] = 1000
):
    """Get reconciliation results with pagination for large datasets"""

    # Try optimized storage first
    results = optimized_reconciliation_storage.get_results(reconciliation_id)

    if not results:
        raise HTTPException(status_code=404, detail="Reconciliation ID not found")

    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    def paginate_results(data_list):
        return data_list[start_idx:end_idx]

    response_data = {
        'reconciliation_id': reconciliation_id,
        'timestamp': results['timestamp'].isoformat(),
        'row_counts': results['row_counts'],
        'pagination': {
            'page': page,
            'page_size': page_size,
            'start_index': start_idx
        }
    }

    if result_type == "all":
        response_data.update({
            'matched': paginate_results(results['matched']),
            'unmatched_file_a': paginate_results(results['unmatched_file_a']),
            'unmatched_file_b': paginate_results(results['unmatched_file_b'])
        })
    elif result_type == "matched":
        response_data['matched'] = paginate_results(results['matched'])
    elif result_type == "unmatched_a":
        response_data['unmatched_file_a'] = paginate_results(results['unmatched_file_a'])
    elif result_type == "unmatched_b":
        response_data['unmatched_file_b'] = paginate_results(results['unmatched_file_b'])
    else:
        raise HTTPException(status_code=400, detail="Invalid result_type. Use: all, matched, unmatched_a, unmatched_b")

    return response_data


@router.get("/download/{reconciliation_id}")
async def download_reconciliation_results_optimized(
        reconciliation_id: str,
        format: str = "csv",
        result_type: str = "all",  # all, matched, unmatched_a, unmatched_b
        compress: bool = True
):
    """Download reconciliation results with optimized streaming for large files"""

    results = optimized_reconciliation_storage.get_results(reconciliation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Reconciliation ID not found")

    try:
        # Convert back to DataFrames for download
        matched_df = pd.DataFrame(results['matched'])
        unmatched_a_df = pd.DataFrame(results['unmatched_file_a'])
        unmatched_b_df = pd.DataFrame(results['unmatched_file_b'])

        if format.lower() == "excel":
            # Create Excel file with streaming for large datasets
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
                if result_type == "all":
                    # Write in chunks for large datasets
                    if len(matched_df) > 0:
                        matched_df.to_excel(writer, sheet_name='Matched Records', index=False)
                    if len(unmatched_a_df) > 0:
                        unmatched_a_df.to_excel(writer, sheet_name='Unmatched FileA', index=False)
                    if len(unmatched_b_df) > 0:
                        unmatched_b_df.to_excel(writer, sheet_name='Unmatched FileB', index=False)

                    # Add summary sheet
                    summary_data = pd.DataFrame({
                        'Metric': ['Total Records FileA', 'Total Records FileB', 'Matched Records',
                                   'Unmatched FileA', 'Unmatched FileB', 'Match Percentage'],
                        'Count': [
                            results['row_counts']['unmatched_a'] + results['row_counts']['matched'],
                            results['row_counts']['unmatched_b'] + results['row_counts']['matched'],
                            results['row_counts']['matched'],
                            results['row_counts']['unmatched_a'],
                            results['row_counts']['unmatched_b'],
                            f"{(results['row_counts']['matched'] / max(results['row_counts']['unmatched_a'] + results['row_counts']['matched'], results['row_counts']['unmatched_b'] + results['row_counts']['matched'], 1) * 100):.2f}%"
                        ]
                    })
                    summary_data.to_excel(writer, sheet_name='Summary', index=False)

                elif result_type == "matched" and len(matched_df) > 0:
                    matched_df.to_excel(writer, sheet_name='Matched Records', index=False)
                elif result_type == "unmatched_a" and len(unmatched_a_df) > 0:
                    unmatched_a_df.to_excel(writer, sheet_name='Unmatched FileA', index=False)
                elif result_type == "unmatched_b" and len(unmatched_b_df) > 0:
                    unmatched_b_df.to_excel(writer, sheet_name='Unmatched FileB', index=False)

            output.seek(0)

            filename = f"reconciliation_{reconciliation_id}_{result_type}.xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        elif format.lower() == "csv":
            # For CSV, return the requested result type
            if result_type == "matched":
                df_to_export = matched_df
            elif result_type == "unmatched_a":
                df_to_export = unmatched_a_df
            elif result_type == "unmatched_b":
                df_to_export = unmatched_b_df
            else:
                # For "all", combine all data
                df_to_export = pd.concat([
                    matched_df.assign(Result_Type='Matched'),
                    unmatched_a_df.assign(Result_Type='Unmatched_FileA'),
                    unmatched_b_df.assign(Result_Type='Unmatched_FileB')
                ], ignore_index=True)

            output = io.StringIO()
            df_to_export.to_csv(output, index=False)
            output.seek(0)

            # Convert to bytes for streaming
            output = io.BytesIO(output.getvalue().encode('utf-8'))
            filename = f"reconciliation_{reconciliation_id}_{result_type}.csv"
            media_type = "text/csv"

        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'excel' or 'csv'")

        return StreamingResponse(
            output,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")


@router.get("/results/{reconciliation_id}/summary")
async def get_reconciliation_summary(reconciliation_id: str):
    """Get a quick summary of reconciliation results"""

    results = optimized_reconciliation_storage.get_results(reconciliation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Reconciliation ID not found")

    row_counts = results['row_counts']
    total_a = row_counts['unmatched_a'] + row_counts['matched']
    total_b = row_counts['unmatched_b'] + row_counts['matched']

    return {
        'reconciliation_id': reconciliation_id,
        'timestamp': results['timestamp'].isoformat(),
        'summary': {
            'total_records_file_a': total_a,
            'total_records_file_b': total_b,
            'matched_records': row_counts['matched'],
            'unmatched_file_a': row_counts['unmatched_a'],
            'unmatched_file_b': row_counts['unmatched_b'],
            'match_percentage': round((row_counts['matched'] / max(total_a, total_b, 1)) * 100, 2),
            'data_quality': {
                'file_a_match_rate': round((row_counts['matched'] / max(total_a, 1)) * 100, 2),
                'file_b_match_rate': round((row_counts['matched'] / max(total_b, 1)) * 100, 2),
                'overall_completeness': round(((total_a + total_b - row_counts['unmatched_a'] - row_counts[
                    'unmatched_b']) / max(total_a + total_b, 1)) * 100, 2)
            }
        }
    }


@router.delete("/results/{reconciliation_id}")
async def delete_reconciliation_results(reconciliation_id: str):
    """Delete reconciliation results to free up memory"""

    results = optimized_reconciliation_storage.get_results(reconciliation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Reconciliation ID not found")

    # Remove from storage
    if reconciliation_id in optimized_reconciliation_storage.storage:
        del optimized_reconciliation_storage.storage[reconciliation_id]
        return {"success": True, "message": f"Reconciliation results {reconciliation_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Reconciliation ID not found in storage")


@router.post("/generate-config/")
async def generate_reconciliation_config(request: dict):
    """Generate reconciliation configuration using AI based on user requirements"""
    
    try:
        requirements = request.get('requirements', '')
        source_files = request.get('source_files', [])
        
        if not requirements:
            raise HTTPException(status_code=400, detail="Requirements are required")
        
        if not source_files or len(source_files) != 2:
            raise HTTPException(status_code=400, detail="Exactly 2 source files are required for reconciliation")
        
        # Import LLM service
        from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage
        
        llm_service = get_llm_service()
        if not llm_service.is_available():
            raise HTTPException(status_code=500, detail=f"LLM service ({llm_service.get_provider_name()}) not configured")
        
        # Get generation parameters from config
        generation_params = get_llm_generation_params()
        
        # Prepare context about source files
        files_context = []
        for i, sf in enumerate(source_files):
            role = f"file_{i}"
            files_context.append(f"File {i + 1} ({role}): {sf['filename']}")
            files_context.append(f"  Columns: {', '.join(sf['columns'])}")
            files_context.append(f"  Rows: {sf['totalRows']}")
        
        files_info = "\\n".join(files_context)
        
        # Create prompt for AI configuration generation
        prompt = f"""
You are an expert financial data reconciliation configuration generator. Based on the user requirements and source file information, generate a JSON configuration for data reconciliation.

Source Files Available:
{files_info}

User Requirements:
{requirements}

Generate a reconciliation configuration with this exact JSON structure:
{{
    "Files": [
        {{
            "Name": "FileA",
            "Extract": [
                {{
                    "ResultColumnName": "extracted_column_name",
                    "SourceColumn": "source_column_name",
                    "MatchType": "regex|exact|contains",
                    "Patterns": ["regex_pattern_if_needed"]
                }}
            ],
            "Filter": [
                {{
                    "ColumnName": "column_to_filter",
                    "MatchType": "equals|contains|greater_than|less_than|not_equals",
                    "Value": "filter_value"
                }}
            ]
        }},
        {{
            "Name": "FileB", 
            "Extract": [],
            "Filter": []
        }}
    ],
    "ReconciliationRules": [
        {{
            "LeftFileColumn": "column_from_fileA",
            "RightFileColumn": "column_from_fileB", 
            "MatchType": "equals|tolerance|fuzzy|date_equals",
            "ToleranceValue": 0.01
        }}
    ],
    "selected_columns_file_a": ["list", "of", "relevant", "columns", "from", "file1"],
    "selected_columns_file_b": ["list", "of", "relevant", "columns", "from", "file2"]
}}

Configuration Rules:
1. ONLY use column names that exist in the source files
2. File 1 columns: {', '.join(source_files[0].get('columns', []))}
3. File 2 columns: {', '.join(source_files[1].get('columns', []))}
4. For exact matches, use MatchType "equals" with ToleranceValue 0
5. For amount/numeric tolerance, use MatchType "tolerance" with appropriate ToleranceValue (e.g., 0.01 for currency)
6. For date matching, use MatchType "date_equals" with ToleranceValue 0
7. For fuzzy/text matching, use MatchType "fuzzy" with ToleranceValue between 0.7-0.9
8. Extract rules are optional - only add if data transformation is needed
9. Filter rules are optional - only add if data filtering is needed
10. Select relevant columns that are needed for reconciliation and reporting

Common Reconciliation Patterns:
- Transaction ID matching: exact match on ID fields
- Amount matching: tolerance match with 0.01 tolerance for currency
- Date matching: date_equals match type for date fields
- Reference number matching: exact or fuzzy match depending on format
- Account matching: exact match on account identifiers

IMPORTANT: Return ONLY the JSON configuration, no additional text or explanation.

Examples of good matching:
- Transaction IDs: {{"LeftFileColumn": "transaction_id", "RightFileColumn": "ref_id", "MatchType": "equals", "ToleranceValue": 0}}
- Amounts: {{"LeftFileColumn": "amount", "RightFileColumn": "value", "MatchType": "tolerance", "ToleranceValue": 0.01}}
- Dates: {{"LeftFileColumn": "date", "RightFileColumn": "transaction_date", "MatchType": "date_equals", "ToleranceValue": 0}}
- Names: {{"LeftFileColumn": "customer_name", "RightFileColumn": "client_name", "MatchType": "fuzzy", "ToleranceValue": 0.8}}
"""
        
        # Call LLM service
        messages = [
            LLMMessage(role="system", content="You are a financial data reconciliation expert. Return only valid JSON configuration."),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = llm_service.generate_text(
            messages=messages,
            **generation_params
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {response.error}")
        
        generated_config_text = response.content
        
        # Parse the JSON response
        import json
        try:
            generated_config = json.loads(generated_config_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response if it contains extra text
            import re
            json_match = re.search(r'\\{.*\\}', generated_config_text, re.DOTALL)
            if json_match:
                generated_config = json.loads(json_match.group())
            else:
                raise HTTPException(status_code=500, detail="Failed to parse AI-generated configuration")
        
        # Validate the generated configuration has required fields
        required_fields = ['Files', 'ReconciliationRules']
        missing_fields = [field for field in required_fields if field not in generated_config]
        if missing_fields:
            raise HTTPException(status_code=500, detail=f"AI generated config missing fields: {missing_fields}")
        
        # Ensure we have exactly 2 files
        if len(generated_config.get('Files', [])) != 2:
            # Fix the configuration
            file_a_columns = source_files[0].get('columns', [])
            file_b_columns = source_files[1].get('columns', [])
            
            generated_config['Files'] = [
                {
                    "Name": "FileA",
                    "Extract": generated_config.get('Files', [{}])[0].get('Extract', []),
                    "Filter": generated_config.get('Files', [{}])[0].get('Filter', [])
                },
                {
                    "Name": "FileB", 
                    "Extract": generated_config.get('Files', [{}, {}])[1].get('Extract', []) if len(generated_config.get('Files', [])) > 1 else [],
                    "Filter": generated_config.get('Files', [{}, {}])[1].get('Filter', []) if len(generated_config.get('Files', [])) > 1 else []
                }
            ]
        
        # Ensure selected columns are present
        if 'selected_columns_file_a' not in generated_config:
            generated_config['selected_columns_file_a'] = source_files[0].get('columns', [])
        if 'selected_columns_file_b' not in generated_config:
            generated_config['selected_columns_file_b'] = source_files[1].get('columns', [])
        
        return {
            "success": True,
            "message": "Reconciliation configuration generated successfully",
            "data": generated_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating AI reconciliation configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration generation error: {str(e)}")


@router.get("/health")
async def reconciliation_health_check():
    """Health check for reconciliation service"""
    
    # Check LLM service status
    try:
        from app.services.llm_service import get_llm_service
        llm_service = get_llm_service()
        llm_status = {
            "provider": llm_service.get_provider_name(),
            "model": llm_service.get_model_name(),
            "available": llm_service.is_available()
        }
    except Exception as e:
        llm_status = {
            "provider": "unknown",
            "model": "unknown",
            "available": False,
            "error": str(e)
        }
    
    storage_count = len(optimized_reconciliation_storage.storage)

    return {
        "status": "healthy",
        "service": "optimized_reconciliation",
        "llm_service": llm_status,
        "active_reconciliations": storage_count,
        "memory_usage": "optimized",
        "features": [
            "hash_based_matching",
            "vectorized_extraction",
            "column_selection",
            "paginated_results",
            "streaming_downloads",
            "batch_processing",
            "json_input_support",
            "file_id_retrieval",
            "ai_configuration_generation"
        ]
    }
