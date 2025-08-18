# backend/app/routes/save_results_routes.py - Save Results to Server Storage
import io
import logging
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.utils.uuid_generator import generate_uuid

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/save-results", tags=["save-results"])


class SaveResultsRequest(BaseModel):
    """Request model for saving results to server"""
    result_id: str  # reconciliation_id or delta_id
    file_id: Optional[str] = None
    result_type: str  # all, unchanged, amended, deleted, newly_added, matched, unmatched_a, unmatched_b, etc.
    file_format: str = "csv"  # csv, excel
    custom_filename: Optional[str] = None
    description: Optional[str] = None
    process_type: str  # 'reconciliation' or 'delta'


class SavedFileInfo(BaseModel):
    """Information about saved file"""
    saved_file_id: str
    original_result_id: str
    process_type: str
    result_type: str
    filename: str
    file_format: str
    file_size_mb: float
    total_rows: int
    total_columns: int
    columns: List[str]
    description: Optional[str]
    created_at: str
    data_types: Dict[str, str]


class SaveResultsResponse(BaseModel):
    """Response model for save results operation"""
    success: bool
    saved_file_info: SavedFileInfo
    message: str
    errors: List[str] = []


class ResultsSaver:
    """Helper class for saving results to server storage"""

    def __init__(self):
        # Import storage service
        from app.services.storage_service import uploaded_files
        self.storage = uploaded_files

    def get_reconciliation_data(self, reconciliation_id: str, result_type: str) -> pd.DataFrame:
        """Get reconciliation data from reconciliation storage"""
        try:
            # Import reconciliation storage from existing routes
            from app.services.reconciliation_service import optimized_reconciliation_storage

            results = optimized_reconciliation_storage.get_results(reconciliation_id)
            if not results:
                raise HTTPException(status_code=404, detail="Reconciliation ID not found")

            if result_type == "matched":
                return pd.DataFrame(results.get('matched', []))
            elif result_type == "unmatched_file_a" or result_type == "unmatched_a":
                return pd.DataFrame(results.get('unmatched_file_a', []))
            elif result_type == "unmatched_file_b" or result_type == "unmatched_b":
                return pd.DataFrame(results.get('unmatched_file_b', []))
            elif result_type == "all":
                # Combine all results
                dfs = []
                if results.get('matched'):
                    matched_df = pd.DataFrame(results['matched'])
                    matched_df['Result_Type'] = 'MATCHED'
                    dfs.append(matched_df)

                if results.get('unmatched_file_a'):
                    unmatched_a_df = pd.DataFrame(results['unmatched_file_a'])
                    unmatched_a_df['Result_Type'] = 'UNMATCHED_A'
                    dfs.append(unmatched_a_df)

                if results.get('unmatched_file_b'):
                    unmatched_b_df = pd.DataFrame(results['unmatched_file_b'])
                    unmatched_b_df['Result_Type'] = 'UNMATCHED_B'
                    dfs.append(unmatched_b_df)

                return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            else:
                raise HTTPException(status_code=400, detail=f"Invalid result_type for reconciliation: {result_type}")

        except ImportError as e:
            logger.error(f"Could not import reconciliation storage: {e}")
            raise HTTPException(status_code=500, detail="Reconciliation storage not available")

    def get_delta_data(self, delta_id: str, result_type: str) -> pd.DataFrame:
        """Get delta data from delta storage"""
        try:
            # Import delta storage from existing routes
            from app.routes.delta_routes import delta_storage

            if delta_id not in delta_storage:
                raise HTTPException(status_code=404, detail="Delta ID not found")

            results = delta_storage[delta_id]

            if result_type == "unchanged":
                return results.get('unchanged', pd.DataFrame())
            elif result_type == "amended":
                return results.get('amended', pd.DataFrame())
            elif result_type == "deleted":
                return results.get('deleted', pd.DataFrame())
            elif result_type == "newly_added":
                return results.get('newly_added', pd.DataFrame())
            elif result_type == "all_changes":
                return results.get('all_changes', pd.DataFrame())
            elif result_type == "all":
                # Combine all results
                dfs = []
                for category in ['unchanged', 'amended', 'deleted', 'newly_added']:
                    df = results.get(category, pd.DataFrame())
                    if len(df) > 0:
                        df_copy = df.copy()
                        df_copy['Delta_Category'] = category.upper()
                        dfs.append(df_copy)

                return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            else:
                raise HTTPException(status_code=400, detail=f"Invalid result_type for delta: {result_type}")

        except ImportError as e:
            logger.error(f"Could not import delta storage: {e}")
            raise HTTPException(status_code=500, detail="Delta storage not available")

    def get_file_trasnformation_data(self, file_trasnformation_id: str, result_type: str) -> pd.DataFrame:
        try:
            # Import delta storage from existing routes
            from app.services.transformation_service import transformation_storage

            if not transformation_storage.isExist(file_trasnformation_id):
                raise HTTPException(status_code=404, detail="File trasnformation ID not found")

            results = transformation_storage.get_results(file_trasnformation_id)

            return results['results']['data']



        except ImportError as e:
            logger.error(f"Could not import delta storage: {e}")
            raise HTTPException(status_code=500, detail="Delta storage not available")

    def save_dataframe_to_storage(self, df: pd.DataFrame, saved_file_id: str, filename: str,
                                  file_format: str, description: str = None) -> SavedFileInfo:
        """Save DataFrame to storage service and return saved file info"""

        try:
            # Calculate file size estimate
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            estimated_size_bytes = len(csv_buffer.getvalue().encode('utf-8'))
            file_size_mb = estimated_size_bytes / (1024 * 1024)

            # Create file info similar to uploaded files structure
            file_info = {
                "file_id": saved_file_id,
                "filename": filename,
                "custom_name": None,
                "file_type": file_format,
                "file_size_mb": round(file_size_mb, 2),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "upload_time": datetime.utcnow().isoformat(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sheet_name": None,
                "is_saved_result": True,  # Flag to identify saved results
                "description": description,
                "created_at": datetime.utcnow().isoformat()
            }

            # Store in the same storage service as uploaded files
            self.storage[saved_file_id] = {
                "info": file_info,
                "data": df
            }

            return SavedFileInfo(
                saved_file_id=saved_file_id,
                original_result_id="",  # Will be set by caller
                process_type="",  # Will be set by caller
                result_type="",  # Will be set by caller
                filename=filename,
                file_format=file_format,
                file_size_mb=round(file_size_mb, 2),
                total_rows=len(df),
                total_columns=len(df.columns),
                columns=list(df.columns),
                description=description,
                created_at=datetime.utcnow().isoformat(),
                data_types={col: str(dtype) for col, dtype in df.dtypes.items()}
            )

        except Exception as e:
            logger.error(f"Error saving DataFrame to storage: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save to storage: {str(e)}")

    def generate_filename(self, result_id: str, process_type: str, result_type: str,
                          file_format: str, custom_filename: Optional[str] = None) -> str:
        """Generate filename for saved results"""
        if custom_filename:
            # Sanitize custom filename
            safe_filename = "".join(c for c in custom_filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            if not safe_filename.endswith(f'.{file_format}'):
                safe_filename += f'.{file_format}'
            return safe_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"saved_{process_type}_{result_type}_{result_id[:8]}_{timestamp}.{file_format}"


@router.post("/save", response_model=SaveResultsResponse)
async def save_results_to_server(request: SaveResultsRequest):
    """Save reconciliation or delta results to server storage"""

    try:
        saver = ResultsSaver()

        # Validate process type
        if request.process_type not in ["reconciliation", "delta", "file-transformation"]:
            raise HTTPException(status_code=400, detail="process_type must be 'reconciliation' or 'delta'")

        # Get the appropriate data based on process type
        if request.process_type == "reconciliation":
            df = saver.get_reconciliation_data(request.result_id, request.result_type)
        elif request.process_type == "file-transformation":
            df = saver.get_file_trasnformation_data(request.result_id, request.result_type)
        else:  # delta
            df = saver.get_delta_data(request.result_id, request.result_type)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.result_type}")

        # Generate filename
        filename = saver.generate_filename(
            request.result_id,
            request.process_type,
            request.result_type,
            request.file_format,
            request.custom_filename
        )

        # Generate saved file ID
        saved_file_id = generate_uuid('saved_result') if request.file_id is None else request.file_id

        # Save DataFrame to storage
        saved_file_info = saver.save_dataframe_to_storage(
            df, saved_file_id, filename, request.file_format, request.description
        )

        # Update saved file info with request details
        saved_file_info.original_result_id = request.result_id
        saved_file_info.process_type = request.process_type
        saved_file_info.result_type = request.result_type

        logger.info(f"Saved {request.process_type} results to storage: {filename} ({len(df):,} rows)")

        return SaveResultsResponse(
            success=True,
            saved_file_info=saved_file_info,
            message=f"Results saved successfully as {filename}. You can now access it from the file library."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")


@router.get("/list")
async def list_saved_results():
    """List all saved results from storage"""

    try:
        from app.services.storage_service import uploaded_files

        saved_files = []
        for file_id, file_data in uploaded_files.items():
            file_info = file_data["info"]
            # Check if this is a saved result (has the flag)
            if file_info.get("is_saved_result", False):
                saved_files.append({
                    "saved_file_id": file_id,
                    "filename": file_info.get("filename"),
                    "description": file_info.get("description"),
                    "file_format": file_info.get("file_type"),
                    "total_rows": file_info.get("total_rows", 0),
                    "total_columns": file_info.get("total_columns", 0),
                    "file_size_mb": file_info.get("file_size_mb", 0),
                    "created_at": file_info.get("created_at", file_info.get("upload_time"))
                })

        # Sort by creation date (newest first)
        saved_files.sort(key=lambda x: x['created_at'], reverse=True)

        return {
            "success": True,
            "saved_files": saved_files,
            "total_count": len(saved_files)
        }

    except Exception as e:
        logger.error(f"Error listing saved results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list saved results: {str(e)}")


@router.get("/info/{saved_file_id}")
async def get_saved_file_info(saved_file_id: str):
    """Get information about a saved file"""

    try:
        from app.services.storage_service import uploaded_files

        if saved_file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="Saved file not found")

        file_data = uploaded_files[saved_file_id]
        file_info = file_data["info"]

        # Check if this is actually a saved result
        if not file_info.get("is_saved_result", False):
            raise HTTPException(status_code=404, detail="File is not a saved result")

        return {
            "success": True,
            "file_info": file_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting saved file info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get saved file info: {str(e)}")


@router.delete("/delete/{saved_file_id}")
async def delete_saved_file(saved_file_id: str):
    """Delete a saved results file"""

    try:
        from app.services.storage_service import uploaded_files

        if saved_file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="Saved file not found")

        file_info = uploaded_files[saved_file_id]["info"]

        # Check if this is actually a saved result
        if not file_info.get("is_saved_result", False):
            raise HTTPException(status_code=404, detail="File is not a saved result")

        # Remove from storage
        del uploaded_files[saved_file_id]

        logger.info(f"Deleted saved file: {file_info['filename']}")

        return {
            "success": True,
            "message": f"Saved file {file_info['filename']} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting saved file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete saved file: {str(e)}")


@router.get("/download/{saved_file_id}")
async def download_saved_file(saved_file_id: str, format: str = "csv"):
    """Download a saved results file"""

    try:
        from app.services.storage_service import uploaded_files

        if saved_file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="Saved file not found")

        file_data = uploaded_files[saved_file_id]
        file_info = file_data["info"]
        df = file_data["data"]

        # Check if this is actually a saved result
        if not file_info.get("is_saved_result", False):
            raise HTTPException(status_code=404, detail="File is not a saved result")

        # Prepare download based on format
        if format.lower() == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Saved Results', index=False)
            output.seek(0)

            filename = file_info['filename'].replace('.csv', '.xlsx')
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        else:  # CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            # Convert to bytes for streaming
            output = io.BytesIO(output.getvalue().encode('utf-8'))
            filename = file_info['filename']
            media_type = "text/csv"

        return StreamingResponse(
            output,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading saved file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download saved file: {str(e)}")


@router.get("/health")
async def save_results_health_check():
    """Health check for save results service"""

    try:
        from app.services.storage_service import uploaded_files, get_storage_info

        # Count saved results
        saved_results_count = 0
        for file_id, file_data in uploaded_files.items():
            if file_data["info"].get("is_saved_result", False):
                saved_results_count += 1

        storage_info = get_storage_info()

        return {
            "status": "healthy",
            "service": "save_results",
            "storage_backend": storage_info.get("backend_class", "Unknown"),
            "storage_type": storage_info.get("storage_type", "local"),
            "saved_results_count": saved_results_count,
            "total_files_in_storage": len(uploaded_files),
            "features": [
                "reuse_existing_storage",
                "delta_results_saving",
                "reconciliation_results_saving",
                "custom_filename_support",
                "description_support",
                "multiple_formats",
                "storage_integration"
            ]
        }

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "error",
            "service": "save_results",
            "error": str(e)
        }
