# backend/app/routes/recent_results_routes.py - Fixed File Generator Support
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/recent-results", tags=["recent-results"])


class RecentResultInfo(BaseModel):
    """Information about a recent result"""
    id: str  # delta_id, reconciliation_id, or generation_id
    process_type: str  # 'delta', 'reconciliation', or 'file_generation'
    status: str  # 'completed', 'processing', 'failed'
    created_at: str  # ISO datetime string
    file_a: Optional[str] = None
    file_b: Optional[str] = None
    source_file: Optional[str] = None  # For file generation
    output_filename: Optional[str] = None  # For file generation
    summary: Optional[Dict[str, Any]] = None
    processing_time_seconds: Optional[float] = None
    row_multiplication_factor: Optional[int] = None  # For file generation


class RecentResultsResponse(BaseModel):
    """Response model for recent results"""
    success: bool
    results: List[RecentResultInfo]
    total_count: int
    message: str


@router.get("/list", response_model=RecentResultsResponse)
async def get_recent_results(limit: int = 5):
    """Get recent Delta Generation, Reconciliation, and File Generation results"""

    try:
        recent_results = []

        # Get Delta Generation results
        try:
            from app.services.delta_storage_service import optimized_delta_storage
            from app.services.storage_service import uploaded_files
            
            # Get all delta result IDs from storage
            # Look for files with delta suffixes (_metadata, _unchanged, etc.)
            delta_file_ids = []
            try:
                for file_id in uploaded_files.list():
                    if file_id.startswith('delta_') and '_metadata' in file_id:
                        # Extract delta ID from metadata file name
                        delta_id = file_id.replace('_metadata', '')
                        if delta_id not in delta_file_ids:
                            delta_file_ids.append(delta_id)
            except Exception as list_error:
                logger.warning(f"Error listing delta files: {list_error}")
                delta_file_ids = []

            # Process each delta result
            for delta_id in delta_file_ids[:limit]:  # Limit to avoid too many results
                try:
                    # Get metadata first to check if this is a valid delta
                    delta_metadata = optimized_delta_storage.get_metadata_only(delta_id)
                    if not delta_metadata:
                        continue

                    recent_results.append(RecentResultInfo(
                        id=delta_id,
                        process_type="delta",
                        status="completed",
                        created_at=delta_metadata.get("timestamp", datetime.now()).isoformat() if hasattr(delta_metadata.get("timestamp", datetime.now()), 'isoformat') else str(delta_metadata.get("timestamp", datetime.now())),
                        file_a="File A",  # We don't store original file names in metadata
                        file_b="File B",  # We don't store original file names in metadata
                        summary={
                            "unchanged_records": delta_metadata["row_counts"]["unchanged"],
                            "amended_records": delta_metadata["row_counts"]["amended"],
                            "deleted_records": delta_metadata["row_counts"]["deleted"],
                            "newly_added_records": delta_metadata["row_counts"]["newly_added"]
                        },
                        processing_time_seconds=None  # Not stored in metadata
                    ))
                    
                except Exception as delta_error:
                    logger.warning(f"Error processing delta result {delta_id}: {delta_error}")
                    continue

        except ImportError as e:
            logger.warning(f"Could not import delta storage: {e}")
        except Exception as e:
            logger.error(f"Error retrieving delta results: {e}")

        # Get Reconciliation results
        try:
            from app.services.reconciliation_service import optimized_reconciliation_storage
            from app.services.storage_service import uploaded_files

            # Get all reconciliation result IDs from storage
            # Look for files with reconciliation suffixes (_matched, _unmatched_a, _unmatched_b)
            recon_file_ids = []
            try:
                for file_id in uploaded_files.list():
                    if file_id.startswith('recon_') and ('_matched' in file_id or '_unmatched_a' in file_id or '_unmatched_b' in file_id):
                        # Extract reconciliation ID from file names like "recon_123_matched" or "recon_123_unmatched_a"
                        if '_matched' in file_id:
                            recon_id = file_id.replace('_matched', '')
                        elif '_unmatched_a' in file_id:
                            recon_id = file_id.replace('_unmatched_a', '')
                        elif '_unmatched_b' in file_id:
                            recon_id = file_id.replace('_unmatched_b', '')
                        else:
                            continue
                        
                        if recon_id not in recon_file_ids:
                            recon_file_ids.append(recon_id)
            except Exception as list_error:
                logger.warning(f"Error listing reconciliation files: {list_error}")
                recon_file_ids = []

            # Process each reconciliation result
            for recon_id in recon_file_ids[:limit]:  # Limit to avoid too many results
                try:
                    # Get metadata first to check if this is a valid reconciliation
                    recon_metadata = optimized_reconciliation_storage.get_metadata_only(recon_id)
                    if not recon_metadata:
                        continue

                    # Get file information using new storage service
                    file_a_name = "Unknown File A"
                    file_b_name = "Unknown File B"

                    try:
                        # Get file list using storage service methods
                        file_list = uploaded_files.list()
                        if len(file_list) >= 2:
                            # Get file info using metadata methods
                            file_a_info = uploaded_files.get_metadata_only(file_list[0])
                            file_b_info = uploaded_files.get_metadata_only(file_list[1])
                                
                            if file_a_info:
                                file_a_name = file_a_info.get("custom_name") or file_a_info.get("filename", "Unknown File A")
                            if file_b_info:
                                file_b_name = file_b_info.get("custom_name") or file_b_info.get("filename", "Unknown File B")
                    except Exception as e:
                        logger.warning(f"Error getting file info for reconciliation {recon_id}: {e}")

                    row_counts = recon_metadata.get("row_counts", {})
                    total_a = row_counts.get("matched", 0) + row_counts.get("unmatched_a", 0)
                    total_b = row_counts.get("matched", 0) + row_counts.get("unmatched_b", 0)
                    match_percentage = 0
                    if max(total_a, total_b) > 0:
                        match_percentage = round((row_counts.get("matched", 0) / max(total_a, total_b)) * 100, 2)

                    recent_results.append(RecentResultInfo(
                        id=recon_id,
                        process_type="reconciliation",
                        status="completed",
                        created_at=recon_metadata.get("timestamp", datetime.now()).isoformat() if hasattr(recon_metadata.get("timestamp", datetime.now()), 'isoformat') else str(recon_metadata.get("timestamp", datetime.now())),
                        file_a=file_a_name,
                        file_b=file_b_name,
                        summary={
                            "matched_records": row_counts.get("matched", 0),
                            "unmatched_file_a": row_counts.get("unmatched_a", 0),
                            "unmatched_file_b": row_counts.get("unmatched_b", 0),
                            "match_percentage": match_percentage,
                            "total_records_file_a": total_a,
                            "total_records_file_b": total_b
                        },
                        processing_time_seconds=None
                    ))
                    
                except Exception as recon_error:
                    logger.warning(f"Error processing reconciliation result {recon_id}: {recon_error}")
                    continue

        except ImportError as e:
            logger.warning(f"Could not import reconciliation storage: {e}")
        except Exception as e:
            logger.error(f"Error retrieving reconciliation results: {e}")

        # Get File Generation/Transformation results
        try:
            from app.services.transformation_service import transformation_storage

            # Check if transformation_storage.storage exists and has items
            if hasattr(transformation_storage, 'storage') and hasattr(transformation_storage.storage, 'items'):
                transformation_items = transformation_storage.storage.items()
            else:
                transformation_items = []

            for file_transformation in transformation_items:
                try:
                    transformation_id = file_transformation[0]
                    transformation_details = file_transformation[1]

                    # Get source file info using new storage service
                    file_ids = []
                    file_names = []
                    
                    source_files_config = transformation_details.get('results', {}).get('config', {}).get('source_files', [])
                    
                    for file_info in source_files_config:
                        file_id = file_info.get('file_id', '')
                        file_ids.append(file_id)
                        
                        # Get actual file name using storage service
                        try:
                            from app.services.storage_service import uploaded_files
                            if uploaded_files.exists(file_id):
                                file_metadata = uploaded_files.get_metadata_only(file_id)
                                if file_metadata:
                                    file_name = file_metadata.get("custom_name") or file_metadata.get("filename", file_id)
                                    file_names.append(file_name)
                                else:
                                    file_names.append(file_id)
                            else:
                                file_names.append(file_id)
                        except Exception as file_error:
                            logger.warning(f"Error getting file info for {file_id}: {file_error}")
                            file_names.append(file_id)

                    recent_results.append(RecentResultInfo(
                        id=transformation_id,
                        process_type="file-transformation",
                        status="completed",
                        created_at=transformation_details['timestamp'].isoformat() if hasattr(transformation_details['timestamp'], 'isoformat') else str(transformation_details['timestamp']),
                        file_a=file_names[0] if file_names else (file_ids[0] if file_ids else "Unknown"),
                        file_b=file_names[1] if len(file_names) > 1 else (file_ids[1] if len(file_ids) > 1 else ''),
                        summary={
                            "input_records": transformation_details.get('results', {}).get('processing_info', {}).get('input_row_count', 0),
                            'output_records': transformation_details.get('results', {}).get('processing_info', {}).get('output_row_count', 0),
                            'columns_generated': 'Unknown',
                            'configuration': transformation_details.get('results', {}).get('config', {}),
                            'processing_info': transformation_details.get('results', {}).get('processing_info', {})
                        },
                        processing_time_seconds=transformation_details.get('results', {}).get('processing_info', {}).get('processing_time', None)
                    ))

                except Exception as gen_error:
                    logger.error(f"Error processing generation result {transformation_id}: {gen_error}")
                    # Add a basic entry even if we can't process all details
                    recent_results.append(RecentResultInfo(
                        id=transformation_id,
                        process_type="file_generation",
                        status="completed",
                        created_at=datetime.now().isoformat(),
                        source_file="Unknown Source",
                        output_filename="generated_file.csv",
                        summary={
                            "total_input_records": 0,
                            "total_output_records": 0,
                            "row_multiplication_factor": 1,
                            "columns_generated": [],
                            "rules_description": "AI File Generation"
                        },
                        processing_time_seconds=None,
                        row_multiplication_factor=1
                    ))

        except ImportError as e:
            logger.warning(f"Could not import file generator storage: {e}")
        except Exception as e:
            logger.error(f"Error retrieving file generation results: {e}")

        # Sort by timestamp (newest first) and limit
        try:
            recent_results.sort(key=lambda x: x.created_at, reverse=True)
        except Exception as sort_error:
            logger.warning(f"Error sorting results: {sort_error}")

        limited_results = recent_results[:limit]

        return RecentResultsResponse(
            success=True,
            results=limited_results,
            total_count=len(recent_results),
            message=f"Retrieved {len(limited_results)} recent results"
        )

    except Exception as e:
        logger.error(f"Error getting recent results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent results: {str(e)}")


@router.get("/delta/{delta_id}/summary")
async def get_delta_result_summary(delta_id: str):
    """Get summary for a specific delta result"""

    try:
        from app.services.delta_storage_service import optimized_delta_storage

        delta_metadata = optimized_delta_storage.get_metadata_only(delta_id)
        if not delta_metadata:
            raise HTTPException(status_code=404, detail="Delta ID not found")

        row_counts = delta_metadata["row_counts"]

        return {
            "success": True,
            "delta_id": delta_id,
            "process_type": "delta",
            "status": "completed",
            "timestamp": delta_data["timestamp"].isoformat() if hasattr(delta_data["timestamp"], 'isoformat') else str(delta_data["timestamp"]),
            "summary": {
                "unchanged_records": row_counts["unchanged"],
                "amended_records": row_counts["amended"],
                "deleted_records": row_counts["deleted"],
                "newly_added_records": row_counts["newly_added"],
                "total_changes": row_counts["amended"] + row_counts["deleted"] + row_counts["newly_added"]
            }
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="Delta storage not available")
    except Exception as e:
        logger.error(f"Error getting delta summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get delta summary: {str(e)}")


@router.get("/reconciliation/{recon_id}/summary")
async def get_reconciliation_result_summary(recon_id: str):
    """Get summary for a specific reconciliation result"""

    try:
        from app.services.reconciliation_service import optimized_reconciliation_storage

        recon_results = optimized_reconciliation_storage.get_results(recon_id)
        if not recon_results:
            raise HTTPException(status_code=404, detail="Reconciliation ID not found")

        row_counts = recon_results.get("row_counts", {})
        total_a = row_counts.get("matched", 0) + row_counts.get("unmatched_a", 0)
        total_b = row_counts.get("matched", 0) + row_counts.get("unmatched_b", 0)
        match_percentage = 0
        if max(total_a, total_b) > 0:
            match_percentage = round((row_counts.get("matched", 0) / max(total_a, total_b)) * 100, 2)

        return {
            "success": True,
            "reconciliation_id": recon_id,
            "process_type": "reconciliation",
            "status": "completed",
            "timestamp": recon_results["timestamp"].isoformat() if hasattr(recon_results["timestamp"], 'isoformat') else str(recon_results["timestamp"]),
            "summary": {
                "matched_records": row_counts.get("matched", 0),
                "unmatched_file_a": row_counts.get("unmatched_a", 0),
                "unmatched_file_b": row_counts.get("unmatched_b", 0),
                "match_percentage": match_percentage,
                "total_records_file_a": total_a,
                "total_records_file_b": total_b
            }
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="Reconciliation storage not available")
    except Exception as e:
        logger.error(f"Error getting reconciliation summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reconciliation summary: {str(e)}")


@router.get("/file-generation/{generation_id}/summary")
async def get_file_generation_result_summary(generation_id: str):
    """Get summary for a specific file generation result"""

    try:
        from app.routes.file_generator import generation_storage

        if generation_id not in generation_storage:
            raise HTTPException(status_code=404, detail="Generation ID not found")

        generation_data = generation_storage[generation_id]

        # Safely extract data
        rules = generation_data.get("rules")
        output_data = generation_data.get("output_data")
        source_filename = generation_data.get("source_filename", "Unknown")
        timestamp = generation_data.get("timestamp", datetime.now())

        # Calculate metrics safely
        total_output_records = len(output_data) if output_data is not None else 0
        row_multiplication_factor = 1
        columns_generated = []
        rules_description = "AI File Generation"
        output_filename = "generated_file.csv"

        # Extract rule information safely
        if rules:
            try:
                if hasattr(rules, 'row_multiplication') and rules.row_multiplication:
                    if rules.row_multiplication.enabled:
                        row_multiplication_factor = rules.row_multiplication.count
                elif isinstance(rules, dict):
                    row_mult = rules.get('row_multiplication', {})
                    if row_mult.get('enabled', False):
                        row_multiplication_factor = row_mult.get('count', 1)

                if hasattr(rules, 'output_filename'):
                    output_filename = rules.output_filename
                elif isinstance(rules, dict):
                    output_filename = rules.get('output_filename', 'generated_file.csv')

                if hasattr(rules, 'description'):
                    rules_description = rules.description
                elif isinstance(rules, dict):
                    rules_description = rules.get('description', 'AI File Generation')
            except:
                pass

        # Calculate input records
        total_input_records = total_output_records // row_multiplication_factor if row_multiplication_factor > 1 else total_output_records

        # Get columns
        if output_data is not None:
            try:
                if hasattr(output_data, 'columns'):
                    columns_generated = output_data.columns.tolist()
                elif hasattr(output_data, 'keys'):
                    columns_generated = list(output_data.keys())
            except:
                pass

        return {
            "success": True,
            "generation_id": generation_id,
            "process_type": "file_generation",
            "status": "completed",
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            "summary": {
                "total_input_records": total_input_records,
                "total_output_records": total_output_records,
                "row_multiplication_factor": row_multiplication_factor,
                "columns_generated": columns_generated,
                "rules_description": rules_description,
                "source_filename": source_filename,
                "output_filename": output_filename
            }
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="File generator storage not available")
    except Exception as e:
        logger.error(f"Error getting file generation summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get file generation summary: {str(e)}")


@router.delete("/clear-old")
async def clear_old_results(keep_count: int = 10):
    """Clear old results, keeping only the most recent ones"""

    try:
        deleted_count = 0

        # Clear old delta results
        try:
            from app.services.delta_storage_service import optimized_delta_storage
            from app.services.storage_service import uploaded_files

            # Get delta IDs and their timestamps
            delta_data = []
            try:
                for file_id in uploaded_files.list():
                    if file_id.startswith('delta_') and '_metadata' in file_id:
                        delta_id = file_id.replace('_metadata', '')
                        metadata = optimized_delta_storage.get_metadata_only(delta_id)
                        if metadata:
                            delta_data.append((delta_id, metadata.get("timestamp", datetime.now())))
            except Exception as e:
                logger.warning(f"Error getting delta data for cleanup: {e}")

            if len(delta_data) > keep_count:
                # Sort by timestamp and keep only recent ones
                sorted_deltas = sorted(delta_data, key=lambda x: x[1], reverse=True)
                
                # Delete old results
                to_delete_count = len(sorted_deltas) - keep_count
                for delta_id, _ in sorted_deltas[keep_count:]:
                    try:
                        success = optimized_delta_storage.delete_results(delta_id)
                        if success:
                            deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Error deleting old delta result {delta_id}: {e}")

        except ImportError:
            pass

        # Clear old reconciliation results
        try:
            from app.services.reconciliation_service import optimized_reconciliation_storage
            from app.services.storage_service import uploaded_files

            # Get reconciliation IDs and their timestamps
            recon_data = []
            try:
                for file_id in uploaded_files.list():
                    if file_id.startswith('recon_') and '_matched' in file_id:
                        recon_id = file_id.replace('_matched', '')
                        metadata = optimized_reconciliation_storage.get_metadata_only(recon_id)
                        if metadata:
                            recon_data.append((recon_id, metadata.get("timestamp", datetime.now())))
            except Exception as e:
                logger.warning(f"Error getting reconciliation data for cleanup: {e}")

            if len(recon_data) > keep_count:
                # Sort by timestamp and delete old ones
                sorted_recons = sorted(recon_data, key=lambda x: x[1], reverse=True)
                to_delete_ids = [item[0] for item in sorted_recons[keep_count:]]
                
                # Delete old reconciliation results
                for recon_id in to_delete_ids:
                    try:
                        optimized_reconciliation_storage.delete_results(recon_id)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Error deleting reconciliation {recon_id}: {e}")

        except ImportError:
            pass

        # Clear old file generation results
        try:
            from app.routes.file_generator import generation_storage

            if len(generation_storage) > keep_count:
                # Sort by timestamp and keep only recent ones
                try:
                    sorted_generations = sorted(
                        generation_storage.items(),
                        key=lambda x: x[1].get("timestamp", datetime.now()),
                        reverse=True
                    )

                    # Keep only the most recent
                    to_keep = dict(sorted_generations[:keep_count])
                    to_delete = len(generation_storage) - len(to_keep)

                    generation_storage.clear()
                    generation_storage.update(to_keep)
                    deleted_count += to_delete
                except Exception as clear_error:
                    logger.warning(f"Error clearing file generation results: {clear_error}")

        except ImportError:
            pass

        return {
            "success": True,
            "message": f"Cleared {deleted_count} old results, kept {keep_count} most recent",
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.error(f"Error clearing old results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear old results: {str(e)}")


@router.get("/health")
async def recent_results_health_check():
    """Health check for recent results service"""

    try:
        delta_count = 0
        recon_count = 0
        generation_count = 0

        try:
            from app.services.storage_service import uploaded_files
            # Count delta results by looking for metadata files
            for file_id in uploaded_files.list():
                if file_id.startswith('delta_') and '_metadata' in file_id:
                    delta_count += 1
        except Exception as e:
            logger.warning(f"Error counting delta results: {e}")

        try:
            from app.services.reconciliation_service import optimized_reconciliation_storage
            from app.services.storage_service import uploaded_files
            
            # Count reconciliation results by looking for matched files (one per reconciliation)
            recon_count = 0
            try:
                for file_id in uploaded_files.list():
                    if file_id.startswith('recon_') and '_matched' in file_id:
                        recon_count += 1
            except Exception as e:
                logger.warning(f"Error counting reconciliation results: {e}")
        except ImportError:
            pass

        try:
            from app.routes.file_generator import generation_storage
            generation_count = len(generation_storage)
        except ImportError:
            pass

        return {
            "status": "healthy",
            "service": "recent_results",
            "available_delta_results": delta_count,
            "available_reconciliation_results": recon_count,
            "available_file_generation_results": generation_count,
            "total_results": delta_count + recon_count + generation_count,
            "features": [
                "recent_results_listing",
                "cross_process_aggregation",
                "timestamp_sorting",
                "result_summaries",
                "old_results_cleanup",
                "file_generation_support"
            ]
        }

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "error",
            "service": "recent_results",
            "error": str(e)
        }
