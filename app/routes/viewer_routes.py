# backend/app/routes/viewer_routes.py - Updated to include filename
import io
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, Response, Request
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from app.services.storage_service import uploaded_files

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/files", tags=["viewer"])


# Pydantic models
class FileDataResponse(BaseModel):
    filename: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int
    current_page: int
    page_size: int
    total_pages: int
    file_info: Dict[str, Any]


class UpdateFileDataRequest(BaseModel):
    data: Dict[str, Any]


class CellChange(BaseModel):
    row_index: int
    column_name: str
    new_value: str


class ColumnOperation(BaseModel):
    type: str  # 'add', 'delete', 'rename'
    column_name: str = None
    old_name: str = None
    new_name: str = None


class PatchFileDataRequest(BaseModel):
    cell_changes: List[CellChange] = []
    added_rows: List[Dict[str, Any]] = []
    deleted_row_indices: List[int] = []
    column_operations: List[ColumnOperation] = []


@router.get("/{file_id}/data")
async def get_file_data(
        file_id: str,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(1000, ge=1, le=5000, description="Items per page"),
        search: str = Query("", description="Search term to filter data across all columns"),
        filter_column: str = Query("", description="Column name for column-specific filtering (deprecated - use filter_<column>=values)"),
        filter_values: str = Query("", description="Comma-separated values for column-specific filtering (deprecated)"),
        request: Request = None
):
    """Get paginated file data for the viewer with filename"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(404, f"File {file_id} not found")

        file_data = uploaded_files[file_id]
        df = file_data["data"]
        file_info = file_data["info"]

        # Apply filtering based on type of search
        # Handle multiple column filters from query parameters (filter_<column>=values)
        column_filters_applied = False
        if request:
            for param_name, param_value in request.query_params.items():
                if param_name.startswith('filter_') and param_value.strip():
                    column_name = param_name[7:]  # Remove 'filter_' prefix
                    if column_name in df.columns:
                        values = [v.strip() for v in param_value.split(',') if v.strip()]
                        if values:
                            # Create mask for rows where the specific column contains any of the filter values
                            mask = df[column_name].astype(str).str.lower().isin([v.lower() for v in values])
                            df = df[mask]
                            column_filters_applied = True
                            logger.info(f"Applied filter on column '{column_name}' with values: {values}")
        
        # Fallback to old single filter format for backwards compatibility
        if not column_filters_applied and filter_column.strip() and filter_values.strip():
            # Column-specific filtering (from dropdown selection) - deprecated
            column_name = filter_column.strip()
            values = [v.strip() for v in filter_values.split(',') if v.strip()]
            
            if column_name in df.columns and values:
                # Create mask for rows where the specific column contains any of the filter values
                mask = df[column_name].astype(str).str.lower().isin([v.lower() for v in values])
                df = df[mask]
                column_filters_applied = True
            
        # Search across all columns if no column filters are applied
        if not column_filters_applied and search.strip():
            # Wildcard search across all columns (from search box)
            search_term = search.strip().lower()
            
            # Create a mask for rows that contain the search term in any column
            mask = df.astype(str).apply(
                lambda row: any(search_term in str(cell).lower() for cell in row), axis=1
            )
            df = df[mask]

        # Calculate pagination on filtered data
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        # Get paginated data from filtered dataset
        paginated_df = df.iloc[start_idx:end_idx]

        # Convert to records (list of dicts)
        rows = []
        for _, row in paginated_df.iterrows():
            # Convert row to dict, handling NaN values
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = ""
                elif isinstance(value, (int, float)):
                    row_dict[col] = value
                else:
                    row_dict[col] = str(value)
            rows.append(row_dict)

        return {
            "success": True,
            "message": f"Retrieved {len(rows)} rows from file",
            "data": FileDataResponse(
                filename=file_info.get("filename", "Unknown File"),
                columns=list(df.columns),
                rows=rows,
                total_rows=total_rows,
                current_page=page,
                page_size=page_size,
                total_pages=total_pages,
                file_info={
                    "file_id": file_id,
                    "filename": file_info.get("filename", "Unknown File"),
                    "file_size_mb": file_info.get("file_size_mb", 0),
                    "upload_time": file_info.get("upload_time", ""),
                    "total_rows": total_rows,
                    "total_columns": len(df.columns)
                }
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file data: {e}")
        raise HTTPException(500, f"Failed to retrieve file data: {str(e)}")


@router.get("/{file_id}/info")
async def get_file_info_for_viewer(file_id: str):
    """Get basic file information for the viewer header"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(404, f"File {file_id} not found")

        file_data = uploaded_files[file_id]
        file_info = file_data["info"]
        df = file_data["data"]

        return {
            "success": True,
            "data": {
                "file_id": file_id,
                "filename": file_info.get("filename", "Unknown File"),
                "file_size_mb": file_info.get("file_size_mb", 0),
                "upload_time": file_info.get("upload_time", ""),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "last_modified": file_info.get("last_modified", file_info.get("upload_time", ""))
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file info: {e}")
        raise HTTPException(500, f"Failed to retrieve file info: {str(e)}")


@router.put("/{file_id}/data")
async def update_file_data(file_id: str, request: UpdateFileDataRequest):
    """Update file data from the viewer"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(404, f"File {file_id} not found")

        # Extract updated data
        updated_data = request.data

        if 'rows' not in updated_data or 'columns' not in updated_data:
            raise HTTPException(400, "Missing 'rows' or 'columns' in request data")

        rows = updated_data['rows']
        columns = updated_data['columns']

        # Create new DataFrame from updated data
        df = pd.DataFrame(rows, columns=columns)

        # Update the stored file data
        uploaded_files[file_id]["data"] = df

        # Update file info
        file_info = uploaded_files[file_id]["info"]
        file_info["total_rows"] = len(df)
        file_info["columns"] = list(df.columns)
        file_info["last_modified"] = datetime.utcnow().isoformat()

        logger.info(
            f"Updated file {file_info.get('filename', file_id)} with {len(df)} rows and {len(df.columns)} columns")

        return {
            "success": True,
            "message": "File data updated successfully",
            "data": {
                "filename": file_info.get("filename", "Unknown File"),
                "total_rows": len(df),
                "columns": len(df.columns),
                "last_modified": file_info["last_modified"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating file data: {e}")
        raise HTTPException(500, f"Failed to update file data: {str(e)}")


@router.patch("/{file_id}/data")
async def patch_file_data(file_id: str, request: PatchFileDataRequest):
    """Apply incremental changes to file data - more efficient than full updates"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(404, f"File {file_id} not found")

        # Get the current DataFrame
        df = uploaded_files[file_id]["data"].copy()
        file_info = uploaded_files[file_id]["info"]
        
        changes_applied = {
            "cell_changes": 0,
            "added_rows": 0,
            "deleted_rows": 0,
            "column_operations": 0
        }

        # Apply column operations first (as they affect structure)
        for operation in request.column_operations:
            if operation.type == "add" and operation.column_name:
                if operation.column_name not in df.columns:
                    df[operation.column_name] = ""
                    changes_applied["column_operations"] += 1
                    logger.info(f"Added column '{operation.column_name}' to file {file_id}")
            
            elif operation.type == "delete" and operation.column_name:
                if operation.column_name in df.columns:
                    df = df.drop(columns=[operation.column_name])
                    changes_applied["column_operations"] += 1
                    logger.info(f"Deleted column '{operation.column_name}' from file {file_id}")
            
            elif operation.type == "rename" and operation.old_name and operation.new_name:
                if operation.old_name in df.columns and operation.new_name not in df.columns:
                    df = df.rename(columns={operation.old_name: operation.new_name})
                    changes_applied["column_operations"] += 1
                    logger.info(f"Renamed column '{operation.old_name}' to '{operation.new_name}' in file {file_id}")

        # Apply cell changes
        for change in request.cell_changes:
            if 0 <= change.row_index < len(df) and change.column_name in df.columns:
                df.iloc[change.row_index, df.columns.get_loc(change.column_name)] = change.new_value
                changes_applied["cell_changes"] += 1

        # Delete rows (do this before adding rows to maintain indices)
        if request.deleted_row_indices:
            # Sort in descending order to delete from end to beginning
            sorted_indices = sorted(request.deleted_row_indices, reverse=True)
            valid_indices = [i for i in sorted_indices if 0 <= i < len(df)]
            if valid_indices:
                df = df.drop(df.index[valid_indices]).reset_index(drop=True)
                changes_applied["deleted_rows"] = len(valid_indices)
                logger.info(f"Deleted {len(valid_indices)} rows from file {file_id}")

        # Add new rows
        if request.added_rows:
            new_rows_df = pd.DataFrame(request.added_rows)
            # Ensure all columns exist in the new rows
            for col in df.columns:
                if col not in new_rows_df.columns:
                    new_rows_df[col] = ""
            # Reorder columns to match existing DataFrame
            new_rows_df = new_rows_df.reindex(columns=df.columns, fill_value="")
            df = pd.concat([df, new_rows_df], ignore_index=True)
            changes_applied["added_rows"] = len(request.added_rows)
            logger.info(f"Added {len(request.added_rows)} rows to file {file_id}")

        # Update the stored file data
        uploaded_files[file_id]["data"] = df

        # Update file info
        file_info["total_rows"] = len(df)
        file_info["columns"] = list(df.columns)
        file_info["last_modified"] = datetime.utcnow().isoformat()

        total_changes = sum(changes_applied.values())
        logger.info(f"Applied {total_changes} changes to file {file_info.get('filename', file_id)}: {changes_applied}")

        return {
            "success": True,
            "message": f"Applied {total_changes} changes successfully",
            "data": {
                "filename": file_info.get("filename", "Unknown File"),
                "total_rows": len(df),
                "columns": len(df.columns),
                "last_modified": file_info["last_modified"],
                "changes_applied": changes_applied
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error patching file data: {e}")
        raise HTTPException(500, f"Failed to patch file data: {str(e)}")


@router.get("/{file_id}/download")
async def download_modified_file(
        file_id: str,
        format: str = Query("csv", regex="^(csv|xlsx)$", description="File format")
):
    """Download the modified file in specified format"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(404, f"File {file_id} not found")

        file_data = uploaded_files[file_id]
        df = file_data["data"]
        file_info = file_data["info"]

        # Get base filename without extension
        original_filename = file_info.get("filename", "data")
        base_filename = os.path.splitext(original_filename)[0]

        if format.lower() == "csv":
            # Generate CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            csv_data = output.getvalue()
            output.close()

            filename = f"{base_filename}_modified.csv"

            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        elif format.lower() == "xlsx":
            # Generate Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            output.close()

            filename = f"{base_filename}_modified.xlsx"

            return Response(
                content=excel_data,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        else:
            raise HTTPException(400, f"Unsupported format: {format}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(500, f"Failed to download file: {str(e)}")


@router.get("/{file_id}/stats")
async def get_file_stats(file_id: str):
    """Get basic statistics about the file"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(404, f"File {file_id} not found")

        file_data = uploaded_files[file_id]
        df = file_data["data"]
        file_info = file_data["info"]

        # Calculate basic statistics
        stats = {
            "filename": file_info.get("filename", "Unknown File"),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": int(df.memory_usage(deep=True).sum()),  # Convert to Python int
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},  # Explicit conversion
            "null_counts": {col: int(count) for col, count in df.isnull().sum().items()},
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "text_columns": list(df.select_dtypes(include=['object']).columns),
            "datetime_columns": list(df.select_dtypes(include=['datetime']).columns)
        }

        # Add basic statistics for numeric columns
        numeric_stats = {}
        for col in stats["numeric_columns"]:
            try:
                col_stats = df[col].describe()
                numeric_stats[col] = {
                    "count": int(col_stats["count"]),
                    "mean": float(col_stats["mean"]) if not pd.isna(col_stats["mean"]) else None,
                    "std": float(col_stats["std"]) if not pd.isna(col_stats["std"]) else None,
                    "min": float(col_stats["min"]) if not pd.isna(col_stats["min"]) else None,
                    "max": float(col_stats["max"]) if not pd.isna(col_stats["max"]) else None,
                    "median": float(col_stats["50%"]) if not pd.isna(col_stats["50%"]) else None
                }
            except Exception:
                numeric_stats[col] = {"error": "Could not calculate statistics"}

        stats["numeric_statistics"] = numeric_stats

        return {
            "success": True,
            "message": "File statistics retrieved",
            "data": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file stats: {e}")
        raise HTTPException(500, f"Failed to get file statistics: {str(e)}")


@router.post("/{file_id}/validate")
async def validate_file_data(file_id: str):
    """Validate file data integrity"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(404, f"File {file_id} not found")

        file_data = uploaded_files[file_id]
        df = file_data["data"]
        file_info = file_data["info"]

        validation_results = {
            "filename": file_info.get("filename", "Unknown File"),
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "duplicate_rows": 0,
                "empty_rows": 0,
                "columns_with_all_nulls": []
            }
        }

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        validation_results["summary"]["duplicate_rows"] = int(duplicate_count)
        if duplicate_count > 0:
            validation_results["warnings"].append(f"Found {duplicate_count} duplicate rows")

        # Check for empty rows (all values are null or empty)
        empty_rows = df.isnull().all(axis=1).sum()
        validation_results["summary"]["empty_rows"] = int(empty_rows)
        if empty_rows > 0:
            validation_results["warnings"].append(f"Found {empty_rows} completely empty rows")

        # Check for columns with all null values
        all_null_columns = df.columns[df.isnull().all()].tolist()
        validation_results["summary"]["columns_with_all_nulls"] = all_null_columns
        if all_null_columns:
            validation_results["warnings"].append(f"Columns with all null values: {', '.join(all_null_columns)}")

        # Check data type consistency
        for col in df.columns:
            try:
                # Try to detect if numeric columns have been corrupted with text
                if df[col].dtype == 'object':
                    # Check if column contains mostly numeric values but has some text
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        numeric_count = 0
                        for val in non_null_values:
                            try:
                                float(str(val))
                                numeric_count += 1
                            except ValueError:
                                pass

                        if numeric_count > 0.8 * len(non_null_values) and numeric_count < len(non_null_values):
                            validation_results["warnings"].append(
                                f"Column '{col}' appears to be mostly numeric but contains some text values"
                            )
            except Exception:
                pass

        return {
            "success": True,
            "message": "File validation completed",
            "data": validation_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        raise HTTPException(500, f"Failed to validate file: {str(e)}")
