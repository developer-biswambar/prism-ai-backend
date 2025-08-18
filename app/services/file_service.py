# backend/app/services/file_service.py
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import aiofiles
import pandas as pd
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FileProcessingService:
    def __init__(self):
        self.temp_dir = os.getenv("TEMP_DIR", "./temp")
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # Default 100MB
        self.allowed_extensions = ['.csv', '.xlsx', '.xls']
        # REMOVED: max_rows - No row limit

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        # In-memory storage for file metadata (use database in production)
        self.file_registry: Dict[str, dict] = {}
        self.file_data_cache: Dict[str, pd.DataFrame] = {}

    async def upload_and_process_file(self, file: UploadFile) -> dict:
        """
        Upload and process Excel/CSV file with no row limit
        """
        try:
            # Validate file
            self._validate_file(file)

            # Generate unique file ID
            file_id = str(uuid.uuid4())

            # Save file temporarily
            file_path = await self._save_temp_file(file, file_id)

            # Process file and extract metadata
            df, file_info = await self._process_file(file_path, file_id, file.filename)

            # Cache the dataframe for processing
            self.file_data_cache[file_id] = df

            # Store metadata
            self.file_registry[file_id] = file_info

            # Clean up temp file
            os.unlink(file_path)

            logger.info(f"Successfully processed file {file.filename} with ID {file_id} - {len(df)} rows")
            return file_info

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file - no row limit"""

        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in self.allowed_extensions:
            raise ValueError(f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}")

        # Check file size (this is approximate, actual size check happens during upload)
        if hasattr(file, 'size') and file.size > self.max_file_size:
            raise ValueError(f"File too large. Maximum size: {self.max_file_size / (1024 * 1024):.1f}MB")

    async def _save_temp_file(self, file: UploadFile, file_id: str) -> str:
        """Save uploaded file to temporary location"""

        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_path = os.path.join(self.temp_dir, f"{file_id}{file_ext}")

        async with aiofiles.open(temp_path, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)

        return temp_path

    async def _process_file(self, file_path: str, file_id: str, filename: str) -> Tuple[pd.DataFrame, dict]:
        """Process the uploaded file and extract metadata - no row limit"""

        try:
            # Determine file type
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext == '.csv':
                # Read CSV with optimized settings for large files
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                file_type = "CSV"
            elif file_ext in ['.xlsx', '.xls']:
                # Read Excel with optimized settings for large files
                df = pd.read_excel(file_path, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
                file_type = "EXCEL"
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")

            # REMOVED: Row count validation
            # Log info about large files
            if len(df) > 10000:
                logger.info(f"Processing large file: {filename} with {len(df)} rows")

            # Clean column names
            df.columns = df.columns.astype(str).str.strip()

            # Get file size
            file_size = os.path.getsize(file_path)

            # Create file info dictionary
            file_info = {
                "file_id": file_id,
                "filename": filename,
                "file_type": file_type,
                "size": file_size,
                "total_rows": len(df),
                "columns": list(df.columns),
                "upload_time": datetime.utcnow().isoformat()
            }

            logger.info(f"Processed file: {filename}, Rows: {len(df)}, Columns: {len(df.columns)}")

            return df, file_info

        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise

    def get_file_info(self, file_id: str) -> Optional[dict]:
        """Get file information by ID"""
        return self.file_registry.get(file_id)

    def get_file_data(self, file_id: str) -> Optional[pd.DataFrame]:
        """Get file data by ID"""
        return self.file_data_cache.get(file_id)

    def get_column_sample(self, file_id: str, column_name: str, sample_size: int = 10) -> List[str]:
        """Get sample data from a specific column"""

        df = self.get_file_data(file_id)
        if df is None:
            raise ValueError(f"File with ID {file_id} not found")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in file")

        # Get non-null sample values
        column_data = df[column_name].dropna().astype(str)

        if len(column_data) == 0:
            return []

        # Return sample
        sample = column_data.head(sample_size).tolist()
        return sample

    def validate_extraction_request(self, file_id: str, source_column: str) -> bool:
        """Validate that extraction request is valid for the file"""

        file_info = self.get_file_info(file_id)
        if not file_info:
            raise ValueError(f"File with ID {file_id} not found")

        if source_column not in file_info["columns"]:
            raise ValueError(
                f"Column '{source_column}' not found in file. Available columns: {', '.join(file_info['columns'])}")

        return True

    def prepare_data_for_extraction(self, file_id: str, source_column: str) -> List[str]:
        """Prepare data for extraction by cleaning and formatting"""

        df = self.get_file_data(file_id)
        if df is None:
            raise ValueError(f"File with ID {file_id} not found")

        # Get the source column data
        column_data = df[source_column].fillna('').astype(str)

        # Clean and filter the data
        cleaned_data = []
        for text in column_data:
            # Remove excessive whitespace
            text = ' '.join(text.split())

            # Skip empty or very short entries
            if len(text.strip()) > 3:
                cleaned_data.append(text.strip())
            else:
                cleaned_data.append('')  # Keep empty to maintain index alignment

        return cleaned_data

    def cleanup_file(self, file_id: str) -> bool:
        """Remove file from cache and registry"""
        try:
            if file_id in self.file_registry:
                del self.file_registry[file_id]

            if file_id in self.file_data_cache:
                del self.file_data_cache[file_id]

            logger.info(f"Cleaned up file with ID {file_id}")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up file {file_id}: {str(e)}")
            return False

    def get_all_files(self) -> List[dict]:
        """Get all uploaded files"""
        return list(self.file_registry.values())


# Singleton instance
file_service = FileProcessingService()
