# backend/app/models/schemas.py
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field, validator


class FileType(str, Enum):
    EXCEL = "excel"
    CSV = "csv"


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileUploadRequest(BaseModel):
    filename: str
    file_type: FileType
    size: int


class ExtractionRequest(BaseModel):
    file_id: str
    extraction_prompt: str = Field(..., min_length=10, max_length=1000)
    source_column: str = Field(..., min_length=1)
    target_columns: Optional[List[str]] = None
    batch_size: Optional[int] = Field(default=100, ge=1, le=1000)

    @validator('extraction_prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Extraction prompt cannot be empty')
        return v.strip()


class ExtractedField(BaseModel):
    field_name: str
    field_value: Optional[Union[str, float, int]]
    confidence: Optional[float] = Field(ge=0.0, le=1.0)
    extraction_method: Optional[str] = None  # "llm", "regex", "manual"


class ExtractionRow(BaseModel):
    row_index: int
    original_text: str
    extracted_fields: List[ExtractedField]
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


class ExtractionResult(BaseModel):
    extraction_id: str
    file_id: str
    status: ExtractionStatus
    total_rows: int
    processed_rows: int
    successful_extractions: int
    failed_extractions: int
    overall_confidence: Optional[float]
    processing_time: float
    extracted_columns: List[str]
    sample_results: List[ExtractionRow]
    created_at: datetime
    completed_at: Optional[datetime]


class FileInfo(BaseModel):
    file_id: str
    filename: str
    file_type: FileType
    size: int
    total_rows: int
    columns: List[str]
    upload_time: datetime


class ValidationResult(BaseModel):
    field_name: str
    is_valid: bool
    validation_message: Optional[str]
    suggested_correction: Optional[str]


class FinancialPattern(BaseModel):
    pattern_name: str
    regex_pattern: str
    description: str
    examples: List[str]


# Response Models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error_details: Optional[Dict[str, Any]] = None


class ExtractionResponse(APIResponse):
    data: Optional[ExtractionResult] = None


class FileUploadResponse(APIResponse):
    data: Optional[FileInfo] = None


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    openai_status: bool
