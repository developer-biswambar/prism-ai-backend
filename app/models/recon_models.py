from typing import List, Optional, Union, Dict, Any

from pydantic import BaseModel, Field


# Pydantic models for request/response
class PatternCondition(BaseModel):
    operator: Optional[str] = None
    pattern: Optional[str] = None
    patterns: Optional[List[str]] = None
    conditions: Optional[List['PatternCondition']] = None


class ExtractRule(BaseModel):
    ResultColumnName: str
    SourceColumn: str
    MatchType: str
    Conditions: Optional[PatternCondition] = None
    # Legacy support
    Patterns: Optional[List[str]] = None


class FilterRule(BaseModel):
    ColumnName: str
    MatchType: str
    Value: Union[str, int, float]


class FileRule(BaseModel):
    Name: str
    SheetName: Optional[str] = None  # For Excel files
    Extract: Optional[List[ExtractRule]] = []  # Made optional with default empty list
    Filter: Optional[List[FilterRule]] = []  # Made optional with default empty list


class ReconciliationRule(BaseModel):
    LeftFileColumn: str
    RightFileColumn: str
    MatchType: str  # Now supports: "equals", "tolerance", "date_equals"
    ToleranceValue: Optional[float] = None


class ColumnSelectionConfig(BaseModel):
    """Configuration for column selection in reconciliation results"""
    file_a_columns: Optional[List[str]] = Field(None, description="Columns to include from File A")
    file_b_columns: Optional[List[str]] = Field(None, description="Columns to include from File B")
    include_mandatory: bool = Field(True, description="Always include columns used in reconciliation rules")
    output_format: str = Field("standard", description="Output format: standard, summary, detailed")


class OptimizedRulesConfig(BaseModel):
    """Enhanced rules configuration with column selection"""
    Files: List[FileRule]
    ReconciliationRules: List[ReconciliationRule]
    ColumnSelection: Optional[ColumnSelectionConfig] = None
    ProcessingOptions: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "batch_size": 1000,
            "use_parallel_processing": True,
            "memory_optimization": True
        }
    )


class ReconciliationSummary(BaseModel):
    total_records_file_a: int
    total_records_file_b: int
    matched_records: int
    unmatched_file_a: int
    unmatched_file_b: int
    match_percentage: float
    processing_time_seconds: float


class DataQualityMetrics(BaseModel):
    """Additional data quality metrics for reconciliation"""
    file_a_match_rate: float = Field(description="Percentage of File A records that found matches")
    file_b_match_rate: float = Field(description="Percentage of File B records that found matches")
    overall_completeness: float = Field(description="Overall data completeness percentage")
    duplicate_records_a: int = Field(default=0, description="Duplicate records found in File A")
    duplicate_records_b: int = Field(default=0, description="Duplicate records found in File B")
    data_consistency_score: float = Field(default=100.0, description="Data consistency score")


class EnhancedReconciliationSummary(ReconciliationSummary):
    """Enhanced summary with additional metrics"""
    data_quality: Optional[DataQualityMetrics] = None
    column_statistics: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class ReconciliationResponse(BaseModel):
    success: bool
    summary: Union[ReconciliationSummary, EnhancedReconciliationSummary]
    reconciliation_id: str
    errors: List[str] = []
    warnings: List[str] = []
    processing_info: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "optimization_used": True,
            "hash_based_matching": True,
            "vectorized_extraction": True
        }
    )


class PaginatedResultsRequest(BaseModel):
    """Request model for paginated results"""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(1000, ge=1, le=10000, description="Records per page")
    result_type: str = Field("all", description="Type of results: all, matched, unmatched_a, unmatched_b")
    include_columns: Optional[List[str]] = Field(None, description="Specific columns to include")


class DownloadRequest(BaseModel):
    """Request model for downloading results"""
    format: str = Field("excel", description="Download format: excel, csv, json")
    result_type: str = Field("all", description="Type of results to download")
    compress: bool = Field(True, description="Whether to compress the download")
    include_summary: bool = Field(True, description="Include summary sheet/data")


class ReconciliationStatus(BaseModel):
    """Status model for tracking reconciliation progress"""
    reconciliation_id: str
    status: str = Field(description="Status: pending, processing, completed, failed")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    current_step: str = Field(description="Current processing step")
    estimated_completion_time: Optional[str] = None
    error_message: Optional[str] = None


class BulkReconciliationRequest(BaseModel):
    """Request model for bulk reconciliation operations"""
    reconciliation_configs: List[OptimizedRulesConfig]
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    priority: str = Field("normal", description="Processing priority: low, normal, high")


# Performance optimization models
class ProcessingMetrics(BaseModel):
    """Metrics for monitoring reconciliation performance"""
    total_processing_time: float
    file_reading_time: float
    extraction_time: float
    filtering_time: float
    reconciliation_time: float
    result_generation_time: float
    memory_usage_mb: float
    records_processed_per_second: float


class CacheConfig(BaseModel):
    """Configuration for caching optimization"""
    enable_pattern_caching: bool = Field(True, description="Cache compiled regex patterns")
    enable_result_caching: bool = Field(True, description="Cache intermediate results")
    cache_ttl_seconds: int = Field(3600, description="Cache time-to-live in seconds")
    max_cache_size_mb: int = Field(512, description="Maximum cache size in MB")


# AI Configuration Generation Models
class ReconciliationConfigRequest(BaseModel):
    """
    Request model for AI-powered reconciliation configuration generation.
    
    Similar to transformation requirements, this allows users to specify
    their reconciliation needs in natural language.
    """
    requirements: str = Field(..., description="Natural language description of reconciliation requirements", 
                             example="Reconcile bank statements with internal transaction records based on transaction ID and amount")
    source_files: List[Dict[str, Any]] = Field(..., description="Information about the two files to be reconciled")


class ReconciliationConfigResponse(BaseModel):
    """
    Response model for AI-generated reconciliation configuration.
    
    Returns a complete reconciliation configuration that can be used
    directly in the reconciliation process.
    """
    success: bool = Field(..., description="Whether configuration generation was successful")
    message: str = Field(..., description="Status message")
    data: Dict[str, Any] = Field(..., description="Generated reconciliation configuration")


class SourceFileInfo(BaseModel):
    """
    Information about a source file for reconciliation configuration generation.
    
    Provides context to the AI about the structure and content of files
    to be reconciled.
    """
    file_id: str = Field(..., description="Unique identifier for the uploaded file", example="file_abc123")
    filename: str = Field(..., description="Name of the uploaded file", example="bank_statements.csv")
    columns: List[str] = Field(..., description="List of column names in the file", 
                              example=["transaction_id", "date", "amount", "description"])
    totalRows: int = Field(..., description="Total number of rows in the file", example=1000)
    role: Optional[str] = Field(None, description="Role of this file in reconciliation", example="primary")
    label: Optional[str] = Field(None, description="Human-readable label for this file", example="Bank Statements")


class ReconciliationTemplateRequest(BaseModel):
    """
    Request model for creating reconciliation templates.
    
    Allows saving frequently used reconciliation configurations
    for reuse across similar data sets.
    """
    name: str = Field(..., description="Template name", example="Bank vs Internal Reconciliation")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(..., description="Template category", example="financial")
    tags: List[str] = Field(default_factory=list, description="Template tags for organization")
    configuration: Dict[str, Any] = Field(..., description="Reconciliation configuration to save as template")
    is_public: bool = Field(default=False, description="Whether template is publicly available")


class ReconciliationTemplate(BaseModel):
    """
    Model for reconciliation templates.
    
    Stores reusable reconciliation configurations that can be applied
    to similar data sets with minimal modification.
    """
    id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    configuration: Dict[str, Any] = Field(..., description="Reconciliation configuration")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    usage_count: int = Field(default=0, description="Number of times template has been used")
    is_public: bool = Field(default=False, description="Whether template is publicly available")


class ValidationResult(BaseModel):
    """
    Result of reconciliation configuration validation.
    
    Provides feedback on whether a configuration is valid and
    suggests improvements if needed.
    """
    is_valid: bool = Field(..., description="Whether the configuration is valid")
    errors: List[str] = Field(default_factory=list, description="Configuration errors")
    warnings: List[str] = Field(default_factory=list, description="Configuration warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    missing_columns: List[str] = Field(default_factory=list, description="Referenced columns not found in source files")
    confidence_score: float = Field(default=1.0, description="Confidence in configuration validity (0-1)")


# Update forward references
PatternCondition.model_rebuild()
OptimizedRulesConfig.model_rebuild()
