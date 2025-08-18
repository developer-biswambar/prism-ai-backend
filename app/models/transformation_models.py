# backend/app/models/transformation_models.py - Updated for rule-based transformations

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field


class ColumnType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    DECIMAL = "decimal"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ARRAY = "array"


class TransformationType(str, Enum):
    DIRECT = "direct"
    EXPRESSION = "expression"
    CUSTOM_FUNCTION = "custom_function"
    LLM_TRANSFORM = "llm_transform"
    AGGREGATE = "aggregate"
    STATIC = "static"
    SEQUENCE = "sequence"
    LOOKUP = "lookup"
    CONDITIONAL = "conditional"


class ExpansionType(str, Enum):
    DUPLICATE = "duplicate"
    FIXED_EXPANSION = "fixed_expansion"
    CONDITIONAL_EXPANSION = "conditional_expansion"
    EXPAND_FROM_LIST = "expand_from_list"
    EXPAND_FROM_FILE = "expand_from_file"
    DYNAMIC_EXPANSION = "dynamic_expansion"


class MappingType(str, Enum):
    DIRECT = "direct"
    STATIC = "static"
    DYNAMIC = "dynamic"


class ConditionOperator(str, Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    STARTS_WITH = "startsWith"
    ENDS_WITH = "endsWith"


class SourceFile(BaseModel):
    """
    Source file configuration for transformation processing.
    
    Represents a source file that will be used in the transformation process.
    Each file must have a unique file_id obtained from the file upload endpoint.
    """
    file_id: str = Field(..., description="Unique identifier for the uploaded file", example="file_abc123")
    alias: str = Field(..., description="Alias name for referencing this file in transformations", example="customers")
    purpose: Optional[str] = Field(None, description="Description of the file's purpose in the transformation", example="Primary customer data source")


class DynamicCondition(BaseModel):
    """
    Individual condition for dynamic column mapping.
    
    Defines a conditional rule where if the condition_column meets the specified
    criteria (operator + condition_value), then output_value is returned.
    """
    id: str = Field(..., description="Unique identifier for this condition", example="cond_001")
    condition_column: str = Field(..., description="Source column to evaluate", example="amount")
    operator: ConditionOperator = Field(default=ConditionOperator.EQUALS, description="Comparison operator", example=">=")
    condition_value: str = Field(..., description="Value to compare against", example="1000")
    output_value: str = Field(..., description="Value to output when condition is met (supports expressions)", example="Premium")


class RuleOutputColumn(BaseModel):
    """
    Configuration for output column in transformation rules.
    
    Defines how an output column should be populated using one of three mapping types:
    - Direct: Copy value directly from a source column
    - Static: Use a fixed value or expression for all rows  
    - Dynamic: Use conditional logic to determine the value
    """
    id: str = Field(..., description="Unique identifier for this column mapping", example="col_001")
    name: str = Field(..., description="Output column name", example="customer_tier")
    mapping_type: MappingType = Field(default=MappingType.DIRECT, description="Type of mapping to use")

    # Direct mapping
    source_column: Optional[str] = Field(None, description="Source column for direct mapping", example="customer_id")

    # Static mapping  
    static_value: Optional[str] = Field(None, description="Static value or expression for all rows", example="{first_name} {last_name}")

    # Dynamic mapping
    dynamic_conditions: Optional[List[DynamicCondition]] = Field(default_factory=list, description="List of conditions for dynamic mapping")
    default_value: Optional[str] = Field(None, description="Default value when no dynamic conditions match", example="Standard")


class ConditionBuilder(BaseModel):
    """Helper for building rule conditions"""
    column: Optional[str] = None
    operator: ConditionOperator = ConditionOperator.EQUALS
    value: Optional[str] = None
    logic: str = Field(default="simple", description="simple or advanced")


class TransformationRule(BaseModel):
    """Individual transformation rule with output column definitions"""
    id: str
    name: str = Field(..., description="Rule display name")
    enabled: bool = Field(default=True)
    priority: int = Field(default=0, description="Execution priority (lower = higher priority)")

    # Condition for when this rule applies
    condition: str = Field(default="", description="Condition expression")
    condition_builder: Optional[ConditionBuilder] = Field(default_factory=ConditionBuilder)

    # Output configuration
    output_columns: List[RuleOutputColumn] = Field(default_factory=list)


# Legacy support - keep existing OutputColumn for backwards compatibility
class OutputColumn(BaseModel):
    id: str
    name: str
    type: ColumnType
    format: Optional[str] = None
    description: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    default_value: Optional[Any] = None
    required: bool = Field(default=True)


class OutputDefinition(BaseModel):
    columns: List[OutputColumn]
    format: str = "csv"  # csv, excel, json, xml, fixed_width
    delimiter: Optional[str] = ","
    include_headers: bool = True


class ExpansionStrategy(BaseModel):
    type: ExpansionType
    config: Dict[str, Any]


class RowGenerationRule(BaseModel):
    """Legacy row generation rule - kept for backwards compatibility"""
    id: str
    name: str
    type: str = "expand"
    enabled: bool = True
    condition: Optional[str] = None
    strategy: ExpansionStrategy
    priority: int = 0  # For ordering multiple rules


class TransformationConfig(BaseModel):
    type: TransformationType
    config: Dict[str, Any]


class ColumnMapping(BaseModel):
    """Legacy column mapping - kept for backwards compatibility"""
    id: str = Field(default_factory=lambda: f"map_{datetime.now().timestamp()}")
    target_column: str
    mapping_type: TransformationType
    source: Optional[str] = None  # For direct mapping
    transformation: Optional[TransformationConfig] = None
    enabled: bool = True


class ValidationRule(BaseModel):
    id: str
    name: str
    type: str  # required, format, range, custom
    config: Dict[str, Any]
    error_message: str
    severity: str = "error"  # error, warning


# New comprehensive transformation configuration
class NewTransformationConfig(BaseModel):
    """Updated transformation configuration with rule-based structure"""
    name: Optional[str] = Field(None, description="Transformation name")
    description: Optional[str] = Field(None, description="Transformation description")

    # Source configuration
    source_files: List[SourceFile] = Field(default_factory=list)

    # New rule-based transformation
    row_generation_rules: List[TransformationRule] = Field(default_factory=list)

    # Output configuration
    merge_datasets: bool = Field(default=True, description="Merge all rule outputs into single dataset")

    # Validation
    validation_rules: Optional[List[ValidationRule]] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# Legacy transformation config for backwards compatibility
class LegacyTransformationConfig(BaseModel):
    """Legacy transformation config structure"""
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    source_files: List[SourceFile]
    output_definition: OutputDefinition
    row_generation_rules: List[RowGenerationRule] = []
    column_mappings: List[ColumnMapping] = []
    validation_rules: List[ValidationRule] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TransformationRequest(BaseModel):
    """
    Request model for processing data transformations.
    
    Contains all the information needed to execute a transformation including
    source files, transformation rules, and processing options.
    """
    process_name: str = Field(..., description="Name of the transformation process", example="Customer Data Standardization")
    description: Optional[str] = Field(None, description="Description of what this transformation does", example="Standardize customer data and calculate totals")
    source_files: List[SourceFile] = Field(..., description="List of source files to transform")
    transformation_config: Union[NewTransformationConfig, LegacyTransformationConfig, Dict[str, Any]] = Field(..., description="Transformation configuration with rules and mappings")
    preview_only: bool = Field(default=False, description="If true, only process a small sample for preview")
    row_limit: Optional[int] = Field(None, description="Maximum number of rows to process (for preview mode)", example=10)


class TransformationResult(BaseModel):
    """
    Result model for transformation processing.
    
    Contains the results and metadata from a completed transformation operation,
    including performance metrics, validation results, and any errors or warnings.
    """
    success: bool = Field(..., description="Whether the transformation completed successfully")
    transformation_id: str = Field(..., description="Unique identifier for this transformation", example="transform_abc123")
    total_input_rows: int = Field(..., description="Total number of input rows processed", example=1000)
    total_output_rows: int = Field(..., description="Total number of output rows generated", example=1000)
    processing_time_seconds: float = Field(..., description="Time taken to process the transformation", example=2.456)
    validation_summary: Dict[str, Any] = Field(..., description="Summary of validation results and processing statistics")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings encountered during processing")
    errors: List[str] = Field(default_factory=list, description="Critical errors encountered during processing")
    preview_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sample of output data (only included in preview mode)")


class DatasetInfo(BaseModel):
    """Information about a generated dataset"""
    name: str
    rule_name: Optional[str] = None
    row_count: int
    column_count: int
    file_id: Optional[str] = None  # ID for accessing via viewer


class TransformationSummary(BaseModel):
    """Summary of transformation results"""
    transformation_id: str
    name: str
    created_at: datetime
    total_input_rows: int
    total_output_rows: int
    processing_time_seconds: float
    datasets: List[DatasetInfo]
    merge_enabled: bool


class TransformationTemplate(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    category: str
    tags: List[str] = []
    source_requirements: Dict[str, Any] = {}  # Expected source columns/structure

    # Support both old and new config structures
    transformation_config: Union[NewTransformationConfig, LegacyTransformationConfig]

    # Legacy fields for backwards compatibility
    output_definition: Optional[OutputDefinition] = None
    row_generation_rules: Optional[List[RowGenerationRule]] = []
    column_mappings: Optional[List[ColumnMapping]] = []
    validation_rules: Optional[List[ValidationRule]] = []

    sample_input: Optional[Dict[str, Any]] = None
    sample_output: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class LLMAssistanceRequest(BaseModel):
    assistance_type: str  # suggest_mappings, generate_transformation, validate_output
    source_columns: Optional[Dict[str, List[str]]] = None
    target_schema: Optional[OutputDefinition] = None
    transformation_rules: Optional[List[TransformationRule]] = None
    output_columns: Optional[List[RuleOutputColumn]] = None
    sample_data: Optional[List[Dict[str, Any]]] = None
    examples: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None


class LLMAssistanceResponse(BaseModel):
    success: bool = True
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    generated_rules: Optional[List[TransformationRule]] = None
    generated_columns: Optional[List[RuleOutputColumn]] = None


class ProcessingMetrics(BaseModel):
    """Metrics from transformation processing"""
    start_time: datetime
    end_time: datetime
    processing_time_seconds: float

    input_metrics: Dict[str, Any]  # row counts, file sizes, etc.
    output_metrics: Dict[str, Any]  # generated rows, columns, etc.

    performance_metrics: Dict[str, Any]  # memory usage, etc.
    quality_metrics: Dict[str, Any]  # validation results, etc.


class TransformationError(BaseModel):
    """Error information from transformation processing"""
    error_type: str
    error_message: str
    error_location: Optional[str] = None  # rule name, column name, etc.
    suggested_fix: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ExpressionValidationResult(BaseModel):
    """Result of validating an expression"""
    is_valid: bool
    error_message: Optional[str] = None
    suggested_correction: Optional[str] = None
    available_columns: List[str] = Field(default_factory=list)


class RuleExecutionResult(BaseModel):
    """Result of executing a single transformation rule"""
    rule_id: str
    rule_name: str
    success: bool
    rows_processed: int
    rows_generated: int
    execution_time_seconds: float
    errors: List[TransformationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
