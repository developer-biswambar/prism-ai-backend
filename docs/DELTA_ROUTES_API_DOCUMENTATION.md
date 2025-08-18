# Delta Routes API Documentation

## Table of Contents
1. [API Overview](#api-overview)
2. [Endpoint Reference](#endpoint-reference)
3. [Data Models](#data-models)
4. [Processing Logic](#processing-logic)
5. [Error Handling](#error-handling)
6. [Performance Features](#performance-features)

## API Overview

The Delta Routes API provides comprehensive file comparison capabilities for financial data processing. It enables users to compare two versions of data files to identify unchanged, amended, deleted, and newly added records through sophisticated matching algorithms.

### Key Features
- **AI-Powered Configuration**: Generate delta rules from natural language requirements
- **Composite Key Matching**: Multi-column key generation with intelligent normalization
- **Advanced Comparison**: Multiple match types (exact, tolerance, date, case-insensitive)
- **Data Filtering**: Pre-processing filters with date-aware handling
- **Result Categorization**: 5 distinct result types with detailed change tracking
- **Streaming Downloads**: Optimized file downloads for large datasets
- **Automatic Persistence**: Results auto-saved to server storage

### Router Configuration
```python
router = APIRouter(prefix="/delta", tags=["delta-generation"])
```

## Endpoint Reference

### 1. AI Configuration Generation
```
POST /delta/generate-config/
```

**Purpose**: Generate delta configuration using AI based on natural language requirements

**Request Body**:
```json
{
    "requirements": "Compare transaction files by ID and amount with 0.01 tolerance",
    "source_files": [
        {
            "filename": "old_transactions.csv",
            "columns": ["id", "amount", "date", "status"],
            "totalRows": 1500,
            "file_id": "file123"
        },
        {
            "filename": "new_transactions.csv", 
            "columns": ["id", "amount", "date", "status"],
            "totalRows": 1600,
            "file_id": "file124"
        }
    ]
}
```

**Response**:
```json
{
    "success": true,
    "message": "Delta configuration generated successfully",
    "data": {
        "Files": [
            {
                "Name": "FileA",
                "Extract": [],
                "Filter": []
            },
            {
                "Name": "FileB",
                "Extract": [],
                "Filter": []
            }
        ],
        "KeyRules": [
            {
                "LeftFileColumn": "id",
                "RightFileColumn": "id",
                "MatchType": "equals",
                "ToleranceValue": null,
                "IsKey": true
            }
        ],
        "ComparisonRules": [
            {
                "LeftFileColumn": "amount",
                "RightFileColumn": "amount",
                "MatchType": "numeric_tolerance",
                "ToleranceValue": 0.01,
                "IsKey": false
            }
        ],
        "selected_columns_file_a": ["id", "amount", "date", "status"],
        "selected_columns_file_b": ["id", "amount", "date", "status"]
    }
}
```

**AI Prompt Structure**:
The endpoint uses sophisticated prompt engineering to generate configurations:

```python
prompt = f"""
You are an expert financial data delta generation configuration generator.
Based on the user requirements and source file information, generate a JSON 
configuration for delta analysis between two data files.

Source Files Available:
{files_info}

User Requirements:
{requirements}

Configuration Rules:
1. ONLY use column names that exist in the source files
2. KeyRules define composite key for record matching - REQUIRED
3. ComparisonRules define optional fields to compare - optional
4. For exact matches, use MatchType "equals" with ToleranceValue null
5. For numeric tolerance, use MatchType "numeric_tolerance" with appropriate value
...
"""
```

### 2. Delta Processing
```
POST /delta/process/
```

**Purpose**: Main delta generation endpoint with comprehensive file comparison

**Request Model**: `JSONDeltaRequest`
```json
{
    "process_type": "delta",
    "process_name": "Transaction Delta Analysis",
    "user_requirements": "Compare transaction files for changes",
    "files": [
        {
            "file_id": "file123",
            "role": "file_0",
            "label": "Old Transactions"
        },
        {
            "file_id": "file124", 
            "role": "file_1",
            "label": "New Transactions"
        }
    ],
    "delta_config": {
        "Files": [...],
        "KeyRules": [...],
        "ComparisonRules": [...],
        "selected_columns_file_a": [...],
        "selected_columns_file_b": [...],
        "file_filters": {
            "file_0": [
                {
                    "column": "status",
                    "values": ["Active", "Pending"]
                }
            ],
            "file_1": [
                {
                    "column": "status",
                    "values": ["Active", "Pending"]
                }
            ]
        }
    }
}
```

**Response Model**: `DeltaResponse`
```json
{
    "success": true,
    "summary": {
        "total_records_file_a": 1500,
        "total_records_file_b": 1600,
        "unchanged_records": 1200,
        "amended_records": 250,
        "deleted_records": 50,
        "newly_added_records": 100,
        "processing_time_seconds": 2.345
    },
    "delta_id": "delta_20240315_143022_789",
    "errors": [],
    "warnings": ["Some result saves failed: unchanged"]
}
```

**Processing Flow**:
1. **File Retrieval**: Load files from storage service by ID
2. **Rule Validation**: Verify all rule columns exist in source files
3. **Data Filtering**: Apply pre-processing filters if provided
4. **Composite Key Creation**: Generate matching keys from multiple columns
5. **Delta Analysis**: Compare records and categorize changes
6. **Result Storage**: Auto-save all 5 result types to server
7. **Response Generation**: Return summary statistics and delta ID

### 3. Results Retrieval
```
GET /delta/results/{delta_id}
```

**Parameters**:
- `delta_id`: Unique identifier for delta operation
- `result_type`: Optional, one of "all", "unchanged", "amended", "deleted", "newly_added", "all_changes"
- `page`: Page number for pagination (default: 1)
- `page_size`: Records per page (default: 1000)

**Response**:
```json
{
    "delta_id": "delta_20240315_143022_789",
    "timestamp": "2024-03-15T14:30:22.789Z",
    "row_counts": {
        "unchanged": 1200,
        "amended": 250,
        "deleted": 50,
        "newly_added": 100
    },
    "filters_applied": {
        "file_0": [{"column": "status", "values": ["Active"]}],
        "file_1": [{"column": "status", "values": ["Active"]}]
    },
    "pagination": {
        "page": 1,
        "page_size": 1000,
        "start_index": 0
    },
    "unchanged": [...],
    "amended": [...],
    "deleted": [...],
    "newly_added": [...]
}
```

### 4. File Downloads
```
GET /delta/download/{delta_id}
```

**Parameters**:
- `delta_id`: Unique identifier for delta operation
- `format`: File format ("csv" or "excel", default: "csv")
- `result_type`: Type of results to download (default: "all")
- `compress`: Enable compression (default: true)

**CSV Download**:
- Single result type: Returns CSV with specified records
- "all" result type: Returns combined CSV with Delta_Category column

**Excel Download**:
- Multiple sheets for different result types
- Summary sheet with statistics and percentages
- Optimized for large file streaming

**Streaming Response**:
```python
return StreamingResponse(
    output,
    media_type=media_type,
    headers={"Content-Disposition": f"attachment; filename={filename}"}
)
```

### 5. Summary Statistics
```
GET /delta/results/{delta_id}/summary
```

**Response**:
```json
{
    "delta_id": "delta_20240315_143022_789",
    "timestamp": "2024-03-15T14:30:22.789Z",
    "filters_applied": {...},
    "summary": {
        "total_records_compared": 1600,
        "unchanged_records": 1200,
        "amended_records": 250,
        "deleted_records": 50,
        "newly_added_records": 100,
        "change_metrics": {
            "stability_percentage": 75.0,
            "amendment_rate": 15.63,
            "deletion_rate": 3.13,
            "addition_rate": 6.25,
            "overall_change_rate": 25.0
        }
    }
}
```

### 6. Resource Cleanup
```
DELETE /delta/results/{delta_id}
```

**Purpose**: Remove delta results from memory to free resources

**Response**:
```json
{
    "success": true,
    "message": "Delta results delta_20240315_143022_789 deleted successfully"
}
```

### 7. Health Check
```
GET /delta/health
```

**Response**:
```json
{
    "status": "healthy",
    "service": "delta_generation",
    "llm_service": {
        "provider": "openai",
        "model": "gpt-4-turbo",
        "available": true
    },
    "active_deltas": 5,
    "features": [
        "composite_key_matching",
        "numeric_normalization", 
        "optional_field_comparison",
        "change_categorization",
        "amendment_tracking",
        "column_selection",
        "data_filtering",
        "date_aware_filtering",
        "paginated_results",
        "streaming_downloads",
        "json_input_support",
        "file_id_retrieval",
        "date_matching",
        "case_insensitive_matching",
        "numeric_tolerance_matching",
        "auto_comparison_rules",
        "ai_configuration_generation"
    ]
}
```

## Data Models

### Core Models

#### DeltaKeyRule
```python
class DeltaKeyRule(BaseModel):
    LeftFileColumn: str                    # Column from older file
    RightFileColumn: str                   # Column from newer file  
    MatchType: str = "equals"              # Matching algorithm
    ToleranceValue: Optional[float] = None # Numeric tolerance
    IsKey: bool = True                     # Always True for key rules
```

**Supported Match Types**:
- `equals`: Exact string matching with numeric normalization
- `case_insensitive`: Case-insensitive text matching
- `numeric_tolerance`: Absolute numeric tolerance matching
- `date_equals`: Date matching with format normalization

#### DeltaComparisonRule
```python
class DeltaComparisonRule(BaseModel):
    LeftFileColumn: str                    # Column from older file
    RightFileColumn: str                   # Column from newer file
    MatchType: str = "equals"              # Comparison algorithm
    ToleranceValue: Optional[float] = None # Numeric tolerance
    IsKey: bool = False                    # Always False for comparison
```

#### DeltaFileRule
```python
class DeltaFileRule(BaseModel):
    Name: str                                        # FileA (older), FileB (newer)
    SheetName: Optional[str] = None                  # Excel sheet name
    Extract: Optional[List[Dict[str, Any]]] = []     # Extraction rules
    Filter: Optional[List[Dict[str, Any]]] = []      # Filter rules
```

#### DeltaGenerationConfig
```python
class DeltaGenerationConfig(BaseModel):
    Files: List[Dict[str, Any]]                      # File configurations
    KeyRules: List[Dict[str, Any]]                   # Composite key rules
    ComparisonRules: Optional[List[Dict[str, Any]]] = []  # Optional comparison
    selected_columns_file_a: Optional[List[str]] = None   # Column selection A
    selected_columns_file_b: Optional[List[str]] = None   # Column selection B
    file_filters: Optional[Dict[str, List[Dict[str, Any]]]] = None  # Pre-filters
    user_requirements: Optional[str] = None          # Natural language requirements
    files: Optional[List[DeltaFileReference]] = None # File metadata
```

### Response Models

#### DeltaSummary
```python
class DeltaSummary(BaseModel):
    total_records_file_a: int            # Total records in older file
    total_records_file_b: int            # Total records in newer file
    unchanged_records: int               # Records with no changes
    amended_records: int                 # Records with field changes
    deleted_records: int                 # Records removed in newer file
    newly_added_records: int             # Records added in newer file  
    processing_time_seconds: float       # Processing duration
```

#### DeltaResponse
```python
class DeltaResponse(BaseModel):
    success: bool                        # Operation success status
    summary: DeltaSummary                # Processing summary
    delta_id: str                        # Unique operation identifier
    errors: List[str] = []               # Processing errors
    warnings: List[str] = []             # Non-critical warnings
```

## Processing Logic

### DeltaProcessor Class

The `DeltaProcessor` class handles the core delta generation logic with these key components:

#### 1. File Loading and Validation
```python
def read_file_from_storage(self, file_id: str):
    """Read file from storage service with error handling"""
    from app.services.storage_service import uploaded_files
    
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
    
    return uploaded_files[file_id]["data"]  # Return DataFrame directly

def validate_rules_against_columns(self, df: pd.DataFrame, columns_needed: List[str], file_name: str):
    """Validate all required columns exist in DataFrame"""
    errors = []
    df_columns = df.columns.tolist()
    
    for column in columns_needed:
        if column not in df_columns:
            errors.append(f"Column '{column}' not found in {file_name}. Available: {df_columns}")
    
    return errors
```

#### 2. Data Filtering System
```python
def apply_filters(self, df: pd.DataFrame, filters: List[Dict[str, Any]], file_label: str):
    """Apply comprehensive filters with date-aware processing"""
    if not filters:
        return df
    
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    for filter_config in filters:
        column = filter_config.get('column')
        values = filter_config.get('values', [])
        
        if not column or not values or column not in filtered_df.columns:
            continue
        
        # Create mask for matching values with date support
        mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        
        for value in values:
            column_mask = self.create_value_mask(filtered_df[column], value)
            mask = mask | column_mask
        
        filtered_df = filtered_df[mask]
    
    return filtered_df
```

**Advanced Value Matching**:
```python
def create_value_mask(self, series: pd.Series, target_value: str):
    """Create boolean mask with intelligent matching"""
    # Handle NaN values
    if target_value.lower() in ['nan', 'null', '']:
        return series.isna()
    
    # Try exact string match first
    string_mask = series.astype(str).str.strip() == str(target_value).strip()
    
    # Try date parsing if exact match fails
    if not string_mask.any():
        date_mask = self.create_date_mask(series, target_value)
        if date_mask.any():
            return date_mask
    
    return string_mask
```

#### 3. Composite Key Generation
```python
def create_composite_key(self, df: pd.DataFrame, key_columns: List[str], rules: List[DeltaKeyRule]):
    """Generate composite keys with intelligent normalization"""
    keys = []
    
    for i, col in enumerate(key_columns):
        if i < len(rules):
            rule = rules[i]
            series_data = df[col].fillna('__NULL__')
            
            if rule.MatchType == "case_insensitive":
                normalized_series = series_data.astype(str).str.lower().str.strip()
            elif rule.MatchType == "numeric_tolerance":
                normalized_series = self.normalize_key_values(series_data)
            else:  # equals
                normalized_series = self.normalize_key_values(series_data)
            
            keys.append(normalized_series)
    
    # Create composite key by joining all components
    composite_keys = []
    for idx in range(len(df)):
        key_parts = [str(keys[j].iloc[idx]) for j in range(len(keys))]
        composite_key = '|'.join(key_parts)
        composite_keys.append(composite_key)
    
    return pd.Series(composite_keys, index=df.index)
```

**Numeric Normalization**:
```python
def normalize_key_values(self, series_data):
    """Handle numeric formatting differences (50 vs 50.0)"""
    normalized_values = []
    
    for value in series_data:
        if pd.isna(value) or value is None:
            normalized_values.append('__NULL__')
        elif self.is_numeric(value):
            try:
                num_val = float(value)
                # Convert to int if whole number, otherwise keep as float
                if num_val.is_integer():
                    normalized_values.append(str(int(num_val)))
                else:
                    normalized_values.append(f"{num_val:g}")
            except (ValueError, TypeError):
                normalized_values.append(str(value).strip())
        else:
            normalized_values.append(str(value).strip())
    
    return pd.Series(normalized_values, index=series_data.index)
```

#### 4. Record Comparison
```python
def compare_records(self, row_a: pd.Series, row_b: pd.Series, comparison_rules: List[DeltaComparisonRule]):
    """Compare records based on comparison rules"""
    changes = []
    
    for rule in comparison_rules:
        col_a = rule.LeftFileColumn
        col_b = rule.RightFileColumn
        
        val_a = row_a.get(col_a)
        val_b = row_b.get(col_b)
        
        # Handle NaN values
        if pd.isna(val_a) and pd.isna(val_b):
            continue
        elif pd.isna(val_a) or pd.isna(val_b):
            changes.append(f"{col_a}: '{val_a}' -> '{val_b}'")
            continue
        
        # Apply comparison based on rule type
        values_match = self.apply_comparison_logic(val_a, val_b, rule)
        
        if not values_match:
            changes.append(f"{col_a}: '{val_a}' -> '{val_b}'")
    
    return len(changes) == 0, changes
```

#### 5. Delta Generation Algorithm
```python
def generate_delta(self, df_a: pd.DataFrame, df_b: pd.DataFrame, key_rules, comparison_rules, ...):
    """Main delta generation with comprehensive categorization"""
    
    # Apply filters if provided
    if file_filters:
        df_a = self.apply_filters(df_a, file_filters.get('file_0', []), "FileA")
        df_b = self.apply_filters(df_b, file_filters.get('file_1', []), "FileB")
    
    # Create composite keys
    df_a['_composite_key'] = self.create_composite_key(df_a, key_columns_a, key_rules)
    df_b['_composite_key'] = self.create_composite_key(df_b, key_columns_b, key_rules)
    
    # Create lookup dictionaries
    dict_a = {}
    dict_b = {}
    
    for idx, row in df_a.iterrows():
        key = row['_composite_key']
        if key not in dict_a:
            dict_a[key] = []
        dict_a[key].append(row.to_dict())
    
    # Process results by category
    common_keys = keys_a.intersection(keys_b)  # UNCHANGED or AMENDED
    deleted_keys = keys_a - keys_b              # DELETED
    new_keys = keys_b - keys_a                  # NEWLY_ADDED
    
    return {
        'unchanged': pd.DataFrame(unchanged_records),
        'amended': pd.DataFrame(amended_records),
        'deleted': pd.DataFrame(deleted_records),
        'newly_added': pd.DataFrame(newly_added_records),
        'all_changes': pd.DataFrame(amended_records + deleted_records + newly_added_records)
    }
```

## Error Handling

### Validation Errors
```python
# Column validation
errors_a = processor.validate_rules_against_columns(df_a, all_columns_a, "FileA")
if errors_a:
    raise HTTPException(status_code=400, detail={"errors": errors_a})

# File existence validation  
if file_id not in uploaded_files:
    raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
```

### Processing Errors
```python
try:
    delta_results = processor.generate_delta(...)
except Exception as e:
    print(f"Delta generation error: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
```

### Warning System
```python
# Duplicate key warnings
if duplicates_a:
    logger.warning(f"Found duplicate composite keys in File A: {duplicates_a[:5]}...")
    self.warnings.append(f"Found {len(duplicates_a)} duplicate composite keys in File A")

# Save operation warnings
if failed_saves:
    processor.warnings.append(f"Some result saves failed: {', '.join(failed_saves)}")
```

## Performance Features

### 1. Memory Optimization
- **Efficient DataFrames**: Use pandas operations optimized for large datasets
- **Streaming Downloads**: Large file downloads via `StreamingResponse`
- **Pagination Support**: Configurable page sizes for result retrieval
- **Memory Cleanup**: Manual cleanup endpoint for resource management

### 2. Processing Optimization
- **Hash-based Lookups**: Use dictionary lookups instead of DataFrame merges
- **Vectorized Operations**: Leverage pandas vectorized operations where possible
- **Batch Processing**: Process records in batches for memory efficiency
- **Index Optimization**: Reset DataFrame indices to prevent performance issues

### 3. Caching and Storage
- **In-Memory Storage**: Fast access to processed results
- **Auto-Save Integration**: Automatic result persistence to server storage
- **Filter Caching**: Cache filter results within processing session
- **Date Parsing Cache**: Shared date normalization cache via utilities

### 4. Streaming and Downloads
```python
# Optimized CSV streaming
output = io.StringIO()
df_to_export.to_csv(output, index=False)
output.seek(0)
output = io.BytesIO(output.getvalue().encode('utf-8'))

# Excel multi-sheet generation
with pd.ExcelWriter(output, engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
    if len(unchanged_df) > 0:
        unchanged_df.to_excel(writer, sheet_name='Unchanged Records', index=False)
```