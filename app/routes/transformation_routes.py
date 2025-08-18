import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.transformation_models import (
    TransformationRequest,
    TransformationResult,
    TransformationTemplate,
    SourceFile
)
from app.services.transformation_service import transformation_storage
from app.utils.uuid_generator import generate_uuid

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/transformation", tags=["transformation"])


async def get_file_dataframe(file_ref: SourceFile) -> pd.DataFrame:
    """Get dataframe from file reference"""
    # Import here to avoid circular imports
    from app.services.storage_service import uploaded_files

    if file_ref.file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail=f"File {file_ref.file_id} not found")

    file_data = uploaded_files[file_ref.file_id]
    return file_data["data"]


def evaluate_expression(expression: str, row_data: Dict[str, Any]) -> Any:
    """Evaluate expressions with variable substitution like {column_name} or calculations"""
    if not expression or not isinstance(expression, str):
        return expression
    
    try:
        # Check if it's an expression with variables (contains curly braces)
        if '{' in expression and '}' in expression:
            import re
            
            # Find all variables in curly braces
            variables = re.findall(r'\{([^}]+)\}', expression)
            result_expression = expression
            
            # Replace each variable with its value
            for var in variables:
                if var in row_data:
                    value = row_data[var]
                    # Handle different types of values
                    if isinstance(value, (int, float)):
                        result_expression = result_expression.replace(f'{{{var}}}', str(value))
                    elif isinstance(value, str):
                        # For string values, wrap in quotes for safe evaluation
                        result_expression = result_expression.replace(f'{{{var}}}', f'"{value}"')
                    else:
                        result_expression = result_expression.replace(f'{{{var}}}', f'"{str(value)}"')
                else:
                    logger.warning(f"Variable '{var}' not found in row data")
                    return expression  # Return original if variable not found
            
            # Try to evaluate as mathematical expression first
            try:
                # Create safe evaluation context
                safe_context = {
                    '__builtins__': {},
                    'abs': abs,
                    'round': round,
                    'min': min, 
                    'max': max,
                    'int': int,
                    'float': float,
                    'str': str
                }
                
                # If it looks like a math expression, evaluate it
                if any(op in result_expression for op in ['+', '-', '*', '/', '(', ')', '%']):
                    result = eval(result_expression, safe_context)
                    return result
                else:
                    # For string concatenation, just remove quotes and return as string
                    # Handle patterns like "John" "Doe" -> "John Doe"
                    result = result_expression.replace('" "', ' ').replace('"', '')
                    return result.strip()
                    
            except Exception as e:
                logger.warning(f"Could not evaluate expression '{result_expression}': {e}")
                # Fallback: simple string concatenation
                result = result_expression.replace('" "', ' ').replace('"', '')
                return result.strip()
        
        else:
            # No variables, return as-is
            return expression
            
    except Exception as e:
        logger.error(f"Error evaluating expression '{expression}': {e}")
        return expression


def evaluate_condition(condition: str, row_data: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Safely evaluate a condition string against row data"""
    if not condition.strip():
        return True

    try:
        # Create a safe evaluation context
        eval_context = {
            **row_data,
            **context,
            'abs': abs,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool
        }

        # Replace column references like file_0.Column with actual values
        safe_condition = condition
        for key, value in eval_context.items():
            if f".{key}" in safe_condition or safe_condition.startswith(key):
                # Handle different value types
                if isinstance(value, str):
                    safe_condition = safe_condition.replace(key, f'"{value}"')
                else:
                    safe_condition = safe_condition.replace(key, str(value))

        # Basic safety check - only allow simple expressions
        allowed_chars = set('0123456789.+-*/()<>=!&|" \t\n')
        if not all(c.isalnum() or c in allowed_chars for c in safe_condition):
            logger.warning(f"Unsafe condition detected: {condition}")
            return False

        return eval(safe_condition)
    except Exception as e:
        logger.warning(f"Error evaluating condition '{condition}': {str(e)}")
        return False


def apply_column_mapping(mapping_config: Dict[str, Any], row_data: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """Apply column mapping configuration to get output value"""
    mapping_type = mapping_config.get('mapping_type', 'direct')

    if mapping_type == 'direct':
        source_column = mapping_config.get('source_column', '')
        return row_data.get(source_column, '')

    elif mapping_type == 'static':
        static_value = mapping_config.get('static_value', '')
        # Evaluate expressions in static values
        return evaluate_expression(static_value, row_data)

    elif mapping_type == 'dynamic':
        # Evaluate dynamic conditions
        conditions = mapping_config.get('dynamic_conditions', [])
        default_value = mapping_config.get('default_value', '')

        for condition in conditions:
            condition_column = condition.get('condition_column', '')
            operator = condition.get('operator', '==')
            condition_value = condition.get('condition_value', '')
            output_value = condition.get('output_value', '')

            if condition_column in row_data:
                row_value = row_data[condition_column]

                # Evaluate condition based on operator
                condition_met = False
                try:
                    # Convert values for comparison - handle type conversion
                    if operator in ['>', '<', '>=', '<=']:
                        # For numerical comparisons, try to convert both to float
                        try:
                            row_value_num = float(str(row_value).replace(',', ''))
                            condition_value_num = float(str(condition_value).replace(',', ''))

                            if operator == '>':
                                condition_met = row_value_num > condition_value_num
                            elif operator == '<':
                                condition_met = row_value_num < condition_value_num
                            elif operator == '>=':
                                condition_met = row_value_num >= condition_value_num
                            elif operator == '<=':
                                condition_met = row_value_num <= condition_value_num
                        except (ValueError, TypeError):
                            # If conversion fails, fall back to string comparison
                            row_value_str = str(row_value)
                            condition_value_str = str(condition_value)

                            if operator == '>':
                                condition_met = row_value_str > condition_value_str
                            elif operator == '<':
                                condition_met = row_value_str < condition_value_str
                            elif operator == '>=':
                                condition_met = row_value_str >= condition_value_str
                            elif operator == '<=':
                                condition_met = row_value_str <= condition_value_str
                    else:
                        # For string comparisons
                        row_value_str = str(row_value)
                        condition_value_str = str(condition_value)

                        if operator == '==':
                            condition_met = row_value_str == condition_value_str
                        elif operator == '!=':
                            condition_met = row_value_str != condition_value_str
                        elif operator == 'contains':
                            condition_met = condition_value_str in row_value_str
                        elif operator == 'startsWith':
                            condition_met = row_value_str.startswith(condition_value_str)
                        elif operator == 'endsWith':
                            condition_met = row_value_str.endswith(condition_value_str)

                    if condition_met:
                        # Evaluate expressions in output values
                        result = evaluate_expression(output_value, row_data)
                        return result

                except Exception as e:
                    logger.warning(f"Error evaluating dynamic condition: {str(e)}")
                    continue
        # Evaluate expressions in default values
        result = evaluate_expression(default_value, row_data)
        return result

    return ''


def process_transformation_rules(source_data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process transformation rules and generate output datasets"""
    rules = config.get('row_generation_rules', [])
    merge_datasets = config.get('merge_datasets', True)

    all_results = []
    rule_results = {}
    processing_errors = []

    # Get all source rows (simplified for single file - no alias prefix needed)
    source_rows = []
    for alias, df in source_data.items():
        for _, row in df.iterrows():
            # Use direct column names since we only have one source file
            row_dict = {}
            for col, value in row.items():
                row_dict[col] = value  # Direct column name, no prefix
            source_rows.append(row_dict)

    for rule in rules:
        if not rule.get('enabled', True):
            continue

        rule_name = rule.get('name', 'Unnamed Rule')
        output_columns = rule.get('output_columns', [])

        if not output_columns:
            logger.warning(f"Rule '{rule_name}' has no output columns defined")
            continue

        rule_output_rows = []
        rule_errors = []

        # Process each source row
        for source_row in source_rows:
            try:
                # Generate output row based on column mappings
                output_row = {}
                # Create combined context that includes source data and progressively built output columns
                combined_row_data = source_row.copy()
                
                for column_config in output_columns:
                    column_name = column_config.get('name', '')
                    if column_name:
                        try:
                            # Pass combined data that includes both source and previously calculated columns
                            result_value = apply_column_mapping(column_config, combined_row_data, {})
                            output_row[column_name] = result_value
                            # Add the newly calculated column to combined_row_data for subsequent calculations
                            combined_row_data[column_name] = result_value
                        except Exception as e:
                            logger.error(f"Error processing column '{column_name}' in rule '{rule_name}': {str(e)}")
                            output_row[column_name] = ''
                            # Still add to combined_row_data as empty value to prevent KeyError
                            combined_row_data[column_name] = ''
                            rule_errors.append(f"Column '{column_name}': {str(e)}")

                if output_row:  # Only add if we have data
                    rule_output_rows.append(output_row)

            except Exception as e:
                logger.error(f"Error processing row in rule '{rule_name}': {str(e)}")
                rule_errors.append(f"Row processing error: {str(e)}")

        # Store rule results
        if rule_output_rows:
            rule_df = pd.DataFrame(rule_output_rows)
            rule_results[rule_name] = {
                'data': rule_df,
                'errors': rule_errors
            }

            if merge_datasets:
                all_results.extend(rule_output_rows)

        # Collect processing errors
        if rule_errors:
            processing_errors.extend([f"Rule '{rule_name}': {error}" for error in rule_errors])

    # Return results with error information
    if merge_datasets:
        return [{
            'merged_output': pd.DataFrame(all_results) if all_results else pd.DataFrame(),
            'errors': processing_errors
        }]
    else:
        return [{
            'rule_name': name,
            'data': result['data'],
            'errors': result['errors']
        } for name, result in rule_results.items()]


@router.post("/process/", response_model=TransformationResult)
async def process_transformation(request: TransformationRequest):
    """Process data transformation based on rule configuration"""

    start_time = datetime.now()

    try:
        # Load source files into dataframes
        source_data = {}
        total_input_rows = 0
        for source_file in request.source_files:
            df = await get_file_dataframe(source_file)
            source_data[source_file.alias] = df
            logger.info(f"Loaded {source_file.alias}: {len(df)} rows")
            total_input_rows += len(df)

        # Process transformation using new rule-based system
        transformation_config = request.transformation_config.dict() if hasattr(request.transformation_config,
                                                                                'dict') else request.transformation_config
        result_datasets = process_transformation_rules(source_data, transformation_config)

        # Calculate totals and collect errors
        total_output_rows = 0
        all_errors = []
        all_warnings = []

        for dataset in result_datasets:
            if 'data' in dataset:
                total_output_rows += len(dataset['data'])
            elif 'merged_output' in dataset:
                total_output_rows += len(dataset['merged_output'])

            # Collect errors from processing
            if 'errors' in dataset and dataset['errors']:
                all_errors.extend(dataset['errors'])

        # Generate transformation ID
        transformation_id = generate_uuid('transform')

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        processing_info = {
            'input_row_count': total_input_rows,
            'output_row_count': total_output_rows,
            'processing_time': processing_time,
            'rules_processed': len(
                [r for r in transformation_config.get('row_generation_rules', []) if r.get('enabled', True)]),
            'datasets_generated': len(result_datasets)
        }

        # Store results if not preview
        if not request.preview_only:
            # Store results in uploaded_files format so they can be viewed with existing viewer
            from app.services.storage_service import uploaded_files

            storage_success = transformation_storage.store_results(
                transformation_id,
                {
                    'datasets': result_datasets,
                    'config': transformation_config,
                    'processing_info': processing_info
                }
            )

            # Also store each dataset as a viewable file
            for i, dataset in enumerate(result_datasets):
                if 'data' in dataset:
                    df = dataset['data']
                    dataset_name = dataset.get('rule_name', f'Rule_{i + 1}')
                elif 'merged_output' in dataset:
                    df = dataset['merged_output']
                    dataset_name = 'Merged_Output'
                else:
                    continue

                # Create file ID for this dataset
                if transformation_config.get('merge_datasets', True):
                    file_id = transformation_id
                    filename = f"{transformation_config.get('name', 'Transformation')}_{dataset_name}.csv"
                else:
                    file_id = f"{transformation_id}_rule_{i}"
                    filename = f"{transformation_config.get('name', 'Transformation')}_{dataset_name}.csv"

                # Store in uploaded_files format
                uploaded_files[file_id] = {
                    "data": df,
                    "info": {
                        "file_id": file_id,
                        "filename": filename,
                        "file_size_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                        "upload_time": datetime.now().isoformat(),
                        "total_rows": len(df),
                        "columns": list(df.columns),
                        "file_type": "transformation_result",
                        "transformation_id": transformation_id,
                        "last_modified": datetime.now().isoformat()
                    }
                }

            if not storage_success:
                logger.warning("Failed to store transformation results")

        # Prepare response
        response = TransformationResult(
            success=True,
            transformation_id=transformation_id,
            total_input_rows=total_input_rows,
            total_output_rows=total_output_rows,
            processing_time_seconds=round(processing_time, 3),
            validation_summary=processing_info,
            warnings=all_warnings,
            errors=all_errors
        )

        # Add preview data if requested
        if request.preview_only and result_datasets:
            preview_data = []
            for dataset in result_datasets[:1]:  # Show first dataset for preview
                if 'data' in dataset:
                    preview_data = dataset['data'].head(request.row_limit or 10).to_dict('records')
                elif 'merged_output' in dataset:
                    preview_data = dataset['merged_output'].head(request.row_limit or 10).to_dict('records')
                break
            response.preview_data = preview_data

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transformation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/results/{transformation_id}")
async def get_transformation_results(
        transformation_id: str,
        page: Optional[int] = 1,
        page_size: Optional[int] = 1000,
        dataset_name: Optional[str] = None
):
    """Get transformation results with pagination"""

    results = transformation_storage.get_results(transformation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Transformation ID not found")

    datasets = results['results']['datasets']

    # If specific dataset requested, return that one
    if dataset_name:
        target_dataset = None
        for dataset in datasets:
            if dataset.get('rule_name') == dataset_name or 'merged_output' in dataset:
                target_dataset = dataset
                break

        if not target_dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

        df = target_dataset.get('data') or target_dataset.get('merged_output')
    else:
        # Return first dataset (merged or first rule)
        if not datasets:
            raise HTTPException(status_code=404, detail="No datasets found")

        first_dataset = datasets[0]
        df = first_dataset.get('data') or first_dataset.get('merged_output')

    total_rows = len(df)

    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get page data
    page_data = df.iloc[start_idx:end_idx].to_dict('records')

    return {
        'transformation_id': transformation_id,
        'timestamp': results['timestamp'].isoformat(),
        'total_rows': total_rows,
        'page': page,
        'page_size': page_size,
        'data': page_data,
        'has_more': end_idx < total_rows,
        'available_datasets': [d.get('rule_name', 'merged_output') for d in datasets]
    }


@router.get("/download/{transformation_id}")
async def download_transformation_results(
        transformation_id: str,
        format: str = "csv",
        dataset_name: Optional[str] = None
):
    """Download transformation results"""

    results = transformation_storage.get_results(transformation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Transformation ID not found")

    datasets = results['results']['datasets']
    config = results['results']['config']

    # Select dataset to download
    if dataset_name:
        target_dataset = None
        for dataset in datasets:
            if dataset.get('rule_name') == dataset_name or (
                    dataset_name == 'merged_output' and 'merged_output' in dataset):
                target_dataset = dataset
                break

        if not target_dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    else:
        if not datasets:
            raise HTTPException(status_code=404, detail="No datasets found")
        target_dataset = datasets[0]

    df = target_dataset.get('data') or target_dataset.get('merged_output')
    dataset_display_name = target_dataset.get('rule_name', 'merged_output')

    try:
        if format.lower() == "csv":
            # Generate CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            # Convert to bytes
            output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
            filename = f"transformation_{transformation_id}_{dataset_display_name}.csv"
            media_type = "text/csv"

        elif format.lower() == "excel":
            # Generate Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name=dataset_display_name[:31], index=False)  # Excel sheet name limit

                # Add metadata sheet
                metadata = pd.DataFrame({
                    'Property': ['Transformation ID', 'Created At', 'Total Rows', 'Dataset', 'Source Files'],
                    'Value': [
                        transformation_id,
                        results['timestamp'].isoformat(),
                        len(df),
                        dataset_display_name,
                        ', '.join([f['alias'] for f in config.get('source_files', [])])
                    ]
                })
                metadata.to_excel(writer, sheet_name='Metadata', index=False)

            output.seek(0)
            output_bytes = output
            filename = f"transformation_{transformation_id}_{dataset_display_name}.xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        elif format.lower() == "json":
            # Generate JSON
            json_data = df.to_json(orient='records', indent=2)
            output_bytes = io.BytesIO(json_data.encode('utf-8'))
            filename = f"transformation_{transformation_id}_{dataset_display_name}.json"
            media_type = "application/json"

        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv', 'excel', or 'json'")

        return StreamingResponse(
            output_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")


@router.post("/templates/", response_model=TransformationTemplate)
async def save_transformation_template(template: TransformationTemplate):
    """Save a transformation template for reuse"""

    # In a real implementation, this would save to a database
    # For now, we'll use in-memory storage
    if not hasattr(save_transformation_template, 'templates'):
        save_transformation_template.templates = {}

    template.id = generate_uuid('template')
    template.updated_at = datetime.now()
    save_transformation_template.templates[template.id] = template

    return template


@router.get("/templates/")
async def list_transformation_templates(
        category: Optional[str] = None,
        search: Optional[str] = None
):
    """List available transformation templates"""

    if not hasattr(save_transformation_template, 'templates'):
        return []

    templates = list(save_transformation_template.templates.values())

    # Filter by category
    if category:
        templates = [t for t in templates if t.category == category]

    # Search filter
    if search:
        search_lower = search.lower()
        templates = [
            t for t in templates
            if search_lower in t.name.lower() or
               (t.description and search_lower in t.description.lower()) or
               any(search_lower in tag.lower() for tag in t.tags)
        ]

    return templates


@router.get("/templates/{template_id}", response_model=TransformationTemplate)
async def get_transformation_template(template_id: str):
    """Get a specific transformation template"""

    if not hasattr(save_transformation_template, 'templates'):
        raise HTTPException(status_code=404, detail="Template not found")

    template = save_transformation_template.templates.get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return template


@router.delete("/results/{transformation_id}")
async def delete_transformation_results(transformation_id: str):
    """Delete transformation results"""

    if transformation_storage.delete_results(transformation_id):
        return {"success": True, "message": f"Transformation {transformation_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Transformation ID not found")


@router.post("/generate-config/")
async def generate_transformation_config(request: dict):
    """Generate transformation configuration using AI based on user requirements"""
    
    try:
        requirements = request.get('requirements', '')
        source_files = request.get('source_files', [])
        
        if not requirements:
            raise HTTPException(status_code=400, detail="Requirements are required")
        
        if not source_files:
            raise HTTPException(status_code=400, detail="Source files information is required")
        
        # Import LLM service
        from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage
        
        llm_service = get_llm_service()
        if not llm_service.is_available():
            raise HTTPException(status_code=500, detail=f"LLM service ({llm_service.get_provider_name()}) not configured")
        
        # Get generation parameters from config
        generation_params = get_llm_generation_params()
        
        # Prepare context about source files
        files_context = []
        for sf in source_files:
            files_context.append(f"File: {sf['filename']} (alias: {sf['alias']})")
            files_context.append(f"  Columns: {', '.join(sf['columns'])}")
            files_context.append(f"  Rows: {sf['totalRows']}")
        
        files_info = "\n".join(files_context)
        
        # Create prompt for OpenAI
        prompt = f"""
You are an expert data transformation configuration generator. Based on the user requirements and source file information, generate a JSON configuration for data transformation.

Source Files Available:
{files_info}

User Requirements:
{requirements}

Generate a transformation configuration with this exact JSON structure:
{{
    "name": "descriptive_name_for_transformation",
    "description": "brief_description_of_what_this_transformation_does",
    "source_files": [
        {{
            "file_id": "actual_file_id",
            "alias": "source_file",
            "purpose": "Primary data source"
        }}
    ],
    "row_generation_rules": [
        {{
            "id": "rule_1",
            "name": "Main Transformation Rule",
            "enabled": true,
            "output_columns": [
                {{
                    "id": "col_1",
                    "name": "output_column_name",
                    "mapping_type": "direct|static|dynamic",
                    "source_column": "source_column_name",
                    "static_value": "static_value_if_applicable",
                    "dynamic_conditions": [
                        {{
                            "condition_column": "column_name",
                            "operator": "==|!=|>|<|>=|<=|contains|startsWith|endsWith",
                            "condition_value": "value_to_compare",
                            "output_value": "value_to_output_if_condition_met"
                        }}
                    ],
                    "default_value": "default_value_for_dynamic"
                }},
                {{
                    "id": "col_2",
                    "name": "another_output_column",
                    "mapping_type": "direct|static|dynamic"
                }}
            ],
            "priority": 0
        }}
    ],
    "merge_datasets": false,
    "validation_rules": []
}}

**CRITICAL: Create only ONE rule in the "row_generation_rules" array. Put ALL requested columns inside the "output_columns" array of this single rule.**

MAPPING TYPE SELECTION RULES:

1. **DIRECT MAPPING** - Use when copying a column as-is:
   - "mapping_type": "direct"
   - "source_column": "existing_column_name"
   - Example: Copy Product_Name directly

2. **STATIC MAPPING** - Use for calculations, expressions, and fixed values:
   - "mapping_type": "static" 
   - "static_value": "expression or fixed value"
   - Use for: Mathematical calculations, string concatenation, fixed text, complex expressions
   - Examples:
     * Calculations: "{{Retail_Price}} - {{Cost_Price}}"
     * Percentages: "({{Retail_Price}} - {{Cost_Price}}) / {{Cost_Price}} * 100"
     * Complex expressions: "({{roi_percentage}} + {{investment_efficiency}} + {{capital_turnover}}) / 3"
     * Concatenation: "{{First_Name}} {{Last_Name}}"
     * Fixed values: "Active Status"
   - **IMPORTANT**: Can reference previously calculated columns within the same rule

3. **DYNAMIC MAPPING** - Use ONLY for conditional logic based on source columns:
   - "mapping_type": "dynamic"
   - "dynamic_conditions": array of condition objects
   - Use for: Categorization, status assignment, conditional text
   - CRITICAL: condition_column MUST be an existing source column, NOT a calculated field

VALID OPERATORS for dynamic conditions:
- "==" (equals)
- "!=" (not equals)
- ">" (greater than)
- "<" (less than)
- ">=" (greater than or equal)  
- "<=" (less than or equal)

INVALID OPERATORS (DO NOT USE):
- "-", "+", "*", "/" (these are for calculations, use static mapping instead)
- "&&", "||" (compound operators not supported)
- ">= && <" (not supported)

CRITICAL RULES:
1. ONLY use column names that exist in source files: {', '.join([col for file_info in source_files for col in file_info.get('columns', [])])}
2. For calculations → Use "static" mapping with mathematical expressions
3. For Mathematical expressions enclose the entire expression with curly brackets '{''}'
4. For categorization → Use "dynamic" mapping with comparison operators on source columns
5. NEVER reference calculated/output columns in condition_column
6. Create calculated fields first, then categorize based on source data
7. **ALWAYS CREATE EXACTLY ONE (1) RULE** - Put all columns in a single rule unless user explicitly asks for multiple rules
8. Never create multiple rules unless the user specifically mentions "multiple rules", "separate rules", or "different rules"
9. Return ONLY valid JSON configuration, no additional text

**IMPORTANT: DEFAULT BEHAVIOR = SINGLE RULE WITH ALL COLUMNS**

CORRECT EXAMPLES:

Example 1 - Mathematical Calculation (STATIC):
{{
  "name": "profit_margin",
  "mapping_type": "static",
  "static_value": "{{Retail_Price}} - {{Cost_Price}}"
}}

Example 2 - Percentage Calculation (STATIC):
{{
  "name": "markup_percentage", 
  "mapping_type": "static",
  "static_value": "({{Retail_Price}} - {{Cost_Price}}) / {{Cost_Price}} * 100"
}}

Example 3 - Conditional Categorization (DYNAMIC):
{{
  "name": "price_tier",
  "mapping_type": "dynamic",
  "dynamic_conditions": [
    {{
      "id": "cond_001",
      "condition_column": "Retail_Price",
      "operator": ">=",
      "condition_value": "1000",
      "output_value": "Premium"
    }},
    {{
      "id": "cond_002", 
      "condition_column": "Retail_Price",
      "operator": ">=", 
      "condition_value": "100",
      "output_value": "Standard"
    }}
  ],
  "default_value": "Budget"
}}

Example 4 - Complex Expression with Calculated Fields (STATIC):
{{
  "name": "composite_score",
  "mapping_type": "static",
  "static_value": "({{roi_percentage}} + {{investment_efficiency}} + {{capital_turnover}}) / 3"
}}

Example 5 - Direct Copy (DIRECT):
{{
  "name": "product_name",
  "mapping_type": "direct",
  "source_column": "Product_Name"
}}

Use {{column_name}} syntax to reference column values in expressions.
**COLUMN DEPENDENCY**: Calculated columns can reference other calculated columns defined earlier in the same rule.
"""
        
        # Call LLM service
        messages = [
            LLMMessage(role="system", content="You are a data transformation expert. Return only valid JSON configuration. ALWAYS create exactly ONE rule with ALL columns inside it unless explicitly asked for multiple rules."),
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
            json_match = re.search(r'\{.*\}', generated_config_text, re.DOTALL)
            if json_match:
                generated_config = json.loads(json_match.group())
            else:
                raise HTTPException(status_code=500, detail="Failed to parse AI-generated configuration")
        
        # Update file IDs with actual values from request (simplified for single file)
        if 'source_files' in generated_config and len(source_files) > 0:
            actual_file_id = source_files[0].get('file_id')
            
            # Update the source_files with actual file_id
            if len(generated_config['source_files']) > 0:
                generated_config['source_files'][0]['file_id'] = actual_file_id
                generated_config['source_files'][0]['alias'] = actual_file_id  # Use file_id as alias
            
            # No need to update column references since we use direct column names now
        
        # Validate the generated configuration has required fields
        required_fields = ['name', 'description', 'source_files', 'row_generation_rules']
        missing_fields = [field for field in required_fields if field not in generated_config]
        if missing_fields:
            raise HTTPException(status_code=500, detail=f"AI generated config missing fields: {missing_fields}")
        
        return {
            "success": True,
            "message": "Configuration generated successfully",
            "data": generated_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating AI configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration generation error: {str(e)}")


@router.get("/health")
async def transformation_health_check():
    """Health check for transformation service"""
    
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

    return {
        "status": "healthy",
        "service": "transformation",
        "llm_service": llm_status,
        "features": [
            "rule_based_transformation",
            "multiple_output_datasets", 
            "conditional_logic",
            "dynamic_column_mapping",
            "dataset_merging",
            "template_system",
            "multiple_output_formats",
            "ai_configuration_generation",
            "expression_evaluation",
            "pluggable_llm_providers"
        ]
    }
