import pandas as pd
import json
import re
from typing import Dict, List, Any, Optional
from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage
import logging

logger = logging.getLogger(__name__)

class EnhancedIntentVerifier:
    """
    Enhanced LLM-based intent verification with detailed column validation,
    intelligent suggestions, and comprehensive feasibility analysis
    """
    
    def __init__(self):
        self.llm_service = get_llm_service()
        
    def verify_intent_detailed(self, user_prompt: str, file_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced LLM-based intent verification with column validation and suggestions
        """
        try:
            logger.info(f"Starting enhanced intent verification for prompt: {user_prompt[:100]}...")
            
            # Phase 1: Extract comprehensive file schemas with sample data
            file_schemas = self._extract_comprehensive_schemas(file_data_list)
            logger.info(f"Extracted schemas for {len(file_schemas)} files")
            
            # Phase 2: LLM-based comprehensive analysis
            comprehensive_analysis = self._perform_comprehensive_analysis(user_prompt, file_schemas)
            logger.info("Completed comprehensive LLM analysis")
            
            # Phase 3: Process and structure the results
            structured_results = self._structure_verification_results(comprehensive_analysis, file_schemas)
            logger.info("Structured verification results")
            
            return structured_results
            
        except Exception as e:
            logger.error(f"Error in enhanced intent verification: {str(e)}")
            return self._create_error_response(str(e))
    
    def _extract_comprehensive_schemas(self, file_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract detailed schemas including column names, types, and sample values
        """
        schemas = []
        
        for i, file_data in enumerate(file_data_list):
            try:
                df = file_data['dataframe']
                filename = file_data.get('filename', f'file_{i+1}')
                
                # Get column information with sample values
                columns_info = {}
                for col in df.columns:
                    # Get sample values (non-null, unique)
                    sample_values = df[col].dropna().unique()[:5]
                    # Convert to strings and clean
                    sample_values = [str(val) for val in sample_values if pd.notna(val)]
                    
                    columns_info[col] = {
                        'type': str(df[col].dtype),
                        'null_count': int(df[col].isnull().sum()),
                        'unique_count': int(df[col].nunique()),
                        'sample_values': sample_values[:3]  # Limit to 3 samples for LLM context
                    }
                
                schema = {
                    'file_index': i + 1,
                    'filename': filename,
                    'table_name': f'file_{i + 1}',
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'columns': list(df.columns),
                    'columns_detailed': columns_info
                }
                
                schemas.append(schema)
                logger.debug(f"Schema extracted for {filename}: {len(df.columns)} columns, {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error extracting schema for file {i}: {str(e)}")
                schemas.append({
                    'file_index': i + 1,
                    'filename': file_data.get('filename', f'file_{i+1}'),
                    'table_name': f'file_{i + 1}',
                    'error': f"Could not extract schema: {str(e)}",
                    'columns': [],
                    'columns_detailed': {}
                })
        
        return schemas
    
    def _format_schemas_for_llm(self, file_schemas: List[Dict[str, Any]]) -> str:
        """
        Format schemas in a comprehensive way for LLM analysis
        """
        formatted_schemas = []
        
        for schema in file_schemas:
            if 'error' in schema:
                formatted_schemas.append(f"FILE {schema['file_index']}: {schema['filename']} - ERROR: {schema['error']}")
                continue
            
            schema_text = f"""
FILE {schema['file_index']}: {schema['filename']} (Table: {schema['table_name']})
- Total Rows: {schema['total_rows']:,}
- Total Columns: {schema['total_columns']}

COLUMNS WITH SAMPLE DATA:"""
            
            for col_name, col_info in schema['columns_detailed'].items():
                sample_text = ', '.join([f'"{val}"' for val in col_info['sample_values']])
                schema_text += f"""
  • {col_name} ({col_info['type']})
    Sample values: {sample_text}
    Unique values: {col_info['unique_count']:,}, Nulls: {col_info['null_count']:,}"""
            
            formatted_schemas.append(schema_text)
        
        return '\n'.join(formatted_schemas)
    
    
    def _perform_comprehensive_analysis(self, user_prompt: str, file_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive LLM analysis with detailed column validation
        """
        schemas_context = self._format_schemas_for_llm(file_schemas)
        
        # Check if user prompt contains existing column mapping suggestions
        has_existing_suggestions = "Column mapping suggestions:" in user_prompt
        existing_mappings_instruction = ""
        if has_existing_suggestions:
            existing_mappings_instruction = """

🚨 CRITICAL INSTRUCTION: The user query contains existing column mapping suggestions at the bottom (after "Column mapping suggestions:").
These mappings have ALREADY been clarified by the user. 
DO NOT suggest alternatives for these mapped columns again.
SKIP all columns that already have mappings provided.
Only validate that the mapped actual columns exist in the schemas and focus on any NEW missing columns that don't have mappings yet.
IGNORE any columns that appear after "Column mapping suggestions:" when creating new suggestions."""

        analysis_prompt = f"""
You are an expert data analyst with deep knowledge of data processing, SQL, and business intelligence. 
Your task is to perform comprehensive feasibility analysis of user queries against actual data schemas.

USER QUERY: "{user_prompt}"

ACTUAL DATA SCHEMAS:
{schemas_context}{existing_mappings_instruction}

COMPREHENSIVE ANALYSIS REQUIRED:

1. COLUMN REFERENCE VALIDATION:
   - Identify ALL column names referenced in the user query (explicit and implicit)
   - Check if each referenced column exists in the available data schemas
   - CRITICAL: If the user query has "Column mapping suggestions:" section, DO NOT suggest alternatives for those already mapped columns
   - For missing columns that are NOT already mapped in the user query, use semantic understanding to suggest similar existing columns
   - Consider common business terminology mappings (e.g., "customer" → "customer_name", "amount" → "transaction_amount")
   - SKIP suggestions for columns that already have mappings provided by the user

2. DERIVED COLUMN DETECTION:
   - Identify if user is requesting calculated/derived columns
   - Determine if these can be computed from existing data
   - Suggest the calculation logic if feasible

3. INTELLIGENT COLUMN SUGGESTIONS:
   - For missing columns, provide 2-3 most likely alternatives from existing columns
   - Use semantic similarity, not just string matching
   - Consider business context and sample data values
   - Rate each suggestion with confidence score (0.0-1.0)

4. JOIN LOGIC VALIDATION:
   - If multiple files involved, validate if joining is possible
   - Check for common columns that could serve as join keys
   - Identify potential join conflicts or issues

5. DATA TYPE COMPATIBILITY:
   - Check if operations requested are compatible with actual data types
   - Flag potential type conversion issues

6. FEASIBILITY ASSESSMENT:
   - Provide nuanced feasibility score (0.0-1.0) instead of binary yes/no
   - List specific issues that might cause problems
   - Suggest alternative approaches if original query has issues

🚨 CRITICAL REQUIREMENTS: 
- Base suggestions on ACTUAL column names and sample data provided above.
- If the user query contains "Column mapping suggestions:" section, do NOT suggest alternatives for those already mapped columns.
- SKIP any columns that have existing mappings after "Column mapping suggestions:"
- Only include NEW missing columns that need suggestions.
- DO NOT DUPLICATE existing mappings provided by the user.

Respond in VALID JSON format with this exact structure (NO COMMENTS, NO // TEXT):
{{
    "column_references": {{
        "found_columns": ["list of columns that exist exactly as referenced"],
        "missing_columns": ["list of columns referenced but don't exist"],
        "ambiguous_references": ["columns that could have multiple interpretations"]
    }},
    "column_suggestions": [
        {{
            "missing_column": "column_name_user_mentioned", 
            "suggestions": [
                {{
                    "actual_column": "existing_column_name",
                    "confidence": 0.85,
                    "reasoning": "why this is a good match based on name and sample data",
                    "sample_values": ["actual", "sample", "values"],
                    "file_source": "file_1"
                }}
            ]
        }}
    ],
    "derived_columns": [
        {{
            "requested": "what_user_wants",
            "feasible": true,
            "calculation": "how to calculate it using existing columns",
            "required_columns": ["columns", "needed", "for", "calculation"],
            "confidence": 0.9
        }}
    ],
    "feasibility_score": 0.85,
    "potential_issues": [
        {{
            "severity": "high",
            "issue": "description of potential problem",
            "impact": "how this affects the query execution",
            "suggestion": "specific recommendation to resolve"
        }}
    ],
    "join_validation": {{
        "joins_required": true,
        "possible_join_keys": ["common_columns_between_files"],
        "join_quality": "excellent",
        "join_issues": ["any problems with joining"],
        "recommended_join_strategy": "specific join approach"
    }},
    "data_type_compatibility": {{
        "compatible": true,
        "type_issues": ["any data type conflicts"],
        "required_conversions": ["type conversions that might be needed"]
    }},
    "alternative_approaches": [
        "suggest alternative ways to achieve similar results"
    ],
    "business_logic_assessment": {{
        "makes_business_sense": true,
        "business_concerns": ["any business logic issues"],
        "recommended_improvements": ["business-focused suggestions"]
    }},
    "confidence": 0.9
}}

CRITICAL: Ensure all column names in suggestions exactly match the actual column names from the schemas above.
"""
        
        try:
            if not self.llm_service.is_available():
                logger.error("LLM service is not available")
                return {"error": "LLM service not available"}
            
            generation_params = get_llm_generation_params()
            # Increase token limit for comprehensive analysis
            generation_params.update({
                'max_tokens': 2000,  # Allow longer responses for detailed analysis
                'temperature': 0.1   # Low temperature for consistency
            })
            
            messages = [
                LLMMessage(role="system", content="You are an expert data analyst who provides detailed, accurate column validation and intelligent suggestions for data queries."),
                LLMMessage(role="user", content=analysis_prompt)
            ]
            
            logger.info("Sending comprehensive analysis request to LLM service")
            response = self.llm_service.generate_text(
                messages=messages,
                **generation_params
            )
            
            if not response.success:
                logger.error(f"LLM generation failed: {response.error}")
                return {"error": f"LLM analysis failed: {response.error}"}
            
            # Parse JSON response
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Raw LLM response length: {len(response_text)}")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    # Remove any comments from JSON
                    json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
                    parsed_json = json.loads(json_str)
                    logger.info("Successfully parsed comprehensive analysis JSON")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Failed JSON snippet: {json_str[:500]}...")
                    return {"error": "Failed to parse LLM response"}
            else:
                logger.warning("No JSON pattern found in LLM response")
                return {"error": "Invalid LLM response format"}
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _structure_verification_results(self, analysis: Dict[str, Any], file_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Structure the LLM analysis results into the format expected by the frontend
        """
        if "error" in analysis:
            return self._create_error_response(analysis["error"])
        
        # Calculate summary metrics
        total_columns_available = sum(len(schema.get('columns', [])) for schema in file_schemas)
        found_columns_count = len(analysis.get('column_references', {}).get('found_columns', []))
        missing_columns_count = len(analysis.get('column_references', {}).get('missing_columns', []))
        suggestions_count = len(analysis.get('column_suggestions', []))
        
        structured_results = {
            # Core verification results
            "verification_status": "completed",
            "feasibility_score": analysis.get('feasibility_score', 0.5),
            "confidence": analysis.get('confidence', 0.5),
            
            # Column validation details
            "column_validation": {
                "found_columns": analysis.get('column_references', {}).get('found_columns', []),
                "missing_columns": analysis.get('column_references', {}).get('missing_columns', []),
                "ambiguous_references": analysis.get('column_references', {}).get('ambiguous_references', []),
                "total_available_columns": total_columns_available,
                "validation_summary": {
                    "found": found_columns_count,
                    "missing": missing_columns_count,
                    "success_rate": found_columns_count / max(1, found_columns_count + missing_columns_count)
                }
            },
            
            # Intelligent suggestions
            "intelligent_suggestions": analysis.get('column_suggestions', []),
            "derived_columns_detected": analysis.get('derived_columns', []),
            
            # Feasibility assessment
            "potential_issues": analysis.get('potential_issues', []),
            "alternative_approaches": analysis.get('alternative_approaches', []),
            
            # Technical validation
            "join_validation": analysis.get('join_validation', {}),
            "data_type_compatibility": analysis.get('data_type_compatibility', {}),
            
            # Business logic assessment
            "business_logic_assessment": analysis.get('business_logic_assessment', {}),
            
            # UI control flags
            "requires_user_confirmation": suggestions_count > 0 or missing_columns_count > 0,
            "has_suggestions": suggestions_count > 0,
            "has_issues": len(analysis.get('potential_issues', [])) > 0,
            
            # File context for UI
            "file_schemas_summary": [
                {
                    "filename": schema.get('filename', f"file_{i+1}"),
                    "columns": schema.get('columns', []),
                    "row_count": schema.get('total_rows', 0)
                }
                for i, schema in enumerate(file_schemas)
            ],
            
            # Processing metadata
            "processing_metadata": {
                "files_analyzed": len(file_schemas),
                "total_columns_analyzed": total_columns_available,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info(f"Structured verification results - Feasibility: {structured_results['feasibility_score']:.2f}, "
                   f"Found columns: {found_columns_count}, Missing: {missing_columns_count}, "
                   f"Suggestions: {suggestions_count}")
        
        return structured_results
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create structured error response when verification fails
        """
        return {
            "verification_status": "error",
            "error_message": error_message,
            "feasibility_score": 0.0,
            "confidence": 0.0,
            "column_validation": {
                "found_columns": [],
                "missing_columns": [],
                "ambiguous_references": [],
                "validation_summary": {"found": 0, "missing": 0, "success_rate": 0.0}
            },
            "intelligent_suggestions": [],
            "derived_columns_detected": [],
            "potential_issues": [
                {
                    "severity": "high",
                    "issue": f"Verification system error: {error_message}",
                    "impact": "Cannot validate query feasibility",
                    "suggestion": "Please try again or contact support"
                }
            ],
            "alternative_approaches": [],
            "requires_user_confirmation": False,
            "has_suggestions": False,
            "has_issues": True,
            "file_schemas_summary": [],
            "processing_metadata": {
                "files_analyzed": 0,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "error": True
            }
        }