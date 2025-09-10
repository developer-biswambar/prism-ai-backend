import pandas as pd
import os
import json
import re
from typing import Dict, List, Any, Optional
from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage
import logging

logger = logging.getLogger(__name__)

class QueryIntentExtractor:
    def __init__(self):
        self.llm_service = get_llm_service()
        
    def extract_intent(self, prompt: str, file_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract comprehensive intent information from user prompt and files
        """
        try:
            # Analyze files first to get sample data and statistics
            file_analysis = self._analyze_files(file_data_list)
            
            # Generate SQL query to understand the operation
            sql_analysis = self._analyze_sql_intent(prompt, file_analysis)
            
            # Ensure sql_analysis is a dictionary
            if not isinstance(sql_analysis, dict):
                logger.error(f"SQL analysis returned unexpected type: {type(sql_analysis)}, value: {sql_analysis}")
                sql_analysis = {"operation_type": "UNKNOWN", "confidence": "LOW", "business_intent": str(sql_analysis)}
            
            # Create data flow steps
            data_flow = self._create_data_flow(sql_analysis, file_analysis)
            
            # Generate plain language summary
            plain_language = self._generate_plain_language_summary(prompt, sql_analysis)
            
            # Calculate processing estimates
            processing_estimates = self._calculate_processing_estimates(file_analysis, sql_analysis)
            
            # Detect risk factors and warnings
            risk_assessment = self._assess_risks(sql_analysis, file_analysis)
            
            return {
                "operation_type": sql_analysis.get("operation_type", "UNKNOWN"),
                "business_intent": sql_analysis.get("business_intent", ""),
                "plain_language_summary": plain_language,
                "data_flow": data_flow,
                "files_involved": file_analysis,
                "matching_logic": sql_analysis.get("matching_logic", {}),
                "expected_output": sql_analysis.get("expected_output", {}),
                "processing_estimates": processing_estimates,
                "confidence": sql_analysis.get("confidence", "MEDIUM"),
                "risk_factors": risk_assessment["risk_factors"],
                "data_quality_warnings": risk_assessment["warnings"]
            }
            
        except Exception as e:
            logger.error(f"Error extracting intent: {str(e)}")
            return self._create_error_response(str(e))
    
    def _analyze_files(self, file_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze file data objects to get sample data and statistics
        """
        file_analysis = []
        
        for i, file_data in enumerate(file_data_list):
            try:
                df = file_data['dataframe']
                filename = file_data.get('filename', f'file_{i+1}')
                
                # Convert sample data to dict format (first 3 rows)
                sample_data = df.head(3).to_dict('records')
                
                # Clean up sample data for JSON serialization
                for row in sample_data:
                    for key, value in row.items():
                        if pd.isna(value):
                            row[key] = None
                        elif isinstance(value, (pd.Timestamp, pd.NaT.__class__)):
                            row[key] = str(value)
                
                # Calculate file size estimate
                file_size_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
                
                file_info = {
                    "file_path": f"uploaded_file_{i+1}",  # Use meaningful path
                    "file_name": filename,
                    "sample_data": sample_data,
                    "statistics": {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "size_mb": file_size_mb
                    },
                    "column_info": {
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                }
                
                file_analysis.append(file_info)
                
            except Exception as e:
                logger.error(f"Error analyzing file data {i}: {str(e)}")
                filename = file_data.get('filename', f'file_{i+1}') if isinstance(file_data, dict) else f'file_{i+1}'
                file_analysis.append({
                    "file_path": f"uploaded_file_{i+1}",
                    "file_name": filename,
                    "error": f"Could not analyze file: {str(e)}",
                    "statistics": {"rows": 0, "columns": 0, "size_mb": 0}
                })
        
        return file_analysis
    
    def _analyze_sql_intent(self, prompt: str, file_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to analyze the SQL intent and operations
        """
        file_context = self._create_file_context_for_llm(file_analysis)
        
        analysis_prompt = f"""
        Analyze this natural language query and the available files to extract SQL intent:
        
        USER QUERY: "{prompt}"
        
        AVAILABLE FILES AND SAMPLE DATA:
        {file_context}
        
        Please analyze and return a JSON response with these exact fields:
        {{
            "operation_type": "JOIN/GROUP_BY/FILTER/AGGREGATE/etc",
            "business_intent": "What business question is being answered",
            "matching_logic": {{
                "column": "name_of_joining_column",
                "type": "exact_match/fuzzy_match/etc",
                "description": "How the files will be matched"
            }},
            "expected_output": {{
                "description": "What the results will contain",
                "estimated_rows": {{
                    "min": 0,
                    "max": 100,
                    "likely": 50
                }}
            }},
            "confidence": "HIGH/MEDIUM/LOW"
        }}
        
        Focus on:
        - Identify the EXACT column names mentioned in the query for joining (e.g., "Reference", "Ref_Number")
        - Determine the type of matching (exact, fuzzy, partial)
        - Estimate realistic result counts based on the sample data
        - Be specific about the matching logic
        
        Return only valid JSON with the exact structure above.
        """
        
        try:
            if not self.llm_service.is_available():
                logger.error("LLM service is not available")
                return {"operation_type": "UNKNOWN", "confidence": "LOW", "business_intent": "LLM service not available"}
            
            generation_params = get_llm_generation_params()
            messages = [
                LLMMessage(role="system", content="You are an expert data analyst who understands SQL operations and business intent."),
                LLMMessage(role="user", content=analysis_prompt)
            ]
            
            logger.info(f"Calling LLM service with {len(messages)} messages")
            logger.debug(f"Messages: {[f'{msg.role}: {msg.content[:50]}...' for msg in messages]}")
            
            response = self.llm_service.generate_text(
                messages=messages,
                **generation_params
            )
            
            logger.info(f"LLM response success: {response.success}")
            if hasattr(response, 'content'):
                logger.debug(f"Response content length: {len(response.content)}")
            
            if not response.success:
                logger.error(f"LLM generation failed: {response.error}")
                return {"operation_type": "UNKNOWN", "confidence": "LOW", "business_intent": "Could not analyze intent"}
            
            # Try to extract JSON from response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Raw LLM response: {response_text[:500]}...")
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    logger.debug(f"Extracted JSON: {json_str[:200]}...")
                    parsed_json = json.loads(json_str)
                    logger.debug(f"Successfully parsed JSON with keys: {list(parsed_json.keys())}")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Failed JSON: {json_str[:200]}...")
                    return {"operation_type": "COMPLEX", "confidence": "LOW", "business_intent": "JSON parsing failed"}
            else:
                logger.warning("No JSON pattern found in LLM response")
                return {"operation_type": "COMPLEX", "confidence": "LOW", "business_intent": response_text[:500]}
        except Exception as e:
            logger.error(f"Error analyzing SQL intent: {str(e)}")
            return {"operation_type": "UNKNOWN", "confidence": "LOW", "business_intent": "Could not analyze intent"}
    
    def _create_file_context_for_llm(self, file_analysis: List[Dict[str, Any]]) -> str:
        """
        Create formatted context about files for LLM analysis
        """
        context = []
        for i, file_info in enumerate(file_analysis, 1):
            if "error" in file_info:
                context.append(f"File {i}: {file_info['file_name']} - ERROR: {file_info['error']}")
                continue
                
            sample_str = json.dumps(file_info['sample_data'][:2], indent=2) if file_info['sample_data'] else "No sample data"
            context.append(f"""
File {i}: {file_info['file_name']}
- Rows: {file_info['statistics']['rows']}, Columns: {file_info['statistics']['columns']}
- Size: {file_info['statistics']['size_mb']} MB
- Columns: {', '.join(file_info['column_info']['columns'])}
- Sample data: {sample_str}
""")
        return '\n'.join(context)
    
    def _create_data_flow(self, sql_analysis: Dict[str, Any], file_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create detailed data flow steps for visualization
        """
        steps = []
        
        # Safety check for sql_analysis
        if not isinstance(sql_analysis, dict):
            sql_analysis = {"operation_type": "UNKNOWN"}
        
        # Add input files with more details
        for i, file_info in enumerate(file_analysis):
            if not isinstance(file_info, dict):
                continue
            
            file_name = file_info.get("file_name", f"file_{i+1}")
            rows = file_info.get("statistics", {}).get("rows", 0)
            columns = file_info.get("statistics", {}).get("columns", 0)
            
            steps.append({
                "type": "input",
                "file": file_name,
                "role": "primary" if i == 0 else "secondary", 
                "rows": rows,
                "columns": columns,
                "description": f"Load {file_name} ({rows:,} rows, {columns} columns)"
            })
        
        # Parse the business intent to understand complexity
        business_intent = sql_analysis.get("business_intent", "").lower()
        matching_logic = sql_analysis.get("matching_logic", {})
        
        # Add detailed processing steps based on intent analysis
        operation_type = sql_analysis.get("operation_type", "PROCESS")
        
        if "reconciliation" in business_intent or "reconcile" in business_intent:
            self._add_reconciliation_steps(steps, matching_logic, business_intent)
        elif "JOIN" in operation_type:
            self._add_join_steps(steps, matching_logic, business_intent)
        elif "GROUP" in operation_type or "aggregate" in business_intent:
            self._add_aggregation_steps(steps, business_intent)
        elif "filter" in business_intent:
            self._add_filter_steps(steps, business_intent)
        else:
            self._add_generic_steps(steps, operation_type, business_intent)
        
        # Add output step
        expected_output = sql_analysis.get("expected_output", {})
        if isinstance(expected_output, dict):
            estimated_rows = expected_output.get("estimated_rows", {})
            if isinstance(estimated_rows, dict):
                output_rows = estimated_rows.get("likely", estimated_rows.get("max", "unknown"))
            else:
                output_rows = estimated_rows
            description = expected_output.get("description", "Processed results")
        else:
            output_rows = "unknown"
            description = "Processed results"
            
        steps.append({
            "type": "output",
            "description": description,
            "estimated_rows": output_rows,
            "final_result": True
        })
        
        return {"steps": steps}
    
    def _add_reconciliation_steps(self, steps: List[Dict], matching_logic: Dict, business_intent: str):
        """Add detailed reconciliation processing steps"""
        
        # Step 1: Prepare data for matching
        steps.append({
            "type": "preparation",
            "name": "Data Preparation",
            "description": "Standardize formats and prepare columns for matching",
            "details": ["Clean reference numbers", "Normalize amounts", "Handle null values"]
        })
        
        # Step 2: Exact matching first
        column_info = matching_logic.get("description", "reference columns")
        steps.append({
            "type": "matching",
            "name": "Exact Reference Match",
            "description": f"Find records with identical {column_info}",
            "condition": matching_logic.get("column", "Reference fields"),
            "match_type": "exact"
        })
        
        # Step 3: Amount tolerance checking (if mentioned)
        if "tolerance" in business_intent or "within" in business_intent:
            steps.append({
                "type": "tolerance",
                "name": "Amount Tolerance Check", 
                "description": "Apply tolerance rules to matched amounts",
                "condition": "Amount difference ≤ tolerance threshold",
                "details": ["Calculate amount differences", "Apply tolerance rules", "Flag tolerance matches"]
            })
        
        # Step 4: Reconciliation logic
        if "unmatched" in business_intent or "missing" in business_intent:
            steps.append({
                "type": "classification",
                "name": "Match Classification",
                "description": "Categorize transactions by match status",
                "categories": ["Exact matches", "Tolerance matches", "Unmatched A", "Unmatched B"]
            })
        else:
            steps.append({
                "type": "join",
                "name": "Combine Matched Data",
                "description": "Merge matching records from both files",
                "join_type": "INNER JOIN with conditions"
            })
    
    def _add_join_steps(self, steps: List[Dict], matching_logic: Dict, business_intent: str):
        """Add detailed join operation steps"""
        
        column_info = matching_logic.get("column", "key columns")
        join_type = "LEFT" if "missing" in business_intent else "INNER"
        
        steps.append({
            "type": "join",
            "name": f"{join_type} JOIN Operation",
            "description": f"Join files using {column_info}",
            "condition": matching_logic.get("description", f"Match on {column_info}"),
            "join_type": join_type
        })
        
        if "filter" in business_intent or "where" in business_intent:
            steps.append({
                "type": "filter",
                "name": "Apply Filters", 
                "description": "Filter results based on specified conditions",
                "details": ["Apply WHERE conditions", "Remove unwanted records"]
            })
    
    def _add_aggregation_steps(self, steps: List[Dict], business_intent: str):
        """Add aggregation processing steps"""
        
        steps.append({
            "type": "grouping",
            "name": "Group Data",
            "description": "Group records by specified columns",
            "details": ["Identify grouping columns", "Create groups"]
        })
        
        steps.append({
            "type": "aggregation", 
            "name": "Calculate Aggregates",
            "description": "Perform calculations on grouped data",
            "operations": ["COUNT", "SUM", "AVG", "MIN/MAX"]
        })
    
    def _add_filter_steps(self, steps: List[Dict], business_intent: str):
        """Add filtering steps"""
        
        steps.append({
            "type": "filter",
            "name": "Apply Filters",
            "description": "Filter data based on specified criteria",
            "details": ["Evaluate filter conditions", "Remove non-matching records"]
        })
    
    def _add_generic_steps(self, steps: List[Dict], operation_type: str, business_intent: str):
        """Add generic processing steps"""
        
        steps.append({
            "type": "operation",
            "name": operation_type.replace("_", " ").title(),
            "description": f"Perform {operation_type.lower()} operation on the data",
            "details": ["Process according to specified logic"]
        })
    
    def _generate_plain_language_summary(self, original_prompt: str, sql_analysis: Dict[str, Any]) -> str:
        """
        Generate business-friendly plain language summary
        """
        # Safety check for sql_analysis
        if not isinstance(sql_analysis, dict):
            return f"Process data as requested: {original_prompt[:100]}..."
            
        business_intent = sql_analysis.get("business_intent", "")
        if business_intent:
            return business_intent
        
        # Fallback to simplified version of original prompt
        return f"Process data as requested: {original_prompt[:100]}..."
    
    def _calculate_processing_estimates(self, file_analysis: List[Dict[str, Any]], sql_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate processing time and resource estimates
        """
        total_rows = sum(file.get("statistics", {}).get("rows", 0) for file in file_analysis)
        total_size_mb = sum(file.get("statistics", {}).get("size_mb", 0) for file in file_analysis)
        
        # Simple estimation logic
        base_time = 0.1  # Base processing time
        row_factor = total_rows / 10000  # Additional time per 10k rows
        size_factor = total_size_mb / 10  # Additional time per 10MB
        
        operation_multiplier = 1.0
        operation_type = sql_analysis.get("operation_type", "")
        if "JOIN" in operation_type:
            operation_multiplier = 1.5
        elif "GROUP" in operation_type:
            operation_multiplier = 2.0
        
        estimated_time = (base_time + row_factor + size_factor) * operation_multiplier
        
        return {
            "execution_time_seconds": {
                "min": max(0.1, estimated_time * 0.5),
                "max": estimated_time * 2.0,
                "likely": estimated_time
            },
            "memory_usage_mb": max(10, total_size_mb * 2),
            "complexity": "LOW" if estimated_time < 1 else "MEDIUM" if estimated_time < 5 else "HIGH"
        }
    
    def _assess_risks(self, sql_analysis: Dict[str, Any], file_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess potential risks and data quality issues
        """
        risk_factors = []
        warnings = []
        
        # Safety check for sql_analysis
        if not isinstance(sql_analysis, dict):
            sql_analysis = {"confidence": "LOW"}
        
        # Check for large files
        large_files = [f for f in file_analysis if isinstance(f, dict) and f.get("statistics", {}).get("size_mb", 0) > 50]
        if large_files:
            risk_factors.append(f"Large files detected: {len(large_files)} files > 50MB")
        
        # Check for many rows
        total_rows = sum(file.get("statistics", {}).get("rows", 0) for file in file_analysis if isinstance(file, dict))
        if total_rows > 100000:
            risk_factors.append(f"High row count: {total_rows:,} total rows")
        
        # Check for file analysis errors
        error_files = [f for f in file_analysis if isinstance(f, dict) and "error" in f]
        if error_files:
            warnings.append(f"Could not analyze {len(error_files)} files")
        
        # Check confidence level
        confidence = sql_analysis.get("confidence", "MEDIUM")
        if confidence == "LOW":
            warnings.append("Query intent unclear - please review carefully")
        
        if not risk_factors:
            risk_factors.append("None detected")
        
        return {
            "risk_factors": risk_factors,
            "warnings": warnings
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create error response when intent extraction fails
        """
        return {
            "operation_type": "ERROR",
            "business_intent": "Could not analyze query intent",
            "plain_language_summary": f"Error analyzing query: {error_message}",
            "data_flow": {"steps": []},
            "files_involved": [],
            "matching_logic": {},
            "expected_output": {},
            "processing_estimates": {"execution_time_seconds": {"min": 0, "max": 0}, "memory_usage_mb": 0, "complexity": "UNKNOWN"},
            "confidence": "LOW",
            "risk_factors": [f"Analysis failed: {error_message}"],
            "data_quality_warnings": ["Could not complete intent analysis"]
        }