"""
Smart Template Execution Service
Handles intelligent template execution with fallback strategies and column mapping.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from difflib import SequenceMatcher

from app.services.dynamodb_templates_service import dynamodb_templates_service
from app.services.miscellaneous_service import MiscellaneousProcessor
from app.services.storage_service import uploaded_files

logger = logging.getLogger(__name__)


class SmartTemplateExecutionError(Exception):
    """Raised when smart template execution fails"""
    pass


class ColumnMappingResult:
    """Result of column mapping analysis"""
    def __init__(self, status: str, mapped_columns: Dict[str, str] = None, 
                 suggestions: Dict[str, List[str]] = None, error: str = None):
        self.status = status  # 'success', 'needs_mapping', 'failed'
        self.mapped_columns = mapped_columns or {}
        self.suggestions = suggestions or {}
        self.error = error


class SmartTemplateExecutionService:
    """Service for intelligent template execution with fallback strategies"""
    
    def __init__(self):
        self.misc_service = MiscellaneousProcessor()
    
    def _get_file_data_by_id(self, file_id: str) -> Dict[str, Any]:
        """Retrieve file data from storage service"""
        if not uploaded_files.exists(file_id):
            raise SmartTemplateExecutionError(f"File with ID {file_id} not found")
        
        try:
            file_data = uploaded_files.get(file_id)
            file_info = file_data["info"]
            df = file_data["data"]
            filename = file_info["filename"]
            
            return {
                'dataframe': df,
                'filename': filename,
                'file_id': file_id,
                'total_rows': len(df),
                'columns': df.columns.tolist()
            }
        except Exception as e:
            raise SmartTemplateExecutionError(f"Error retrieving file {file_id}: {str(e)}")
        
    def execute_template(self, template_id: str, files: List[Dict[str, Any]], 
                        parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a template with EXACT execution only - no automatic AI fallback
        
        Args:
            template_id: ID of the template to execute
            files: List of files to process
            parameters: Runtime parameters for the template
            
        Returns:
            Execution result with status and detailed error information
        """
        try:
            # Load template
            template = dynamodb_templates_service.get_template(template_id)
            if not template:
                raise SmartTemplateExecutionError(f"Template {template_id} not found")
            
            # Extract template metadata
            template_config = template.get('template_config', {})
            
            # Get file schemas for analysis
            file_schemas = self._extract_file_schemas(files)
            
            # ONLY try exact execution - no automatic fallback
            logger.info(f"Attempting exact execution for template {template_id}")
            exact_result = self._try_exact_execution(template, files, parameters)
            
            if exact_result.get('success') or exact_result.get('status') == 'success':
                logger.info("Exact execution succeeded")
                
                # Generate process ID if not already present
                process_id = exact_result.get('process_id')
                if not process_id:
                    from app.utils.uuid_generator import generate_uuid
                    process_id = generate_uuid('use_case')
                
                result_data = exact_result.get('data', exact_result.get('results', []))
                
                # Store results using the miscellaneous service storage method
                if result_data:
                    storage_data = {
                        'data': result_data,
                        'generated_sql': exact_result.get('generated_sql'),
                        'row_count': exact_result.get('row_count', len(result_data) if isinstance(result_data, list) else 0),
                        'processing_info': {
                            'template_name': template.get('name', 'Unknown'),
                            'execution_method': 'exact',
                            'template_id': template_id
                        }
                    }
                    self.misc_service.store_results(process_id, storage_data)
                
                # Return in the same format as /api/miscellaneous/process/
                return {
                    'success': True,
                    'message': f"Successfully executed use case: {template.get('name', 'Unknown')}",
                    'process_id': process_id,
                    'generated_sql': exact_result.get('generated_sql'),
                    'row_count': exact_result.get('row_count'),
                    'processing_time_seconds': exact_result.get('processing_time_seconds', 0),
                    'errors': exact_result.get('errors', []),
                    'warnings': exact_result.get('warnings', []),
                    'data': result_data,
                    'execution_method': 'exact'
                }
            
            logger.info("Exact execution failed - analyzing error and providing suggestions")
            
            # Analyze the failure and provide detailed suggestions
            error_analysis = self._analyze_execution_error(exact_result.get('error', ''), file_schemas, template)
            
            return {
                # Required fields for SmartExecutionResponse
                'success': False,
                'message': f"Use case execution requires user intervention: {error_analysis.get('user_hint', 'Column mismatch detected')}",
                'process_id': None,
                'generated_sql': None,
                'row_count': 0,
                'processing_time_seconds': 0.0,
                'errors': [exact_result.get('error', 'Unknown error')],
                'warnings': [],
                'data': None,
                
                # Error-specific fields for user intervention
                'template_id': template_id,
                'execution_error': exact_result.get('error', 'Unknown error'),
                'error_analysis': error_analysis,
                'available_options': [
                    {
                        'option': 'ai_assisted',
                        'label': 'Try with AI Assistance',
                        'description': 'Let AI analyze your data and adapt the query automatically'
                    },
                    {
                        'option': 'column_mapping', 
                        'label': 'Manual Column Mapping',
                        'description': 'Map your file columns to the expected columns manually'
                    },
                    {
                        'option': 'cancel',
                        'label': 'Cancel',
                        'description': 'Stop execution and return to use case selection'
                    }
                ],
                'file_schemas': file_schemas
            }
            
        except Exception as e:
            logger.error(f"Error executing template {template_id}: {e}")
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'process_id': None,
                'generated_sql': None,
                'row_count': 0,
                'processing_time_seconds': 0.0,
                'errors': [str(e)],
                'warnings': [],
                'data': None
            }
    
    def _try_exact_execution(self, template: Dict[str, Any], files: List[Dict[str, Any]], 
                           parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Try executing template with original SQL/prompt"""
        try:
            # Use existing template content or generate from prompt
            template_content = template.get('template_content', template.get('description', ''))
            
            if not template_content:
                return {'status': 'failed', 'error': 'No executable content in template'}
            
            # Retrieve actual file data
            retrieved_files = []
            for i, file_info in enumerate(files):
                file_data = self._get_file_data_by_id(file_info['file_id'])
                file_data['role'] = f'file_{i}'
                file_data['label'] = file_info.get('filename', f'File {i+1}')
                retrieved_files.append(file_data)
            
            # Execute using miscellaneous service
            result = self.misc_service.process_core_request(
                user_prompt=template_content,
                files_data=retrieved_files,
                output_format='json',
                execute_exact_sql=True,
                exact_sql_query= template['template_metadata']['processing_context']['generated_sql']
            )
            
            if result.get('success'):
               return result
            else:
                return {
                    'status': 'failed',
                    'error': result.get('error', 'Execution failed')
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'process_id': None,
                'generated_sql': None,
                'row_count': 0,
                'processing_time_seconds': 0.0,
                'errors': [str(e)],
                'warnings': [],
                'data': None
            }
    
    def _extract_file_schemas(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract column information from files"""
        schemas = []
        for file_info in files:
            schema = {
                'filename': file_info.get('filename', ''),
                'columns': file_info.get('columns', []),
                'total_rows': file_info.get('total_rows', 0)
            }
            schemas.append(schema)
        return schemas
    
    def _analyze_column_mapping(self, stored_mapping: Dict[str, List[str]], 
                              current_schemas: List[Dict[str, Any]]) -> ColumnMappingResult:
        """
        Analyze if stored column mapping can be applied to current schemas
        """
        try:
            # Get all current columns from all files
            current_columns = []
            for schema in current_schemas:
                current_columns.extend(schema.get('columns', []))
            current_columns = list(set(current_columns))  # Remove duplicates
            
            if not stored_mapping:
                # No stored mapping, try to suggest mappings
                return ColumnMappingResult(
                    status='needs_mapping',
                    suggestions=self._suggest_column_mappings(current_columns)
                )
            
            mapped_columns = {}
            needs_mapping = {}
            
            # Try to map each stored column to current columns
            for template_column, possible_matches in stored_mapping.items():
                best_match = self._find_best_column_match(template_column, possible_matches, current_columns)
                
                if best_match:
                    mapped_columns[template_column] = best_match
                else:
                    # Suggest possible matches using fuzzy matching
                    suggestions = self._fuzzy_match_columns(template_column, current_columns)
                    if suggestions:
                        needs_mapping[template_column] = suggestions
            
            if needs_mapping:
                return ColumnMappingResult(
                    status='needs_mapping',
                    mapped_columns=mapped_columns,
                    suggestions=needs_mapping
                )
            else:
                return ColumnMappingResult(
                    status='success',
                    mapped_columns=mapped_columns
                )
                
        except Exception as e:
            logger.error(f"Error analyzing column mapping: {e}")
            return ColumnMappingResult(
                status='failed',
                error=str(e)
            )
    
    def _find_best_column_match(self, template_column: str, possible_matches: List[str], 
                               current_columns: List[str]) -> Optional[str]:
        """Find the best match for a template column in current columns"""
        # First try exact matches
        for match in possible_matches:
            if match in current_columns:
                return match
        
        # Then try fuzzy matching
        for match in possible_matches:
            fuzzy_matches = self._fuzzy_match_columns(match, current_columns, threshold=0.8)
            if fuzzy_matches:
                return fuzzy_matches[0]['column']
        
        return None
    
    def _fuzzy_match_columns(self, target_column: str, available_columns: List[str], 
                           threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Find fuzzy matches for a column name"""
        matches = []
        
        for column in available_columns:
            # Calculate similarity score
            similarity = SequenceMatcher(None, target_column.lower(), column.lower()).ratio()
            
            if similarity >= threshold:
                matches.append({
                    'column': column,
                    'similarity': similarity,
                    'match_type': 'fuzzy'
                })
        
        # Sort by similarity score
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def _analyze_execution_error(self, error_message: str, file_schemas: List[Dict[str, Any]], 
                                template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the execution error and provide helpful suggestions
        """
        analysis = {
            'error_type': 'unknown',
            'missing_columns': [],
            'available_columns': [],
            'suggestions': [],
            'user_hint': ''
        }
        
        try:
            # Get all available columns from user files
            all_available_columns = []
            for schema in file_schemas:
                all_available_columns.extend(schema.get('columns', []))
            all_available_columns = list(set(all_available_columns))
            analysis['available_columns'] = all_available_columns
            
            # Parse SQL binder errors to find missing columns
            if 'Binder Error' in error_message:
                analysis['error_type'] = 'missing_column'
                
                # Extract missing column name from different error patterns
                import re
                
                # Pattern 1: "Referenced column "column_name" not found"
                column_match = re.search(r'Referenced column "([^"]+)" not found', error_message)
                
                # Pattern 2: "does not have a column named "column_name""
                if not column_match:
                    column_match = re.search(r'does not have a column named "([^"]+)"', error_message)
                
                # Pattern 3: "column "column_name" not found"
                if not column_match:
                    column_match = re.search(r'column "([^"]+)" not found', error_message)
                
                if column_match:
                    missing_column = column_match.group(1)
                    analysis['missing_columns'] = [missing_column]
                    
                    # Find similar columns using fuzzy matching
                    similar_columns = self._fuzzy_match_columns(missing_column, all_available_columns, threshold=0.4)
                    
                    if similar_columns:
                        analysis['suggestions'] = [
                            f"Did you mean '{col['column']}'? (similarity: {col['similarity']:.1%})" 
                            for col in similar_columns[:3]
                        ]
                        analysis['user_hint'] = f"The template expects a column named '{missing_column}', but your file has similar columns like: {', '.join([col['column'] for col in similar_columns[:3]])}"
                    else:
                        analysis['user_hint'] = f"The template expects a column named '{missing_column}', but it's not found in your data. Your file has columns: {', '.join(all_available_columns[:5])}"
                
                # Extract candidate bindings (available columns from error message)
                candidates_match = re.search(r'Candidate bindings: ([^\\n]+)', error_message)
                if candidates_match:
                    candidates_text = candidates_match.group(1)
                    # Parse out column names from "file_1.column_name" format
                    candidate_columns = re.findall(r'"[^"]*\.([^"]+)"', candidates_text)
                    if candidate_columns:
                        analysis['available_columns'] = candidate_columns
            
            elif 'Missing required columns' in error_message:
                analysis['error_type'] = 'missing_required_columns'
                
                # Extract missing columns list
                columns_match = re.search(r'Missing required columns: \[(.*?)\]', error_message)
                if columns_match:
                    columns_text = columns_match.group(1)
                    missing_columns = [col.strip().strip("'\"") for col in columns_text.split(',')]
                    analysis['missing_columns'] = missing_columns
                    
                    analysis['user_hint'] = f"Your template requires these columns: {', '.join(missing_columns)}, but they're missing from your data. Available columns: {', '.join(all_available_columns[:5])}"
                    
                    # Find suggestions for each missing column
                    suggestions = []
                    for missing_col in missing_columns:
                        similar = self._fuzzy_match_columns(missing_col, all_available_columns, threshold=0.4)
                        if similar:
                            suggestions.append(f"For '{missing_col}', try '{similar[0]['column']}' (similarity: {similar[0]['similarity']:.1%})")
                    
                    analysis['suggestions'] = suggestions
            
            else:
                analysis['error_type'] = 'general_sql_error'
                analysis['user_hint'] = "There was an SQL execution error. This might be due to data format issues or incompatible column types."
                analysis['suggestions'] = [
                    "Check if your data format matches what the template expects",
                    "Verify that column names and data types are compatible",
                    "Try using AI assistance to automatically adapt the query to your data"
                ]
            
        except Exception as e:
            logger.error(f"Error analyzing execution error: {e}")
            analysis['user_hint'] = "Unable to analyze the error automatically. Try using AI assistance or manual column mapping."
        
        return analysis
    
    def execute_with_ai_assistance(self, template_id: str, files: List[Dict[str, Any]], 
                                  parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute template with AI assistance - only called with explicit user consent
        """
        try:
            logger.info(f"Starting AI-assisted execution for template {template_id}")
            
            # Load template
            template = dynamodb_templates_service.get_template(template_id)
            if not template:
                raise SmartTemplateExecutionError(f"Template {template_id} not found")
            
            # Get template content
            template_content = template.get('template_content', template.get('description', ''))
            if not template_content:
                return {
                    'status': 'failed',
                    'error': 'No executable content in template for AI assistance'
                }
            
            # Retrieve actual file data
            retrieved_files = []
            for i, file_info in enumerate(files):
                file_data = self._get_file_data_by_id(file_info['file_id'])
                file_data['role'] = f'file_{i}'
                file_data['label'] = file_info.get('filename', f'File {i+1}')
                retrieved_files.append(file_data)
            
            # Execute using AI with natural language processing
            logger.info("Executing with AI assistance - adapting query to user data")
            result = self.misc_service.process_core_request(
                user_prompt=f"ADAPT THE FOLLOWING TEMPLATE TO THE USER'S DATA:\n\n{template_content}\n\nPlease analyze the user's file structure and adapt this template accordingly. Be flexible with column names and data formats.",
                files_data=retrieved_files,
                output_format='json'
            )
            
            if result.get('success'):
                logger.info("AI-assisted execution succeeded")
                
                # Generate process ID if not already present
                process_id = result.get('process_id')
                if not process_id:
                    from app.utils.uuid_generator import generate_uuid
                    process_id = generate_uuid('use_case')
                
                result_data = result.get('data', result.get('results', []))
                
                # Store results using the miscellaneous service storage method
                if result_data:
                    storage_data = {
                        'data': result_data,
                        'generated_sql': result.get('generated_sql'),
                        'row_count': result.get('row_count', len(result_data) if isinstance(result_data, list) else 0),
                        'processing_info': {
                            'template_name': template.get('name', 'Unknown'),
                            'execution_method': 'ai_assisted',
                            'template_id': template_id,
                            'ai_adaptations': 'AI adapted the template to work with your specific data structure'
                        }
                    }
                    self.misc_service.store_results(process_id, storage_data)
                
                # Return in the same format as /api/miscellaneous/process/
                return {
                    'success': True,
                    'message': f"Successfully executed use case with AI assistance: {template.get('name', 'Unknown')}",
                    'process_id': process_id,
                    'generated_sql': result.get('generated_sql'),
                    'row_count': result.get('row_count'),
                    'processing_time_seconds': result.get('processing_time_seconds', 0),
                    'errors': result.get('errors', []),
                    'warnings': result.get('warnings', []),
                    'data': result_data,
                    'execution_method': 'ai_assisted',
                    'ai_adaptations': 'AI adapted the template to work with your specific data structure'
                }
            else:
                logger.error("AI-assisted execution failed")
                return {
                    'success': False,
                    'message': f"AI-assisted execution failed: {result.get('error', 'Unknown error')}",
                    'process_id': None,
                    'generated_sql': None,
                    'row_count': 0,
                    'processing_time_seconds': 0.0,
                    'errors': [result.get('error', 'AI-assisted execution failed')],
                    'warnings': [],
                    'data': None
                }
                
        except Exception as e:
            logger.error(f"Error in AI-assisted execution: {e}")
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'process_id': None,
                'generated_sql': None,
                'row_count': 0,
                'processing_time_seconds': 0.0,
                'errors': [str(e)],
                'warnings': [],
                'data': None
            }
    
    def _suggest_column_mappings(self, current_columns: List[str]) -> Dict[str, List[str]]:
        """Suggest initial column mappings based on common patterns"""
        suggestions = {}
        
        # Common column patterns
        patterns = {
            'customer': ['customer', 'client', 'cust', 'customer_name', 'client_name'],
            'amount': ['amount', 'value', 'revenue', 'sales', 'price', 'total'],
            'date': ['date', 'timestamp', 'created_at', 'transaction_date', 'order_date'],
            'id': ['id', 'identifier', 'key', 'reference', 'number'],
            'name': ['name', 'title', 'description', 'label'],
            'status': ['status', 'state', 'condition', 'flag']
        }
        
        for pattern_name, pattern_keywords in patterns.items():
            matching_columns = []
            for column in current_columns:
                column_lower = column.lower()
                for keyword in pattern_keywords:
                    if keyword in column_lower:
                        matching_columns.append(column)
                        break
            
            if matching_columns:
                suggestions[pattern_name] = matching_columns
        
        return suggestions
    
    def _execute_with_mapping(self, template: Dict[str, Any], files: List[Dict[str, Any]], 
                            parameters: Dict[str, Any], column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Execute template with applied column mapping"""
        try:
            # Get template content
            template_content = template.get('template_content', template.get('description', ''))
            
            # Apply column mapping to the template content
            mapped_content = self._apply_column_mapping_to_content(template_content, column_mapping)
            
            # Retrieve actual file data
            retrieved_files = []
            for i, file_info in enumerate(files):
                file_data = self._get_file_data_by_id(file_info['file_id'])
                file_data['role'] = f'file_{i}'
                file_data['label'] = file_info.get('filename', f'File {i+1}')
                retrieved_files.append(file_data)
            
            # Execute with mapped content
            result = self.misc_service.process_core_request(
                user_prompt=mapped_content,
                files_data=retrieved_files,
                output_format='json'
            )
            
            if result.get('success'):
                # Generate process ID if not already present
                process_id = result.get('process_id')
                if not process_id:
                    from app.utils.uuid_generator import generate_uuid
                    process_id = generate_uuid('use_case')
                
                result_data = result.get('data', result.get('results', []))
                
                # Store results using the miscellaneous service storage method
                if result_data:
                    storage_data = {
                        'data': result_data,
                        'generated_sql': result.get('generated_sql'),
                        'row_count': result.get('row_count', len(result_data) if isinstance(result_data, list) else 0),
                        'processing_info': {
                            'template_name': template.get('name', 'Unknown'),
                            'execution_method': 'mapped',
                            'template_id': template.get('id', 'Unknown'),
                            'applied_mapping': column_mapping
                        }
                    }
                    self.misc_service.store_results(process_id, storage_data)
                
                # Return in the same format as /api/miscellaneous/process/
                return {
                    'success': True,
                    'message': f"Successfully executed use case with column mapping: {template.get('name', 'Unknown')}",
                    'process_id': process_id,
                    'generated_sql': result.get('generated_sql'),
                    'row_count': result.get('row_count'),
                    'processing_time_seconds': result.get('processing_time_seconds', 0),
                    'errors': result.get('errors', []),
                    'warnings': result.get('warnings', []),
                    'data': result_data,
                    'execution_method': 'mapped',
                    'applied_mapping': column_mapping
                }
            else:
                return {
                    'status': 'failed',
                    'error': result.get('error', 'Mapped execution failed')
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'process_id': None,
                'generated_sql': None,
                'row_count': 0,
                'processing_time_seconds': 0.0,
                'errors': [str(e)],
                'warnings': [],
                'data': None
            }
    
    def _apply_column_mapping_to_content(self, content: str, mapping: Dict[str, str]) -> str:
        """Apply column mapping to template content"""
        mapped_content = content
        
        # Replace column references in the content
        for template_col, actual_col in mapping.items():
            # Simple replacement - could be made more sophisticated
            mapped_content = mapped_content.replace(template_col, actual_col)
        
        # Add mapping information to the prompt
        mapping_info = ", ".join([f"{k} is now {v}" for k, v in mapping.items()])
        mapped_content += f"\n\nNote: Column mapping applied - {mapping_info}"
        
        return mapped_content
    
    def _update_template_mapping(self, template_id: str, new_mapping: Dict[str, str]):
        """Update template with successful column mapping"""
        try:
            template = dynamodb_templates_service.get_template(template_id)
            if not template:
                return
            
            # Update column mapping in template config
            template_config = template.get('template_config', {})
            stored_mapping = template_config.get('column_mapping', {})
            
            # Add new mappings to stored mapping
            for template_col, actual_col in new_mapping.items():
                if template_col not in stored_mapping:
                    stored_mapping[template_col] = []
                
                if actual_col not in stored_mapping[template_col]:
                    stored_mapping[template_col].append(actual_col)
            
            # Update template
            template_config['column_mapping'] = stored_mapping
            template_config['last_mapping_update'] = datetime.utcnow().isoformat()
            
            updates = {
                'template_config': template_config,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            dynamodb_templates_service.update_template(template_id, updates)
            logger.info(f"Updated column mapping for template {template_id}")
            
        except Exception as e:
            logger.error(f"Error updating template mapping: {e}")
    
    def apply_user_column_mapping(self, template_id: str, user_mapping: Dict[str, str], 
                                 files: List[Dict[str, Any]], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply user-provided column mapping and execute template
        """
        try:
            template = dynamodb_templates_service.get_template(template_id)
            if not template:
                raise SmartTemplateExecutionError(f"Template {template_id} not found")
            
            # Execute with user mapping
            result = self._execute_with_mapping(template, files, parameters, user_mapping)
            
            if result.get('success') == True:
                # Update template with successful mapping
                self._update_template_mapping(template_id, user_mapping)
                
            return result
            
        except Exception as e:
            logger.error(f"Error applying user column mapping: {e}")
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'process_id': None,
                'generated_sql': None,
                'row_count': 0,
                'processing_time_seconds': 0.0,
                'errors': [str(e)],
                'warnings': [],
                'data': None
            }


# Create singleton instance
smart_template_execution_service = SmartTemplateExecutionService()