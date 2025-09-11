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
        Execute a template with smart fallback strategies
        
        Args:
            template_id: ID of the template to execute
            files: List of files to process
            parameters: Runtime parameters for the template
            
        Returns:
            Execution result with status and data
        """
        try:
            # Load template
            template = dynamodb_templates_service.get_template(template_id)
            if not template:
                raise SmartTemplateExecutionError(f"Template {template_id} not found")
            
            # Extract template metadata
            template_config = template.get('template_config', {})
            column_mapping = template_config.get('column_mapping', {})
            primary_sql =  template['template_metadata']['processing_context']['generated_sql']
            fallback_strategy = template_config.get('fallback_strategy', 'fuzzy_match')
            
            # Get file schemas
            file_schemas = self._extract_file_schemas(files)
            
            # Strategy 1: Try exact execution
            logger.info(f"Attempting exact execution for template {template_id}")
            exact_result = self._try_exact_execution(template, files, parameters)
            if exact_result['success']:
                logger.info("Exact execution succeeded")
                return exact_result
            
            logger.info("Exact execution failed, trying column mapping")
            
            # Strategy 2: Try with column mapping
            mapping_result = self._analyze_column_mapping(column_mapping, file_schemas)
            
            if mapping_result.status == 'success':
                # Apply mapping and try execution
                mapped_result = self._execute_with_mapping(template, files, parameters, mapping_result.mapped_columns)
                if mapped_result['status'] == 'success':
                    # Update template with successful mapping
                    self._update_template_mapping(template_id, mapping_result.mapped_columns)
                    logger.info("Execution with column mapping succeeded")
                    return mapped_result
            
            # Strategy 3: Return mapping suggestions for user intervention
            if mapping_result.status == 'needs_mapping':
                logger.info("Template needs user intervention for column mapping")
                return {
                    'status': 'needs_mapping',
                    'template_id': template_id,
                    'suggestions': mapping_result.suggestions,
                    'error': 'Column mapping required - some columns could not be automatically matched'
                }
            
            # All strategies failed
            logger.error(f"All execution strategies failed for template {template_id}")
            return {
                'status': 'failed',
                'error': f"Template execution failed: {exact_result.get('error', 'Unknown error')}"
            }
            
        except Exception as e:
            logger.error(f"Error executing template {template_id}: {e}")
            return {
                'status': 'failed',
                'error': str(e)
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
                'status': 'failed',
                'error': str(e)
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
                return {
                    'status': 'success',
                    'data': result.get('results'),
                    'process_id': result.get('process_id'),
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
                'status': 'failed',
                'error': str(e)
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
            
            if result['status'] == 'success':
                # Update template with successful mapping
                self._update_template_mapping(template_id, user_mapping)
                
            return result
            
        except Exception as e:
            logger.error(f"Error applying user column mapping: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }


# Create singleton instance
smart_template_execution_service = SmartTemplateExecutionService()