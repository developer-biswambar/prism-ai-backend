"""
Template Application Service
Applies templates to user data and generates executable queries from templates.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from app.services.dynamodb_templates_service import dynamodb_templates_service
from app.services.llm_service import get_llm_service, get_llm_generation_params, LLMMessage

logger = logging.getLogger(__name__)


class TemplateApplicationError(Exception):
    """Raised when template application fails"""
    pass


class TemplateApplicationService:
    """Service for applying templates to user queries and data"""
    
    def __init__(self):
        self.llm_service = get_llm_service()
        self.generation_params = get_llm_generation_params()
    
    def suggest_templates(self, user_prompt: str, file_schemas: List[Dict[str, Any]], 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest templates based on user prompt and file schemas using AI analysis
        
        Args:
            user_prompt: User's natural language query
            file_schemas: List of file schemas with column information
            limit: Maximum number of templates to return
            
        Returns:
            List of suggested templates with match scores
        """
        try:
            # Analyze user intent and data structure
            intent_analysis = self._analyze_user_intent(user_prompt, file_schemas)
            
            # Get all templates and score them
            all_templates = dynamodb_templates_service.list_templates(limit=200)  # Get more for scoring
            scored_templates = []
            
            for template in all_templates:
                score = self._calculate_template_match_score(
                    template, intent_analysis, user_prompt, file_schemas
                )
                if score > 0.3:  # Only suggest templates with decent match
                    template['match_score'] = score
                    template['match_reasons'] = self._get_match_reasons(template, intent_analysis)
                    scored_templates.append(template)
            
            # Sort by score and return top suggestions
            scored_templates.sort(key=lambda x: x['match_score'], reverse=True)
            return scored_templates[:limit]
            
        except Exception as e:
            logger.error(f"Error suggesting templates: {e}")
            return []
    
    def apply_template(self, template_id: str, user_data: Dict[str, Any], 
                      user_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply a template to user data and generate executable query
        
        Args:
            template_id: ID of template to apply
            user_data: User's data context (files, columns, etc.)
            user_params: User-provided parameter values
            
        Returns:
            Dictionary containing generated query and application details
        """
        try:
            # Get template
            template = dynamodb_templates_service.get_template(template_id)
            if not template:
                raise TemplateApplicationError(f"Template {template_id} not found")
            
            # Validate user data against template requirements
            validation_result = self._validate_template_requirements(template, user_data)
            if not validation_result['valid']:
                raise TemplateApplicationError(
                    f"Template requirements not met: {validation_result['errors']}"
                )
            
            # Map user data to template parameters
            column_mapping = self._map_columns_to_template(template, user_data)
            
            # Merge user parameters with defaults
            parameters = self._resolve_template_parameters(template, user_params or {})
            
            # Generate executable query
            generated_query = self._generate_query_from_template(
                template, column_mapping, parameters, user_data
            )
            
            # Mark template as used
            dynamodb_templates_service.mark_template_as_used(template_id)
            
            return {
                'success': True,
                'template_id': template_id,
                'template_name': template['name'],
                'generated_query': generated_query,
                'column_mapping': column_mapping,
                'parameters': parameters,
                'validation_result': validation_result,
                'applied_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error applying template {template_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'template_id': template_id
            }
    
    def create_template_from_successful_query(self, query_data: Dict[str, Any], 
                                            template_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new template from a successful query execution
        
        Args:
            query_data: Successful query execution data
            template_metadata: Template metadata (name, description, etc.)
            
        Returns:
            Created template information
        """
        try:
            # Extract template configuration from successful query
            template_config = self._extract_template_config(query_data)
            
            # Create template data with all required fields
            current_time = datetime.utcnow().isoformat()
            template_data = {
                'id': f"template_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'name': template_metadata['name'],
                'description': template_metadata['description'],
                'template_type': template_metadata.get('template_type', 'data_processing'),
                'category': template_metadata.get('category', 'Custom'),
                'tags': template_metadata.get('tags', []),
                'template_config': template_config,
                'version': '1.0',
                'is_public': template_metadata.get('is_public', False),
                'created_by': template_metadata.get('created_by'),
                'created_at': current_time,
                'updated_at': current_time,
                'usage_count': 0,
                'last_used_at': None,
                'rating': 0.0,
                'rating_count': 0
            }
            
            # Save template
            success = dynamodb_templates_service.save_template(template_data)
            if not success:
                raise TemplateApplicationError("Failed to save template")
            
            return {
                'success': True,
                'template_id': template_data['id'],
                'template': template_data
            }
            
        except Exception as e:
            logger.error(f"Error creating template from query: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_user_intent(self, user_prompt: str, file_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user intent using AI"""
        try:
            schema_context = self._format_schema_context(file_schemas)
            
            messages = [
                LLMMessage(
                    role="system",
                    content="""You are an expert data analyst. Analyze the user's query and data structure to determine their intent.
                    
                    Categorize the intent into one of these types:
                    - reconciliation: Matching/comparing data between sources
                    - analysis: Statistical analysis, trends, insights
                    - transformation: Data cleaning, formatting, restructuring  
                    - reporting: Summaries, dashboards, formatted outputs
                    - data_processing: General data manipulation
                    
                    Return JSON with:
                    {
                        "intent_type": "string",
                        "confidence": "high|medium|low", 
                        "key_operations": ["list", "of", "operations"],
                        "data_patterns": ["patterns", "detected"],
                        "suggested_categories": ["Finance", "Operations", etc.],
                        "complexity": "simple|medium|complex"
                    }"""
                ),
                LLMMessage(
                    role="user", 
                    content=f"User Query: {user_prompt}\n\nData Structure:\n{schema_context}"
                )
            ]
            
            response = self.llm_service.generate(messages, **self.generation_params)
            
            # Parse JSON response
            import json
            return json.loads(response.content)
            
        except Exception as e:
            logger.error(f"Error analyzing user intent: {e}")
            return {
                'intent_type': 'data_processing',
                'confidence': 'low',
                'key_operations': [],
                'data_patterns': [],
                'suggested_categories': ['Custom'],
                'complexity': 'medium'
            }
    
    def _calculate_template_match_score(self, template: Dict[str, Any], 
                                      intent_analysis: Dict[str, Any],
                                      user_prompt: str, file_schemas: List[Dict[str, Any]]) -> float:
        """Calculate how well a template matches user requirements"""
        try:
            score = 0.0
            
            # Intent type match (40% weight)
            if template.get('template_type') == intent_analysis.get('intent_type'):
                score += 0.4
            elif template.get('template_type') in ['data_processing'] and intent_analysis.get('intent_type'):
                score += 0.2  # Generic templates get partial score
            
            # Category match (20% weight)
            template_categories = [template.get('category', '').lower()]
            intent_categories = [cat.lower() for cat in intent_analysis.get('suggested_categories', [])]
            if any(cat in template_categories for cat in intent_categories):
                score += 0.2
            
            # Tag/keyword match (20% weight)
            template_tags = [tag.lower() for tag in template.get('tags', [])]
            user_words = user_prompt.lower().split()
            keyword_matches = sum(1 for tag in template_tags if any(word in tag or tag in word for word in user_words))
            if template_tags:
                score += 0.2 * (keyword_matches / len(template_tags))
            
            # Column requirement match (15% weight)
            required_columns = template.get('template_config', {}).get('required_columns', [])
            if required_columns:
                available_columns = []
                for schema in file_schemas:
                    available_columns.extend(schema.get('columns', []))
                
                column_matches = sum(1 for req_col in required_columns 
                                   if any(req_col.lower() in col.lower() or col.lower() in req_col.lower() 
                                         for col in available_columns))
                score += 0.15 * (column_matches / len(required_columns))
            else:
                score += 0.15  # No requirements is good
            
            # Popularity boost (5% weight)
            usage_count = template.get('usage_count', 0)
            rating = template.get('rating', 0)
            popularity_score = min(1.0, (usage_count / 100.0) + (rating / 5.0))
            score += 0.05 * popularity_score
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating template match score: {e}")
            return 0.0
    
    def _get_match_reasons(self, template: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasons why template matches"""
        reasons = []
        
        if template.get('template_type') == intent_analysis.get('intent_type'):
            reasons.append(f"Perfect match for {intent_analysis['intent_type']} operations")
        
        if template.get('category') in intent_analysis.get('suggested_categories', []):
            reasons.append(f"Matches {template['category']} domain")
        
        if template.get('usage_count', 0) > 50:
            reasons.append("Popular template with proven track record")
        
        if template.get('rating', 0) > 4.0:
            reasons.append("Highly rated by users")
        
        return reasons
    
    def _validate_template_requirements(self, template: Dict[str, Any], 
                                      user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that user data meets template requirements"""
        try:
            errors = []
            warnings = []
            
            template_config = template.get('template_config', {})
            required_columns = template_config.get('required_columns', [])
            validation_rules = template_config.get('validation_rules', [])
            
            # Check required columns
            user_columns = []
            for file_info in user_data.get('files', []):
                user_columns.extend(file_info.get('columns', []))
            
            missing_columns = []
            for req_col in required_columns:
                if not any(req_col.lower() in col.lower() or col.lower() in req_col.lower() 
                          for col in user_columns):
                    missing_columns.append(req_col)
            
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check validation rules
            for rule in validation_rules:
                rule_result = self._check_validation_rule(rule, user_data)
                if not rule_result['valid']:
                    if rule_result['severity'] == 'error':
                        errors.append(rule_result['message'])
                    else:
                        warnings.append(rule_result['message'])
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Error validating template requirements: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def _map_columns_to_template(self, template: Dict[str, Any], 
                               user_data: Dict[str, Any]) -> Dict[str, str]:
        """Map user columns to template column requirements"""
        try:
            mapping = {}
            template_config = template.get('template_config', {})
            required_columns = template_config.get('required_columns', [])
            optional_columns = template_config.get('optional_columns', [])
            
            # Get all available columns
            user_columns = {}  # column_name -> file_info
            for file_info in user_data.get('files', []):
                for col in file_info.get('columns', []):
                    user_columns[col] = file_info
            
            # Map required columns
            for req_col in required_columns:
                best_match = self._find_best_column_match(req_col, list(user_columns.keys()))
                if best_match:
                    mapping[req_col] = best_match
            
            # Map optional columns if available
            for opt_col in optional_columns:
                if opt_col not in mapping:  # Don't overwrite required mappings
                    best_match = self._find_best_column_match(opt_col, list(user_columns.keys()))
                    if best_match:
                        mapping[opt_col] = best_match
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error mapping columns to template: {e}")
            return {}
    
    def _find_best_column_match(self, target_column: str, available_columns: List[str]) -> Optional[str]:
        """Find the best matching column from available columns"""
        target_lower = target_column.lower()
        
        # Exact match
        for col in available_columns:
            if col.lower() == target_lower:
                return col
        
        # Substring match
        for col in available_columns:
            if target_lower in col.lower() or col.lower() in target_lower:
                return col
        
        # Common synonyms
        synonyms = {
            'amount': ['value', 'total', 'sum', 'price', 'cost'],
            'reference': ['ref', 'id', 'number', 'identifier'],
            'date': ['time', 'timestamp', 'created', 'updated'],
            'description': ['desc', 'name', 'title', 'details']
        }
        
        for col in available_columns:
            col_lower = col.lower()
            if target_lower in synonyms:
                if any(synonym in col_lower for synonym in synonyms[target_lower]):
                    return col
            
            # Reverse check
            for synonym_key, synonym_list in synonyms.items():
                if synonym_key in col_lower and target_lower in synonym_list:
                    return col
        
        return None
    
    def _resolve_template_parameters(self, template: Dict[str, Any], 
                                   user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template parameters with user values and defaults"""
        try:
            resolved_params = {}
            template_config = template.get('template_config', {})
            parameter_definitions = template_config.get('parameters', [])
            
            for param_def in parameter_definitions:
                param_name = param_def['name']
                
                # Use user-provided value if available
                if param_name in user_params:
                    resolved_params[param_name] = user_params[param_name]
                # Otherwise use default
                elif 'default' in param_def:
                    resolved_params[param_name] = param_def['default']
                # Required parameter without default or user value
                else:
                    logger.warning(f"Required parameter '{param_name}' not provided")
                    resolved_params[param_name] = None
            
            return resolved_params
            
        except Exception as e:
            logger.error(f"Error resolving template parameters: {e}")
            return {}
    
    def _generate_query_from_template(self, template: Dict[str, Any], 
                                    column_mapping: Dict[str, str],
                                    parameters: Dict[str, Any],
                                    user_data: Dict[str, Any]) -> str:
        """Generate executable query from template"""
        try:
            template_config = template.get('template_config', {})
            prompt_template = template_config.get('prompt_template', '')
            
            # Replace column placeholders
            query = prompt_template
            for template_col, user_col in column_mapping.items():
                query = query.replace(f"{{{template_col}}}", user_col)
            
            # Replace parameter placeholders
            for param_name, param_value in parameters.items():
                if param_value is not None:
                    query = query.replace(f"{{{param_name}}}", str(param_value))
            
            # Add file context information
            file_context = ""
            for i, file_info in enumerate(user_data.get('files', [])):
                file_context += f"File {i+1} ({file_info.get('filename', 'unnamed')}): "
                file_context += f"columns {file_info.get('columns', [])}\n"
            
            # Enhance query with AI if needed
            if '{AI_ENHANCE}' in query:
                enhanced_query = self._ai_enhance_query(query, file_context, template)
                query = enhanced_query
            
            return query.strip()
            
        except Exception as e:
            logger.error(f"Error generating query from template: {e}")
            return f"Error applying template: {str(e)}"
    
    def _ai_enhance_query(self, base_query: str, file_context: str, template: Dict[str, Any]) -> str:
        """Use AI to enhance/optimize the generated query"""
        try:
            messages = [
                LLMMessage(
                    role="system",
                    content="""You are an expert data analyst. Enhance the given query template with the file context provided.
                    Make the query more specific and optimized for the actual data structure.
                    Return only the enhanced natural language query, no explanations."""
                ),
                LLMMessage(
                    role="user",
                    content=f"Base Query Template: {base_query}\n\nFile Context:\n{file_context}\n\nTemplate Type: {template.get('template_type')}"
                )
            ]
            
            response = self.llm_service.generate(messages, **self.generation_params)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error AI-enhancing query: {e}")
            return base_query.replace('{AI_ENHANCE}', '')
    
    def _check_validation_rule(self, rule: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check a validation rule against user data"""
        # Simple validation rule checking
        # You can expand this with more sophisticated rule parsing
        
        if rule == 'amount_positive':
            return {'valid': True, 'severity': 'warning', 'message': 'Ensure amount columns contain positive values'}
        elif rule == 'reference_not_null':
            return {'valid': True, 'severity': 'warning', 'message': 'Ensure reference columns are not empty'}
        elif rule == 'date_format_valid':
            return {'valid': True, 'severity': 'warning', 'message': 'Ensure date columns have valid date format'}
        
        return {'valid': True, 'severity': 'info', 'message': f'Rule {rule} not implemented'}
    
    def _extract_template_config(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract template configuration from successful query data"""
        try:
            # Analyze the successful query to create template config
            user_prompt = query_data.get('user_prompt', '')
            file_schemas = query_data.get('file_schemas', [])
            
            # Extract column requirements
            required_columns = []
            optional_columns = []
            
            # Simple heuristic: extract column names mentioned in prompt
            for schema in file_schemas:
                for col in schema.get('columns', []):
                    if col.lower() in user_prompt.lower():
                        required_columns.append(col)
            
            # Extract parameters from prompt (look for numeric values, dates, etc.)
            parameters = []
            
            # Look for tolerance values
            tolerance_match = re.search(r'(\d+\.?\d*)\s*tolerance', user_prompt.lower())
            if tolerance_match:
                parameters.append({
                    'name': 'tolerance',
                    'type': 'decimal',
                    'default': float(tolerance_match.group(1)),
                    'description': 'Tolerance value for matching'
                })
            
            return {
                'prompt_template': self._parameterize_prompt(user_prompt, required_columns, parameters),
                'required_columns': required_columns,
                'optional_columns': optional_columns,
                'parameters': parameters,
                'validation_rules': ['amount_positive', 'reference_not_null'],
                'output_format': {'type': 'table'},
                'sample_data': {}
            }
            
        except Exception as e:
            logger.error(f"Error extracting template config: {e}")
            return {
                'prompt_template': query_data.get('user_prompt', ''),
                'required_columns': [],
                'optional_columns': [],
                'parameters': [],
                'validation_rules': [],
                'output_format': {},
                'sample_data': {}
            }
    
    def _parameterize_prompt(self, prompt: str, columns: List[str], parameters: List[Dict[str, Any]]) -> str:
        """Convert a specific prompt into a parameterized template"""
        parameterized = prompt
        
        # Replace specific column names with placeholders
        for col in columns:
            parameterized = parameterized.replace(col, f"{{{col}}}")
        
        # Replace parameter values with placeholders
        for param in parameters:
            if param['type'] == 'decimal':
                # Replace numeric values
                parameterized = re.sub(
                    rf"\b{param['default']}\b",
                    f"{{{param['name']}}}",
                    parameterized
                )
        
        return parameterized
    
    def _format_schema_context(self, file_schemas: List[Dict[str, Any]]) -> str:
        """Format file schemas for AI context"""
        context = ""
        for i, schema in enumerate(file_schemas):
            context += f"File {i+1}:\n"
            context += f"  Name: {schema.get('filename', 'unknown')}\n"
            context += f"  Columns: {', '.join(schema.get('columns', []))}\n"
            if schema.get('sample_data'):
                context += f"  Sample: {schema['sample_data']}\n"
            context += "\n"
        return context


# Global instance
template_application_service = TemplateApplicationService()