import ast
import logging
import operator
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd

from app.models.transformation_models import (
    TransformationConfig,
    ColumnMapping,
    ExpansionType,
    TransformationType,
    ValidationRule,
    OutputDefinition, RowGenerationRule
)

logger = logging.getLogger(__name__)


class TransformationEngine:
    """Generic data transformation engine for creating declaration files"""

    def __init__(self):
        self.data_context = {}
        self.sequence_counters = {}
        self.errors = []
        self.warnings = []

    def process_transformation(
            self,
            source_data: Dict[str, pd.DataFrame],
            config: TransformationConfig,
            preview_only: bool = False,
            row_limit: Optional[int] = None
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Process the transformation based on configuration"""

        # Reset state
        self.data_context = source_data
        self.sequence_counters = {}
        self.errors = []
        self.warnings = []

        try:
            # Step 1: Apply row generation rules
            logger.info("Applying row generation rules...")
            generated_rows = self._apply_row_generation(config.row_generation_rules)

            # Step 2: Apply column mappings
            logger.info("Applying column mappings...")
            result_df = self._apply_column_mappings(generated_rows, config.column_mappings)

            # Step 3: Apply validation rules
            logger.info("Validating output...")
            validation_results = self._validate_output(result_df, config.validation_rules, config.output_definition)

            # Step 4: Format output according to schema
            result_df = self._format_output(result_df, config.output_definition)

            # Apply row limit for preview
            if preview_only and row_limit:
                result_df = result_df.head(row_limit)

            return result_df, {
                'validation_results': validation_results,
                'errors': self.errors,
                'warnings': self.warnings,
                'output_row_count': len(result_df)
            }

        except Exception as e:
            logger.error(f"Transformation error: {str(e)}")
            self.errors.append(f"Transformation failed: {str(e)}")
            raise

    def _apply_row_generation(self, rules: List[RowGenerationRule]) -> List[Dict[str, Any]]:
        """Apply row generation rules to create output rows"""

        if not rules:
            # No expansion rules, return original rows
            primary_df = self._get_primary_dataframe()
            return primary_df.to_dict('records')

        generated_rows = []
        primary_df = self._get_primary_dataframe()

        # Sort rules by priority
        sorted_rules = sorted(rules, key=lambda r: r.priority)

        for idx, row in primary_df.iterrows():
            row_dict = row.to_dict()

            # Check if any rule applies to this row
            rule_applied = False

            for rule in sorted_rules:
                if not rule.enabled:
                    continue

                # Evaluate condition if present
                if rule.condition and not self._evaluate_condition(rule.condition, row_dict):
                    continue

                # Apply expansion strategy
                expanded_rows = self._apply_expansion_strategy(row_dict, rule.strategy)
                generated_rows.extend(expanded_rows)
                rule_applied = True

                # If a rule is applied, don't apply other rules (unless specified)
                break

            # If no rule applied, include the original row
            if not rule_applied:
                generated_rows.append(row_dict)

        return generated_rows

    def _apply_expansion_strategy(self, source_row: Dict[str, Any], strategy: Any) -> List[Dict[str, Any]]:
        """Apply specific expansion strategy to a row"""

        expanded_rows = []

        if strategy.type == ExpansionType.DUPLICATE:
            # Simple duplication
            count = strategy.config.get('count', 1)
            for i in range(count):
                new_row = source_row.copy()
                new_row['_expansion_index'] = i
                expanded_rows.append(new_row)

        elif strategy.type == ExpansionType.FIXED_EXPANSION:
            # Expand with fixed values
            expansions = strategy.config.get('expansions', [])
            for expansion in expansions:
                new_row = source_row.copy()
                # Apply set_values
                if 'set_values' in expansion:
                    new_row.update(expansion['set_values'])
                expanded_rows.append(new_row)

        elif strategy.type == ExpansionType.EXPAND_FROM_LIST:
            # Expand from a list of values
            values = strategy.config.get('values', [])
            target_column = strategy.config.get('target_column', 'expansion_value')
            for value in values:
                new_row = source_row.copy()
                new_row[target_column] = value
                expanded_rows.append(new_row)

        elif strategy.type == ExpansionType.CONDITIONAL_EXPANSION:
            # Conditional expansion
            condition = strategy.config.get('condition', 'True')
            if self._evaluate_condition(condition, source_row):
                true_expansions = strategy.config.get('true_expansions', [])
                for expansion in true_expansions:
                    new_row = source_row.copy()
                    if 'set_values' in expansion:
                        new_row.update(expansion['set_values'])
                    expanded_rows.append(new_row)
            else:
                false_expansions = strategy.config.get('false_expansions', [])
                for expansion in false_expansions:
                    new_row = source_row.copy()
                    if 'set_values' in expansion:
                        new_row.update(expansion['set_values'])
                    expanded_rows.append(new_row)

        elif strategy.type == ExpansionType.EXPAND_FROM_FILE:
            # Expand from another file
            file_alias = strategy.config.get('file_alias')
            join_on = strategy.config.get('join_on')
            expand_columns = strategy.config.get('expand_columns', [])

            if file_alias in self.data_context:
                lookup_df = self.data_context[file_alias]
                # Implement join logic here
                # This is simplified - you'd need more complex join handling
                expanded_rows.append(source_row)
            else:
                self.warnings.append(f"File alias '{file_alias}' not found for expansion")
                expanded_rows.append(source_row)

        else:
            # Default: return original row
            expanded_rows.append(source_row)

        return expanded_rows

    def _apply_column_mappings(self, rows: List[Dict[str, Any]], mappings: List[ColumnMapping]) -> pd.DataFrame:
        """Apply column mappings to create output dataframe"""

        if not rows:
            return pd.DataFrame()

        # Convert rows to dataframe for easier manipulation
        working_df = pd.DataFrame(rows)
        result_data = []

        for idx, row in working_df.iterrows():
            output_row = {}

            for mapping in mappings:
                if not mapping.enabled:
                    continue

                try:
                    value = self._apply_single_mapping(row, mapping, idx)
                    output_row[mapping.target_column] = value
                except Exception as e:
                    logger.error(f"Error applying mapping for column {mapping.target_column}: {str(e)}")
                    self.errors.append(f"Mapping error for {mapping.target_column}: {str(e)}")
                    output_row[mapping.target_column] = None

            result_data.append(output_row)

        return pd.DataFrame(result_data)

    def _apply_single_mapping(self, row: pd.Series, mapping: ColumnMapping, row_index: int) -> Any:
        """Apply a single column mapping"""

        if mapping.mapping_type == TransformationType.DIRECT:
            # Direct mapping from source
            if mapping.source:
                return self._get_value_from_path(row, mapping.source)
            return None

        elif mapping.mapping_type == TransformationType.STATIC:
            # Static value
            return mapping.transformation.config.get('value')

        elif mapping.mapping_type == TransformationType.SEQUENCE:
            # Sequential number
            config = mapping.transformation.config
            counter_key = f"{mapping.target_column}_seq"

            if counter_key not in self.sequence_counters:
                self.sequence_counters[counter_key] = config.get('start', 1)

            current = self.sequence_counters[counter_key]
            self.sequence_counters[counter_key] += config.get('increment', 1)

            prefix = config.get('prefix', '')
            suffix = config.get('suffix', '')
            padding = config.get('padding', 0)

            if padding > 0:
                return f"{prefix}{str(current).zfill(padding)}{suffix}"
            return f"{prefix}{current}{suffix}"

        elif mapping.mapping_type == TransformationType.EXPRESSION:
            # Expression-based transformation
            formula = mapping.transformation.config.get('formula', '')
            variables = mapping.transformation.config.get('variables', {})

            # Replace variables in formula
            for var_name, var_path in variables.items():
                value = self._get_value_from_path(row, var_path)
                formula = formula.replace(f"{{{var_name}}}", str(value))

            # Evaluate expression
            try:
                return self._safe_eval(formula, row)
            except Exception as e:
                logger.error(f"Expression evaluation error: {str(e)}")
                return None

        elif mapping.mapping_type == TransformationType.CONDITIONAL:
            # Conditional transformation
            config = mapping.transformation.config
            condition = config.get('condition', 'True')

            if self._evaluate_condition(condition, row.to_dict()):
                return config.get('true_value')
            else:
                return config.get('false_value')

        elif mapping.mapping_type == TransformationType.LOOKUP:
            # Lookup transformation
            config = mapping.transformation.config
            source_value = self._get_value_from_path(row, config.get('source_value', ''))
            lookup_file = config.get('lookup_file')
            lookup_key = config.get('lookup_key')
            lookup_return = config.get('lookup_return')
            default_value = config.get('default_value')

            if lookup_file in self.data_context:
                lookup_df = self.data_context[lookup_file]
                matches = lookup_df[lookup_df[lookup_key] == source_value]
                if not matches.empty:
                    return matches.iloc[0][lookup_return]

            return default_value

        elif mapping.mapping_type == TransformationType.CUSTOM_FUNCTION:
            # Custom function transformation
            code = mapping.transformation.config.get('code', '')
            try:
                # Create a safe execution environment
                local_vars = {'row': row.to_dict(), 'context': self.data_context}
                exec(code, {"__builtins__": {}}, local_vars)
                if 'transform' in local_vars:
                    return local_vars['transform'](row.to_dict(), self.data_context)
            except Exception as e:
                logger.error(f"Custom function error: {str(e)}")
                return None

        return None

    def _get_value_from_path(self, row: Union[pd.Series, Dict], path: str) -> Any:
        """Get value from a path like 'file_alias.column_name' or just 'column_name'"""

        if isinstance(row, pd.Series):
            row = row.to_dict()

        # Check if path contains file alias
        if '.' in path:
            parts = path.split('.', 1)
            file_alias = parts[0]
            column_name = parts[1]

            if column_name in row.keys():
                return row[column_name]

            # If file_alias is in data context, look there
            if file_alias in self.data_context:
                # This is simplified - in reality you'd need row matching logic
                return None
            else:
                # Try to get from current row
                return row.get(column_name)
        else:
            # Direct column reference
            return row.get(path)

    def _evaluate_condition(self, condition: str, row_data: Dict[str, Any]) -> bool:
        """Safely evaluate a condition"""
        try:
            # Replace column references with actual values
            for key, value in row_data.items():
                # Handle different value types
                if isinstance(value, str):
                    condition = condition.replace(f"{key}", f"'{value}'")
                elif value is None:
                    condition = condition.replace(f"{key}", "None")
                else:
                    condition = condition.replace(f"{key}", str(value))

            # Safe evaluation
            return self._safe_eval(condition, row_data)
        except Exception as e:
            logger.error(f"Condition evaluation error: {str(e)}")
            return False

    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate an expression"""

        # Define allowed names
        allowed_names = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'True': True,
            'False': False,
            'None': None,
        }

        # Add context variables
        allowed_names.update(context)

        # Parse and evaluate
        try:
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body, allowed_names)
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    def _eval_node(self, node, allowed_names):
        """Recursively evaluate AST nodes"""

        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Name '{node.id}' is not allowed")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, allowed_names)
            right = self._eval_node(node.right, allowed_names)
            return self._apply_operator(node.op, left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, allowed_names)
            return self._apply_unary_operator(node.op, operand)
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, allowed_names)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, allowed_names)
                if not self._apply_comparison(op, left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.IfExp):
            test = self._eval_node(node.test, allowed_names)
            if test:
                return self._eval_node(node.body, allowed_names)
            else:
                return self._eval_node(node.orelse, allowed_names)
        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func, allowed_names)
            args = [self._eval_node(arg, allowed_names) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def _apply_operator(self, op, left, right):
        """Apply binary operator"""
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        return operators[type(op)](left, right)

    def _apply_unary_operator(self, op, operand):
        """Apply unary operator"""
        operators = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
            ast.Not: operator.not_,
        }
        return operators[type(op)](operand)

    def _apply_comparison(self, op, left, right):
        """Apply comparison operator"""
        operators = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y,
        }
        return operators[type(op)](left, right)

    def _validate_output(
            self,
            df: pd.DataFrame,
            validation_rules: List[ValidationRule],
            output_definition: OutputDefinition
    ) -> Dict[str, Any]:
        """Validate the output dataframe"""

        validation_results = {
            'errors': [],
            'warnings': [],
            'passed': True
        }

        # Check required columns
        required_columns = [col.id for col in output_definition.columns]
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            validation_results['passed'] = False

        # Check column types and formats
        for col_def in output_definition.columns:
            if col_def.id not in df.columns:
                continue

            # Type validation
            # if col_def.type == 'number' or col_def.type == 'decimal':
            #     non_numeric = df[~df[col_def.name].apply(lambda x: isinstance(x, (int, float)) or pd.isna(x))]
            #     if not non_numeric.empty:
            #         validation_results['errors'].append(
            #             f"Column '{col_def.name}' contains non-numeric values"
            #         )

            # Allowed values validation
            if col_def.allowed_values:
                invalid_values = df[~df[col_def.name].isin(col_def.allowed_values + [None, np.nan])]
                if not invalid_values.empty:
                    validation_results['warnings'].append(
                        f"Column '{col_def.name}' contains values not in allowed list"
                    )

        # Apply custom validation rules
        for rule in validation_rules:
            try:
                if rule.type == 'required':
                    # Check for null values in specified columns
                    columns = rule.config.get('columns', [])
                    for col in columns:
                        if col in df.columns:
                            null_count = df[col].isna().sum()
                            if null_count > 0:
                                msg = f"Column '{col}' has {null_count} null values"
                                if rule.severity == 'error':
                                    validation_results['errors'].append(msg)
                                    validation_results['passed'] = False
                                else:
                                    validation_results['warnings'].append(msg)

                elif rule.type == 'format':
                    # Check format patterns
                    column = rule.config.get('column')
                    pattern = rule.config.get('pattern')
                    if column in df.columns and pattern:
                        invalid = df[~df[column].astype(str).str.match(pattern)]
                        if not invalid.empty:
                            msg = f"Column '{column}' has {len(invalid)} values not matching pattern"
                            if rule.severity == 'error':
                                validation_results['errors'].append(msg)
                                validation_results['passed'] = False
                            else:
                                validation_results['warnings'].append(msg)

            except Exception as e:
                logger.error(f"Validation rule error: {str(e)}")
                validation_results['warnings'].append(f"Could not apply validation rule '{rule.name}'")

        return validation_results

    def _format_output(self, df: pd.DataFrame, output_definition: OutputDefinition) -> pd.DataFrame:
        """Format the output dataframe according to the schema"""

        # Reorder columns according to schema
        column_order = [col.id for col in output_definition.columns if col.id in df.columns]
        df = df[column_order]

        # Apply formatting
        for col_def in output_definition.columns:
            if col_def.id not in df.columns:
                # Add missing columns with default values
                df[col_def.name] = col_def.default_value
                continue

            # Apply type formatting
            if col_def.type == 'decimal' and col_def.format:
                try:
                    decimals = int(col_def.format.split('.')[-1])
                    df[col_def.name] = df[col_def.id].round(decimals)
                except:
                    pass

            elif col_def.type == 'date' and col_def.format:
                try:
                    from app.utils.date_utils import normalize_date_value
                    
                    # Use shared date utilities for consistent date handling
                    def normalize_and_format_date(value):
                        if pd.isna(value):
                            return None
                        # First normalize using shared utilities
                        normalized_date_str = normalize_date_value(value)
                        if normalized_date_str is not None:
                            # Parse normalized date and format according to col_def.format
                            parsed_date = pd.to_datetime(normalized_date_str)
                            return parsed_date.strftime(col_def.format)
                        return None
                    
                    df[col_def.name] = df[col_def.id].apply(normalize_and_format_date)
                except Exception as e:
                    logger.warning(f"Date formatting failed for column {col_def.name}: {str(e)}")
                    # Fallback to original logic if shared utilities fail
                    try:
                        df[col_def.name] = pd.to_datetime(df[col_def.id]).dt.strftime(col_def.format)
                    except:
                        pass

            else:
                df[col_def.name] = df[col_def.id].astype(str)

        return df

    def _get_primary_dataframe(self) -> pd.DataFrame:
        """Get the primary dataframe from context"""

        # Try to find primary_data or first available dataframe
        if 'primary_data' in self.data_context:
            return self.data_context['primary_data']

        # Return first dataframe
        for key, value in self.data_context.items():
            if isinstance(value, pd.DataFrame):
                return value

        return pd.DataFrame()


class TransformationStorage:
    """Storage for transformation results"""

    def __init__(self):
        self.storage = {}

    def store_results(self, transformation_id: str, results: Dict[str, Any]) -> bool:
        """Store transformation results"""
        try:
            self.storage[transformation_id] = {
                'timestamp': datetime.now(),
                'results': results
            }
            return True
        except Exception as e:
            logger.error(f"Failed to store results: {str(e)}")
            return False

    def get_results(self, transformation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve transformation results"""
        return self.storage.get(transformation_id)

    def isExist(self, transformation_id: str) -> bool:
        """Check if transformation id exists"""
        return transformation_id in self.storage

    def delete_results(self, transformation_id: str) -> bool:
        """Delete transformation results"""
        if transformation_id in self.storage:
            del self.storage[transformation_id]
            return True
        return False


# Global storage instance
transformation_storage = TransformationStorage()
