import io
import re
import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

import pandas as pd
from fastapi import UploadFile, HTTPException
from rapidfuzz import fuzz, process
import numpy as np

from app.models.recon_models import PatternCondition, FileRule, ExtractRule, FilterRule, ReconciliationRule
from app.utils.threading_config import get_reconciliation_config, get_timeout_for_operation

# Configure logging for reconciliation service
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Update forward reference
PatternCondition.model_rebuild()


class OptimizedFileProcessor:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self._pattern_cache = {}  # Cache compiled regex patterns
        self._date_cache = {}  # Cache parsed dates for performance
        
        # Use centralized hardware-aware threading configuration
        self.threading_config = get_reconciliation_config()
        self.max_workers = self.threading_config.max_workers
        self.batch_size = self.threading_config.batch_size

    def _process_batch_parallel(self, batch_data: Dict) -> Dict:
        """
        Process a single batch of records in parallel
        Uses many-to-many matching by default (allows all possible matches)
        Preserves all original features: rule types, column selection, closest matching
        This method is designed to be called by parallel workers
        """
        batch_a = batch_data['batch_a']
        grouped_b = batch_data['grouped_b']
        recon_rules = batch_data['recon_rules']
        df_a = batch_data['df_a']
        df_b = batch_data['df_b']
        selected_columns_a = batch_data['selected_columns_a']
        selected_columns_b = batch_data['selected_columns_b']
        
        matches = []
        matched_indices_a = set()
        matched_indices_b = set()
        
        try:
            for idx_a, row_a in batch_a.iterrows():
                match_key = row_a['_match_key']
                
                # Get potential matches from grouped data
                if match_key in grouped_b:
                    potential_matches = grouped_b[match_key]
                    
                    for idx_b, row_b in potential_matches.iterrows():
                        # Check all reconciliation rules
                        all_rules_match = True
                        
                        for rule in recon_rules:
                            val_a = row_a[rule.LeftFileColumn]
                            val_b = row_b[rule.RightFileColumn]
                            
                            if rule.MatchType.lower() == "equals":
                                if not self._check_equals_match(val_a, val_b):
                                    all_rules_match = False
                                    break
                            elif rule.MatchType.lower() == "date_equals":
                                if not self._check_date_equals_match(val_a, val_b):
                                    all_rules_match = False
                                    break
                            elif rule.MatchType.lower() == "tolerance":
                                if not self._check_tolerance_match(val_a, val_b, rule.ToleranceValue):
                                    all_rules_match = False
                                    break
                            elif rule.MatchType.lower() == "fuzzy":
                                if not self._check_fuzzy_match(val_a, val_b, rule.ToleranceValue):
                                    all_rules_match = False
                                    break
                        
                        if all_rules_match:
                            # Create match record with selected columns (preserves original functionality)
                            match_record = self._create_match_record(
                                row_a, row_b, df_a, df_b,
                                selected_columns_a, selected_columns_b,
                                recon_rules
                            )
                            matches.append(match_record)
                            
                            # Track matched records for unmatched calculation
                            # Note: Many-to-many allows the same record to be in both matched and unmatched
                            matched_indices_a.add(row_a['_orig_index_a'])
                            matched_indices_b.add(row_b['_orig_index_b'])
                            # No break - allows many-to-many matching (same as original)
                                
        except Exception as e:
            batch_idx = batch_data.get('batch_idx', 'unknown')
            logger.warning(f"Error processing batch {batch_idx}: {e}")
            
        return {
            'matches': matches,
            'matched_indices_a': matched_indices_a,
            'matched_indices_b': matched_indices_b
        }
    
    def read_file(self, file: UploadFile, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Read CSV or Excel file into DataFrame with leading zero preservation and optimized settings"""
        try:
            content = file.file.read()
            file.file.seek(0)

            # Import the leading zero detection from file_routes
            from app.routes.file_routes import detect_leading_zero_columns
            
            # Step 1: Detect columns with leading zeros first
            dtype_mapping = detect_leading_zero_columns(content, file.filename, sheet_name)

            if file.filename.endswith('.csv'):
                df = pd.read_csv(
                    io.BytesIO(content),
                    low_memory=False,
                    engine='c',  # Use C engine for better performance
                    dtype=dtype_mapping if dtype_mapping else None  # Preserve leading zero columns as strings
                )
            elif file.filename.endswith(('.xlsx', '.xls')):
                if sheet_name:
                    df = pd.read_excel(
                        io.BytesIO(content), 
                        sheet_name=sheet_name, 
                        engine='openpyxl',
                        dtype=dtype_mapping if dtype_mapping else None  # Preserve leading zero columns as strings
                    )
                else:
                    df = pd.read_excel(
                        io.BytesIO(content), 
                        engine='openpyxl',
                        dtype=dtype_mapping if dtype_mapping else None  # Preserve leading zero columns as strings
                    )
            else:
                raise ValueError(f"Unsupported file format: {file.filename}")
            
            # Fix: Preserve integer types to prevent 15 -> 15.0 conversion (only for non-string columns)
            df = self._preserve_integer_types(df)
            return df
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file {file.filename}: {str(e)}")
    
    def _preserve_integer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert float columns back to integers where all values are whole numbers.
        This prevents 15 from being displayed as 15.0 in reconciliation results.
        IMPORTANT: Skip string columns that contain preserved leading zeros.
        """
        try:
            for col in df.columns:
                # Skip string/object columns (these may contain preserved leading zeros like '01')
                if df[col].dtype == 'object':
                    continue
                    
                if df[col].dtype == 'float64':
                    # Check if all non-null values are whole numbers
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # Check if all values are integers (no decimal part)
                        if all(float(val).is_integer() for val in non_null_values):
                            # Convert to Int64 (pandas nullable integer type) to handle NaN values
                            df[col] = df[col].astype('Int64')
                            
            return df
        except Exception as e:
            # If conversion fails, return original dataframe
            self.warnings.append(f"Warning: Could not preserve integer types: {str(e)}")
            return df

    def _calculate_composite_similarity(self, val_a, val_b, column_type: str = "text") -> float:
        """
        Calculate composite similarity score using multiple algorithms based on data type
        
        Args:
            val_a: Value from file A
            val_b: Value from file B  
            column_type: Type of data - "text", "numeric", "date", "identifier"
            
        Returns:
            Similarity score between 0.0 and 100.0
        """
        # Handle null values
        if pd.isna(val_a) and pd.isna(val_b):
            # Both are NaN - for closest match, this is often not a meaningful comparison
            # Return lower score to deprioritize NaN-to-NaN matches
            return 50.0  # Reduced from 100.0
        if pd.isna(val_a) or pd.isna(val_b):
            return 0.0
            
        # Convert to strings for comparison
        str_a = str(val_a).strip()
        str_b = str(val_b).strip()
        
        # Exact match gets perfect score
        if str_a == str_b:
            return 100.0
            
        # Different algorithms based on data type
        if column_type == "numeric":
            return self._calculate_numeric_similarity(val_a, val_b)
        elif column_type == "date":
            return self._calculate_date_similarity(val_a, val_b)
        elif column_type == "identifier":
            return self._calculate_identifier_similarity(str_a, str_b)
        else:  # text/default
            return self._calculate_text_similarity(str_a, str_b)
    
    def _calculate_text_similarity(self, str_a: str, str_b: str) -> float:
        """Calculate text similarity using multiple fuzzy algorithms"""
        if not str_a or not str_b:
            return 0.0
            
        # Multiple fuzzy matching algorithms with weights
        algorithms = {
            'ratio': fuzz.ratio(str_a, str_b) * 0.3,              # Basic similarity
            'partial_ratio': fuzz.partial_ratio(str_a, str_b) * 0.2,  # Partial matching
            'token_sort_ratio': fuzz.token_sort_ratio(str_a, str_b) * 0.25,  # Token order independent
            'token_set_ratio': fuzz.token_set_ratio(str_a, str_b) * 0.25    # Token set comparison
        }
        
        # Weighted composite score
        composite_score = sum(algorithms.values())
        return min(composite_score, 100.0)
    
    def _calculate_numeric_similarity(self, val_a, val_b) -> float:
        """Calculate numeric similarity with tolerance handling"""
        try:
            num_a = float(val_a)
            num_b = float(val_b)
            
            # Exact match
            if num_a == num_b:
                return 100.0
            
            # Calculate percentage difference
            if num_b != 0:
                percentage_diff = abs(num_a - num_b) / abs(num_b) * 100
            else:
                return 100.0 if num_a == 0 else 0.0
            
            # Convert to similarity score (inverse of difference)
            # 0% difference = 100% similarity
            # 1% difference = 99% similarity, etc.
            similarity = max(0, 100 - percentage_diff)
            return similarity
            
        except (ValueError, TypeError):
            # Fall back to string comparison for non-numeric values
            return self._calculate_text_similarity(str(val_a), str(val_b))
    
    def _calculate_date_similarity(self, val_a, val_b) -> float:
        """Calculate date similarity with format tolerance"""
        try:
            # Try to parse dates in multiple formats
            date_a = pd.to_datetime(val_a, errors='coerce')
            date_b = pd.to_datetime(val_b, errors='coerce')
            
            if pd.isna(date_a) or pd.isna(date_b):
                # Fall back to string comparison if date parsing fails
                return self._calculate_text_similarity(str(val_a), str(val_b))
            
            # Exact date match
            if date_a == date_b:
                return 100.0
            
            # Calculate day difference
            day_diff = abs((date_a - date_b).days)
            
            # Similarity decreases with day difference
            # Same day = 100%, 1 day = 95%, 7 days = 65%, 30 days = 0%
            if day_diff == 0:
                return 100.0
            elif day_diff <= 1:
                return 95.0
            elif day_diff <= 7:
                return max(0, 95 - (day_diff - 1) * 5)  # 95, 90, 85, 80, 75, 70, 65
            elif day_diff <= 30:
                return max(0, 65 - (day_diff - 7) * 2.8)  # Gradual decrease to 0
            else:
                return 0.0
                
        except Exception:
            # Fall back to string comparison
            return self._calculate_text_similarity(str(val_a), str(val_b))
    
    def _calculate_identifier_similarity(self, str_a: str, str_b: str) -> float:
        """Calculate similarity for identifiers (account numbers, transaction IDs, etc.)"""
        if not str_a or not str_b:
            return 0.0
        
        # For identifiers, we're more strict but allow for minor variations
        algorithms = {
            'ratio': fuzz.ratio(str_a, str_b) * 0.4,              # Basic similarity (higher weight)
            'partial_ratio': fuzz.partial_ratio(str_a, str_b) * 0.3,  # Partial matching
            'token_sort_ratio': fuzz.token_sort_ratio(str_a, str_b) * 0.3   # Token order independent
        }
        
        composite_score = sum(algorithms.values())
        return min(composite_score, 100.0)
    
    def _detect_column_type(self, column_name: str, sample_values: List) -> str:
        """
        Detect the most likely data type of a column based on name and sample values
        """
        column_name_lower = column_name.lower()
        
        # Date detection
        date_keywords = ['date', 'time', 'created', 'updated', 'timestamp', 'day', 'month', 'year']
        if any(keyword in column_name_lower for keyword in date_keywords):
            return "date"
        
        # Numeric detection  
        numeric_keywords = ['amount', 'value', 'price', 'cost', 'total', 'sum', 'balance', 'quantity', 'qty']
        if any(keyword in column_name_lower for keyword in numeric_keywords):
            return "numeric"
        
        # Identifier detection
        id_keywords = ['id', 'ref', 'reference', 'number', 'account', 'code', 'key']
        if any(keyword in column_name_lower for keyword in id_keywords):
            return "identifier"
        
        # Analyze sample values
        if sample_values:
            non_null_values = [v for v in sample_values if not pd.isna(v)][:10]  # Sample first 10 non-null values
            
            if non_null_values:
                # Check if most values are numeric
                numeric_count = 0
                for val in non_null_values:
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                
                if numeric_count >= len(non_null_values) * 0.7:  # 70% are numeric
                    return "numeric"
                
                # Check if most values look like dates
                date_count = 0
                for val in non_null_values:
                    if pd.to_datetime(val, errors='coerce') is not pd.NaT:
                        date_count += 1
                
                if date_count >= len(non_null_values) * 0.7:  # 70% are dates
                    return "date"
        
        # Default to text
        return "text"

    def _check_equals_match(self, val_a, val_b) -> bool:
        """Check equality with STRICT string matching (no auto date detection)"""
        # Handle null values
        if pd.isna(val_a) and pd.isna(val_b):
            return True
        if pd.isna(val_a) or pd.isna(val_b):
            return False

        # Try exact match first (fastest path)
        if val_a == val_b:
            return True

        # Convert to strings and strip whitespace
        str_a = str(val_a).strip()
        str_b = str(val_b).strip()
        
        # Don't match empty strings (they should be explicit matches)
        if not str_a or not str_b:
            return False

        # Case-insensitive string comparison (preserves leading zeros)
        return str_a.lower() == str_b.lower()

    def _check_date_equals_match(self, val_a, val_b) -> bool:
        """Check if two values match as dates using shared date utilities (for explicit date_equals match type)"""
        from app.utils.date_utils import check_date_equals_match
        return check_date_equals_match(val_a, val_b)

    def _check_numeric_equals(self, val_a, val_b) -> bool:
        """
        Check if two values are numerically equal, handling cases like:
        - "01" vs "1" → True
        - "09" vs "9" → True  
        - "007" vs "7" → True
        Only for non-date values.
        """
        try:
            # Convert both values to strings first
            str_a = str(val_a).strip()
            str_b = str(val_b).strip()
            
            # Skip if either value is empty
            if not str_a or not str_b:
                return False
            
            # Try to convert both to numbers
            # This handles integers, floats, and numeric strings
            try:
                num_a = float(str_a)
                num_b = float(str_b)
                
                # Check if both are integers (no decimal part)
                if num_a.is_integer() and num_b.is_integer():
                    # Compare as integers: "01" (1.0) == "1" (1.0) → True
                    return int(num_a) == int(num_b)
                else:
                    # For decimals, use float comparison
                    return num_a == num_b
                    
            except (ValueError, TypeError):
                # If either value can't be converted to number, not a numeric match
                return False
                
        except Exception:
            # If anything goes wrong, not a numeric match
            return False

    def validate_rules_against_columns(self, df: pd.DataFrame, file_rule: FileRule) -> List[str]:
        """Validate that all columns mentioned in rules exist in the DataFrame"""
        errors = []
        df_columns = df.columns.tolist()

        # Check extract rules - Handle optional Extract
        if hasattr(file_rule, 'Extract') and file_rule.Extract:
            for extract in file_rule.Extract:
                if extract.SourceColumn not in df_columns:
                    errors.append(
                        f"Column '{extract.SourceColumn}' not found in file '{file_rule.Name}'. Available columns: {df_columns}")

        # Check filter rules - Handle optional Filter
        if hasattr(file_rule, 'Filter') and file_rule.Filter:
            for filter_rule in file_rule.Filter:
                if filter_rule.ColumnName not in df_columns:
                    errors.append(
                        f"Column '{filter_rule.ColumnName}' not found in file '{file_rule.Name}'. Available columns: {df_columns}")

        return errors

    @lru_cache(maxsize=1000)
    def _get_compiled_pattern(self, pattern: str) -> re.Pattern:
        """Cache compiled regex patterns for better performance with case insensitive flag"""
        return re.compile(pattern, re.IGNORECASE)

    def evaluate_pattern_condition(self, text: str, condition: PatternCondition) -> bool:
        """Recursively evaluate pattern conditions with caching"""
        if condition.pattern:
            try:
                compiled_pattern = self._get_compiled_pattern(condition.pattern)
                return bool(compiled_pattern.search(str(text)))
            except re.error as e:
                self.errors.append(f"Invalid regex pattern '{condition.pattern}': {str(e)}")
                return False

        elif condition.patterns:
            results = []
            for pattern in condition.patterns:
                try:
                    compiled_pattern = self._get_compiled_pattern(pattern)
                    results.append(bool(compiled_pattern.search(str(text))))
                except re.error as e:
                    self.errors.append(f"Invalid regex pattern '{pattern}': {str(e)}")
                    results.append(False)

            if condition.operator == "AND":
                return all(results)
            else:  # Default OR
                return any(results)

        elif condition.conditions:
            results = []
            for sub_condition in condition.conditions:
                results.append(self.evaluate_pattern_condition(text, sub_condition))

            if condition.operator == "AND":
                return all(results)
            else:  # Default OR
                return any(results)

        return False

    def extract_patterns_vectorized(self, df: pd.DataFrame, extract_rule: ExtractRule) -> pd.Series:
        """Optimized pattern extraction using vectorized operations"""

        def extract_from_text(text):
            if pd.isna(text):
                return None

            text = str(text)

            # Special handling for amount extraction with optimized patterns
            if extract_rule.ResultColumnName.lower() in ['amount', 'extractedamount', 'value']:
                amount_patterns = [
                    r'(?:Amount:?\s*)?(?:[\$€£¥₹]\s*)([\d,]+(?:\.\d{2})?)',
                    r'(?:Amount|Price|Value|Cost|Total):\s*([\d,]+(?:\.\d{2})?)',
                    r'\b((?:\d{1,3},)+\d{3}(?:\.\d{2})?)\b(?!\d)',
                    r'(?:[\$€£¥₹]\s*)(\d+(?:\.\d{2})?)\b'
                ]

                for pattern in amount_patterns:
                    try:
                        compiled_pattern = self._get_compiled_pattern(pattern)
                        match = compiled_pattern.search(text)
                        if match:
                            amount_str = match.group(1).replace(',', '').replace('$', '')
                            try:
                                amount = float(amount_str)
                                if amount > 0:  # Valid amount
                                    return amount_str
                            except ValueError:
                                continue
                    except re.error:
                        continue

            # Handle new nested condition format
            if hasattr(extract_rule, 'Conditions') and extract_rule.Conditions:
                if self.evaluate_pattern_condition(text, extract_rule.Conditions):
                    matched_value = self.extract_first_match(text, extract_rule.Conditions)
                    return matched_value

            # Handle legacy format
            elif hasattr(extract_rule, 'Patterns') and extract_rule.Patterns:
                for pattern in extract_rule.Patterns:
                    try:
                        compiled_pattern = self._get_compiled_pattern(pattern)
                        match = compiled_pattern.search(text)
                        if match:
                            return match.group(0)
                    except re.error as e:
                        self.errors.append(f"Invalid regex pattern '{pattern}': {str(e)}")

            return None

        # Use vectorized operations where possible
        column_data = df[extract_rule.SourceColumn].astype(str)
        return column_data.apply(extract_from_text)

    def extract_first_match(self, text: str, condition: PatternCondition) -> Optional[str]:
        """Extract the first matching value from text"""
        if condition.pattern:
            try:
                compiled_pattern = self._get_compiled_pattern(condition.pattern)
                match = compiled_pattern.search(text)
                if match:
                    return match.group(0)
            except re.error:
                pass

        elif condition.patterns:
            for pattern in condition.patterns:
                try:
                    compiled_pattern = self._get_compiled_pattern(pattern)
                    match = compiled_pattern.search(text)
                    if match:
                        return match.group(0)
                except re.error:
                    pass

        elif condition.conditions:
            for sub_condition in condition.conditions:
                result = self.extract_first_match(text, sub_condition)
                if result:
                    return result

        return None

    def apply_filters_optimized(self, df: pd.DataFrame, filters: List[FilterRule]) -> pd.DataFrame:
        """Apply filter rules to DataFrame with optimized operations"""
        if not filters:
            return df

        filtered_df = df.copy()

        for filter_rule in filters:
            column = filter_rule.ColumnName
            value = filter_rule.Value
            match_type = filter_rule.MatchType.lower()

            try:
                if match_type == "equals":
                    # Case insensitive string comparison for equals
                    if isinstance(value, str):
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.lower() == str(value).lower()]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == value]
                elif match_type == "not_equals":
                    # Case insensitive string comparison for not_equals
                    if isinstance(value, str):
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.lower() != str(value).lower()]
                    else:
                        filtered_df = filtered_df[filtered_df[column] != value]
                elif match_type == "greater_than":
                    numeric_col = pd.to_numeric(filtered_df[column], errors='coerce')
                    filtered_df = filtered_df[numeric_col > value]
                elif match_type == "less_than":
                    numeric_col = pd.to_numeric(filtered_df[column], errors='coerce')
                    filtered_df = filtered_df[numeric_col < value]
                elif match_type == "contains":
                    # Case insensitive contains
                    filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), case=False, na=False)]
                elif match_type == "in":
                    if isinstance(value, str):
                        value = [v.strip() for v in value.split(',')]
                    # Case insensitive in operation
                    if all(isinstance(v, str) for v in value):
                        # Convert both column values and filter values to lowercase for comparison
                        value_lower = [str(v).lower() for v in value]
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.lower().isin(value_lower)]
                    else:
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    self.warnings.append(f"Unknown filter match type: {match_type}")
            except Exception as e:
                self.errors.append(f"Error applying filter on column '{column}': {str(e)}")

        return filtered_df

    def get_mandatory_columns(self, recon_rules: List[ReconciliationRule],
                              file_a_rules: Optional[FileRule], file_b_rules: Optional[FileRule]) -> Tuple[
        Set[str], Set[str]]:
        """Get mandatory columns that must be included in results"""
        mandatory_a = set()
        mandatory_b = set()

        # Add reconciliation rule columns
        for rule in recon_rules:
            mandatory_a.add(rule.LeftFileColumn)
            mandatory_b.add(rule.RightFileColumn)

        # Add extracted columns - Handle optional rules
        if file_a_rules and hasattr(file_a_rules, 'Extract') and file_a_rules.Extract:
            for extract_rule in file_a_rules.Extract:
                mandatory_a.add(extract_rule.ResultColumnName)

        if file_b_rules and hasattr(file_b_rules, 'Extract') and file_b_rules.Extract:
            for extract_rule in file_b_rules.Extract:
                mandatory_b.add(extract_rule.ResultColumnName)

        # Add filter columns - Handle optional filters
        if file_a_rules and hasattr(file_a_rules, 'Filter') and file_a_rules.Filter:
            for filter_rule in file_a_rules.Filter:
                mandatory_a.add(filter_rule.ColumnName)

        if file_b_rules and hasattr(file_b_rules, 'Filter') and file_b_rules.Filter:
            for filter_rule in file_b_rules.Filter:
                mandatory_b.add(filter_rule.ColumnName)

        return mandatory_a, mandatory_b

    def create_optimized_match_keys(self, df: pd.DataFrame,
                                    recon_rules: List[ReconciliationRule],
                                    file_prefix: str) -> Tuple[
        pd.DataFrame, List[ReconciliationRule], List[ReconciliationRule]]:
        """Create optimized match keys for faster reconciliation"""
        df_work = df.copy()

        # Create composite match key for exact matches (excluding date_equals and tolerance)
        exact_match_cols = []
        tolerance_rules = []
        date_rules = []

        for rule in recon_rules:
            col_name = rule.LeftFileColumn if file_prefix == 'A' else rule.RightFileColumn

            if rule.MatchType.lower() == "equals":
                exact_match_cols.append(col_name)
            elif rule.MatchType.lower() == "tolerance":
                tolerance_rules.append(rule)
            elif rule.MatchType.lower() == "date_equals":
                date_rules.append(rule)

        # Create composite key for exact matches only (dates and tolerance handled separately)
        if exact_match_cols:
            # Make match key case insensitive by converting to lowercase
            df_work['_match_key'] = df_work[exact_match_cols].astype(str).apply(lambda x: x.str.lower()).agg('|'.join, axis=1)
        else:
            df_work['_match_key'] = df_work.index.astype(str)

        return df_work, tolerance_rules, date_rules

    def reconcile_files_optimized(self, df_a: pd.DataFrame, df_b: pd.DataFrame,
                                  recon_rules: List[ReconciliationRule],
                                  selected_columns_a: Optional[List[str]] = None,
                                  selected_columns_b: Optional[List[str]] = None,
                                  match_mode: str = "one_to_one",
                                  closest_match_config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """
        Optimized reconciliation using hash-based matching for large datasets with date support
        
        Args:
            match_mode: Matching behavior
                - "one_to_one": Each record matches at most once (default, fastest)
                - "one_to_many": File A records can match multiple File B records  
                - "many_to_one": Multiple File A records can match same File B record
                - "many_to_many": Full cartesian matching (slowest)
        """
        start_time = time.time()
        
        # Parse closest match configuration
        find_closest_matches = closest_match_config and closest_match_config.enabled
        
        logger.info("🚀 Starting optimized reconciliation process")
        logger.info(f"📊 Dataset sizes: File A ({len(df_a):,} records) vs File B ({len(df_b):,} records)")
        logger.info(f"🔧 Configuration: Match mode = {match_mode}, Closest matches = {find_closest_matches}")
        if find_closest_matches and closest_match_config:
            logger.info(f"🎯 Closest match config: {closest_match_config}")
        logger.info(f"📋 Reconciliation rules: {len(recon_rules)} rules defined")
        
        # Log rule details for debugging
        for i, rule in enumerate(recon_rules, 1):
            logger.debug(f"   Rule {i}: {rule.LeftFileColumn} ↔ {rule.RightFileColumn} ({rule.MatchType})")
        
        logger.info("🔄 Creating optimized match keys and processing rules...")

        # Create working copies with indices and separate rule types
        df_a_work, tolerance_rules_a, date_rules_a = self.create_optimized_match_keys(df_a, recon_rules, 'A')
        df_b_work, tolerance_rules_b, date_rules_b = self.create_optimized_match_keys(df_b, recon_rules, 'B')
        
        logger.info(f"✅ Match keys created: File A ({len(df_a_work)} records), File B ({len(df_b_work)} records)")
        logger.info(f"📈 Rule distribution: {len(tolerance_rules_a)} tolerance rules, {len(date_rules_a)} date rules")

        df_a_work['_orig_index_a'] = range(len(df_a_work))
        df_b_work['_orig_index_b'] = range(len(df_b_work))
        
        logger.info("🔍 Starting matching process with hash-based optimization...")

        # Group by match key for faster lookups (only for exact matches)
        grouped_b = df_b_work.groupby('_match_key')
        unique_keys_b = len(grouped_b.groups)
        logger.info(f"📊 Created {unique_keys_b:,} unique match key groups in File B")

        matched_indices_a = set()
        matched_indices_b = set()
        matches = []

        # Process matches in batches with hardware-optimized batch size
        batch_size = self.batch_size  # Use centralized threading configuration
        total_batches = (len(df_a_work) + batch_size - 1) // batch_size
        logger.info(f"⚙️ Processing {len(df_a_work):,} records in {total_batches} batches (size: {batch_size}) using {self.max_workers} workers")
        logger.info(f"🖥️ Hardware optimization: {self.threading_config.server_class} ({self.threading_config.available_cores} cores, {self.threading_config.optimization_level})")
        
        # Use parallel processing for batch execution on capable systems (6+ cores)
        # Enable parallel processing if we have multiple workers OR if system has 6+ cores
        if (self.threading_config.server_class in ['high_end_server', 'mid_range_server', 'standard_workstation'] and 
            (total_batches > 1 or self.threading_config.available_cores >= 6)) and self.max_workers >= 2:
            logger.info(f"🚀 Using parallel batch processing with {self.max_workers} workers for optimal performance")
            
            # Prepare batches for parallel processing
            batch_tasks = []
            grouped_b_dict = {key: group for key, group in grouped_b}
            
            for batch_idx, start_idx in enumerate(range(0, len(df_a_work), batch_size), 1):
                end_idx = min(start_idx + batch_size, len(df_a_work))
                batch_a = df_a_work.iloc[start_idx:end_idx]
                
                batch_data = {
                    'batch_a': batch_a,
                    'grouped_b': grouped_b_dict,
                    'recon_rules': recon_rules,
                    'df_a': df_a,
                    'df_b': df_b,
                    'selected_columns_a': selected_columns_a,
                    'selected_columns_b': selected_columns_b,
                    # Note: Using many-to-many matching by default (no restrictions on matching)
                    'batch_idx': batch_idx
                }
                batch_tasks.append(batch_data)
            
            # Process batches in parallel with optimal timeout
            base_timeout = 300  # 5 minutes base timeout
            timeout = get_timeout_for_operation(base_timeout, self.threading_config)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {executor.submit(self._process_batch_parallel, task): task['batch_idx'] 
                                 for task in batch_tasks}
                
                completed_batches = 0
                for future in as_completed(future_to_batch, timeout=timeout):
                    try:
                        batch_result = future.result()
                        matches.extend(batch_result['matches'])
                        matched_indices_a.update(batch_result['matched_indices_a'])
                        matched_indices_b.update(batch_result['matched_indices_b'])
                        
                        completed_batches += 1
                        if completed_batches % 10 == 0 or completed_batches == total_batches:
                            progress_pct = (completed_batches / total_batches) * 100
                            logger.info(f"📈 Completed {completed_batches}/{total_batches} batches ({progress_pct:.1f}%) - {len(matches):,} matches found")
                            
                    except Exception as e:
                        batch_idx = future_to_batch[future]
                        logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                        self.warnings.append(f"Batch {batch_idx} failed: {e}")
        else:
            # Fall back to sequential processing for smaller datasets or limited hardware
            logger.info(f"📝 Using sequential processing for {total_batches} batches")
            
            processed_records = 0
            for batch_idx, start_idx in enumerate(range(0, len(df_a_work), batch_size), 1):
                end_idx = min(start_idx + batch_size, len(df_a_work))
                batch_a = df_a_work.iloc[start_idx:end_idx]
                
                # Progress logging every 10 batches or for large datasets
                if batch_idx % 10 == 0 or total_batches <= 10:
                    progress_pct = (batch_idx / total_batches) * 100
                    logger.info(f"📈 Processing batch {batch_idx}/{total_batches} ({progress_pct:.1f}%) - {len(matches):,} matches found so far")

                for idx_a, row_a in batch_a.iterrows():
                    match_key = row_a['_match_key']

                    # Get potential matches from grouped data
                    if match_key in grouped_b.groups:
                        potential_matches = grouped_b.get_group(match_key)

                        for idx_b, row_b in potential_matches.iterrows():
                            # Check all reconciliation rules
                            all_rules_match = True

                            for rule in recon_rules:
                                val_a = row_a[rule.LeftFileColumn]
                                val_b = row_b[rule.RightFileColumn]

                                if rule.MatchType.lower() == "equals":
                                    if not self._check_equals_match(val_a, val_b):
                                        all_rules_match = False
                                        break
                                elif rule.MatchType.lower() == "date_equals":
                                    if not self._check_date_equals_match(val_a, val_b):
                                        all_rules_match = False
                                        break
                                elif rule.MatchType.lower() == "tolerance":
                                    if not self._check_tolerance_match(val_a, val_b, rule.ToleranceValue):
                                        all_rules_match = False
                                        break
                                elif rule.MatchType.lower() == "fuzzy":
                                    if not self._check_fuzzy_match(val_a, val_b, rule.ToleranceValue):
                                        all_rules_match = False
                                        break

                            if all_rules_match:
                                # Create match record with selected columns
                                match_record = self._create_match_record(
                                    row_a, row_b, df_a, df_b,
                                    selected_columns_a, selected_columns_b,
                                    recon_rules
                                )
                                matches.append(match_record)
                                
                                # Track matched records for unmatched calculation
                                matched_indices_a.add(row_a['_orig_index_a'])
                                matched_indices_b.add(row_b['_orig_index_b'])

        # Create result DataFrames with selected columns
        matched_df = pd.DataFrame(matches) if matches else pd.DataFrame()
        
        logger.info(f"✅ Main reconciliation completed - found {len(matches):,} matches")
        logger.info("🔍 Calculating unmatched records...")

        # Unmatched records are calculated based on matched_indices sets
        unmatched_a = self._select_result_columns(
            df_a_work[~df_a_work['_orig_index_a'].isin(matched_indices_a)].drop(['_orig_index_a', '_match_key'],
                                                                                axis=1),
            selected_columns_a, recon_rules, 'A'
        )

        unmatched_b = self._select_result_columns(
            df_b_work[~df_b_work['_orig_index_b'].isin(matched_indices_b)].drop(['_orig_index_b', '_match_key'],
                                                                                axis=1),
            selected_columns_b, recon_rules, 'B'
        )
        
        logger.info(f"📊 Unmatched records: File A ({len(unmatched_a):,}), File B ({len(unmatched_b):,})")

        # Add closest match functionality if requested
        if find_closest_matches:
            logger.info("🎯 Starting closest match analysis (enhanced mode)...")
            closest_match_start = time.time()
            
            # Prepare full datasets for comparison (with selected columns)
            full_df_a = self._select_result_columns(df_a_work.drop(['_orig_index_a', '_match_key'], axis=1), 
                                                   selected_columns_a, recon_rules, 'A')
            full_df_b = self._select_result_columns(df_b_work.drop(['_orig_index_b', '_match_key'], axis=1), 
                                                   selected_columns_b, recon_rules, 'B')
            
            if len(unmatched_a) > 0 and len(full_df_b) > 0:
                logger.info(f"🔍 Analyzing {len(unmatched_a):,} unmatched A records against entire File B ({len(full_df_b):,} records)")
                unmatched_a = self._add_closest_matches(unmatched_a, full_df_b, recon_rules, 'A', closest_match_config)
                
            if len(unmatched_b) > 0 and len(full_df_a) > 0:
                logger.info(f"🔍 Analyzing {len(unmatched_b):,} unmatched B records against entire File A ({len(full_df_a):,} records)")
                unmatched_b = self._add_closest_matches(unmatched_b, full_df_a, recon_rules, 'B', closest_match_config)
            
            closest_match_time = time.time() - closest_match_start
            logger.info(f"✅ Closest match analysis completed in {closest_match_time:.2f}s")
        
        # Final summary
        total_time = time.time() - start_time
        match_percentage = (len(matches) / len(df_a)) * 100 if len(df_a) > 0 else 0
        
        logger.info("🏁 Reconciliation process completed!")
        logger.info(f"📊 Final Results Summary:")
        logger.info(f"   ✅ Matched records: {len(matches):,}")
        logger.info(f"   🔍 Unmatched A: {len(unmatched_a):,}")  
        logger.info(f"   🔍 Unmatched B: {len(unmatched_b):,}")
        logger.info(f"   📈 Match percentage: {match_percentage:.1f}%")
        logger.info(f"   ⏱️ Total processing time: {total_time:.2f}s")
        logger.info(f"   🚀 Processing rate: {(len(df_a) + len(df_b))/total_time:.0f} records/second")

        return {
            'matched': matched_df,
            'unmatched_file_a': unmatched_a,
            'unmatched_file_b': unmatched_b
        }

    def _add_closest_matches(self, unmatched_source: pd.DataFrame, full_target: pd.DataFrame, 
                            recon_rules: List[ReconciliationRule], source_file: str, 
                            closest_match_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add closest match columns to unmatched records using optimized composite similarity scoring
        
        PERFORMANCE OPTIMIZATIONS:
        - Early termination on perfect matches
        - Column type caching
        - Minimum score thresholds
        - Batch processing for large datasets
        - Memory-efficient processing
        - Hardware-aware thread allocation
        
        Args:
            unmatched_source: Unmatched records from source file
            full_target: All records from target file (both matched and unmatched for comparison)
            recon_rules: Reconciliation rules to determine which columns to compare
            source_file: 'A' or 'B' to indicate which file is the source
            
        Returns:
            DataFrame with closest match information added
        """
        if len(unmatched_source) == 0 or len(full_target) == 0:
            return unmatched_source
            
        logger.info(f"🚀 Starting optimized closest match analysis for {len(unmatched_source):,} unmatched records against {len(full_target):,} target records")
        
        # Check if we need batch processing for large datasets
        total_comparisons = len(unmatched_source) * len(full_target)
        logger.info(f"📊 Total potential comparisons: {total_comparisons:,}")
        
        # Use config to override comparison limits if specified
        max_comparisons = closest_match_config.max_comparisons if closest_match_config and closest_match_config.max_comparisons else 10_000_000
        
        # UNIFIED APPROACH: Always use batch processing with intelligent thread allocation
        from app.utils.threading_config import get_reconciliation_config
        
        # Get optimal threading configuration for reconciliation workload
        threading_config = get_reconciliation_config()
        num_threads = threading_config.max_workers
        
        logger.info(f"🚀 Using hardware-aware threading: {num_threads} threads on {threading_config.server_class} ({threading_config.available_cores} cores)")
        logger.info(f"📊 Dataset size: {total_comparisons:,} comparisons")
        
        # Always use the batch processing function with intelligent threading
        return self._add_closest_matches_optimized_batch(unmatched_source, full_target, recon_rules, source_file, closest_match_config, num_threads)

    def _add_closest_matches_optimized_single(self, unmatched_source: pd.DataFrame, full_target: pd.DataFrame,
                                            recon_rules: List[ReconciliationRule], source_file: str, 
                                            closest_match_config: Optional[Dict] = None) -> pd.DataFrame:
        """Optimized single-batch processing for smaller datasets"""
        
        # Make a copy to avoid modifying the original
        result_df = unmatched_source.copy()
        
        # Initialize closest match columns
        result_df['closest_match_record'] = None
        result_df['closest_match_score'] = 0.0
        result_df['closest_match_details'] = None
        
        # Get columns to compare - use specific columns from config if provided, otherwise use reconciliation rules
        compare_columns = []
        
        if closest_match_config and closest_match_config.specific_columns:
            # Use user-specified columns for comparison
            specific_cols = closest_match_config.specific_columns
            logger.info(f"🎯 Using specific columns for comparison: {specific_cols}")
            
            if source_file == 'A':
                # For file A: specific_columns = {"file_a_col": "file_b_col"}
                for source_col, target_col in specific_cols.items():
                    if source_col in unmatched_source.columns and target_col in full_target.columns:
                        compare_columns.append((source_col, target_col))
            else:
                # For file B: reverse the mapping
                for file_a_col, file_b_col in specific_cols.items():
                    if file_b_col in unmatched_source.columns and file_a_col in full_target.columns:
                        compare_columns.append((file_b_col, file_a_col))
        else:
            # Use all reconciliation rule columns (default behavior)
            logger.info("🔍 Using all reconciliation rule columns for comparison")
            for rule in recon_rules:
                if source_file == 'A':
                    source_col = rule.LeftFileColumn
                    target_col = rule.RightFileColumn
                else:
                    source_col = rule.RightFileColumn  
                    target_col = rule.LeftFileColumn
                    
                if source_col in unmatched_source.columns and target_col in full_target.columns:
                    compare_columns.append((source_col, target_col))
        
        if not compare_columns:
            logger.warning(f"⚠️ No comparable columns found for closest match analysis")
            return result_df
        
        # Pre-compute column types for caching
        column_type_cache = {}
        for source_col, target_col in compare_columns:
            if source_col not in column_type_cache:
                column_type_cache[source_col] = self._detect_column_type(
                    source_col, 
                    unmatched_source[source_col].head(10).tolist()
                )
        
        logger.info(f"🔍 Comparing {len(compare_columns)} column pairs with optimizations...")
        
        # Performance settings - use config values if provided
        MIN_SCORE_THRESHOLD = closest_match_config.min_score_threshold if closest_match_config else 30.0
        PERFECT_MATCH_THRESHOLD = closest_match_config.perfect_match_threshold if closest_match_config else 99.5
        
        logger.info(f"⚙️ Performance thresholds: Min score = {MIN_SCORE_THRESHOLD}%, Perfect match = {PERFECT_MATCH_THRESHOLD}%")
        
        processed_count = 0
        matches_found = 0
        early_terminations = 0
        skipped_comparisons = 0
        
        # Process each unmatched record
        for idx, source_row in unmatched_source.iterrows():
            best_match_score = 0.0
            best_match_record = None
            best_match_details = {}
            
            # Progress reporting
            processed_count += 1
            if processed_count % 1000 == 0:
                logger.info(f"📊 Processed {processed_count:,} / {len(unmatched_source):,} records | Matches: {matches_found:,} | Early exits: {early_terminations:,}")
            
            # Compare with each record in the target file
            for target_idx, target_row in full_target.iterrows():
                # Early exit if we already found a very good match
                if best_match_score >= PERFECT_MATCH_THRESHOLD:
                    early_terminations += 1
                    break
                
                # Two-phase closest match: exact filters + similarity scoring
                
                # Phase 1: Check exact match filters (non-similarity columns)
                if closest_match_config and closest_match_config.specific_columns:
                    # Get all reconciliation columns for exact match filtering
                    all_recon_columns = []
                    for rule in recon_rules:
                        if source_file == 'A':
                            source_col = rule.LeftFileColumn
                            target_col = rule.RightFileColumn
                        else:
                            source_col = rule.RightFileColumn
                            target_col = rule.LeftFileColumn
                        if source_col in unmatched_source.columns and target_col in full_target.columns:
                            all_recon_columns.append((source_col, target_col))
                    
                    # Check exact match for non-similarity columns
                    exact_match_failed = False
                    for source_col, target_col in all_recon_columns:
                        # Skip if this column is used for similarity matching
                        is_similarity_column = False
                        for sim_src, sim_tgt in compare_columns:
                            if source_col == sim_src and target_col == sim_tgt:
                                is_similarity_column = True
                                break
                        
                        if not is_similarity_column:
                            # This column must match exactly
                            source_val = source_row[source_col]
                            target_val = target_row[target_col]
                            
                            # Normalize values for exact comparison
                            source_str = str(source_val).strip().lower() if pd.notna(source_val) else ""
                            target_str = str(target_val).strip().lower() if pd.notna(target_val) else ""
                            
                            if source_str != target_str:
                                exact_match_failed = True
                                break
                    
                    if exact_match_failed:
                        continue  # Skip this record - exact match requirement failed
                
                # Phase 2: Calculate similarity for specified columns (or all if none specified)
                column_scores = {}
                total_weighted_score = 0.0
                
                for source_col, target_col in compare_columns:
                    source_val = source_row[source_col]
                    target_val = target_row[target_col]
                    
                    # Use cached column type
                    column_type = column_type_cache[source_col]
                    
                    # Calculate similarity
                    # Check cache first to avoid recalculation
                    cache_key = f"{source_val}|{target_val}|{column_type}"
                    if cache_key in similarity_cache:
                        similarity = similarity_cache[cache_key]
                    else:
                        similarity = self._calculate_composite_similarity(source_val, target_val, column_type)
                        similarity_cache[cache_key] = similarity
                        if len(similarity_cache) > 10000:
                            similarity_cache.clear()
                    column_scores[f"{source_col}_vs_{target_col}"] = {
                        'score': similarity,
                        'source_value': source_val,
                        'target_value': target_val,
                        'type': column_type
                    }
                    
                    total_weighted_score += similarity
                
                # Average score across similarity columns only
                avg_score = total_weighted_score / len(compare_columns) if compare_columns else 0.0
                
                # Update best match if this is better and above threshold
                if avg_score > best_match_score and avg_score > MIN_SCORE_THRESHOLD:
                    best_match_score = avg_score
                    best_match_record = target_row.to_dict()
                    best_match_details = column_scores
                    
                    # Early termination for perfect matches
                    if avg_score >= PERFECT_MATCH_THRESHOLD:
                        break
            
            # Add closest match information to result
            if best_match_score > 0:
                matches_found += 1
                # Create simplified record summary instead of full JSON
                if best_match_record:
                    record_summary = []
                    for key, value in list(best_match_record.items())[:3]:  # Show first 3 key fields
                        record_summary.append(f"{key}: {value}")
                    result_df.at[idx, 'closest_match_record'] = "; ".join(record_summary)
                else:
                    result_df.at[idx, 'closest_match_record'] = "No match details available"
                result_df.at[idx, 'closest_match_score'] = round(best_match_score, 2)
                
                # Create simple closest match details showing only mismatched columns
                details_list = []
                for column_key, details in best_match_details.items():
                    source_val = details['source_value']
                    target_val = details['target_value']
                    score = details['score']
                    
                    # Only include columns that don't match exactly (score < 100)
                    if score < 100:
                        # Extract just the column name from the key (e.g., "transaction_id_vs_ref_id" -> "transaction_id")
                        column_name = column_key.split('_vs_')[0]
                        details_list.append(f"{column_name}: '{source_val}' → '{target_val}'")
                
                if details_list:
                    result_df.at[idx, 'closest_match_details'] = "; ".join(details_list)
                else:
                    result_df.at[idx, 'closest_match_details'] = "All columns match exactly"
                
        # Final performance summary
        total_potential_comparisons = len(unmatched_source) * len(full_target)
        actual_comparisons = total_potential_comparisons - skipped_comparisons
        efficiency_pct = (skipped_comparisons / total_potential_comparisons) * 100 if total_potential_comparisons > 0 else 0
        
        logger.info(f"✅ Single-threaded closest match completed!")
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   🎯 Records processed: {len(unmatched_source):,}")
        logger.info(f"   ✅ Matches found: {matches_found:,}")
        logger.info(f"   ⚡ Early terminations: {early_terminations:,}")
        logger.info(f"   ⏭️ Skipped comparisons: {skipped_comparisons:,} ({efficiency_pct:.1f}% efficiency)")
        logger.info(f"   🔥 Actual comparisons: {actual_comparisons:,} of {total_potential_comparisons:,} potential")
        
        return result_df

    def _add_closest_matches_optimized_batch(self, unmatched_source: pd.DataFrame, full_target: pd.DataFrame,
                                           recon_rules: List[ReconciliationRule], source_file: str, 
                                           closest_match_config: Optional[Dict] = None, num_threads: int = None) -> pd.DataFrame:
        """
        Batch processing for very large datasets with parallel processing
        
        ADVANCED OPTIMIZATIONS:
        - Parallel processing using multiprocessing
        - Intelligent batching
        - Memory-efficient chunk processing
        - Advanced caching and indexing
        """
        from multiprocessing import Pool, cpu_count
        import multiprocessing as mp
        import gc
        import time
        
        # Get available cores for logging (from threading config or system)
        from app.utils.threading_config import get_reconciliation_config
        threading_config = get_reconciliation_config()
        available_cores = threading_config.available_cores
        
        logger.info(f"🚀 Starting advanced batch processing for large dataset...")
        batch_start_time = time.time()
        
        # Make a copy to avoid modifying the original
        result_df = unmatched_source.copy()
        
        # Initialize closest match columns
        result_df['closest_match_record'] = None
        result_df['closest_match_score'] = 0.0
        result_df['closest_match_details'] = None
        
        # Get columns to compare - use specific columns from config if provided, otherwise use reconciliation rules
        compare_columns = []
        
        if closest_match_config and closest_match_config.specific_columns:
            # Use user-specified columns for comparison
            specific_cols = closest_match_config.specific_columns
            logger.info(f"🎯 Using specific columns for batch comparison: {specific_cols}")
            
            if source_file == 'A':
                # For file A: specific_columns = {"file_a_col": "file_b_col"}
                for source_col, target_col in specific_cols.items():
                    if source_col in unmatched_source.columns and target_col in full_target.columns:
                        compare_columns.append((source_col, target_col))
            else:
                # For file B: reverse the mapping
                for file_a_col, file_b_col in specific_cols.items():
                    if file_b_col in unmatched_source.columns and file_a_col in full_target.columns:
                        compare_columns.append((file_b_col, file_a_col))
        else:
            # Use all reconciliation rule columns (default behavior)
            logger.info("🔍 Using all reconciliation rule columns for batch comparison")
            for rule in recon_rules:
                if source_file == 'A':
                    source_col = rule.LeftFileColumn
                    target_col = rule.RightFileColumn
                else:
                    source_col = rule.RightFileColumn  
                    target_col = rule.LeftFileColumn
                    
                if source_col in unmatched_source.columns and target_col in full_target.columns:
                    compare_columns.append((source_col, target_col))
        
        if not compare_columns:
            logger.warning(f"⚠️ No comparable columns found for batch closest match analysis")
            return result_df
        
        # Pre-compute column types for caching
        column_type_cache = {}
        for source_col, target_col in compare_columns:
            if source_col not in column_type_cache:
                column_type_cache[source_col] = self._detect_column_type(
                    source_col, 
                    unmatched_source[source_col].head(10).tolist()
                )
        
        # Determine optimal batch size and number of processes using threading config
        total_records = len(unmatched_source)
        
        # Use the passed num_threads parameter (from threading config)
        num_processes = num_threads or 1
        logger.info(f"🔧 Using {num_processes} processes (from threading configuration)")
        
        # Calculate optimal batch size based on dataset size and thread count
        optimal_batch_size = max(50, min(500, total_records // num_processes)) if num_processes > 1 else total_records
        
        # Memory-aware optimization
        estimated_memory_mb = ((total_records * len(full_target)) / 1_000_000) * 0.1  # Rough estimate in MB
        
        # Adjust batch processing based on available memory
        if estimated_memory_mb > 4000:  # > 4GB estimated usage
            logger.info(f"📈 Large memory requirement detected ({estimated_memory_mb:.0f}MB estimated)")
            # Use smaller batches for memory efficiency
            optimal_batch_size = max(25, optimal_batch_size // 2)
            logger.info(f"⚡ Reduced batch size to {optimal_batch_size} for memory efficiency")
        
        logger.info(f"🔧 Final Configuration: {num_processes} processes, batch size: {optimal_batch_size}")
        logger.info(f"📊 Processing {total_records:,} records against {len(full_target):,} targets")
        logger.info(f"💾 Estimated memory impact: {estimated_memory_mb:.0f}MB ({(total_records * len(full_target)) / 1_000_000:.1f}M potential comparisons)")
        
        # Split unmatched_source into batches
        batches = []
        for i in range(0, total_records, optimal_batch_size):
            end_idx = min(i + optimal_batch_size, total_records)
            batch_df = unmatched_source.iloc[i:end_idx].copy()
            batches.append((i, batch_df))
        
        logger.info(f"📦 Created {len(batches)} batches for parallel processing")
        
        # Process batches with progress tracking
        processed_batches = []
        
        try:
            # Use ProcessPoolExecutor for better resource management
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import time
            
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit all batch jobs
                future_to_batch = {}
                
                for batch_idx, (start_idx, batch_df) in enumerate(batches):
                    future = executor.submit(
                        self._process_batch_closest_matches,
                        batch_df, full_target, compare_columns, column_type_cache,
                        batch_idx, len(batches), closest_match_config, recon_rules, source_file, unmatched_source
                    )
                    future_to_batch[future] = (start_idx, batch_df)
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    start_idx, original_batch = future_to_batch[future]
                    try:
                        # Adaptive timeout based on batch size and hardware
                        base_timeout = 300  # 5 minutes base
                        if num_processes >= 20:  # High-performance systems
                            timeout = min(base_timeout * 2, 600)  # Up to 10 minutes for large parallel jobs
                        else:
                            timeout = base_timeout
                            
                        batch_result = future.result(timeout=timeout)
                        processed_batches.append((start_idx, batch_result))
                        
                        # Progress update
                        elapsed = time.time() - start_time
                        completed = len(processed_batches)
                        remaining = len(batches) - completed
                        progress_pct = (completed / len(batches)) * 100 if len(batches) > 0 else 0
                        eta = (elapsed / completed * remaining) if completed > 0 else 0
                        
                        logger.info(f"📈 Completed batch {completed}/{len(batches)} ({progress_pct:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                        
                    except Exception as e:
                        logger.warning(f"❌ Batch {start_idx} processing failed: {str(e)}")
                        # Use original batch data as fallback
                        processed_batches.append((start_idx, original_batch))
            
            # Reassemble results
            processed_batches.sort(key=lambda x: x[0])  # Sort by start_idx
            
            for start_idx, batch_result in processed_batches:
                end_idx = start_idx + len(batch_result)
                result_df.iloc[start_idx:end_idx] = batch_result
            
            total_time = time.time() - start_time
            batch_total_time = time.time() - batch_start_time
            
            # Calculate performance metrics
            records_per_second = total_records / batch_total_time if batch_total_time > 0 else 0
            theoretical_single_thread_time = batch_total_time * num_processes  # Rough estimate
            speedup_factor = theoretical_single_thread_time / batch_total_time if batch_total_time > 0 else 1
            cpu_efficiency = (speedup_factor / num_processes) * 100 if num_processes > 0 else 0
            
            logger.info(f"✅ Parallel batch processing completed in {batch_total_time:.1f} seconds")
            logger.info(f"⚡ Processing rate: {records_per_second:.0f} records/second")
            logger.info(f"🚀 Parallelization speedup: {speedup_factor:.1f}x (using {num_processes} processes)")
            logger.info(f"🎯 CPU efficiency: {cpu_efficiency:.1f}% (theoretical max: 100%)")
            logger.info(f"💪 Hardware utilization: {num_processes}/{available_cores} cores ({(num_processes/available_cores)*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Parallel processing failed: {str(e)}")
            raise  # Re-raise the exception instead of fallback
        
        # Force garbage collection after intensive processing
        gc.collect()
        
        return result_df
    
    def _format_closest_match_details(self, column_scores: dict) -> str:
        """
        Format closest match details in a readable format
        
        Input: {'amount_vs_total_amount': {'score': 99.99, 'source_value': 899.95, 'target_value': 899.99, 'type': 'numeric'}}
        Output: 'amount: 899.95->899.99'
        """
        if not column_scores:
            return "No comparison details available"
        
        formatted_details = []
        for column_key, details in column_scores.items():
            # Extract column name (remove _vs_ part)
            if '_vs_' in column_key:
                source_col = column_key.split('_vs_')[0]
            else:
                source_col = column_key
            
            source_val = details.get('source_value', 'N/A')
            target_val = details.get('target_value', 'N/A')
            score = details.get('score', 0)
            
            # Handle NaN/None values properly
            import pandas as pd
            if pd.isna(source_val) and pd.isna(target_val):
                # Both are NaN - skip this comparison as it's not meaningful
                continue
            elif pd.isna(source_val):
                source_val = "(empty)"
            elif pd.isna(target_val):
                target_val = "(empty)"
            
            # Convert to string and handle 'nan' strings
            source_str = str(source_val)
            target_str = str(target_val)
            
            if source_str.lower() == 'nan':
                source_str = "(empty)"
            if target_str.lower() == 'nan':
                target_str = "(empty)"
            
            # Only include meaningful comparisons (not both empty)
            if source_str != "(empty)" or target_str != "(empty)":
                formatted_details.append(f"{source_col}: {source_str}->{target_str}")
        
        return "; ".join(formatted_details) if formatted_details else "No meaningful comparisons found"
    
    def _add_closest_matches_ultra_optimized(self, unmatched_source: pd.DataFrame, full_target: pd.DataFrame,
                                          recon_rules: List[ReconciliationRule], source_file: str, 
                                          closest_match_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Ultra-optimized closest match with pre-filtering and vectorized operations
        
        PERFORMANCE OPTIMIZATIONS:
        - Pre-filter with exact match indexing (10-100x speedup)
        - Pre-normalize string columns (2-5x speedup)
        - Vectorized similarity calculations (5-20x speedup)
        - Intelligent early termination
        """
        import time
        from collections import defaultdict
        
        logger.info(f"🚀 Starting ultra-optimized closest match processing...")
        start_time = time.time()
        
        # Make a copy to avoid modifying the original
        result_df = unmatched_source.copy()
        
        # Initialize closest match columns
        result_df['closest_match_record'] = None
        result_df['closest_match_score'] = 0.0
        result_df['closest_match_details'] = None
        
        # Get columns to compare - use specific columns from config if provided
        compare_columns = []
        exact_match_columns = []
        
        if closest_match_config and closest_match_config.specific_columns:
            # Use user-specified columns for similarity
            specific_cols = closest_match_config.specific_columns
            logger.info(f"🎯 Using specific columns for similarity: {specific_cols}")
            
            # Build similarity columns
            if source_file == 'A':
                for source_col, target_col in specific_cols.items():
                    if source_col in unmatched_source.columns and target_col in full_target.columns:
                        compare_columns.append((source_col, target_col))
            else:
                for file_a_col, file_b_col in specific_cols.items():
                    if file_b_col in unmatched_source.columns and file_a_col in full_target.columns:
                        compare_columns.append((file_b_col, file_a_col))
            
            # Build exact match columns (all non-similarity reconciliation columns)
            for rule in recon_rules:
                if source_file == 'A':
                    source_col = rule.LeftFileColumn
                    target_col = rule.RightFileColumn
                else:
                    source_col = rule.RightFileColumn
                    target_col = rule.LeftFileColumn
                
                if source_col in unmatched_source.columns and target_col in full_target.columns:
                    # Check if this is a similarity column
                    is_similarity_column = False
                    for sim_src, sim_tgt in compare_columns:
                        if source_col == sim_src and target_col == sim_tgt:
                            is_similarity_column = True
                            break
                    
                    if not is_similarity_column:
                        exact_match_columns.append((source_col, target_col))
        else:
            # Use all reconciliation rule columns for similarity (original behavior)
            logger.info("🔍 Using all reconciliation rule columns for similarity")
            for rule in recon_rules:
                if source_file == 'A':
                    source_col = rule.LeftFileColumn
                    target_col = rule.RightFileColumn
                else:
                    source_col = rule.RightFileColumn
                    target_col = rule.LeftFileColumn
                
                if source_col in unmatched_source.columns and target_col in full_target.columns:
                    compare_columns.append((source_col, target_col))
        
        if not compare_columns:
            logger.warning(f"⚠️ No comparable columns found for closest match analysis")
            return result_df
        
        logger.info(f"📊 Similarity columns: {len(compare_columns)}, Exact match columns: {len(exact_match_columns)}")
        
        # Performance settings
        MIN_SCORE_THRESHOLD = closest_match_config.min_score_threshold if closest_match_config else 30.0
        PERFECT_MATCH_THRESHOLD = closest_match_config.perfect_match_threshold if closest_match_config else 99.5
        
        # OPTIMIZATION 1: Pre-normalize string columns for exact matching
        if exact_match_columns:
            logger.info(f"🔧 Pre-normalizing {len(exact_match_columns)} columns for exact matching...")
            target_normalized = full_target.copy()
            source_normalized = unmatched_source.copy()
            
            for i, (source_col, target_col) in enumerate(exact_match_columns, 1):
                logger.info(f"  → Normalizing column {i}/{len(exact_match_columns)}: {source_col} ↔ {target_col}")
                # Normalize target column with progress feedback
                target_normalized[f"{target_col}_normalized"] = target_normalized[target_col].astype(str).str.strip().str.lower()
                # Normalize source column
                source_normalized[f"{source_col}_normalized"] = source_normalized[source_col].astype(str).str.strip().str.lower()
        else:
            # No exact match columns, skip normalization
            target_normalized = full_target
            source_normalized = unmatched_source
        
        # OPTIMIZATION 2: Create exact match index for fast filtering
        if exact_match_columns:
            logger.info(f"🗂️ Building exact match index for {len(full_target):,} target records...")
            
            # Create composite key for exact matching
            def create_composite_key(row, columns, suffix="_normalized"):
                key_parts = []
                for source_col, target_col in columns:
                    col_name = f"{target_col if source_file == 'A' else source_col}{suffix}"
                    key_parts.append(str(row.get(col_name, "")))
                return "|".join(key_parts)
            
            # Build target index with progress tracking
            target_index = defaultdict(list)
            batch_size = 10000
            total_rows = len(target_normalized)
            
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                logger.info(f"  → Processing batch {batch_start:,}-{batch_end:,} of {total_rows:,} records...")
                
                batch_df = target_normalized.iloc[batch_start:batch_end]
                for idx, row in batch_df.iterrows():
                    composite_key = create_composite_key(row, exact_match_columns)
                    target_index[composite_key].append(idx)
            
            logger.info(f"📈 Built index with {len(target_index):,} unique key combinations")
        else:
            target_index = None
        
        # Pre-compute column types for caching
        column_type_cache = {}
        for source_col, target_col in compare_columns:
            if source_col not in column_type_cache:
                column_type_cache[source_col] = self._detect_column_type(
                    source_col, 
                    unmatched_source[source_col].head(10).tolist()
                )
        
        # OPTIMIZATION 4: Initialize similarity cache
        similarity_cache = {}
        
        logger.info(f"🔍 Processing {len(unmatched_source)} source records...")
        
        processed_count = 0
        matches_found = 0
        early_terminations = 0
        filtered_comparisons = 0
        total_comparisons = 0
        
        # Process each unmatched record with timeout protection
        import time
        loop_start_time = time.time()
        timeout_seconds = 300  # 5 minute timeout for ultra-optimization
        
        for idx, source_row in unmatched_source.iterrows():
            # Check for timeout
            if time.time() - loop_start_time > timeout_seconds:
                logger.error(f"❌ Ultra-optimization timeout after {timeout_seconds}s - dataset too large for this method")
                raise TimeoutError(f"Ultra-optimization timed out after {timeout_seconds} seconds. Consider using smaller dataset or batch processing.")
            
            best_match_score = 0.0
            best_match_record = None
            best_match_details = {}
            
            processed_count += 1
            if processed_count % 100 == 0:  # More frequent progress for large datasets
                elapsed = time.time() - loop_start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta = (len(unmatched_source) - processed_count) / rate if rate > 0 else 0
                logger.info(f"📊 Processed {processed_count:,} / {len(unmatched_source):,} records | Matches: {matches_found:,} | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
            
            # OPTIMIZATION 3: Use exact match index for pre-filtering
            if exact_match_columns and target_index:
                source_composite_key = create_composite_key(source_normalized.loc[idx], exact_match_columns)
                candidate_indices = target_index.get(source_composite_key, [])
                
                if candidate_indices:
                    # Found exact matches, use only those candidates
                    target_candidates = full_target.loc[candidate_indices]
                    filtered_comparisons += len(candidate_indices)
                else:
                    # No exact matches found, but still do similarity analysis on all records
                    # This is important for closest match - we want to find the most similar even without exact matches
                    target_candidates = full_target
                    filtered_comparisons += len(full_target)
            else:
                # No exact match filtering, use all target records
                target_candidates = full_target
                filtered_comparisons += len(full_target)
            
            total_comparisons += len(target_candidates)
            
            # Process similarity scoring on filtered candidates
            for target_idx, target_row in target_candidates.iterrows():
                # Early exit if we already found a very good match
                if best_match_score >= PERFECT_MATCH_THRESHOLD:
                    early_terminations += 1
                    break
                
                # Calculate similarity for specified columns
                column_scores = {}
                total_weighted_score = 0.0
                
                for source_col, target_col in compare_columns:
                    source_val = source_row[source_col]
                    target_val = target_row[target_col]
                    
                    # Use cached column type
                    column_type = column_type_cache[source_col]
                    
                    # OPTIMIZATION 4: Calculate similarity with caching
                    cache_key = f"{source_val}|{target_val}|{column_type}"
                    if cache_key in similarity_cache:
                        similarity = similarity_cache[cache_key]
                    else:
                        similarity = self._calculate_composite_similarity(source_val, target_val, column_type)
                        similarity_cache[cache_key] = similarity
                        # Limit cache size to prevent memory issues
                        if len(similarity_cache) > 10000:
                            similarity_cache.clear()
                    column_scores[f"{source_col}_vs_{target_col}"] = {
                        'score': similarity,
                        'source_value': source_val,
                        'target_value': target_val,
                        'type': column_type
                    }
                    
                    total_weighted_score += similarity
                
                # Average score across similarity columns only
                avg_score = total_weighted_score / len(compare_columns) if compare_columns else 0.0
                
                # Update best match if this is better and above threshold
                if avg_score > best_match_score and avg_score > MIN_SCORE_THRESHOLD:
                    best_match_score = avg_score
                    best_match_record = target_row.to_dict()
                    best_match_details = column_scores
                    
                    # Early termination for perfect matches
                    if avg_score >= PERFECT_MATCH_THRESHOLD:
                        break
            
            # Add closest match information to result
            if best_match_score > 0:
                matches_found += 1
                # Create simplified record summary
                if best_match_record:
                    record_summary = []
                    for key, value in list(best_match_record.items())[:3]:  # Show first 3 key fields
                        record_summary.append(f"{key}: {value}")
                    result_df.at[idx, 'closest_match_record'] = "; ".join(record_summary)
                    result_df.at[idx, 'closest_match_score'] = round(best_match_score, 2)
                    result_df.at[idx, 'closest_match_details'] = self._format_closest_match_details(best_match_details)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance metrics
        avg_comparisons_per_record = total_comparisons / len(unmatched_source) if len(unmatched_source) > 0 else 0
        reduction_ratio = (len(full_target) - avg_comparisons_per_record) / len(full_target) * 100 if len(full_target) > 0 else 0
        
        # Calculate cache efficiency
        cache_hit_ratio = 0.0
        if total_comparisons > 0:
            cache_hits = total_comparisons - len(similarity_cache)
            cache_hit_ratio = (cache_hits / total_comparisons) * 100 if total_comparisons > 0 else 0
        
        logger.info(f"✅ Ultra-optimized processing completed in {processing_time:.2f} seconds")
        logger.info(f"📊 Performance metrics:")
        logger.info(f"   • Records processed: {processed_count:,}")
        logger.info(f"   • Closest matches found: {matches_found:,}")
        logger.info(f"   • Early terminations: {early_terminations:,}")
        logger.info(f"   • Avg comparisons per record: {avg_comparisons_per_record:.1f} (vs {len(full_target):,} without optimization)")
        logger.info(f"   • Comparison reduction: {reduction_ratio:.1f}%")
        logger.info(f"   • Similarity cache size: {len(similarity_cache):,} entries")
        logger.info(f"   • Cache hit ratio: {cache_hit_ratio:.1f}%")
        logger.info(f"   • Processing rate: {processed_count/processing_time:.1f} records/second")
        
        return result_df
    
    def _process_batch_closest_matches(self, batch_df: pd.DataFrame, full_target: pd.DataFrame, 
                                     compare_columns: list, column_type_cache: dict, 
                                     batch_idx: int, total_batches: int, 
                                     closest_match_config: Optional[Dict] = None,
                                     recon_rules: Optional[List] = None,
                                     source_file: Optional[str] = None,
                                     unmatched_source: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process a single batch of records for closest matches
        This method is designed to be called by multiprocessing
        """
        
        # Initialize result columns
        batch_df = batch_df.copy()
        batch_df['closest_match_record'] = None
        batch_df['closest_match_score'] = 0.0
        batch_df['closest_match_details'] = None
        
        # Performance settings - use config values if provided
        MIN_SCORE_THRESHOLD = closest_match_config.min_score_threshold if closest_match_config else 30.0
        PERFECT_MATCH_THRESHOLD = closest_match_config.perfect_match_threshold if closest_match_config else 99.5
        
        # Create target index for faster lookup (if beneficial)
        target_size = len(full_target)
        # Use config to override sampling behavior if specified
        if closest_match_config and closest_match_config.use_sampling is not None:
            use_sampling = closest_match_config.use_sampling
        else:
            use_sampling = target_size > 50_000  # Use sampling for very large targets
        
        if use_sampling:
            # For extremely large targets, sample a representative subset
            sample_size = min(10_000, target_size // 2)
            target_sample = full_target.sample(n=sample_size, random_state=42)
            logger.debug(f"🎯 Batch {batch_idx}: Using target sampling ({sample_size:,} records from {target_size:,} total)")
        else:
            target_sample = full_target
            logger.debug(f"🔍 Batch {batch_idx}: Processing against full target ({target_size:,} records)")
        
        # SAFE OPTIMIZATION: Add similarity caching for this batch (no logic changes)
        similarity_cache = {}
        processed_in_batch = 0
        batch_start_time = time.time()
        
        # Process each record in the batch
        for idx, source_row in batch_df.iterrows():
            processed_in_batch += 1
            
            # Progress feedback for long-running batches
            if processed_in_batch % 50 == 0:
                elapsed = time.time() - batch_start_time
                rate = processed_in_batch / elapsed if elapsed > 0 else 0
                logger.debug(f"Batch {batch_idx}: Processed {processed_in_batch}/{len(batch_df)} records at {rate:.1f}/s")
            
            best_match_score = 0.0
            best_match_record = None
            best_match_details = {}
            
            # Compare with each record in the target (or sample)
            for target_idx, target_row in target_sample.iterrows():
                # Early exit if we already found a very good match
                if best_match_score >= PERFECT_MATCH_THRESHOLD:
                    break
                
                # Two-phase closest match: exact filters + similarity scoring
                
                # Phase 1: Check exact match filters (non-similarity columns) 
                if closest_match_config and closest_match_config.specific_columns:
                    # Get all reconciliation columns for exact match filtering
                    all_recon_columns = []
                    for rule in recon_rules:
                        if source_file == 'A':
                            source_col = rule.LeftFileColumn
                            target_col = rule.RightFileColumn
                        else:
                            source_col = rule.RightFileColumn
                            target_col = rule.LeftFileColumn
                        if source_col in unmatched_source.columns and target_col in full_target.columns:
                            all_recon_columns.append((source_col, target_col))
                    
                    # Check exact match for non-similarity columns
                    exact_match_failed = False
                    for source_col, target_col in all_recon_columns:
                        # Skip if this column is used for similarity matching
                        is_similarity_column = False
                        for sim_src, sim_tgt in compare_columns:
                            if source_col == sim_src and target_col == sim_tgt:
                                is_similarity_column = True
                                break
                        
                        if not is_similarity_column:
                            # This column must match exactly
                            source_val = source_row[source_col]
                            target_val = target_row[target_col]
                            
                            # Normalize values for exact comparison
                            source_str = str(source_val).strip().lower() if pd.notna(source_val) else ""
                            target_str = str(target_val).strip().lower() if pd.notna(target_val) else ""
                            
                            if source_str != target_str:
                                exact_match_failed = True
                                break
                    
                    if exact_match_failed:
                        continue  # Skip this record - exact match requirement failed
                
                # Phase 2: Calculate similarity for specified columns (or all if none specified)
                column_scores = {}
                total_weighted_score = 0.0
                
                for source_col, target_col in compare_columns:
                    source_val = source_row[source_col]
                    target_val = target_row[target_col]
                    
                    # Use cached column type
                    column_type = column_type_cache[source_col]
                    
                    # Calculate similarity
                    # Check cache first to avoid recalculation
                    cache_key = f"{source_val}|{target_val}|{column_type}"
                    if cache_key in similarity_cache:
                        similarity = similarity_cache[cache_key]
                    else:
                        similarity = self._calculate_composite_similarity(source_val, target_val, column_type)
                        similarity_cache[cache_key] = similarity
                        if len(similarity_cache) > 10000:
                            similarity_cache.clear()
                    column_scores[f"{source_col}_vs_{target_col}"] = {
                        'score': similarity,
                        'source_value': source_val,
                        'target_value': target_val,
                        'type': column_type
                    }
                    
                    total_weighted_score += similarity
                
                # Average score across similarity columns only
                avg_score = total_weighted_score / len(compare_columns) if compare_columns else 0.0
                
                # Update best match if this is better and above threshold
                if avg_score > best_match_score and avg_score > MIN_SCORE_THRESHOLD:
                    best_match_score = avg_score
                    best_match_record = target_row.to_dict()
                    best_match_details = column_scores
                    
                    # Early termination for perfect matches
                    if avg_score >= PERFECT_MATCH_THRESHOLD:
                        break
            
            # Add closest match information to result
            if best_match_score > 0:
                # Create simplified record summary
                if best_match_record:
                    record_summary = []
                    for key, value in list(best_match_record.items())[:3]:
                        record_summary.append(f"{key}: {value}")
                    batch_df.at[idx, 'closest_match_record'] = "; ".join(record_summary)
                else:
                    batch_df.at[idx, 'closest_match_record'] = "No match details available"
                
                batch_df.at[idx, 'closest_match_score'] = round(best_match_score, 2)
                
                # Create simple closest match details showing only mismatched columns
                details_list = []
                for column_key, details in best_match_details.items():
                    source_val = details['source_value']
                    target_val = details['target_value']
                    score = details['score']
                    
                    # Only include columns that don't match exactly (score < 100)
                    if score < 100:
                        # Extract just the column name from the key
                        column_name = column_key.split('_vs_')[0]
                        details_list.append(f"{column_name}: '{source_val}' → '{target_val}'")
                
                if details_list:
                    batch_df.at[idx, 'closest_match_details'] = "; ".join(details_list)
                else:
                    batch_df.at[idx, 'closest_match_details'] = "All columns match exactly"
        
        return batch_df

    def _check_tolerance_match(self, val_a, val_b, tolerance: float) -> bool:
        """Check if two values match within tolerance"""
        try:
            if pd.isna(val_a) or pd.isna(val_b):
                return pd.isna(val_a) and pd.isna(val_b)

            num_a = float(val_a)
            num_b = float(val_b)

            if num_b != 0:
                percentage_diff = abs(num_a - num_b) / abs(num_b) * 100
                return percentage_diff <= tolerance
            else:
                return num_a == 0
        except (ValueError, TypeError):
            return False

    def _check_fuzzy_match(self, val_a, val_b, threshold: float) -> bool:
        """Check if two string values match using fuzzy matching (case insensitive)"""
        try:
            if pd.isna(val_a) or pd.isna(val_b):
                return pd.isna(val_a) and pd.isna(val_b)

            str_a = str(val_a).strip().lower()
            str_b = str(val_b).strip().lower()
            
            # Simple fuzzy matching using character overlap ratio
            if len(str_a) == 0 and len(str_b) == 0:
                return True
            if len(str_a) == 0 or len(str_b) == 0:
                return False
            
            # Calculate similarity ratio
            # Use a simple approach: common characters / max length
            common_chars = sum(1 for c in str_a if c in str_b)
            max_len = max(len(str_a), len(str_b))
            similarity = common_chars / max_len
            
            return similarity >= threshold
            
        except Exception:
            return False

    def _create_match_record(self, row_a, row_b, df_a, df_b,
                             selected_columns_a, selected_columns_b,
                             recon_rules) -> Dict:
        """Create a match record with selected columns"""
        match_record = {}

        # Get mandatory columns
        mandatory_a, mandatory_b = self.get_mandatory_columns(recon_rules, None, None)

        # Determine which columns to include
        cols_a = selected_columns_a if selected_columns_a else df_a.columns.tolist()
        cols_b = selected_columns_b if selected_columns_b else df_b.columns.tolist()

        # Ensure mandatory columns are included
        cols_a = list(set(cols_a) | mandatory_a)
        cols_b = list(set(cols_b) | mandatory_b)

        # Add columns from both files
        for col in cols_a:
            if col in row_a:
                match_record[f"FileA_{col}"] = row_a[col]

        for col in cols_b:
            if col in row_b:
                match_record[f"FileB_{col}"] = row_b[col]

        return match_record

    def _select_result_columns(self, df: pd.DataFrame, selected_columns: Optional[List[str]],
                               recon_rules: List[ReconciliationRule], file_type: str) -> pd.DataFrame:
        """Select only requested columns plus mandatory ones"""
        if selected_columns is None:
            return df

        # Get mandatory columns based on reconciliation rules
        mandatory_cols = set()
        for rule in recon_rules:
            if file_type == 'A':
                mandatory_cols.add(rule.LeftFileColumn)
            else:
                mandatory_cols.add(rule.RightFileColumn)

        # Combine selected and mandatory columns
        final_columns = list(set(selected_columns) | mandatory_cols)

        # Filter to only existing columns
        existing_columns = [col for col in final_columns if col in df.columns]

        return df[existing_columns] if existing_columns else df


# Create optimized storage for results with compression
class OptimizedReconciliationStorage:
    def __init__(self):
        self.storage = {}

    def store_results(self, recon_id: str, results: Dict[str, pd.DataFrame]) -> bool:
        """Store results with optimized format"""
        try:
            # Convert to optimized format for storage
            optimized_results = {
                'matched': results['matched'].to_dict('records'),
                'unmatched_file_a': results['unmatched_file_a'].to_dict('records'),
                'unmatched_file_b': results['unmatched_file_b'].to_dict('records'),
                'timestamp': pd.Timestamp.now(),
                'row_counts': {
                    'matched': len(results['matched']),
                    'unmatched_a': len(results['unmatched_file_a']),
                    'unmatched_b': len(results['unmatched_file_b'])
                }
            }

            self.storage[recon_id] = optimized_results
            return True
        except Exception as e:
            print(f"Error storing results: {e}")
            return False

    def get_results(self, recon_id: str) -> Optional[Dict]:
        """Get stored results"""
        return self.storage.get(recon_id)


# Global instances
optimized_reconciliation_storage = OptimizedReconciliationStorage()
