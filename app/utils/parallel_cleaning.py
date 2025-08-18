#!/usr/bin/env python3
"""
High-performance parallel data cleaning for large datasets
Optimized for files with millions of rows and hundreds of columns
"""
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
from typing import List, Dict, Tuple, Any
import time
import functools
from app.utils.threading_config import get_cleaning_config

logger = logging.getLogger(__name__)

class ParallelDataCleaner:
    """
    High-performance data cleaner using multi-threading and vectorized operations
    Optimized for large datasets with millions of rows and hundreds of columns
    """
    
    def __init__(self, max_workers: int = None):
        """
        Initialize the parallel data cleaner with centralized hardware-aware thread allocation
        
        Args:
            max_workers: Maximum number of worker threads (defaults to centralized config)
        """
        # Use centralized threading configuration
        self.threading_config = get_cleaning_config(max_workers_override=max_workers)
        self.max_workers = self.threading_config.max_workers
        self.batch_size = self.threading_config.batch_size
        
        self.stats = {
            'timing': {},
            'performance': {},
            'threading_config': {
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'available_cores': self.threading_config.available_cores,
                'server_class': self.threading_config.server_class,
                'optimization_level': self.threading_config.optimization_level
            }
        }
        
    def clean_dataframe_parallel(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform complete data cleaning using parallel processing
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned_df, cleanup_stats)
        """
        start_time = time.time()
        original_shape = df.shape
        
        logger.info(f"🚀 Starting parallel data cleaning for {original_shape[0]:,} rows × {original_shape[1]} columns using {self.max_workers} threads")
        
        # Step 1: Fast empty column detection (parallel)
        step_start = time.time()
        empty_columns = self._detect_empty_columns_parallel(df)
        if empty_columns:
            logger.info(f"  🗑️  Removing {len(empty_columns)} empty columns: {empty_columns[:5]}{'...' if len(empty_columns) > 5 else ''}")
            df = df.drop(columns=empty_columns)
        self.stats['timing']['empty_columns'] = time.time() - step_start
        
        # Step 2: Fast empty row detection (vectorized)
        step_start = time.time()
        empty_row_mask = self._detect_empty_rows_vectorized(df)
        empty_row_count = empty_row_mask.sum()
        if empty_row_count > 0:
            logger.info(f"  📋 Removing {empty_row_count:,} empty rows")
            df = df[~empty_row_mask].reset_index(drop=True)
        self.stats['timing']['empty_rows'] = time.time() - step_start
        
        # Step 3: Parallel column name cleaning
        step_start = time.time()
        df = self._clean_column_names_parallel(df)
        self.stats['timing']['column_names'] = time.time() - step_start
        
        # Step 4: Parallel data value cleaning with adaptive chunking
        step_start = time.time()
        cleaned_values_count = self._clean_data_values_parallel(df)
        self.stats['timing']['data_values'] = time.time() - step_start
        
        # Step 5: Preserve integer types (prevent 15 -> 15.0)
        step_start = time.time()
        converted_int_columns = self._preserve_integer_types_parallel(df)
        self.stats['timing']['integer_preservation'] = time.time() - step_start
        
        # Step 6: Parallel date normalization
        step_start = time.time()
        date_columns = self._normalize_date_columns_parallel(df)
        self.stats['timing']['date_normalization'] = time.time() - step_start
        
        total_time = time.time() - start_time
        
        # Calculate comprehensive performance metrics
        total_cells_processed = original_shape[0] * original_shape[1]
        cells_per_second = int(total_cells_processed / total_time) if total_time > 0 else 0
        
        # Add performance metadata to stats
        self.stats['performance'] = {
            'max_workers': self.max_workers,
            'total_cells_processed': total_cells_processed,
            'cells_per_second': cells_per_second,
            'megacells_per_second': round(cells_per_second / 1000000, 2),
            'processing_method': 'parallel_multi_threaded',
            'optimization_level': 'high_performance'
        }
        
        # Compile cleanup statistics
        cleanup_stats = {
            'original_rows': int(original_shape[0]),
            'original_columns': int(original_shape[1]),
            'removed_rows': int(empty_row_count),
            'removed_columns': int(len(empty_columns)),
            'empty_column_names': list(empty_columns),
            'final_rows': int(len(df)),
            'final_columns': int(len(df.columns)),
            'cleaned_values': int(cleaned_values_count),
            'preserved_integer_columns': list(converted_int_columns),
            'normalized_date_columns': list(date_columns),
            'processing_time_seconds': round(total_time, 2),
            'performance_stats': self.stats
        }
        
        logger.info(f"✅ Parallel cleaning completed in {total_time:.2f}s")
        logger.info(f"   Final size: {len(df):,} rows × {len(df.columns)} columns")
        logger.info(f"   Performance: {original_shape[0] * original_shape[1] / total_time / 1000:.0f}K cells/second")
        
        return df, cleanup_stats
    
    def _detect_empty_columns_parallel(self, df: pd.DataFrame) -> List[str]:
        """
        Detect empty columns using parallel processing
        Much faster for datasets with many columns
        """
        def check_column_empty(col_name):
            """Check if a single column is completely empty"""
            try:
                col_data = df[col_name]
                
                # Fast check: if all values are NaN
                if col_data.isna().all():
                    return col_name
                
                # Check for empty strings/whitespace (only for object columns)
                if col_data.dtype == 'object':
                    non_na_values = col_data.dropna()
                    if len(non_na_values) == 0:
                        return col_name
                    
                    # Vectorized string cleaning check
                    string_vals = non_na_values.astype(str).str.strip()
                    if (string_vals == '').all():
                        return col_name
                
                return None
            except Exception as e:
                logger.warning(f"Error checking column '{col_name}': {e}")
                return None
        
        empty_columns = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_col = {executor.submit(check_column_empty, col): col for col in df.columns}
            
            for future in as_completed(future_to_col):
                result = future.result()
                if result:
                    empty_columns.append(result)
        
        return empty_columns
    
    def _detect_empty_rows_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect empty rows using highly optimized vectorized operations
        Much faster than row-by-row iteration
        """
        if len(df) == 0:
            return pd.Series([], dtype=bool)
        
        # For numeric columns: check if all values are NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_empty = df[numeric_cols].isna().all(axis=1)
        else:
            numeric_empty = pd.Series([True] * len(df), index=df.index)
        
        # For object columns: check if all values are NaN or empty strings
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            # Vectorized approach: convert to string, strip, check if empty
            object_data = df[object_cols].fillna('').astype(str)
            object_stripped = object_data.apply(lambda x: x.str.strip())
            object_empty = (object_stripped == '').all(axis=1)
        else:
            object_empty = pd.Series([True] * len(df), index=df.index)
        
        # Row is empty if both numeric and object parts are empty
        return numeric_empty & object_empty
    
    def _clean_column_names_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names using parallel processing
        """
        def clean_single_column_name(col_info):
            """Clean a single column name"""
            i, col = col_info
            if col is None:
                return f"Unnamed_{i}"
            cleaned = str(col).strip()
            if cleaned == "":
                return f"Unnamed_{i}"
            return cleaned
        
        original_columns = list(df.columns)
        
        # Parallel cleaning
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            cleaned_columns = list(executor.map(clean_single_column_name, enumerate(original_columns)))
        
        # Handle duplicates sequentially (fast operation)
        final_columns = []
        seen = set()
        
        for col in cleaned_columns:
            original_col = col
            counter = 1
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            seen.add(col)
            final_columns.append(col)
        
        df.columns = final_columns
        return df
    
    def _clean_data_values_parallel(self, df: pd.DataFrame) -> int:
        """
        Clean data values using parallel processing with chunking
        Optimized for large datasets with millions of rows
        """
        total_cleaned = 0
        string_columns = []
        
        # Identify string columns efficiently
        for col in df.columns:
            if df[col].dtype == 'object':
                # Sample check to see if column contains mostly strings
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    string_count = sum(1 for val in sample if isinstance(val, str))
                    if string_count >= len(sample) * 0.5:
                        string_columns.append(col)
        
        if not string_columns:
            return 0
        
        logger.info(f"  🧼 Cleaning {len(string_columns)} string columns with parallel processing...")
        
        def clean_column_chunk(args):
            """Clean a chunk of data for a specific column"""
            col, chunk_start, chunk_end = args
            try:
                # Get the chunk
                chunk = df.loc[chunk_start:chunk_end, col].copy()
                
                # Vectorized string cleaning
                mask = chunk.notna() & (chunk.astype(str) != chunk.astype(str).str.strip())
                if mask.any():
                    df.loc[chunk_start:chunk_end, col] = chunk.astype(str).str.strip()
                    return mask.sum()
                return 0
            except Exception as e:
                logger.warning(f"Error cleaning column '{col}' chunk {chunk_start}-{chunk_end}: {e}")
                return 0
        
        # Use centralized batch sizing configuration
        chunk_size = max(self.batch_size // 2,  # Use half the batch size for chunk processing
                        len(df) // (self.max_workers * 4))  # Ensure reasonable distribution
        
        for col in string_columns:
            tasks = []
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size - 1, len(df) - 1)
                tasks.append((col, start, end))
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(clean_column_chunk, tasks))
                total_cleaned += sum(results)
        
        return total_cleaned
    
    def _normalize_date_columns_parallel(self, df: pd.DataFrame) -> List[str]:
        """
        Normalize date columns using parallel processing
        Integrates with the parallel date utils for maximum performance
        """
        try:
            from app.utils.parallel_date_utils import normalize_datetime_columns_fast
            
            # Use parallel date normalization
            df_normalized, converted_columns = normalize_datetime_columns_fast(df, max_workers=self.max_workers)
            
            # Update the original dataframe in place
            for col in converted_columns:
                df[col] = df_normalized[col]
            
            if converted_columns:
                logger.info(f"  📅 Normalized {len(converted_columns)} date columns in parallel: {converted_columns}")
            else:
                logger.debug("  ℹ️  No date columns detected for normalization")
                
            return converted_columns
            
        except ImportError:
            # Fallback to standard date normalization if parallel utils not available
            logger.warning("  ⚠️  Parallel date utils not available, skipping date normalization")
            return []
        except Exception as e:
            logger.error(f"  ❌ Error in parallel date normalization: {e}")
            return []
    
    def _preserve_integer_types_parallel(self, df: pd.DataFrame) -> List[str]:
        """
        Preserve integer types in parallel processing
        Prevents 15 -> 15.0 conversion for better data integrity
        """
        def check_and_convert_column(col_info):
            """Check if a column should be converted from float to integer"""
            col_name, dtype = col_info
            
            if dtype != 'float64':
                return None
                
            try:
                # Check if all non-null values are whole numbers
                non_null_values = df[col_name].dropna()
                if len(non_null_values) == 0:
                    return None
                
                # Vectorized check for integer values
                is_integer_column = all(
                    float(val).is_integer() for val in non_null_values 
                    if pd.notna(val) and isinstance(val, (int, float))
                )
                
                if is_integer_column:
                    # Convert to Int64 (pandas nullable integer type)
                    df[col_name] = df[col_name].astype('Int64')
                    return col_name
                    
                return None
                
            except (ValueError, TypeError):
                return None
        
        logger.info(f"🔢 Preserving integer types in parallel across {len(df.columns)} columns...")
        
        # Get column info for parallel processing
        column_info = [(col, str(df[col].dtype)) for col in df.columns]
        
        # Process columns in parallel
        converted_columns = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(check_and_convert_column, column_info))
            converted_columns = [col for col in results if col is not None]
        
        if converted_columns:
            logger.info(f"  ✅ Preserved integer types in {len(converted_columns)} columns: {converted_columns[:5]}{'...' if len(converted_columns) > 5 else ''}")
        else:
            logger.debug("  ℹ️  No float columns needed integer type preservation")
            
        return converted_columns


def clean_dataframe_fast(df: pd.DataFrame, max_workers: int = None) -> Tuple[pd.DataFrame, Dict]:
    """
    High-performance data cleaning function optimized for large datasets
    Hardware-aware threading for Intel Xeon Platinum 8260 and similar high-end servers
    
    Args:
        df: Input DataFrame
        max_workers: Maximum number of worker threads (auto-detected for optimal performance)
        
    Returns:
        Tuple of (cleaned_df, cleanup_stats)
    """
    cleaner = ParallelDataCleaner(max_workers=max_workers)
    return cleaner.clean_dataframe_parallel(df)


# Backwards compatibility functions that use parallel processing
def remove_empty_rows_and_columns_fast(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Fast version of empty row/column removal using hardware-optimized parallel processing"""
    start_time = time.time()
    cleaner = ParallelDataCleaner()  # Uses hardware-aware threading
    
    # Step 1: Remove empty columns in parallel
    empty_columns = cleaner._detect_empty_columns_parallel(df)
    if empty_columns:
        df = df.drop(columns=empty_columns)
    
    # Step 2: Remove empty rows using vectorized operations
    empty_row_mask = cleaner._detect_empty_rows_vectorized(df)
    empty_row_count = empty_row_mask.sum()
    if empty_row_count > 0:
        df = df[~empty_row_mask].reset_index(drop=True)
    
    processing_time = time.time() - start_time
    
    cleanup_stats = {
        'original_rows': int(df.shape[0] + empty_row_count),
        'original_columns': int(df.shape[1] + len(empty_columns)),
        'removed_rows': int(empty_row_count),
        'removed_columns': int(len(empty_columns)),
        'empty_column_names': list(empty_columns),
        'final_rows': int(len(df)),
        'final_columns': int(len(df.columns)),
        'processing_time_seconds': round(processing_time, 2)
    }
    
    return df, cleanup_stats


def clean_column_names_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Fast version of column name cleaning with hardware-optimized threading"""
    cleaner = ParallelDataCleaner()  # Uses hardware-aware threading
    return cleaner._clean_column_names_parallel(df)


def clean_data_values_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Fast version of data value cleaning with hardware-optimized threading"""
    cleaner = ParallelDataCleaner()  # Uses hardware-aware threading
    cleaner._clean_data_values_parallel(df)
    return df