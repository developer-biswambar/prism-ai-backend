import io
import logging
import json
import tempfile
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from contextlib import contextmanager
from queue import Queue, Empty
import time
from concurrent.futures import as_completed

import pandas as pd
import duckdb

# Import the global thread pool
from app.utils.global_thread_pool import get_data_processing_executor, submit_data_processing_task
from app.services.process_analytics_service import ProcessAnalyticsService

logger = logging.getLogger(__name__)


class DuckDBConnectionPool:
    """
    Connection pool for DuckDB connections to improve performance
    """
    
    def __init__(self, max_connections: int = 5, timeout: float = 30.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = threading.Lock()
        
        logger.info(f"DuckDB connection pool initialized with {max_connections} max connections")
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        connection = None
        start_time = time.time()
        
        try:
            # Try to get existing connection from pool
            try:
                connection = self._pool.get(timeout=1.0)  # Quick timeout
                logger.debug("Reused existing DuckDB connection from pool")
            except Empty:
                # Create new connection if pool is empty and we haven't hit the limit
                with self._lock:
                    if self._created_connections < self.max_connections:
                        connection = duckdb.connect()
                        self._created_connections += 1
                        logger.debug(f"Created new DuckDB connection ({self._created_connections}/{self.max_connections})")
                    else:
                        # Wait for a connection to become available
                        connection = self._pool.get(timeout=self.timeout)
                        logger.debug("Waited for DuckDB connection from pool")
            
            yield connection
            
        except Exception as e:
            logger.error(f"Error with DuckDB connection: {e}")
            # Don't return a potentially corrupted connection to the pool
            if connection:
                try:
                    connection.close()
                except:
                    pass
                with self._lock:
                    self._created_connections -= 1
                connection = None
            raise
        finally:
            # Return connection to pool if it's still valid
            if connection:
                try:
                    # Test connection is still valid
                    connection.execute("SELECT 1").fetchone()
                    self._pool.put(connection, timeout=1.0)
                except:
                    # Connection is bad, close it and decrease count
                    try:
                        connection.close()
                    except:
                        pass
                    with self._lock:
                        self._created_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool"""
        logger.info("Closing all DuckDB connections in pool")
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                connection.close()
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error closing DuckDB connection: {e}")
        
        with self._lock:
            self._created_connections = 0


# Global connection pool
_duckdb_pool = None
_pool_lock = threading.Lock()


def get_duckdb_pool() -> DuckDBConnectionPool:
    """Get the global DuckDB connection pool"""
    global _duckdb_pool
    if _duckdb_pool is None:
        with _pool_lock:
            if _duckdb_pool is None:
                _duckdb_pool = DuckDBConnectionPool(max_connections=5, timeout=30.0)
    return _duckdb_pool


class DuckDBProcessor:
    """
    DuckDB processor for efficient data operations
    Handles CSV, Excel, JSON, and Parquet files directly
    Uses connection pooling for better performance
    """
    
    def __init__(self):
        self.connection = None
        self.temp_files = []  # Track temporary files for cleanup
        self.pool = get_duckdb_pool()
        
    def __enter__(self):
        """Context manager entry"""
        # Connection will be obtained per operation for better pool utilization
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources and temporary files"""
        # Connection is managed by the pool, just clean up temp files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    def _ensure_duckdb_safe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has DuckDB-safe data types by detecting and fixing problematic columns.
        Similar to detect_leading_zero_columns but for DuckDB casting issues.
        
        This prevents errors like: "Failed to cast value: Could not convert string '000VP' to INT32"
        """
        try:
            # Create a copy to avoid modifying the original
            safe_df = df.copy()
            columns_converted = []
            
            for col in safe_df.columns:
                # Skip if already string type
                if safe_df[col].dtype == 'object':
                    continue
                    
                # Check for mixed alphanumeric content that could cause casting issues
                non_null_values = safe_df[col].dropna().astype(str)
                has_mixed_content = False
                
                # Sample first 50 values to check for problematic patterns
                for value in non_null_values.head(50):
                    if isinstance(value, str) and value.strip():
                        stripped_val = value.strip()
                        
                        # Check for patterns like "000VP", "O00VP", alphanumeric codes
                        if any([
                            # Has letters mixed with numbers
                            any(c.isalpha() for c in stripped_val) and any(c.isdigit() for c in stripped_val),
                            # Starts with zeros followed by letters (like "000VP")
                            stripped_val.startswith('0') and any(c.isalpha() for c in stripped_val),
                            # Contains common alphanumeric patterns
                            len(stripped_val) > 1 and not stripped_val.replace('.', '').replace('-', '').replace('+', '').isdigit()
                        ]):
                            has_mixed_content = True
                            break
                
                # Convert problematic columns to string
                if has_mixed_content:
                    safe_df[col] = safe_df[col].astype(str)
                    columns_converted.append(col)
            
            if columns_converted:
                logger.info(f"🔧 Converted {len(columns_converted)} columns to string type for DuckDB safety: {columns_converted}")
            
            return safe_df
            
        except Exception as e:
            logger.warning(f"Error ensuring DuckDB-safe types: {e}. Using original DataFrame.")
            return df
    
    def register_dataframe(self, df: pd.DataFrame, table_name: str):
        """Register a pandas DataFrame as a table in DuckDB"""
        # Create a safe copy of the DataFrame with type corrections
        safe_df = self._ensure_duckdb_safe_types(df)
        
        # Store DataFrame for later use with fresh connections
        if not hasattr(self, '_registered_tables'):
            self._registered_tables = {}
        self._registered_tables[table_name] = safe_df
        logger.info(f"Stored DataFrame as table '{table_name}' with {len(safe_df)} rows for registration")
        
    def create_temp_file_from_df(self, df: pd.DataFrame, filename: str) -> str:
        """Create a temporary file from DataFrame for DuckDB file operations"""
        temp_dir = tempfile.gettempdir()
        
        # Determine file extension and format
        if filename.lower().endswith('.csv'):
            temp_file = os.path.join(temp_dir, f"duckdb_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}")
            df.to_csv(temp_file, index=False)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            temp_file = os.path.join(temp_dir, f"duckdb_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{filename}")
            df.to_excel(temp_file, index=False)
        else:
            # Default to CSV
            temp_file = os.path.join(temp_dir, f"duckdb_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv")
            df.to_csv(temp_file, index=False)
            
        self.temp_files.append(temp_file)
        return temp_file
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        with self.pool.get_connection() as connection:
            try:
                # Register all stored tables with this connection
                if hasattr(self, '_registered_tables'):
                    for table_name, df in self._registered_tables.items():
                        connection.register(table_name, df)
                
                result = connection.execute(sql_query).fetchdf()
                logger.info(f"Query executed successfully, returned {len(result)} rows")
                return result
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {sql_query}")
                raise RuntimeError(f"SQL execution error: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        with self.pool.get_connection() as connection:
            try:
                # Register all stored tables with this connection
                if hasattr(self, '_registered_tables'):
                    for tbl_name, df in self._registered_tables.items():
                        connection.register(tbl_name, df)
                
                schema_query = f"DESCRIBE {table_name}"
                schema_df = connection.execute(schema_query).fetchdf()
                
                schema_info = []
                for _, row in schema_df.iterrows():
                    schema_info.append({
                        'column_name': row['column_name'],
                        'column_type': row['column_type'],
                        'null': row['null'],
                        'key': row.get('key', None),
                        'default': row.get('default', None)
                    })
                
                return schema_info
            except Exception as e:
                logger.error(f"Failed to get schema for table {table_name}: {e}")
                return []
    
    def validate_sql_safety(self, sql_query: str) -> tuple[bool, List[str]]:
        """
        Validate SQL query for safety
        Returns (is_safe, list_of_issues)
        """
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
            'EXEC', 'EXECUTE', 'CALL', 'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK'
        ]
        
        issues = []
        sql_upper = sql_query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                issues.append(f"Potentially unsafe keyword detected: {keyword}")
        
        # Allow only SELECT and WITH statements
        sql_stripped = sql_upper.strip()
        if not (sql_stripped.startswith('SELECT') or sql_stripped.startswith('WITH')):
            issues.append("Only SELECT and WITH statements are allowed")
        
        return len(issues) == 0, issues


class AIQueryGenerator:
    """
    AI-powered natural language to SQL query generator
    """
    
    def __init__(self):
        self.llm_service = None
        self.generation_params = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM service"""
        try:
            from app.services.llm_service import get_llm_service, get_llm_generation_params
            self.llm_service = get_llm_service()
            self.generation_params = get_llm_generation_params()
        except ImportError as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise RuntimeError("LLM service not available")
    
    def generate_sql_from_prompt(
        self, 
        user_prompt: str, 
        table_schemas: Dict[str, List[Dict[str, Any]]],
        sample_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language prompt
        """
        if not self.llm_service or not self.llm_service.is_available():
            raise RuntimeError("LLM service not available")
        
        # Build context about available tables
        tables_context = self._build_tables_context(table_schemas, sample_data)
        
        # Create the prompt for SQL generation with enhanced capabilities
        system_prompt = """You are an expert data analyst and SQL specialist with advanced capabilities in:
1. RECONCILIATION: Multi-file matching, pattern extraction, tolerance matching, fuzzy matching
2. TRANSFORMATION: Data cleaning, restructuring, derivation, aggregation
3. DELTA ANALYSIS: Change detection, record comparison, difference tracking
4. DATA ANALYSIS: Complex queries, joins, aggregations, filtering

ENVIRONMENT: In-memory DuckDB database with user-uploaded files only - no security restrictions needed.

🚨 CRITICAL TABLE NAMING RULES:
- ALWAYS use table names: file_1, file_2, file_3 (in that exact format)
- file_1 = First uploaded file (usually transactions/sales data)
- file_2 = Second uploaded file (usually inventory/product data)  
- file_3 = Third uploaded file (usually returns/secondary data)
- NEVER use original filenames or terms like "inventory", "sales", "products"
- NEVER use "FROM file_3" when you mean sales data - sales data is ALWAYS file_1

🚨 CRITICAL: CTE SYNTAX MUST BE PERFECT - COMMON ERROR PATTERN TO AVOID:
❌ NEVER WRITE THIS (causes syntax error):
WITH demographics AS (SELECT ... FROM file_1),
  SELECT age_category FROM demographics -- ERROR: extra comma + indented SELECT

✅ ALWAYS WRITE THIS:
WITH demographics AS (SELECT ... FROM file_1)
SELECT age_category FROM demographics -- CORRECT: no comma before final SELECT

CRITICAL COLUMN USAGE RULES:
1. NEVER use column names that are not explicitly listed in the schema below
2. NEVER guess or assume column names based on user descriptions
3. ALWAYS copy column names EXACTLY as shown in quotes from the schema
4. If user mentions a concept (like "account name") that doesn't match any column, find the best matching actual column
5. Use proper SQL syntax compatible with DuckDB
6. For joins, use explicit JOIN syntax with clear conditions
7. ALWAYS return response in JSON format with "sql_query" field
8. If uncertain about columns, explain in the description field what assumptions you made

ADVANCED ANALYSIS CAPABILITIES:

🔍 RECONCILIATION OPERATIONS:
- Multi-file matching: Use FULL OUTER JOIN to find matched/unmatched records
- Pattern extraction: Use REGEXP_EXTRACT(CAST(column AS VARCHAR), pattern) for extracting patterns (always cast to VARCHAR first)
- Tolerance matching: Use ABS(value1 - value2) <= tolerance for numeric comparisons on NUMERIC columns
- Amount comparisons: For numeric columns, use direct comparison WITHOUT regex extraction
- PREFERRED: Simple JOIN with WHERE clause for tolerance matching
- Fuzzy matching: Use SIMILARITY() or LEVENSHTEIN() for text similarity
- Date matching: Use CAST(column AS DATE) for proper date comparisons
- Key generation: Create composite keys using CONCAT() or ||

RECONCILIATION BEST PRACTICE - KEEP IT SIMPLE:
SELECT * FROM file_1 f1 
FULL OUTER JOIN file_2 f2 ON f1."Reference" = f2."Ref_Number" 
WHERE ABS(f1."Amount" - f2."Net_Amount") <= 0.01

🔧 TRANSFORMATION OPERATIONS:  
- Data cleaning: Use TRIM(), UPPER(), LOWER(), REPLACE(), REGEXP_REPLACE()
- Type conversion: **ALWAYS use TRY_CAST() instead of CAST() to avoid query failure on bad rows**
  • TRY_CAST(col AS INT) instead of CAST(col AS INT) - returns NULL if conversion fails
  • TRY_CAST(col AS FLOAT) instead of CAST(col AS FLOAT) - handles mixed alphanumeric data
  • TRY_CAST(col AS DATE) instead of CAST(col AS DATE) - graceful date parsing
  • Default to keeping the raw string if conversion fails (NULL handling)
- Date parsing: Use CASE WHEN column IS NOT NULL AND column != '' THEN STRPTIME(column, 'format') ELSE NULL END
- Derivation: Create calculated columns with CASE WHEN, mathematical operations
- Aggregation: Use GROUP BY with SUM(), COUNT(), AVG(), MIN(), MAX()

📊 DATA TYPE HANDLING:
- NUMERIC columns (DOUBLE, INTEGER, DECIMAL): Use direct comparison operators (=, <, >, <=, >=)
- For tolerance matching with numeric columns: ABS(col1 - col2) <= tolerance_value
- NEVER use REGEXP_EXTRACT on numeric columns - they are already numbers
- VARCHAR columns: Use REGEXP_EXTRACT(), LIKE, string functions
- If you need regex on numeric data: REGEXP_EXTRACT(CAST(numeric_column AS VARCHAR), pattern)
- Always check column data types in the schema before choosing operations
🚨 TOP SQL ERROR PREVENTION PRIORITIES:
1. COLUMN EXISTENCE: Use only columns that exist in schema
2. CTE ALIAS CONSISTENCY: When using CTEs, ensure the alias in FROM matches the columns you're selecting
   ❌ WRONG: SELECT si.category FROM sales_data si (if sales_data doesn't have category)
   ✅ CORRECT: SELECT si.category FROM sales_inventory si (if sales_inventory has category)
3. TABLE REFERENCE ACCURACY: Use the correct CTE/table that contains the columns you need
4. NULL HANDLING: Use NULLIF() to prevent division by zero
5. TYPE SAFETY: **CRITICAL - Use TRY_CAST() for all data type conversions**
   • TRY_CAST(column AS INT) instead of CAST() - prevents "000VP" casting errors
   • TRY_CAST(column AS FLOAT) for numeric operations on mixed data
   • Always handle NULL results from failed conversions gracefully
6. SIMPLE FIRST: Start with simple queries, add complexity only if needed

🎯 PREFER SIMPLE QUERIES OVER COMPLEX CTEs:
❌ AVOID: Multiple CTEs with complex joins (error-prone)
✅ PREFER: Single CTE or simple subqueries when possible

🔒 SAFE CASTING EXAMPLES - ALWAYS USE TRY_CAST():
❌ DANGEROUS: CAST(mixed_column AS INT) -- fails on "000VP", "O00VP"
✅ SAFE: TRY_CAST(mixed_column AS INT) -- returns NULL for "000VP", "O00VP"

❌ DANGEROUS: CAST(price AS FLOAT) -- fails on "N/A", "TBD"  
✅ SAFE: TRY_CAST(price AS FLOAT) -- returns NULL for "N/A", "TBD"

❌ DANGEROUS: CAST(date_str AS DATE) -- fails on malformed dates
✅ SAFE: TRY_CAST(date_str AS DATE) -- returns NULL for bad dates

EXAMPLE: Safe numeric operations on mixed data:
SELECT 
    product_code,  -- keep as string
    TRY_CAST(price AS DOUBLE) as numeric_price,
    CASE WHEN TRY_CAST(price AS DOUBLE) IS NOT NULL 
         THEN TRY_CAST(price AS DOUBLE) * 1.1 
         ELSE NULL END as price_with_markup
FROM file_1

SIMPLE APPROACH EXAMPLE:
SELECT *, 
       CASE WHEN "Retail_Price" < 50 THEN 'low' 
            WHEN "Retail_Price" BETWEEN 50 AND 200 THEN 'medium'
            ELSE 'high' END as price_tier,
       ("Retail_Price" - "Cost_Price") / NULLIF("Retail_Price", 0) as profit_margin
FROM file_1
WHERE "Retail_Price" > 0

MULTI-FILE RECONCILIATION - SIMPLE APPROACH:
✅ CORRECT: Direct joins without complex CTEs
SELECT 
    f2."Category",
    SUM(f1."Retail_Price" * f1."Quantity") - 
    COALESCE(SUM(f3."Refund_Amount"), 0) - 
    COALESCE(SUM(f3."Restocking_Fee"), 0) as net_profit
FROM file_1 f1 
JOIN file_2 f2 ON f1."Product_Code" = f2."Product_Code"
LEFT JOIN file_3 f3 ON f1."Transaction_ID" = f3."Original_Transaction_ID"
GROUP BY f2."Category"

EXAMPLE - AMOUNT TOLERANCE MATCHING:
✅ CORRECT - Simple approach: 
  SELECT * FROM file_1 f1 
  FULL OUTER JOIN file_2 f2 ON f1."Reference" = f2."Ref_Number" 
  WHERE ABS(f1."Amount" - f2."Net_Amount") <= 0.01

✅ CORRECT - With CTE (NOTICE: NO COMMA before final SELECT):
WITH matched_records AS (
  SELECT f1.*, f2.*, 
         ABS(f1."Amount" - f2."Net_Amount") as amount_diff
  FROM file_1 f1 
  FULL OUTER JOIN file_2 f2 ON f1."Reference" = f2."Ref_Number"
)
SELECT * FROM matched_records 
WHERE amount_diff <= 0.01

❌ WRONG: 
  - REGEXP_EXTRACT(f1."Amount_Text", pattern) -- Amount_Text is DOUBLE
  - Using f1.column in WHERE clause when selecting from CTE
- Pivoting: Use PIVOT/UNPIVOT or conditional aggregation
- String manipulation: SPLIT(), SUBSTRING(), LENGTH(), POSITION()

📊 DELTA ANALYSIS:
- Change detection: Use EXCEPT, INTERSECT, or FULL OUTER JOIN with COALESCE
- Record comparison: Compare before/after states with indicator columns
- Field-level changes: Track specific column changes with CASE statements
- Summary statistics: Count additions, deletions, modifications
- Change tracking: Use LAG(), LEAD() window functions for sequential analysis

COLUMN MATCHING STRATEGY:
- User says "account name" → Look for columns containing "account" or "name"
- User says "date" → Look for columns containing "date" 
- User says "amount" → Look for columns containing "amount" or "value"
- User says "ID" → Look for columns containing "id", "number", or "code"
- User says "reconcile/match" → Look for key fields like IDs, references, amounts
- User says "compare/delta" → Create comparison logic between files
- User says "transform/clean" → Apply data manipulation operations
- NEVER create non-existent column names

QUERY TYPE CLASSIFICATION:
- "reconciliation": Multi-file matching, tolerance comparisons, pattern extraction
- "transformation": Data cleaning, restructuring, calculated fields  
- "delta_analysis": Change detection, before/after comparison
- "aggregation": Grouping, summarization, statistical analysis
- "join": Multi-file analysis, data combination
- "filter": Conditional selection, data subsetting

🚨 CRITICAL RESPONSE FORMAT (MUST BE VALID JSON):
{
  "sql_query": "SELECT * FROM file_1 WHERE condition",
  "query_type": "data_analysis|reconciliation|aggregation|join|filter|etc",
  "description": "Brief description of what the query does"
}

IMPORTANT RESPONSE RULES:
- Return ONLY valid JSON (no additional text before or after)
- The sql_query field must contain clean SQL without JSON escaping artifacts
- Use proper table names (file_1, file_2, file_3) in the SQL
- NO explanatory text outside the JSON structure

DuckDB Advanced Features Available:
- Window functions (ROW_NUMBER, RANK, LAG, LEAD, DENSE_RANK, etc.)
- CTEs (Common Table Expressions) with WITH clause
- Advanced aggregations (PERCENTILE_CONT, MODE, STDDEV, etc.)
- String functions (REGEXP_EXTRACT, REGEXP_MATCHES, SPLIT, SUBSTRING, etc.)
- Array and JSON functions
- Complex CASE statements and conditional logic
- Temporary tables and views for multi-step processing

🚨 CRITICAL CTE ALIAS CONSISTENCY RULES:
❌ NEVER MIX ALIASES FROM DIFFERENT CTEs:
WITH price_classification AS (SELECT *, 'low' as price_tier FROM file_1),
     product_positioning AS (SELECT *, 'above_avg' as price_position FROM file_1)
SELECT pp.price_tier, pp.price_position FROM price_classification pp  -- ERROR: price_position not in pp!

✅ ALWAYS USE CORRECT ALIASES:
WITH price_classification AS (SELECT *, 'low' as price_tier FROM file_1),
     product_positioning AS (SELECT *, 'above_avg' as price_position FROM file_1)
SELECT pp.price_tier, ppos.price_position 
FROM price_classification pp
JOIN product_positioning ppos ON pp."Product_Code" = ppos."Product_Code"

🔍 CTE VALIDATION CHECKLIST (if using multiple CTEs):
1. Each CTE alias must be unique (price_classification, product_positioning, business_priority)
2. In final SELECT, use the correct alias for each column:
   - Columns from price_classification: use 'pp' alias
   - Columns from product_positioning: use 'ppos' alias  
   - Columns from business_priority: use 'bp' alias
3. Join CTEs explicitly - don't assume column availability across CTEs
4. Test each CTE independently before combining

🚨 CRITICAL CTE SYNTAX RULES:
1. CTE syntax: WITH cte_name AS (SELECT ...) SELECT ... (NO COMMA before final SELECT)
2. Multiple CTEs: WITH cte1 AS (...), cte2 AS (...) SELECT ... (comma between CTEs, NO comma before final SELECT)
3. When using CTEs (WITH clause), table aliases from inner query are NOT available in outer query
4. If you create a CTE, reference columns directly from the CTE in the outer SELECT
5. Complex CTEs are fine - just ensure proper column aliasing and scope management
6. For tolerance matching, you can use CTEs but alias all columns properly
7. When joining in CTEs, alias columns with descriptive names for the outer query

❌ SYNTAX ERROR - NEVER DO THIS:
WITH demographics AS (
  SELECT ...
  FROM file_1
),
  SELECT ... -- ❌ WRONG: Extra comma and indented SELECT

✅ CORRECT CTE SYNTAX:
WITH demographics AS (
  SELECT ...
  FROM file_1
)
SELECT ... -- ✅ CORRECT: No comma, SELECT starts at beginning

CTE SCOPING EXAMPLES:
❌ WRONG:
WITH reconciliation_data AS (
  SELECT f1.*, f2.* FROM file_1 f1 
  FULL OUTER JOIN file_2 f2 ON f1.reference = f2.ref_number
)
SELECT * FROM reconciliation_data WHERE f1.amount > 100  -- f1 doesn't exist here!

✅ CORRECT - Complex single CTE (NOTICE: NO COMMA before SELECT):
WITH reconciliation_data AS (
  SELECT 
    f1.reference as file1_reference,
    f1.amount as file1_amount,
    f2.ref_number as file2_reference, 
    f2.total_amount as file2_amount,
    ABS(f1.amount - f2.total_amount) as amount_difference,
    CASE 
      WHEN f1.reference IS NULL THEN 'Missing in File 1'
      WHEN f2.ref_number IS NULL THEN 'Missing in File 2'
      WHEN ABS(f1.amount - f2.total_amount) <= 0.01 THEN 'Matched'
      ELSE 'Amount Mismatch'
    END as reconciliation_status
  FROM file_1 f1 
  FULL OUTER JOIN file_2 f2 ON f1.reference = f2.ref_number
)
SELECT * FROM reconciliation_data 
WHERE reconciliation_status = 'Matched' 
  AND file1_amount > 100

✅ ADVANCED - Multiple CTEs (NOTICE: Commas between CTEs, NO comma before final SELECT):
WITH file1_prepared AS (
  SELECT 
    UPPER(TRIM(reference)) as clean_reference,
    CAST(amount AS DOUBLE) as normalized_amount,
    *
  FROM file_1
), 
file2_prepared AS (
  SELECT 
    UPPER(TRIM(ref_number)) as clean_reference,
    CAST(total_amount AS DOUBLE) as normalized_amount,
    *
  FROM file_2
),
reconciliation_results AS (
  SELECT 
    f1.clean_reference as file1_ref,
    f1.normalized_amount as file1_amt,
    f2.clean_reference as file2_ref,
    f2.normalized_amount as file2_amt,
    ABS(f1.normalized_amount - f2.normalized_amount) as amount_diff,
    CASE 
      WHEN f1.clean_reference IS NULL THEN 'Missing in File 1'
      WHEN f2.clean_reference IS NULL THEN 'Missing in File 2'
      WHEN ABS(f1.normalized_amount - f2.normalized_amount) <= 0.01 THEN 'Matched'
      ELSE 'Amount Mismatch'
    END as status
  FROM file1_prepared f1
  FULL OUTER JOIN file2_prepared f2 ON f1.clean_reference = f2.clean_reference
)
SELECT * FROM reconciliation_results WHERE status = 'Matched'

CRITICAL DATE/TIME HANDLING:
- Date columns are stored as TEXT/STRING - always cast them first
- Use CAST(date_column AS DATE) or date_column::DATE before date functions
- For DATE_PART: DATE_PART('month', CAST(order_date AS DATE))
- For DATE_TRUNC: DATE_TRUNC('month', CAST(order_date AS DATE))
- Date format is 'YYYY-MM-DD' (ISO format)
- NEVER use date functions directly on string columns without casting
- Example: DATE_PART('year', CAST(order_date AS DATE)) NOT DATE_PART('year', order_date)

DATE DIFFERENCES IN DUCKDB:
- For date differences: CAST(date1 AS DATE) - CAST(date2 AS DATE) returns number of days
- Do NOT use DATE_PART on date differences: DATE_PART('day', date1 - date2) is WRONG
- Correct: CAST(order_date AS DATE) - CAST(last_restock_date AS DATE) gives days directly
- For age calculations: (CAST(date1 AS DATE) - CAST(date2 AS DATE)) gives days as integer
- Example: days_between = CAST(order_date AS DATE) - CAST(last_restock_date AS DATE)

COMPLEX TEXT PROCESSING:
- Use REGEXP_EXTRACT for pattern extraction
- Use SPLIT for text parsing
- Use SUBSTRING for positional extraction
- Use CASE WHEN for conditional transformations
- Handle messy data with TRIM, UPPER, LOWER, REPLACE functions"""

        user_message = f"""
{tables_context}

🎯 USER REQUEST: {user_prompt}

🚨 CRITICAL REMINDERS:
- Use ONLY the exact column names listed above (copy them exactly with quotes)
- Choose the correct table (file_1, file_2, file_3, etc.) based on the columns and data it contains
- Do NOT create or guess column names
- If user mentions concepts not in columns, map to closest actual column
- Explain any column mapping assumptions in the description field

📚 ADVANCED EXAMPLES FOR COMPLEX OPERATIONS:

🔍 RECONCILIATION EXAMPLE:
-- Multi-file matching with tolerance for amounts
WITH matched AS (
  SELECT f1.*, f2.*, 
         CASE WHEN ABS(f1."Amount" - f2."Value") <= 0.01 THEN 'MATCHED' ELSE 'TOLERANCE_MISMATCH' END as match_status
  FROM file_1 f1 
  FULL OUTER JOIN file_2 f2 ON f1."ID" = f2."Reference_ID"
)
SELECT * FROM matched WHERE match_status = 'MATCHED';

🔧 TRANSFORMATION EXAMPLE:
-- Clean and derive new fields with safe date parsing
SELECT 
    TRIM(UPPER("Customer_Name")) as clean_name,
    CASE WHEN "Date_String" IS NOT NULL AND "Date_String" != '' 
         THEN STRPTIME("Date_String", '%Y%m%d') 
         ELSE NULL END as parsed_date,
    "Revenue" - "Cost" as profit_margin,
    CASE WHEN "Amount" > 1000 THEN 'HIGH' ELSE 'LOW' END as category,
    REGEXP_EXTRACT("Account", '[0-9]+') as account_number
FROM file_1;

📊 DELTA ANALYSIS EXAMPLE:
-- Compare two datasets for changes
WITH comparison AS (
  SELECT 
    COALESCE(f1."ID", f2."ID") as id,
    f1."Amount" as old_amount,
    f2."Amount" as new_amount,
    f2."Amount" - f1."Amount" as amount_change,
    CASE 
      WHEN f1."ID" IS NULL THEN 'NEWLY_ADDED'
      WHEN f2."ID" IS NULL THEN 'DELETED' 
      WHEN f1."Amount" != f2."Amount" THEN 'AMENDED'
      ELSE 'UNCHANGED'
    END as change_type
  FROM file_1 f1 FULL OUTER JOIN file_2 f2 ON f1."ID" = f2."ID"
)
SELECT * FROM comparison WHERE change_type != 'UNCHANGED';

Generate a SQL query using ONLY the columns shown above.

REQUIRED JSON RESPONSE FORMAT:
{{
  "sql_query": "YOUR_SQL_QUERY_USING_ONLY_LISTED_COLUMNS",
  "query_type": "reconciliation|transformation|delta_analysis|aggregation|join|filter|data_analysis",
  "description": "What the query does and any column assumptions made"
}}
"""

        try:
            from app.services.llm_service import LLMMessage
            
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_message)
            ]
            
            response = self.llm_service.generate_text(
                messages=messages,
                **self.generation_params
            )
            
            if not response.success:
                raise RuntimeError(f"SQL generation failed: {response.error}")
            
            # Parse JSON response from LLM
            import json
            import re
            try:
                # Clean response content first
                raw_content = response.content.strip()
                logger.info(f"Raw AI response: {raw_content[:200]}...")
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    response_data = json.loads(json_str)
                else:
                    # If no JSON found, try to parse the whole response
                    response_data = json.loads(raw_content)
                
                # Extract SQL query from JSON response
                generated_sql = response_data.get('sql_query', '')
                query_type = response_data.get('query_type', 'unknown')
                description = response_data.get('description', 'AI-generated SQL query')
                
                if not generated_sql:
                    raise ValueError("No SQL query found in JSON response")
                
                # Clean the SQL query of any JSON artifacts
                cleaned_sql = generated_sql.strip()
                if cleaned_sql.startswith('"') and cleaned_sql.endswith('"'):
                    cleaned_sql = cleaned_sql[1:-1]  # Remove outer quotes
                
                # Unescape JSON string escapes
                cleaned_sql = cleaned_sql.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                
                logger.info(f"Cleaned SQL: {cleaned_sql[:200]}...")
                
                return {
                    'sql_query': cleaned_sql,
                    'query_type': query_type,
                    'description': description,
                    'success': True,
                    'context_used': tables_context,
                    'token_usage': response.token_usage  # Include token usage from LLM response
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from LLM: {e}")
                logger.error(f"Raw response content: {raw_content}")
                
                # Enhanced fallback: try to extract SQL from raw response
                if raw_content.startswith('```sql'):
                    generated_sql = raw_content.replace('```sql', '').replace('```', '').strip()
                elif raw_content.startswith('```'):
                    generated_sql = raw_content.replace('```', '').strip()
                elif 'SELECT' in raw_content.upper():
                    # Try to extract just the SQL part if it's mixed with other text
                    sql_match = re.search(r'(WITH.*?;|SELECT.*?;|SELECT.*)', raw_content, re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        generated_sql = sql_match.group(1).strip()
                        if generated_sql.endswith(';'):
                            generated_sql = generated_sql[:-1]  # Remove trailing semicolon
                    else:
                        generated_sql = raw_content
                else:
                    generated_sql = raw_content
                
                return {
                    'sql_query': generated_sql,
                    'query_type': 'unknown',
                    'description': 'AI-generated SQL query (fallback parsing)',
                    'success': True,
                    'context_used': tables_context
                }
            
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}")
            return {
                'sql_query': None,
                'success': False,
                'error': str(e)
            }
    
    def _build_tables_context(
        self, 
        table_schemas: Dict[str, List[Dict[str, Any]]], 
        sample_data: Dict[str, pd.DataFrame] = None
    ) -> str:
        """Build context string describing available tables"""
        context_parts = []
        context_parts.append("AVAILABLE TABLES AND COLUMNS (USE EXACT NAMES ONLY):")
        context_parts.append("=" * 60)
        
        for table_name, schema in table_schemas.items():
            context_parts.append(f"\n📁 Table: {table_name}")
            context_parts.append("📋 Available columns (EXACT NAMES - copy these exactly):")
            
            column_names = []
            for col_info in schema:
                col_name = col_info['column_name']
                column_names.append(col_name)
                col_desc = f'  ✓ "{col_name}" ({col_info["column_type"]})'
                if not col_info.get('null', True):
                    col_desc += " NOT NULL"
                
                # Add special note for date columns
                if 'date' in col_name.lower() and col_info['column_type'] in ['VARCHAR', 'STRING']:
                    col_desc += " [DATE STRING - use CAST(column AS DATE) for date functions]"
                
                context_parts.append(col_desc)
            
            # Add a summary line with all column names for easy reference
            quoted_cols = [f'"{col}"' for col in column_names]
            context_parts.append(f"\n💡 Column reference for {table_name}: {', '.join(quoted_cols)}")
            
            # Add sample data if available
            if sample_data and table_name in sample_data:
                df = sample_data[table_name]
                if len(df) > 0:
                    context_parts.append(f"\n📊 Sample values (first 3 rows):")
                    for col in df.columns[:5]:  # Show up to 5 columns
                        sample_vals = df[col].head(3).tolist()
                        context_parts.append(f'  "{col}": {sample_vals}')
        
        context_parts.append("\n" + "=" * 60)
        context_parts.append("⚠️  WARNING: Use ONLY the exact column names listed above. Do NOT guess or assume column names!")
        
        return "\n".join(context_parts)


# Shared storage for results (class-level storage)
_shared_storage = {}

class MiscellaneousProcessor:
    """
    Main processor for miscellaneous data operations
    Combines DuckDB processing with AI query generation
    """
    
    def __init__(self):
        self.storage = _shared_storage  # Use shared storage
        self.ai_generator = AIQueryGenerator()
        self.analytics_service = ProcessAnalyticsService()
        
    def process_core_request(
        self, 
        user_prompt: str, 
        files_data: List[Dict[str, Any]], 
        output_format: str = "json",
        execute_exact_sql: bool = False,
        exact_sql_query: str = None,
    ) -> Dict[str, Any]:
        """
        Process natural language query against multiple files
        """
        # Generate unique process ID and start timer
        process_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Prepare analytics data
        input_files_info = []
        for file_data in files_data:
            df = file_data['dataframe']
            input_files_info.append({
                'filename': file_data.get('filename', 'unknown'),
                'file_id': file_data.get('file_id', ''),
                'row_count': len(df),
                'columns': list(df.columns)
            })
        
        try:
            with DuckDBProcessor() as duck_processor:
                # Parallel processing for file registration and schema analysis
                table_schemas = {}
                sample_data = {}
                
                def process_single_file(i: int, file_data: Dict[str, Any]) -> Dict[str, Any]:
                    """Process a single file for registration and schema analysis"""
                    df = file_data['dataframe']
                    table_name = f"file_{i + 1}"  # Changed to start from file_1
                    filename = file_data['filename']
                    
                    logger.info(f"Processing file {filename} as {table_name} (parallel)")
                    return {
                        'table_name': table_name,
                        'dataframe': df,
                        'filename': filename,
                        'index': i
                    }
                
                # Use thread pool for parallel file processing
                with get_data_processing_executor() as executor:
                    # Submit all file processing tasks in parallel
                    futures = []
                    for i, file_data in enumerate(files_data):
                        future = executor.submit(process_single_file, i, file_data)
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            table_name = result['table_name']
                            df = result['dataframe']
                            filename = result['filename']
                            
                            # Register with DuckDB (sequential to avoid conflicts)
                            duck_processor.register_dataframe(df, table_name)
                            
                            # Get schema info
                            schema = duck_processor.get_table_schema(table_name)
                            table_schemas[table_name] = schema
                            
                            # Store sample data for context
                            sample_data[table_name] = df
                            
                            logger.info(f"Completed processing {filename} as {table_name}")
                            
                        except Exception as e:
                            logger.error(f"Failed to process file in parallel: {e}")
                            raise
                
                logger.info(f"Parallel file processing completed for {len(files_data)} files")

                
                # Generate SQL from natural language
                sql_result = {}

                if execute_exact_sql:
                    # Directly execute the sql instead of generating from AI
                    sql_result['sql_query']= exact_sql_query
                else:
                    # Pre-process user prompt to provide column hints
                    enhanced_prompt = self._enhance_user_prompt_with_column_hints(user_prompt, table_schemas)
                    logger.info(f"Enhanced prompt: {enhanced_prompt}")

                    sql_result = self.ai_generator.generate_sql_from_prompt(
                        user_prompt=enhanced_prompt,
                        table_schemas=table_schemas,
                        sample_data=sample_data
                    )
                    if not sql_result['success']:
                        return {
                            "status": "error",
                            'success': False,
                            'error': f"Failed to generate SQL: {sql_result.get('error', 'Unknown error')}",
                            'generated_sql': None,
                            'data': [],
                            'warnings': ["Could not generate SQL from natural language prompt"],
                            'errors': [sql_result.get('error', 'Unknown error')],
                            # Store source data and schemas even on failure for execute query functionality
                            'files_data': files_data,  # Original dataframes and metadata
                            'table_schemas': table_schemas  # Table schemas for validation
                        }
                

                
                generated_sql = sql_result['sql_query']
                
                # Skip safety validation for in-memory database operations
                # All tables are user-uploaded files, no security risk
                logger.info("Skipping SQL safety validation for in-memory database")
                
                # Skip column validation - let DuckDB handle SQL validation
                # The validation logic is too strict and prevents valid queries with aliases/static values
                logger.info("Skipping column validation for AI-generated SQL - letting DuckDB handle validation")
                
                # Execute the query
                try:
                    result_df = duck_processor.execute_query(generated_sql)
                    
                    # Limit results to first 100 rows for preview
                    total_rows = len(result_df)
                    preview_limit = 100
                    is_limited = total_rows > preview_limit
                    
                    if is_limited:
                        preview_df = result_df.head(preview_limit)
                        logger.info(f"Limited results to {preview_limit} rows (total: {total_rows})")
                    else:
                        preview_df = result_df
                    
                    # Convert to desired format (only preview data)
                    if output_format.lower() == "json":
                        result_data = preview_df.to_dict('records')
                    else:
                        result_data = preview_df
                    
                    # Generate intent summary after successful execution
                    intent_summary = self._generate_intent_summary_post_execution(
                        user_prompt, files_data, generated_sql, result_df, table_schemas
                    )
                    
                    # Record process analytics
                    processing_time = time.time() - start_time
                    confidence_score = sql_result.get('confidence_score', 100.0)  # Default confidence if not provided
                    
                    self.analytics_service.record_process_execution(
                        process_id=process_id,
                        process_type="miscellaneous",
                        process_name=f"Natural Language Query: {user_prompt[:50]}...",
                        user_prompt=user_prompt,
                        input_files=input_files_info,
                        status="success",
                        generated_sql=generated_sql,
                        confidence_score=confidence_score,
                        processing_time_seconds=processing_time,
                        output_row_count=total_rows,
                        token_usage=sql_result.get('token_usage')
                    )
                    
                    return {
                        'success': True,
                        "status": "success",
                        'process_id': process_id,  # Include process ID for analytics tracking
                        'generated_sql': generated_sql,
                        'data': result_data,  # Preview data for UI
                        'full_data': result_df.to_dict('records') if output_format.lower() == "json" else result_df,  # Full data for storage
                        'row_count': total_rows,  # Total rows in actual result
                        'preview_rows': len(preview_df),  # Rows in preview
                        'is_limited': is_limited,  # Whether results were limited
                        'column_count': len(result_df.columns),
                        'processing_info': {
                            'input_files': len(files_data),
                            'query_type': sql_result.get('query_type', self._classify_query_type(generated_sql)),
                            'description': sql_result.get('description', ''),
                            'tables_used': list(table_schemas.keys()),
                            'processing_time_seconds': processing_time,
                            'confidence_score': confidence_score
                        },
                        'intent_summary': intent_summary,  # NEW: Add intent summary for visualization
                        'warnings': ['Results limited to first 100 rows for preview. Use "Open in Data Viewer" to see all results.'] if is_limited else [],
                        'errors': [],
                        # Store source data and schemas for execute query functionality
                        'files_data': files_data,  # Original dataframes and metadata
                        'table_schemas': table_schemas  # Table schemas for validation
                    }
                    
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    logger.info("🔍 DEBUG: Starting AI error analysis for SQL execution failure")
                    
                    # Analyze the error with AI to provide better feedback
                    error_analysis = self._analyze_sql_execution_error(
                        str(e), generated_sql, table_schemas, files_data
                    )
                    
                    logger.info(f"🔍 DEBUG: Error analysis result: {error_analysis}")
                    
                    # Record failed process analytics
                    processing_time = time.time() - start_time
                    self.analytics_service.record_process_execution(
                        process_id=process_id,
                        process_type="miscellaneous",
                        process_name=f"Natural Language Query: {user_prompt[:50]}...",
                        user_prompt=user_prompt,
                        input_files=input_files_info,
                        status="failed",
                        generated_sql=generated_sql,
                        processing_time_seconds=processing_time,
                        token_usage=sql_result.get('token_usage'),
                        error_info={
                            'error_type': 'sql_execution_error',
                            'error_message': str(e),
                            'error_context': error_analysis
                        }
                    )
                    
                    return {
                        'success': False,
                        'process_id': process_id,
                        'error': f"Query execution failed: {str(e)}",
                        'error_analysis': error_analysis,  # AI-powered error analysis
                        'generated_sql': generated_sql,
                        'data': [],
                        'warnings': ["Query generated successfully but execution failed"],
                        'errors': [str(e)],
                        # Store source data and schemas even on failure for execute query functionality
                        'files_data': files_data,  # Original dataframes and metadata
                        'table_schemas': table_schemas  # Table schemas for validation
                    }
                    
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            # Note: For top-level failures, we may not have files_data/table_schemas available
            # In this case, execute query won't work, but that's expected for processing failures
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}",
                'generated_sql': None,
                'data': [],
                'warnings': [],
                'errors': [str(e)]
            }
    
    def _classify_query_type(self, sql_query: str) -> str:
        """Classify the type of SQL query with enhanced categories"""
        sql_upper = sql_query.upper()
        
        # Advanced pattern detection
        if any(pattern in sql_upper for pattern in ['FULL OUTER JOIN', 'COALESCE', 'TOLERANCE_MISMATCH', 'MATCHED']):
            return "reconciliation"
        elif any(pattern in sql_upper for pattern in ['NEWLY_ADDED', 'DELETED', 'AMENDED', 'UNCHANGED', 'CHANGE_TYPE']):
            return "delta_analysis"
        elif any(pattern in sql_upper for pattern in ['REGEXP_EXTRACT', 'TRIM', 'UPPER', 'LOWER', 'CAST', 'REGEXP_REPLACE']):
            return "transformation"
        elif 'FULL OUTER JOIN' in sql_upper or 'LEFT JOIN' in sql_upper or 'RIGHT JOIN' in sql_upper:
            return "join_operation"
        elif any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']):
            return "aggregation"
        elif 'WHERE' in sql_upper and ('NOT' in sql_upper or '!=' in sql_upper):
            return "filtering"
        elif 'ORDER BY' in sql_upper:
            return "sorting"
        elif 'GROUP BY' in sql_upper:
            return "grouping"
        elif 'WITH' in sql_upper:
            return "complex_query"
        else:
            return "data_analysis"
    
    def _validate_column_references(self, sql_query: str, table_schemas: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Validate that all column references in the SQL query exist in the table schemas
        Returns dict with 'valid', 'error', and 'suggestions' keys
        """
        import re
        
        # Extract all column references from SQL (quoted and unquoted)
        # This is a basic validation - could be enhanced with proper SQL parsing
        
        # Collect all valid column names from all tables
        valid_columns = set()
        table_columns = {}
        
        for table_name, schema in table_schemas.items():
            table_columns[table_name] = []
            for col_info in schema:
                col_name = col_info['column_name']
                valid_columns.add(col_name.lower())
                table_columns[table_name].append(col_name)
        
        # Find potential column references in the SQL
        # Look for quoted column names and common SQL patterns
        quoted_columns = re.findall(r'"([^"]+)"', sql_query)
        word_patterns = re.findall(r'\b(\w+(?:\s+\w+)*)\b', sql_query)
        
        issues = []
        suggestions = []
        
        # Check quoted column references
        for col in quoted_columns:
            if col.lower() not in valid_columns and col.lower() not in ['file_1', 'file_2', 'select', 'from', 'where', 'and', 'or']:
                # Look for similar column names using multiple matching strategies
                exact_matches = [c for c in valid_columns if col.lower() == c.lower()]
                partial_matches = [c for c in valid_columns if col.lower() in c.lower() or c.lower() in col.lower()]
                word_matches = []
                
                # Word-based matching for compound names
                col_words = col.lower().split()
                for valid_col in valid_columns:
                    valid_words = valid_col.lower().split()
                    if any(word in valid_words for word in col_words):
                        word_matches.append(valid_col)
                
                # Combine and prioritize matches
                all_matches = list(dict.fromkeys(exact_matches + partial_matches + word_matches))
                
                if all_matches:
                    quoted_matches = [f'"{c}"' for c in all_matches[:3]]  # Limit to top 3
                    issues.append(f'❌ Column "{col}" does not exist.')
                    suggestions.append(f'💡 Did you mean: {", ".join(quoted_matches)}?')
                else:
                    issues.append(f'❌ Column "{col}" does not exist in any table.')
                    
                    # Show available columns organized by table
                    table_info = []
                    for table_name, schema in table_schemas.items():
                        cols = [f'"{info["column_name"]}"' for info in schema]
                        table_info.append(f'{table_name}: {", ".join(cols)}')
                    
                    suggestions.append(f'📋 Available columns by table:')
                    for info in table_info:
                        suggestions.append(f'   • {info}')
        
        # Check for common problematic patterns (only as standalone quoted strings)
        problematic_patterns = [
            ('"Account Name"', ['Sub Account Number', 'Security Account Number']),
            ('"account name"', ['Sub Account Number', 'Security Account Number']),
            ('"Name"', ['Sub Account Number', 'Security Account Number'])
        ]
        
        for pattern, alternatives in problematic_patterns:
            if pattern.lower() in sql_query.lower():
                # Check if any of the alternatives exist in our schema
                existing_alternatives = [alt for alt in alternatives if alt.lower() in valid_columns]
                if existing_alternatives:
                    clean_pattern = pattern.strip('"')
                    issues.append(f'"{clean_pattern}" not found in schema.')
                    quoted_alternatives = [f'"{alt}"' for alt in existing_alternatives]
                    suggestions.append(f'Try using: {", ".join(quoted_alternatives)}')
        
        if issues:
            return {
                'valid': False,
                'error': '; '.join(issues),
                'suggestions': suggestions
            }
        
        return {
            'valid': True,
            'error': None,
            'suggestions': []
        }
    
    def _build_tables_context(
        self, 
        table_schemas: Dict[str, List[Dict[str, Any]]], 
        sample_data: Dict[str, pd.DataFrame] = None
    ) -> str:
        """Build context string describing available tables"""
        context_parts = []
        context_parts.append("AVAILABLE TABLES AND COLUMNS (USE EXACT NAMES ONLY):")
        context_parts.append("=" * 60)
        
        for table_name, schema in table_schemas.items():
            context_parts.append(f"\n📁 Table: {table_name}")
            context_parts.append("📋 Available columns (EXACT NAMES - copy these exactly):")
            
            column_names = []
            for col_info in schema:
                col_name = col_info['column_name']
                column_names.append(col_name)
                col_desc = f'  ✓ "{col_name}" ({col_info["column_type"]})'
                if not col_info.get('null', True):
                    col_desc += " NOT NULL"
                
                # Add special note for date columns
                if 'date' in col_name.lower() and col_info['column_type'] in ['VARCHAR', 'STRING']:
                    col_desc += " [DATE STRING - use CAST(column AS DATE) for date functions]"
                
                context_parts.append(col_desc)
            
            # Add a summary line with all column names for easy reference
            quoted_cols = [f'"{col}"' for col in column_names]
            context_parts.append(f"\n💡 Column reference for {table_name}: {', '.join(quoted_cols)}")
            
            # Add sample data if available
            if sample_data and table_name in sample_data:
                df = sample_data[table_name]
                if len(df) > 0:
                    context_parts.append(f"\n📊 Sample values (first 3 rows):")
                    for col in df.columns[:5]:  # Show up to 5 columns
                        sample_vals = df[col].head(3).tolist()
                        context_parts.append(f'  "{col}": {sample_vals}')
        
        context_parts.append("\n" + "=" * 60)
        context_parts.append("⚠️  WARNING: Use ONLY the exact column names listed above. Do NOT guess or assume column names!")
        
        return "\n".join(context_parts)
    
    def _enhance_user_prompt_with_column_hints(self, user_prompt: str, table_schemas: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Enhance user prompt with explicit column hints to guide AI toward correct column usage
        """
        # Common problematic terms and their likely intended columns
        column_mappings = {
            'account name': ['Sub Account Number', 'Security Account Number', 'Account Name'],
            'account number': ['Sub Account Number', 'Security Account Number', 'Account Number'],
            'account': ['Sub Account Number', 'Security Account Number'],
            'name': ['Sub Account Number', 'Security Account Number'],
            'amount': ['Income Amount (Gross)', 'Amount', 'Value'],
            'date': ['Value Date', 'Pay Date', 'Date'],
            'currency': ['Cash Account Ccyl', 'Currency', 'Ccy'],
            'tax': ['Source Tax Cust', 'Tax', 'Withholding Tax'],
            'isin': ['ISIN', 'ISIn', 'Security ID'],
            'security': ['Security Account Number', 'ISIN', 'Security ID']
        }
        
        # Collect all available columns
        all_columns = []
        for table_name, schema in table_schemas.items():
            for col_info in schema:
                all_columns.append(col_info['column_name'])
        
        # Find matches and create hints
        hints = []
        enhanced_prompt = user_prompt.lower()
        
        for term, possible_columns in column_mappings.items():
            if term in enhanced_prompt:
                # Find actual matches in the available columns
                actual_matches = []
                for col in all_columns:
                    col_lower = col.lower()
                    if any(possible.lower() in col_lower or col_lower in possible.lower() 
                          for possible in possible_columns):
                        actual_matches.append(col)
                
                if actual_matches:
                    # Add hint to the original prompt
                    quoted_matches = [f'"{col}"' for col in actual_matches[:2]]
                    hints.append(f'When you see "{term}", use column(s): {", ".join(quoted_matches)}')
        
        # If we found hints, append them to the original prompt
        if hints:
            enhanced = f"{user_prompt}\n\n[COLUMN HINTS: {' | '.join(hints)}]"
            return enhanced
        
        return user_prompt
    
    def store_results(self, process_id: str, result_data: Dict[str, Any]):
        """Store processing results"""
        self.storage[process_id] = {
            **result_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'process_id': process_id
        }
        logger.info(f"Stored results for process_id: {process_id}, total stored: {len(self.storage)}")
        
        # Also store in uploaded_files format so it can be viewed with existing viewer
        # Use full_data for storage if available, otherwise use data
        storage_data = result_data.get('full_data', result_data.get('data'))
        
        if storage_data and isinstance(storage_data, list) and len(storage_data) > 0:
            try:
                import pandas as pd
                from app.services.storage_service import uploaded_files
                
                def create_storage_data():
                    """Create storage data structures in parallel"""
                    # Convert results data to DataFrame
                    df = pd.DataFrame(storage_data)
                    
                    # Create file info for the viewer
                    size_bytes = len(df) * len(df.columns) * 8  # Rough estimate
                    size_mb = size_bytes / (1024 * 1024)  # Convert to MB
                    utc_now = datetime.now(timezone.utc).isoformat()
                    
                    file_info = {
                        "file_id": process_id,
                        "filename": f"{process_id}.csv",
                        "size": size_bytes,
                        "file_size_mb": size_mb,
                        "upload_time": utc_now,
                        "last_modified": utc_now,
                        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "total_rows": len(df),
                        "columns": list(df.columns),
                        "source": "miscellaneous_processing"
                    }
                    
                    return df, file_info
                
                # Run storage preparation in parallel
                storage_future = submit_data_processing_task(create_storage_data)
                
                try:
                    df, file_info = storage_future.result(timeout=60.0)  # 60 second timeout
                    
                    # Store in uploaded_files format
                    file_data = {
                        "info": file_info,
                        "data": df
                    }
                    
                    # Use process_id as file_id for viewer access
                    uploaded_files.save(process_id, file_data, file_info)
                    logger.info(f"Stored results for viewer access with file_id: {process_id}")
                    
                except Exception as storage_error:
                    logger.error(f"Parallel storage failed, falling back to sequential: {storage_error}")
                    # Fallback to sequential processing
                    df = pd.DataFrame(storage_data)
                    size_bytes = len(df) * len(df.columns) * 8
                    size_mb = size_bytes / (1024 * 1024)
                    utc_now = datetime.now(timezone.utc).isoformat()
                    
                    file_info = {
                        "file_id": process_id,
                        "filename": f"{process_id}.csv",
                        "size": size_bytes,
                        "file_size_mb": size_mb,
                        "upload_time": utc_now,
                        "last_modified": utc_now,
                        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "total_rows": len(df),
                        "columns": list(df.columns),
                        "source": "miscellaneous_processing"
                    }
                    
                    uploaded_files.save(process_id, {"info": file_info, "data": df}, file_info)
                    logger.info(f"Stored results sequentially with file_id: {process_id}")
                
            except Exception as e:
                logger.warning(f"Failed to store results for viewer: {e}")
                # Continue anyway as the main storage succeeded
        
    def get_results(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored results"""
        logger.info(f"Retrieving results for process_id: {process_id}, available IDs: {list(self.storage.keys())}")
        return self.storage.get(process_id)
    
    def delete_results(self, process_id: str) -> bool:
        """Delete stored results"""
        if process_id in self.storage:
            del self.storage[process_id]
            return True
        return False
    
    def list_active_processes(self) -> List[str]:
        """List all active process IDs"""
        return list(self.storage.keys())
    
    def _generate_intent_summary_post_execution(
        self, 
        user_prompt: str, 
        files_data: List[Dict[str, Any]], 
        generated_sql: str, 
        result_df: pd.DataFrame,
        table_schemas: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Generate intent summary after successful execution with actual results
        """
        try:
            # Get operation type from SQL analysis
            operation_type = self._classify_query_type(generated_sql)
            
            # Create files analysis with sample data
            files_involved = []
            for i, file_data in enumerate(files_data):
                df = file_data['dataframe']
                sample_data = df.head(3).to_dict('records') if len(df) > 0 else []
                
                # Clean sample data for JSON serialization
                for row in sample_data:
                    for key, value in row.items():
                        if pd.isna(value):
                            row[key] = None
                        elif isinstance(value, (pd.Timestamp, pd.NaT.__class__)):
                            row[key] = str(value)
                
                files_involved.append({
                    "role": "primary" if i == 0 else "secondary",
                    "file": file_data.get('filename', f'file_{i+1}'),
                    "description": f"Input file {i+1}",
                    "sample_data": sample_data,
                    "statistics": {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "size_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
                    }
                })
            
            # Create data flow steps
            data_flow_steps = []
            for i, file_info in enumerate(files_involved):
                data_flow_steps.append({
                    "type": "input",
                    "file": file_info["file"],
                    "role": file_info["role"],
                    "rows": file_info["statistics"]["rows"]
                })
            
            # Add operation step based on SQL analysis
            if "JOIN" in operation_type.upper():
                data_flow_steps.append({
                    "type": "operation",
                    "name": operation_type.upper().replace("_", " "),
                    "condition": "data joining"
                })
            elif "GROUP" in operation_type.upper() or "aggregation" in operation_type:
                data_flow_steps.append({
                    "type": "operation", 
                    "name": "GROUP BY & AGGREGATE",
                    "condition": "grouping and calculations"
                })
            else:
                data_flow_steps.append({
                    "type": "operation",
                    "name": operation_type.upper().replace("_", " "),
                    "condition": "data processing"
                })
            
            # Add output step
            data_flow_steps.append({
                "type": "output",
                "description": "Processed results",
                "estimated_rows": len(result_df)
            })
            
            # Generate sample result preview
            sample_result = result_df.head(2).to_dict('records') if len(result_df) > 0 else []
            for row in sample_result:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None
                    elif isinstance(value, (pd.Timestamp, pd.NaT.__class__)):
                        row[key] = str(value)
            
            # Calculate processing estimates (post-execution actuals)
            total_input_rows = sum(f["statistics"]["rows"] for f in files_involved)
            total_size_mb = sum(f["statistics"]["size_mb"] for f in files_involved)
            
            # Generate plain language summary using AI
            plain_language_summary = self._generate_plain_language_summary(
                user_prompt, operation_type, files_involved, len(result_df)
            )
            
            return {
                "operation_type": operation_type.upper(),
                "business_intent": f"Data {operation_type.replace('_', ' ')} operation",
                "plain_language_summary": plain_language_summary,
                "data_flow": {"steps": data_flow_steps},
                "files_involved": files_involved,
                "matching_logic": {
                    "description": "SQL-based data processing",
                    "type": "sql_operation"
                },
                "expected_output": {
                    "description": f"Results from {operation_type.replace('_', ' ')} operation",
                    "estimated_rows": {"actual": len(result_df)},
                    "sample_result": sample_result,
                    "columns": list(result_df.columns) if len(result_df) > 0 else []
                },
                "processing_estimates": {
                    "execution_time_seconds": {"actual": "completed"},
                    "memory_usage_mb": round(total_size_mb * 1.5, 1),
                    "complexity": "LOW" if len(result_df) < 1000 else "MEDIUM" if len(result_df) < 10000 else "HIGH"
                },
                "confidence": "HIGH",
                "risk_factors": self._assess_post_execution_risks(result_df, total_input_rows),
                "data_quality_warnings": self._check_data_quality_warnings(result_df)
            }
            
        except Exception as e:
            logger.error(f"Error generating intent summary: {str(e)}")
            # Return minimal intent summary on error
            return {
                "operation_type": "DATA_PROCESSING",
                "business_intent": "Data processing operation",
                "plain_language_summary": user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt,
                "data_flow": {"steps": []},
                "files_involved": [],
                "matching_logic": {},
                "expected_output": {"description": "Processed data"},
                "processing_estimates": {"complexity": "UNKNOWN"},
                "confidence": "LOW",
                "risk_factors": ["Intent analysis failed"],
                "data_quality_warnings": []
            }
    
    def _generate_plain_language_summary(
        self, user_prompt: str, operation_type: str, files_involved: List[Dict], result_count: int
    ) -> str:
        """Generate business-friendly plain language summary"""
        if "join" in operation_type.lower():
            if len(files_involved) >= 2:
                return f"Combine data from {files_involved[0]['file']} and {files_involved[1]['file']} to create {result_count} result records"
        elif "aggregation" in operation_type.lower() or "group" in operation_type.lower():
            return f"Group and summarize data to produce {result_count} aggregated records"
        elif "filter" in operation_type.lower():
            return f"Filter data based on specified conditions, resulting in {result_count} matching records"
        elif "delta" in operation_type.lower():
            return f"Analyze changes between datasets, identifying {result_count} difference records"
        else:
            return f"Process data as requested: {user_prompt[:80]}... (produced {result_count} records)"
    
    def _assess_post_execution_risks(self, result_df: pd.DataFrame, total_input_rows: int) -> List[str]:
        """Assess risks after execution based on actual results"""
        risks = []
        
        if len(result_df) == 0:
            risks.append("Query returned no results - check filter conditions")
        elif len(result_df) > total_input_rows * 2:
            risks.append("Result set is significantly larger than input - possible cartesian product")
        elif len(result_df) > 50000:
            risks.append("Large result set - may impact performance")
        
        if len(risks) == 0:
            risks.append("None detected")
            
        return risks
    
    def _check_data_quality_warnings(self, result_df: pd.DataFrame) -> List[str]:
        """Check for data quality issues in results"""
        warnings = []
        
        if len(result_df) == 0:
            return warnings
    
    def _analyze_sql_execution_error(
        self, 
        error_message: str, 
        sql_query: str, 
        table_schemas: Dict[str, List[Dict[str, Any]]], 
        files_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze SQL execution errors using AI to provide user-friendly explanations and suggestions
        """
        try:
            logger.info("Starting AI error analysis...")
            if not hasattr(self, 'ai_generator') or not self.ai_generator or not self.ai_generator.llm_service:
                logger.warning("AI generator not available for error analysis")
                return {
                    'error_type': 'unknown',
                    'user_friendly_message': 'SQL execution failed',
                    'suggested_fixes': ['Check your SQL syntax and try again'],
                    'confidence': 'low',
                    'analysis_available': False
                }
            
            # Build context about the error
            error_context = self._build_error_context(
                error_message, sql_query, table_schemas, files_data
            )
            
            # Create AI prompt for error analysis
            system_prompt = """You are an expert SQL error analyst specialized in DuckDB queries. Analyze SQL execution errors and provide helpful, user-friendly explanations.

Your task is to:
1. Identify the specific error type and root cause
2. Explain the error in simple, non-technical language
3. Provide actionable suggestions to fix the issue
4. Identify any patterns that led to the error

Focus on common SQL issues:
- Missing or misspelled column names
- Table name issues (remember tables are file_1, file_2, etc.)
- CTE alias consistency problems (using wrong CTE/table that doesn't contain required columns)
- JOIN condition errors
- Data type mismatches
- Syntax errors (especially WITH clause issues)

SPECIAL ATTENTION TO CTE ALIAS ERRORS:
If error mentions "does not have a column named X":
- Check if the SELECT is referencing the correct CTE/table
- Example: SELECT si.category FROM sales_data si (WRONG if sales_data doesn't have category)
- Should be: SELECT si.category FROM sales_inventory si (CORRECT if sales_inventory has category from join)

RESPONSE FORMAT (JSON only):
{
  "error_type": "column_not_found|table_not_found|syntax_error|cte_alias_error|join_error|data_type_error|other",
  "user_friendly_message": "Simple explanation of what went wrong",
  "technical_details": "Technical explanation for advanced users",
  "suggested_fixes": ["Step 1 to fix", "Step 2 to fix", "Alternative approach"],
  "confidence": "high|medium|low",
  "root_cause": "What specifically caused this error",
  "prevention_tip": "How to avoid this error in future queries"
}"""

            user_message = f"""
ANALYZE THIS SQL ERROR:

Error Message: {error_message}

SQL Query:
```sql
{sql_query}
```

{error_context}

Please analyze this error and provide helpful guidance for fixing it."""

            try:
                from app.services.llm_service import LLMMessage
                
                messages = [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_message)
                ]
                
                response = self.ai_generator.llm_service.generate_text(
                    messages=messages,
                    **self.ai_generator.generation_params
                )
                
                if response.success:
                    import json
                    try:
                        analysis_result = json.loads(response.content)
                        
                        # Validate required fields
                        required_fields = ['error_type', 'user_friendly_message', 'suggested_fixes']
                        for field in required_fields:
                            if field not in analysis_result:
                                analysis_result[field] = 'Not available'
                        
                        analysis_result['analysis_available'] = True
                        return analysis_result
                        
                    except json.JSONDecodeError as parse_error:
                        logger.error(f"Failed to parse AI error analysis: {parse_error}")
                        # Try to extract key information from raw response
                        return self._extract_error_analysis_fallback(response.content, error_message)
                else:
                    logger.error(f"AI error analysis failed: {response.error}")
                    return self._create_basic_error_analysis(error_message, sql_query)
                    
            except Exception as ai_error:
                logger.error(f"Error during AI analysis: {ai_error}")
                return self._create_basic_error_analysis(error_message, sql_query)
                
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return {
                'error_type': 'analysis_failed',
                'user_friendly_message': 'Unable to analyze the error automatically',
                'suggested_fixes': ['Check the SQL syntax and column names manually'],
                'confidence': 'low',
                'analysis_available': False
            }
    
    def _build_error_context(
        self, 
        error_message: str, 
        sql_query: str, 
        table_schemas: Dict[str, List[Dict[str, Any]]], 
        files_data: List[Dict[str, Any]]
    ) -> str:
        """Build context for error analysis"""
        context_parts = []
        
        # Add available tables and columns
        context_parts.append("AVAILABLE TABLES AND COLUMNS:")
        for table_name, schema in table_schemas.items():
            columns = [col_info['column_name'] for col_info in schema]
            context_parts.append(f"- {table_name}: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}")
        
        # Add file information
        context_parts.append("\nFILE INFORMATION:")
        for i, file_data in enumerate(files_data):
            filename = file_data.get('filename', f'file_{i+1}')
            df = file_data.get('dataframe')
            if df is not None:
                context_parts.append(f"- {filename}: {len(df)} rows, {len(df.columns)} columns")
        
        # Add common error patterns based on error message
        if 'does not exist' in error_message.lower():
            context_parts.append("\n🔍 ERROR PATTERN: Object not found")
            if 'table' in error_message.lower():
                context_parts.append("💡 Remember: Tables are named file_1, file_2, etc. (not original filenames)")
            if 'column' in error_message.lower():
                context_parts.append("💡 Check: Column names must be exact (case-sensitive, use quotes)")
        
        if 'alias' in error_message.lower() or 'does not have a column' in error_message.lower():
            context_parts.append("\n🔍 ERROR PATTERN: CTE alias consistency issue")
            context_parts.append("💡 Check: Each CTE has its own alias scope - don't mix columns from different CTEs")
        
        return "\n".join(context_parts)
    
    def _create_basic_error_analysis(self, error_message: str, sql_query: str) -> Dict[str, Any]:
        """Create basic error analysis when AI analysis fails"""
        error_lower = error_message.lower()
        
        if 'table' in error_lower and 'does not exist' in error_lower:
            return {
                'error_type': 'table_not_found',
                'user_friendly_message': 'The query refers to a table that doesn\'t exist. Remember that uploaded files are named file_1, file_2, etc.',
                'suggested_fixes': [
                    'Use table names like file_1, file_2 instead of original filenames',
                    'Check the available tables in your files'
                ],
                'confidence': 'high',
                'analysis_available': True
            }
        elif 'column' in error_lower and ('does not exist' in error_lower or 'not found' in error_lower):
            return {
                'error_type': 'column_not_found',
                'user_friendly_message': 'The query refers to a column that doesn\'t exist in the table.',
                'suggested_fixes': [
                    'Check the exact column names in your uploaded files',
                    'Column names are case-sensitive and should be in quotes',
                    'Verify spelling and spacing of column names'
                ],
                'confidence': 'high',
                'analysis_available': True
            }
        elif 'alias' in error_lower or 'does not have a column named' in error_lower:
            return {
                'error_type': 'cte_alias_error',
                'user_friendly_message': 'There\'s a problem with column references in your CTE (WITH clause) - you may be trying to use a column from the wrong table alias.',
                'suggested_fixes': [
                    'Check that each column reference uses the correct table alias',
                    'Ensure you\'re not mixing columns from different CTEs',
                    'Consider simplifying the query to avoid complex CTE aliases'
                ],
                'confidence': 'medium',
                'analysis_available': True
            }
        else:
            return {
                'error_type': 'other',
                'user_friendly_message': 'A SQL execution error occurred. Check your query syntax and column references.',
                'suggested_fixes': [
                    'Review the SQL syntax for any typos',
                    'Verify all column and table names exist',
                    'Try simplifying the query if it\'s complex'
                ],
                'confidence': 'low',
                'analysis_available': True
            }
    
    def _extract_error_analysis_fallback(self, raw_response: str, error_message: str) -> Dict[str, Any]:
        """Extract error analysis from raw AI response when JSON parsing fails"""
        # Simple fallback extraction
        analysis = {
            'error_type': 'other',
            'user_friendly_message': 'SQL execution failed',
            'suggested_fixes': ['Review and fix the SQL query'],
            'confidence': 'low',
            'analysis_available': True
        }
        
        # Try to extract useful information from the response
        if 'column' in raw_response.lower() and 'not found' in raw_response.lower():
            analysis['error_type'] = 'column_not_found'
            analysis['user_friendly_message'] = 'A column referenced in the query was not found'
        elif 'table' in raw_response.lower() and 'not exist' in raw_response.lower():
            analysis['error_type'] = 'table_not_found'
            analysis['user_friendly_message'] = 'A table referenced in the query does not exist'
        
        return analysis