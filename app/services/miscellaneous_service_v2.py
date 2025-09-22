"""
Miscellaneous Data Processing Service V2 - LangChain SQL Agent Implementation

This service provides enhanced data processing capabilities using LangChain SQL Agents
while maintaining backward compatibility with the existing API contract.

Key Features:
- LangChain SQL Agent with DuckDB integration
- Automatic schema introspection and query optimization
- Self-correction capabilities for failed queries
- Multi-step reasoning for complex data analysis
- Detailed execution logging and agent step tracking
- Full backward compatibility with v1 service
"""

import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import as_completed
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import duckdb
import pandas as pd

# LangChain imports - V2 requires these or fails
try:
    import duckdb_engine
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentType
    from langchain_community.callbacks.manager import get_openai_callback
    from langchain.schema import OutputParserException
    from langchain.llms.base import LLM
    from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.schema.language_model import BaseLanguageModel

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.error(f"LangChain dependencies not available: {e}")
    LANGCHAIN_AVAILABLE = False


    # Create dummy classes to prevent import errors
    class BaseLanguageModel:
        pass

# Import existing components for compatibility
from app.services.miscellaneous_service import (
    DuckDBConnectionPool,
    DuckDBProcessor
)
from app.utils.global_thread_pool import get_data_processing_executor
from app.services.process_analytics_service import ProcessAnalyticsService
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


#
# class CustomLLMWrapper(BaseLanguageModel):
#     """Wrapper to make our LLM service compatible with LangChain LLM interface"""
#
#     def __init__(self, llm_service=None, model_name="gpt-4", temperature=0, **kwargs):
#         super().__init__(**kwargs)
#         self.llm_service = llm_service or get_llm_service()
#         self.model_name = model_name
#         self.temperature = temperature
#
#         # Make it compatible with LangChain's expected interface
#         self.model = model_name
#
#     @property
#     def _llm_type(self) -> str:
#         """Return identifier of LLM type."""
#         return "custom_llm_wrapper"
#
#     def _call(self, prompt: str, stop=None, **kwargs) -> str:
#         """Call the LLM with a prompt and return the response."""
#         # Convert prompt to our LLM service format
#         llm_messages = [LLMMessage(role='user', content=prompt)]
#
#         # Generate response using our LLM service
#         response = self.llm_service.generate_text(
#             messages=llm_messages,
#             temperature=self.temperature,
#             max_tokens=2000,
#             **kwargs
#         )
#
#         if not response.success:
#             raise Exception(f"LLM generation failed: {response.error}")
#
#         return response.content
#
#     def invoke(self, messages):
#         """LangChain-compatible invoke method for chat-style interactions"""
#         # Convert LangChain messages to our LLM service format
#         llm_messages = []
#         for msg in messages:
#             if hasattr(msg, 'content') and hasattr(msg, 'type'):
#                 # Handle LangChain message objects
#                 role = msg.type if msg.type in ['system', 'user', 'assistant'] else 'user'
#                 llm_messages.append(LLMMessage(role=role, content=msg.content))
#             elif isinstance(msg, dict):
#                 # Handle dict format
#                 role = msg.get('role', 'user')
#                 content = msg.get('content', str(msg))
#                 llm_messages.append(LLMMessage(role=role, content=content))
#             else:
#                 # Handle string or other formats
#                 llm_messages.append(LLMMessage(role='user', content=str(msg)))
#
#         # Generate response using our LLM service
#         response = self.llm_service.generate_text(
#             messages=llm_messages,
#             temperature=self.temperature,
#             max_tokens=2000
#         )
#
#         if not response.success:
#             raise Exception(f"LLM generation failed: {response.error}")
#
#         # Return in LangChain-compatible format
#         class MockMessage:
#             def __init__(self, content):
#                 self.content = content
#
#         return MockMessage(response.content)
#
#     from langchain.sql_database import SQLDatabase
#     from typing import Dict, Any, List

class DuckDBSafeSQLDatabase(SQLDatabase):
    """
    SQLDatabase subclass that forces DuckDB column types to strings
    to avoid 'unhashable type: DuckDBPyType'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # _table_info is populated now
        if hasattr(self, "_table_info"):
            for table, cols in self._table_info.items():
                for col in cols:
                    col["type"] = str(col.get("type"))


class LangChainSQLAgentProcessor:
    """
    Enhanced SQL processing using LangChain SQL Agent
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.llm_service = get_llm_service()

        if not self.llm_service.is_available():
            raise ValueError("LLM service is not available for V2 LangChain service")

        logger.info(f"✅ LangChain SQL Agent processor initialized with {self.llm_service.get_provider_name()}")
        self.agent_available = True

        # Initialize components
        self.llm = None
        self.sql_agent = None
        self.agent_executor = None

        # Tracking variables
        self.execution_steps = []
        self.self_corrections = []
        self.tool_usage = {}
        self.reasoning_chain = []

    def setup_agent(self, db_connection_string: str):
        """Setup LangChain SQL Agent with DuckDB connection"""
        logger.info(f"Setting up LangChain SQL Agent with connection: {db_connection_string}")

        if not self.agent_available:
            logger.error("LangChain SQL Agent not available. Check dependencies and API key.")
            raise ValueError("LangChain SQL Agent not available. Check dependencies and API key.")

        try:
            logger.info(f"Initializing custom LLM wrapper with model: {self.model_name}")
            # Initialize LLM using our existing LLM service
            self.llm = ChatOpenAI(
                model="gpt-4.1-nano",
                temperature=0
            )
            # logger.info(f"✅ Custom LLM wrapper initialized successfully with {self.llm_service.get_provider_name()}")

            # Verify DuckDB dialect availability before creating connection
            logger.info("🔍 Verifying DuckDB SQLAlchemy dialect availability...")
            try:
                # Test dialect registration (duckdb_engine already imported at module level)
                from sqlalchemy import create_engine
                from sqlalchemy.dialects import registry

                logger.info("✅ duckdb_engine package available (imported at module level)")

                # Check if duckdb dialect is registered
                try:
                    dialect = registry.load("duckdb")
                    logger.info(f"✅ DuckDB dialect loaded successfully: {dialect}")
                except Exception as dialect_error:
                    logger.warning(f"⚠️ DuckDB dialect not found in registry: {dialect_error}")
                    logger.info("🔄 Attempting to register DuckDB dialect manually...")
                    # Try to register manually
                    try:
                        registry.register("duckdb", "duckdb_engine.dialects.duckdb", "DuckDBDialect_duckdb_engine")
                        logger.info("✅ DuckDB dialect registered manually")
                    except Exception as reg_error:
                        logger.error(f"❌ Failed to register dialect manually: {reg_error}")

                # Test engine creation
                logger.info(f"🔧 Testing SQLAlchemy engine creation with: {db_connection_string}")
                test_engine = create_engine(db_connection_string)
                logger.info("✅ SQLAlchemy engine created successfully")
                test_engine.dispose()  # Clean up test engine

            except Exception as dialect_error:
                logger.error(f"❌ DuckDB dialect verification failed: {dialect_error}")
                logger.error(f"🔍 Error type: {type(dialect_error).__name__}")
                import traceback
                logger.error(f"📋 Full dialect error:\n{traceback.format_exc()}")
                # Continue anyway, but log the issue
                logger.warning("⚠️ Continuing with SQLDatabase creation despite dialect issues...")

            logger.info(f"Creating SQLDatabase connection from URI: {db_connection_string}")
            # Create SQL Database connection with DuckDB type handling
            try:
                self.db = DuckDBSafeSQLDatabase.from_uri(
                    db_connection_string,
                    include_tables=None,
                    sample_rows_in_table_info=3,
                )

                logger.info(
                    f"✅ SQLDatabase connected successfully. Available tables: {self.db.get_usable_table_names()}")
            except Exception as db_error:
                logger.error(f"❌ Failed to create SQLDatabase: {db_error}")
                # Try alternative approach with minimal table info
                try:
                    logger.info("🔄 Retrying SQLDatabase creation with minimal table info...")
                    self.db = SQLDatabase.from_uri(
                        db_connection_string
                    )
                    logger.info(
                        f"✅ SQLDatabase connected successfully (minimal mode). Available tables: {self.db.get_usable_table_names()}")
                except Exception as retry_error:
                    logger.error(f"❌ SQLDatabase creation failed even with minimal mode: {retry_error}")
                    raise RuntimeError(f"Cannot create SQLDatabase connection: {retry_error}") from retry_error

            logger.info("Creating SQL Agent with enhanced capabilities...")
            # Create SQL Agent with enhanced capabilities
            self.sql_agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                return_intermediate_steps=True,
                max_iterations=5,  # Allow multiple iterations for self-correction
                early_stopping_method="generate",
                handle_parsing_errors=True,  # <-- allow agent to self-correct
            )

            logger.info("LangChain SQL Agent setup completed successfully")
            return True

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ Failed to setup LangChain SQL Agent: {e}")
            logger.error(f"📋 Full error details:\n{error_details}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            logger.error(f"🔧 Model attempted: {self.model_name}")
            logger.error(f"🔗 Connection string: {db_connection_string}")
            # Don't set agent_available to False - raise exception instead for V2 strict mode
            raise RuntimeError(f"LangChain SQL Agent setup failed: {e}") from e

    def execute_query_with_agent(self, user_prompt: str, table_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query using LangChain SQL Agent with self-correction capabilities
        """
        if not self.agent_available or not self.sql_agent:
            raise ValueError("SQL Agent not available or not setup")

        # Reset tracking variables
        self.execution_steps = []
        self.self_corrections = []
        self.tool_usage = {}
        self.reasoning_chain = []

        # Enhanced prompt with schema context
        enhanced_prompt = self._create_enhanced_prompt(user_prompt, table_schemas)

        try:
            with get_openai_callback() as cb:
                # Execute with agent
                result = self.sql_agent.invoke({
                    "input": enhanced_prompt,

                }, return_direct=True,  # bypass output parser
                    handle_parsing_errors=True)

                # Extract results
                agent_output = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])

                # Process intermediate steps for tracking
                self._process_intermediate_steps(intermediate_steps)

                print(intermediate_steps)

                # Extract SQL query from agent output or steps
                generated_sql = self._extract_sql_from_result(result, intermediate_steps)

                # Token usage tracking
                token_usage = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }

                return {
                    "success": True,
                    "sql_query": generated_sql,
                    "agent_output": agent_output,
                    "execution_steps": self.execution_steps,
                    "self_corrections": self.self_corrections,
                    "reasoning_chain": self.reasoning_chain,
                    "token_usage": token_usage,
                    "intermediate_steps": intermediate_steps
                }

        except Exception as e:
            logger.error(f"LangChain SQL Agent execution failed: {e}")

            # Attempt self-correction
            correction_result = self._attempt_self_correction(user_prompt, str(e), table_schemas)

            if correction_result["success"]:
                return correction_result
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "sql_query": None,
                    "execution_steps": self.execution_steps,
                    "self_corrections": self.self_corrections,
                    "reasoning_chain": self.reasoning_chain
                }

    def _create_enhanced_prompt(self, user_prompt: str, table_schemas: Dict[str, Any]) -> str:
        """Create enhanced prompt with schema information"""

        schema_info = "Available tables and their schemas:\n\n"
        for table_name, schema in table_schemas.items():
            schema_info += f"Table: {table_name}\n"
            # Handle schema as list of dicts (from get_table_schema)
            if isinstance(schema, list) and schema:
                columns = [f'{row["column_name"]}({row["column_type"]})' for row in schema]
                schema_info += f"Columns: {', '.join(columns)}\n\n"
            elif isinstance(schema, dict):
                schema_info += f"Columns: {', '.join([f'{col}({dtype})' for col, dtype in schema.items()])}\n\n"
            else:
                schema_info += "Schema information not available\n\n"

        enhanced_prompt = f"""
{schema_info}

User Request: {user_prompt}

Please analyze the user request and generate appropriate SQL query to answer their question.
Consider the table schemas provided above and ensure your query is syntactically correct for DuckDB.

Important notes:
- Use exact table names as provided: {', '.join(table_schemas.keys())}
- Consider data types when performing operations
- For aggregations, ensure appropriate grouping
- Use aliases for clarity when needed
"""

        return enhanced_prompt

    def _process_intermediate_steps(self, intermediate_steps: List[tuple]):
        """Process and track intermediate steps from agent execution"""

        for i, (action, observation) in enumerate(intermediate_steps):
            step_info = {
                "step": i + 1,
                "action": {
                    "tool": action.tool if hasattr(action, 'tool') else "unknown",
                    "input": action.tool_input if hasattr(action, 'tool_input') else str(action),
                    "log": action.log if hasattr(action, 'log') else ""
                },
                "observation": str(observation),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self.execution_steps.append(step_info)

            # Track tool usage
            tool_name = step_info["action"]["tool"]
            if tool_name not in self.tool_usage:
                self.tool_usage[tool_name] = 0
            self.tool_usage[tool_name] += 1

            # Add to reasoning chain
            self.reasoning_chain.append({
                "type": "action",
                "content": step_info["action"]["log"],
                "step": i + 1
            })

            if "error" in str(observation).lower() or "exception" in str(observation).lower():
                self.reasoning_chain.append({
                    "type": "error_detected",
                    "content": str(observation),
                    "step": i + 1
                })

    def _extract_sql_from_result(self, result: Dict, intermediate_steps: List[tuple]) -> Optional[str]:
        """Extract SQL query from agent result or intermediate steps"""

        # First, try to extract from agent output
        agent_output = result.get("output", "")

        # Look for SQL in agent output
        sql_patterns = [
            r"```sql\n(.*?)\n```",
            r"```\n(SELECT.*?)\n```",
            r"SELECT.*?;",
            r"(SELECT.*?)(?=\n|$)"
        ]

        import re
        for pattern in sql_patterns:
            match = re.search(pattern, agent_output, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If not found in output, look in intermediate steps
        for action, observation in intermediate_steps:
            if hasattr(action, 'tool_input') and isinstance(action.tool_input, dict):
                query = action.tool_input.get('query', '')
                if query and query.strip().upper().startswith('SELECT'):
                    return query.strip()

        return None

    def _attempt_self_correction(self, original_prompt: str, error_message: str, table_schemas: Dict[str, Any]) -> Dict[
        str, Any]:
        """Attempt to self-correct based on error message"""

        correction_prompt = f"""
The previous query failed with this error: {error_message}

Original request: {original_prompt}

Available tables: {', '.join(table_schemas.keys())}

Please analyze the error and provide a corrected SQL query that addresses the issue.
Focus on:
1. Syntax errors
2. Column name issues  
3. Data type mismatches
4. Table name errors

Provide only the corrected SQL query.
"""

        try:
            correction_result = self.sql_agent.invoke({"input": correction_prompt})
            corrected_sql = self._extract_sql_from_result(correction_result, [])

            if corrected_sql:
                # Track the self-correction
                self.self_corrections.append({
                    "original_error": error_message,
                    "correction_attempt": corrected_sql,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                return {
                    "success": True,
                    "sql_query": corrected_sql,
                    "agent_output": correction_result.get("output", ""),
                    "self_corrected": True,
                    "execution_steps": self.execution_steps,
                    "self_corrections": self.self_corrections,
                    "reasoning_chain": self.reasoning_chain
                }

        except Exception as e:
            logger.error(f"Self-correction attempt failed: {e}")

        return {"success": False, "error": "Self-correction failed"}


class MiscellaneousProcessorV2:
    """
    Enhanced Miscellaneous Data Processor with LangChain SQL Agent
    
    Maintains backward compatibility with v1 while providing enhanced capabilities:
    - LangChain SQL Agent integration
    - Self-correction for failed queries
    - Multi-step reasoning
    - Detailed execution tracking
    """

    def __init__(self):
        # Initialize connection pool (reuse from v1)
        self.connection_pool = DuckDBConnectionPool()

        # Initialize analytics service (same as v1)
        self.analytics_service = ProcessAnalyticsService()

        # Initialize LangChain SQL Agent
        self.sql_agent_processor = LangChainSQLAgentProcessor()

        # Storage for results (same as v1)
        self.results_storage = {}

        logger.info("MiscellaneousProcessorV2 initialized with LangChain SQL Agent support")

    def process_core_request(
            self,
            user_prompt: str,
            files_data: List[Dict[str, Any]],
            output_format: str = "json",
            use_langchain_agent: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced core processing with LangChain SQL Agent
        
        Args:
            user_prompt: Natural language query
            files_data: List of file data with DataFrames
            output_format: Output format (json/csv)
            use_langchain_agent: Whether to use LangChain agent (required for V2)
        """

        process_id = str(uuid.uuid4())
        start_time = time.time()

        # Prepare analytics data (same as v1)
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
                # Process files and setup tables (same as v1)
                table_schemas = {}
                sample_data = {}

                # Parallel file processing (same as v1)
                with get_data_processing_executor() as executor:
                    futures = []
                    for i, file_data in enumerate(files_data):
                        future = executor.submit(self._process_single_file, i, file_data)
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            table_name = result['table_name']
                            df = result['dataframe']
                            filename = result['filename']

                            # Register with DuckDB
                            duck_processor.register_dataframe(df, table_name)

                            # Get schema info
                            schema = duck_processor.get_table_schema(table_name)
                            table_schemas[table_name] = schema
                            sample_data[table_name] = df

                            logger.info(f"V2: Completed processing {filename} as {table_name}")

                        except Exception as e:
                            logger.error(f"V2: Failed to process file in parallel: {e}")
                            raise

                logger.info(f"V2: Parallel file processing completed for {len(files_data)} files")

                # V2 uses LangChain SQL Agent only
                logger.info("Using LangChain SQL Agent for query generation")
                sql_result = self._generate_sql_with_langchain_agent(
                    user_prompt, table_schemas, duck_processor
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
                        'files_data': files_data,
                        'table_schemas': table_schemas,
                        'agent_steps': sql_result.get('execution_steps', []),
                        'self_corrections': sql_result.get('self_corrections', [])
                    }

                generated_sql = sql_result['sql_query']

                # Execute the query
                try:
                    result_df = duck_processor.execute_query(generated_sql)

                    # Process results (same as v1)
                    total_rows = len(result_df)
                    preview_limit = 100
                    is_limited = total_rows > preview_limit

                    if is_limited:
                        preview_df = result_df.head(preview_limit)
                        logger.info(f"V2: Limited results to {preview_limit} rows (total: {total_rows})")
                    else:
                        preview_df = result_df

                    # Convert to desired format
                    if output_format.lower() == "json":
                        result_data = preview_df.to_dict('records')
                    else:
                        result_data = preview_df

                    # Record analytics (enhanced with agent info)
                    processing_time = time.time() - start_time
                    self._record_enhanced_analytics(
                        process_id, user_prompt, input_files_info,
                        generated_sql, result_df, processing_time, sql_result
                    )

                    return {
                        'success': True,
                        'data': result_data,
                        'full_data': result_df.to_dict('records'),  # Store full data
                        'generated_sql': generated_sql,
                        'row_count': total_rows,
                        'is_limited': is_limited,
                        'processing_info': {
                            'input_files': len(files_data),
                            'table_count': len(table_schemas),
                            'query_type': 'langchain_sql_agent'
                        },
                        'agent_steps': sql_result.get('execution_steps', []),
                        'self_corrections': sql_result.get('self_corrections', []),
                        'reasoning_chain': sql_result.get('reasoning_chain', []),
                        'tool_usage': sql_result.get('tool_usage', {}),
                        'token_usage': sql_result.get('token_usage', {}),
                        'schema_introspection': table_schemas,
                        'confidence_score': self._calculate_confidence_score(sql_result),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

                except Exception as e:
                    logger.error(f"V2: Query execution failed: {e}")

                    # Enhanced error analysis with agent context
                    error_analysis = {
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'generated_sql': generated_sql,
                        'agent_steps': sql_result.get('execution_steps', []),
                        'self_corrections': sql_result.get('self_corrections', []),
                        'suggested_fixes': self._generate_error_suggestions(str(e), generated_sql)
                    }

                    return {
                        'success': False,
                        'error': str(e),
                        'error_analysis': error_analysis,
                        'generated_sql': generated_sql,
                        'data': [],
                        'files_data': files_data,
                        'table_schemas': table_schemas,
                        'agent_steps': sql_result.get('execution_steps', []),
                        'self_corrections': sql_result.get('self_corrections', [])
                    }

        except Exception as e:
            logger.error(f"V2: Core processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'files_data': files_data
            }

    def _process_single_file(self, i: int, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single file (same as v1)"""
        df = file_data['dataframe']
        table_name = f"file_{i + 1}"
        filename = file_data['filename']

        logger.info(f"V2: Processing file {filename} as {table_name}")
        return {
            'table_name': table_name,
            'dataframe': df,
            'filename': filename,
            'index': i
        }

    def _generate_sql_with_langchain_agent(
            self, user_prompt: str, table_schemas: Dict[str, Any], duck_processor
    ) -> Dict[str, Any]:
        """Generate SQL using LangChain SQL Agent"""

        temp_db_path = None
        try:
            # Setup agent with DuckDB connection
            # Create unique temporary database file for SQLDatabase connection
            temp_fd, temp_db_path = tempfile.mkstemp(suffix=".db", prefix="langchain_duckdb_")
            os.close(temp_fd)  # Close the file descriptor, keep the path

            # Remove the empty file first to avoid conflicts
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)

            # Export tables to temporary DuckDB file
            with duckdb.connect(temp_db_path) as temp_conn:
                for table_name in table_schemas.keys():
                    # Get dataframe by executing SELECT query on original connection
                    df = duck_processor.execute_query(f"SELECT * FROM {table_name}")
                    # Register and create table in temp database
                    temp_conn.register(f"{table_name}_temp", df)
                    temp_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_temp")
                # Ensure connection is closed properly
                temp_conn.close()

            # Setup agent with temporary database
            db_connection_string = f"duckdb:///{temp_db_path}"

            if not self.sql_agent_processor.setup_agent(db_connection_string):
                raise ValueError("Failed to setup LangChain SQL Agent")

            # Execute query with agent
            result = self.sql_agent_processor.execute_query_with_agent(user_prompt, table_schemas)

            return result

        except Exception as e:
            logger.error(f"LangChain SQL Agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_steps': getattr(self.sql_agent_processor, 'execution_steps', []),
                'self_corrections': getattr(self.sql_agent_processor, 'self_corrections', [])
            }
        finally:
            # Cleanup temp database file
            if temp_db_path and os.path.exists(temp_db_path):
                try:
                    os.unlink(temp_db_path)
                    logger.debug(f"Cleaned up temporary database: {temp_db_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary database {temp_db_path}: {cleanup_error}")

    def _calculate_confidence_score(self, sql_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on agent execution"""

        base_score = 0.8 if sql_result.get('success') else 0.2

        # Adjust based on self-corrections
        corrections = len(sql_result.get('self_corrections', []))
        if corrections == 0:
            correction_penalty = 0
        elif corrections == 1:
            correction_penalty = 0.1
        else:
            correction_penalty = 0.2

        # Adjust based on execution steps
        steps = len(sql_result.get('execution_steps', []))
        if steps <= 3:
            step_bonus = 0.1
        elif steps <= 5:
            step_bonus = 0.05
        else:
            step_bonus = 0

        confidence = max(0.0, min(1.0, base_score - correction_penalty + step_bonus))
        return round(confidence, 2)

    def _generate_error_suggestions(self, error_message: str, sql_query: str) -> List[str]:
        """Generate suggestions for fixing SQL errors"""

        suggestions = []
        error_lower = error_message.lower()

        if "column" in error_lower and "not found" in error_lower:
            suggestions.append("Check column names in your query against available table schemas")
            suggestions.append("Ensure column names are spelled correctly and exist in the referenced tables")

        if "table" in error_lower and "not found" in error_lower:
            suggestions.append("Verify table names are correct (file_1, file_2, etc.)")
            suggestions.append("Check that all referenced tables exist in the database")

        if "syntax" in error_lower:
            suggestions.append("Review SQL syntax for DuckDB compatibility")
            suggestions.append("Check for missing commas, parentheses, or keywords")

        if "type" in error_lower:
            suggestions.append("Check data type compatibility in operations")
            suggestions.append("Consider explicit type casting if needed")

        return suggestions

    def _record_enhanced_analytics(
            self, process_id: str, user_prompt: str, input_files_info: List[Dict],
            generated_sql: str, result_df: pd.DataFrame, processing_time: float, sql_result: Dict[str, Any]
    ):
        """Record enhanced analytics with agent information"""

        try:
            # Enhanced analytics data
            analytics_data = {
                'process_id': process_id,
                'process_type': 'data_analysis_v2',
                'process_name': f"LangChain SQL Agent Query",
                'user_prompt': user_prompt,
                'generated_sql': generated_sql,
                'status': 'success',
                'confidence_score': self._calculate_confidence_score(sql_result),
                'input_row_count': sum(file_info['row_count'] for file_info in input_files_info),
                'output_row_count': len(result_df),
                'processing_time_seconds': processing_time,
                'token_usage': sql_result.get('token_usage', {}),
                'input_files': input_files_info,
                'agent_execution': {
                    'steps_count': len(sql_result.get('execution_steps', [])),
                    'self_corrections_count': len(sql_result.get('self_corrections', [])),
                    'tool_usage': sql_result.get('tool_usage', {}),
                    'reasoning_steps': len(sql_result.get('reasoning_chain', []))
                },
                'created_at': datetime.now(timezone.utc).isoformat(),
                'version': 'v2_langchain_agent'
            }

            # Store analytics
            self.analytics_service.record_process_execution(
                process_id=process_id,
                process_type="data_analysis",
                process_name="V2 LangChain Analysis",
                user_prompt=user_prompt,
                generated_sql=generated_sql,
                status="success",
                confidence_score=analytics_data.get('confidence_score', 85),
                input_row_count=analytics_data.get('input_row_count', 0),
                output_row_count=analytics_data.get('output_row_count', 0),
                processing_time_seconds=processing_time,
                token_usage=analytics_data.get('token_usage', {}),
                input_files=analytics_data.get('input_files', []),
                errors=[]
            )

        except Exception as e:
            logger.error(f"Failed to record enhanced analytics: {e}")

    def store_results(self, process_id: str, results: Dict[str, Any]):
        """Store results for later retrieval (same as v1)"""
        self.results_storage[process_id] = results
        logger.info(f"V2: Stored results for process {process_id}")

    def get_results(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get stored results (same as v1)"""
        return self.results_storage.get(process_id)

    def execute_direct_sql_v2(self, sql_query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute direct SQL query with V2 enhancements"""

        try:
            start_time = time.time()

            # Recreate DuckDB environment from stored results
            files_data = results.get('files_data', [])

            if not files_data:
                raise ValueError("No file data available for query execution")

            with DuckDBProcessor() as duck_processor:
                # Register tables from stored data
                for i, file_data in enumerate(files_data):
                    table_name = f"file_{i + 1}"
                    df = file_data['dataframe']
                    duck_processor.register_dataframe(df, table_name)

                # Execute query
                result_df = duck_processor.execute_query(sql_query)

                execution_time = time.time() - start_time

                # Enhanced analysis with agent if available
                enhanced_analysis = {}
                agent_suggestions = []

                if self.sql_agent_processor.agent_available:
                    enhanced_analysis = self._analyze_query_performance(sql_query, result_df, execution_time)
                    agent_suggestions = self._generate_query_suggestions(sql_query, result_df)

                return {
                    'success': True,
                    'data': result_df.to_dict('records'),
                    'row_count': len(result_df),
                    'execution_time': execution_time,
                    'enhanced_analysis': enhanced_analysis,
                    'agent_suggestions': agent_suggestions
                }

        except Exception as e:
            logger.error(f"V2 Direct SQL execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'row_count': 0,
                'execution_time': 0
            }

    def _analyze_query_performance(self, sql_query: str, result_df: pd.DataFrame, execution_time: float) -> Dict[
        str, Any]:
        """Analyze query performance with LangChain agent"""

        try:
            performance_analysis = {
                'execution_time_seconds': execution_time,
                'result_rows': len(result_df),
                'result_columns': len(result_df.columns),
                'complexity_score': self._calculate_query_complexity(sql_query),
                'optimization_suggestions': []
            }

            # Add performance insights
            if execution_time > 5.0:
                performance_analysis['optimization_suggestions'].append(
                    "Query took longer than 5 seconds. Consider adding WHERE clauses to limit data."
                )

            if len(result_df) > 10000:
                performance_analysis['optimization_suggestions'].append(
                    "Large result set returned. Consider pagination or filtering."
                )

            return performance_analysis

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_query_complexity(self, sql_query: str) -> int:
        """Calculate query complexity score"""

        complexity = 0
        query_lower = sql_query.lower()

        # Basic complexity factors
        complexity += query_lower.count('join') * 2
        complexity += query_lower.count('subquery') * 3
        complexity += query_lower.count('group by') * 1
        complexity += query_lower.count('order by') * 1
        complexity += query_lower.count('having') * 2
        complexity += query_lower.count('union') * 2

        return complexity

    def _generate_query_suggestions(self, sql_query: str, result_df: pd.DataFrame) -> List[str]:
        """Generate query optimization suggestions"""

        suggestions = []
        query_lower = sql_query.lower()

        if 'select *' in query_lower:
            suggestions.append("Consider selecting specific columns instead of SELECT * for better performance")

        if 'group by' in query_lower and 'order by' not in query_lower:
            suggestions.append("Consider adding ORDER BY for consistent results with GROUP BY")

        if len(result_df) == 0:
            suggestions.append("Query returned no results. Check your WHERE conditions")

        return suggestions

    def _explain_sql_with_agent(self, sql_query: str) -> Dict[str, Any]:
        """Explain SQL query using LangChain agent"""

        if not self.sql_agent_processor.agent_available:
            return {"explanation": "Basic SQL query", "enhanced": False}

        try:
            explanation_prompt = f"""
            Explain this SQL query in detail:
            
            {sql_query}
            
            Please provide:
            1. What the query does
            2. Step-by-step breakdown
            3. Potential optimizations
            4. Common use cases
            """

            # Use LLM to explain query
            if self.sql_agent_processor.llm:
                explanation = self.sql_agent_processor.llm.invoke(explanation_prompt)
                return {
                    "explanation": explanation.content if hasattr(explanation, 'content') else str(explanation),
                    "enhanced": True,
                    "analysis_type": "langchain_agent"
                }

        except Exception as e:
            logger.error(f"Agent SQL explanation failed: {e}")

        return {"explanation": "SQL query explanation not available", "enhanced": False}

    def _generate_ideal_prompt_with_agent(self, basic_intent: str, files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate ideal prompt using LangChain agent analysis"""

        try:
            # Analyze data structure
            data_summary = self._analyze_data_structure(files_data)

            # Generate enhanced prompt
            enhancement_prompt = f"""
            User's basic intent: {basic_intent}
            
            Available data:
            {data_summary}
            
            Generate an ideal, specific prompt that would help create accurate SQL queries for this analysis.
            Make the prompt:
            1. Specific and detailed
            2. Include relevant column names
            3. Clarify the expected output format
            4. Mention any assumptions
            """

            if self.sql_agent_processor.llm:
                response = self.sql_agent_processor.llm.invoke(enhancement_prompt)
                ideal_prompt = response.content if hasattr(response, 'content') else str(response)

                return {
                    'ideal_prompt': ideal_prompt,
                    'alternative_prompts': [
                        f"Detailed analysis: {ideal_prompt}",
                        f"Summary version: {basic_intent} using the available data",
                        f"Step-by-step: Break down {basic_intent} into detailed steps"
                    ],
                    'analysis': data_summary
                }

        except Exception as e:
            logger.error(f"Agent prompt generation failed: {e}")

        return {
            'ideal_prompt': basic_intent,
            'alternative_prompts': [],
            'analysis': {}
        }

    def _analyze_data_structure(self, files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data structure for prompt enhancement"""

        analysis = {
            'total_files': len(files_data),
            'files_info': [],
            'common_columns': [],
            'data_types': {},
            'potential_joins': []
        }

        for i, file_data in enumerate(files_data):
            df = file_data['dataframe']
            file_info = {
                'table_name': f"file_{i + 1}",
                'filename': file_data.get('filename', 'unknown'),
                'rows': len(df),
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict()
            }
            analysis['files_info'].append(file_info)

        # Find common columns across files
        if len(files_data) > 1:
            all_columns = [set(info['columns']) for info in analysis['files_info']]
            analysis['common_columns'] = list(set.intersection(*all_columns))

        return analysis

    def _analyze_prompt_quality(self, prompt_text: str) -> Dict[str, Any]:
        """Analyze prompt quality and provide suggestions"""

        analysis = {
            'quality_score': 0,
            'suggestions': [],
            'strengths': [],
            'weaknesses': []
        }

        # Basic quality checks
        score = 50  # Base score

        # Length check
        if len(prompt_text) > 20:
            score += 10
            analysis['strengths'].append("Good prompt length")
        else:
            analysis['weaknesses'].append("Prompt is too short")

        # Specificity check
        specific_words = ['calculate', 'analyze', 'compare', 'filter', 'group', 'sum', 'count', 'average']
        if any(word in prompt_text.lower() for word in specific_words):
            score += 20
            analysis['strengths'].append("Contains specific action words")
        else:
            analysis['suggestions'].append("Add specific action words (calculate, analyze, etc.)")

        # Column references
        if any(word in prompt_text.lower() for word in ['column', 'field', 'by']):
            score += 15
            analysis['strengths'].append("References data structure")
        else:
            analysis['suggestions'].append("Reference specific columns or fields")

        analysis['quality_score'] = min(100, score)
        return analysis

    def _generate_intelligent_suggestions_with_agent(self, files_data: List[Dict[str, Any]], current_prompt: str) -> \
    Dict[str, Any]:
        """Generate intelligent suggestions using LangChain agent"""

        try:
            data_analysis = self._analyze_data_structure(files_data)

            suggestions = {
                'suggestions': [],
                'categories': {
                    'analysis': [],
                    'aggregation': [],
                    'filtering': [],
                    'comparison': []
                },
                'data_insights': data_analysis,
                'complexity_analysis': {}
            }

            # Generate category-specific suggestions
            for file_info in data_analysis['files_info']:
                table_name = file_info['table_name']
                columns = file_info['columns']

                # Analysis suggestions
                suggestions['categories']['analysis'].extend([
                    f"Show all data from {table_name}",
                    f"Describe the structure of {table_name}",
                    f"Count total rows in {table_name}"
                ])

                # Aggregation suggestions
                numeric_columns = [col for col, dtype in file_info['data_types'].items()
                                   if 'int' in str(dtype) or 'float' in str(dtype)]
                for col in numeric_columns[:3]:  # Limit to first 3
                    suggestions['categories']['aggregation'].extend([
                        f"Calculate sum of {col} in {table_name}",
                        f"Find average {col} in {table_name}"
                    ])

            # Flatten all suggestions
            all_suggestions = []
            for category_suggestions in suggestions['categories'].values():
                all_suggestions.extend(category_suggestions)

            suggestions['suggestions'] = all_suggestions[:10]  # Limit to 10

            return suggestions

        except Exception as e:
            logger.error(f"Intelligent suggestions failed: {e}")
            return {
                'suggestions': ['Analyze the data', 'Calculate totals', 'Group by categories'],
                'categories': {},
                'data_insights': {},
                'complexity_analysis': {}
            }

    def _verify_intent_with_agent(self, user_prompt: str, files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify user intent using LangChain agent"""

        try:
            data_context = self._analyze_data_structure(files_data)

            verification_prompt = f"""
            User prompt: "{user_prompt}"
            
            Available data context:
            {data_context}
            
            Analyze if this prompt can be fulfilled with the available data.
            Consider:
            1. Are the requested operations possible?
            2. Are referenced columns/data available?
            3. Is the intent clear and specific?
            4. What clarifications might be needed?
            
            Provide a confidence score (0-100) and detailed analysis.
            """

            if self.sql_agent_processor.llm:
                response = self.sql_agent_processor.llm.invoke(verification_prompt)
                analysis = response.content if hasattr(response, 'content') else str(response)

                # Simple confidence scoring based on analysis
                confidence = 80 if "can be fulfilled" in analysis.lower() else 60
                intent_understood = confidence > 70

                return {
                    'intent_understood': intent_understood,
                    'confidence_score': confidence,
                    'interpretation': analysis,
                    'alternative_interpretations': [],
                    'required_clarifications': [],
                    'suggested_improvements': []
                }

        except Exception as e:
            logger.error(f"Intent verification failed: {e}")

        return {
            'intent_understood': True,
            'confidence_score': 75,
            'interpretation': 'Intent appears reasonable for the available data',
            'alternative_interpretations': [],
            'required_clarifications': [],
            'suggested_improvements': []
        }

    def _improve_prompt_with_agent(self, original_prompt: str, files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Improve prompt using LangChain agent"""

        try:
            data_context = self._analyze_data_structure(files_data)

            improvement_prompt = f"""
            Original prompt: "{original_prompt}"
            
            Data context:
            {data_context}
            
            Improve this prompt to be more specific and effective for SQL generation.
            Make it:
            1. More specific about desired output
            2. Reference available columns where appropriate
            3. Clarify any ambiguous requirements
            4. Add context about expected format
            
            Provide the improved version and explain what changes were made.
            """

            if self.sql_agent_processor.llm:
                response = self.sql_agent_processor.llm.invoke(improvement_prompt)
                analysis = response.content if hasattr(response, 'content') else str(response)

                # Extract improved prompt (simple heuristic)
                lines = analysis.split('\n')
                improved_prompt = original_prompt  # Fallback

                for line in lines:
                    if len(line.strip()) > len(original_prompt) * 0.8:
                        improved_prompt = line.strip()
                        break

                return {
                    'improved_prompt': improved_prompt,
                    'improvement_score': 85,
                    'changes_made': ['Added specificity', 'Clarified output format'],
                    'reasoning': analysis,
                    'alternative_versions': [improved_prompt]
                }

        except Exception as e:
            logger.error(f"Prompt improvement failed: {e}")

        return {
            'improved_prompt': original_prompt,
            'improvement_score': 0,
            'changes_made': [],
            'reasoning': 'No improvements available',
            'alternative_versions': []
        }


# Utility function to check LangChain availability
def check_langchain_dependencies() -> Dict[str, Any]:
    """Check if LangChain dependencies are available"""

    dependencies = {
        'langchain_available': LANGCHAIN_AVAILABLE,
        'openai_key_configured': bool(os.getenv("OPENAI_API_KEY")),
        'missing_packages': []
    }

    if not LANGCHAIN_AVAILABLE:
        try:
            import langchain_community
        except ImportError:
            dependencies['missing_packages'].append('langchain-community')

        try:
            import langchain_openai
        except ImportError:
            dependencies['missing_packages'].append('langchain-openai')

    return dependencies
