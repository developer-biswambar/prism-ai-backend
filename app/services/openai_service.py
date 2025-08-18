# backend/app/services/openai_service.py
import asyncio
import json
import logging
import time
from typing import List, Dict, Any

from app.models.schemas import ExtractedField, ExtractionRow
from app.services.llm_service import get_llm_service, LLMMessage

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self):
        self.llm_service = get_llm_service()

    async def extract_financial_data(
            self,
            text_batch: List[str],
            extraction_prompt: str,
            source_column: str
    ) -> List[ExtractionRow]:
        """
        Extract financial data from a batch of text using OpenAI GPT
        """
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(text_batch, extraction_prompt, source_column)

            start_time = time.time()

            response = await self._call_llm_api(system_prompt, user_prompt)

            processing_time = time.time() - start_time

            # Parse response and create ExtractionRow objects
            extraction_rows = self._parse_openai_response(
                response, text_batch, processing_time
            )

            return extraction_rows

        except Exception as e:
            logger.error(f"Error in OpenAI extraction: {str(e)}")
            # Return failed extraction rows
            return self._create_failed_rows(text_batch, str(e))

    async def call_llm_generic(
            self,
            system_prompt: str = None,
            user_prompt: str = None,
            messages: List[Dict[str, str]] = None,
            temperature: float = None,
            max_tokens: int = None,
            response_format: Dict[str, str] = None
    ) -> str:
        """
        Generic LLM API call without predefined prompts
        Can be used for any custom AI assistance tasks
        """
        try:
            # Build messages array
            if messages:
                # Convert dict messages to LLMMessage objects
                llm_messages = [LLMMessage(role=msg["role"], content=msg["content"]) for msg in messages]
            else:
                llm_messages = []
                if system_prompt:
                    llm_messages.append(LLMMessage(role="system", content=system_prompt))
                if user_prompt:
                    llm_messages.append(LLMMessage(role="user", content=user_prompt))

            if not llm_messages:
                raise ValueError("Must provide either messages array or system/user prompts")

            # Make the LLM call
            response = self.llm_service.generate_text(
                messages=llm_messages,
                temperature=temperature or 0.3,
                max_tokens=max_tokens or 2000
            )

            if not response.success:
                raise ValueError(f"LLM generation failed: {response.error}")

            return response.content

        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            raise

    async def suggest_transformation_rules(
            self,
            source_columns: Dict[str, List[str]],
            output_schema: Dict[str, Any],
            transformation_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Use OpenAI to analyze data structure and suggest intelligent transformation rules
        """
        try:
            system_prompt = self._build_transformation_system_prompt()
            user_prompt = self._build_transformation_user_prompt(
                source_columns, output_schema, transformation_context
            )

            start_time = time.time()

            response = await self.call_llm_generic(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for more consistent suggestions
                response_format={"type": "json_object"}
            )

            processing_time = time.time() - start_time

            # Parse and validate response
            suggestions = self._parse_transformation_response(response)

            logger.info(f"Generated transformation suggestions in {processing_time:.2f}s")

            return {
                "success": True,
                "suggestions": suggestions,
                "processing_time": processing_time,
                "model_used": self.llm_service.get_provider_name()
            }

        except Exception as e:
            logger.error(f"Error generating transformation suggestions: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "suggestions": {"row_generation": [], "column_mappings": []}
            }

    def _build_transformation_system_prompt(self) -> str:
        """Build system prompt for transformation rule suggestions"""
        return """
You are an expert data transformation specialist. Your task is to analyze source data structures and suggest intelligent transformation rules for converting data into a target schema.

TRANSFORMATION CAPABILITIES:
1. ROW GENERATION RULES:
   - duplicate: Simple row duplication
   - fixed_expansion: Create multiple rows with fixed values
   - conditional_expansion: Create rows based on conditions
   - expand_from_list: Expand rows for each value in a list
   - localization_expansion: Expand for different locales/regions

2. COLUMN MAPPING TYPES:
   - direct: Direct column-to-column mapping
   - static: Set fixed values
   - expression: Mathematical/logical expressions
   - conditional: If-then-else logic
   - sequence: Generate sequential numbers/IDs
   - custom_function: JavaScript functions for complex logic

ANALYSIS FOCUS:
- Identify patterns in column names (tax, amount, id, date, status, etc.)
- Detect business logic requirements (calculations, validations, formatting)
- Suggest data normalization and denormalization patterns
- Recommend row expansion for business scenarios
- Propose intelligent default values and transformations

OUTPUT FORMAT:
Return a JSON object with this structure:
{
  "row_generation": [
    {
      "confidence": 0.85,
      "title": "Tax Line Item Expansion",
      "description": "Create separate line items for tax calculations",
      "reasoning": "Detected tax columns - expand for detailed reporting",
      "rule_type": "fixed_expansion",
      "auto_config": {
        "name": "Tax Line Items",
        "type": "expand",
        "strategy": {
          "type": "fixed_expansion",
          "config": {
            "expansions": [
              {"set_values": {"line_type": "base_amount"}},
              {"set_values": {"line_type": "tax_amount"}}
            ]
          }
        }
      }
    }
  ],
  "column_mappings": [
    {
      "confidence": 0.9,
      "target_column": "total_amount",
      "title": "Calculate Total Amount",
      "description": "Sum base amount and tax amount",
      "reasoning": "Detected total pattern with component amounts available",
      "mapping_type": "expression",
      "auto_config": {
        "mapping_type": "expression",
        "transformation": {
          "type": "expression",
          "config": {
            "formula": "{base_amount} + {tax_amount}",
            "variables": {
              "base_amount": "file_0.amount",
              "tax_amount": "file_0.tax"
            }
          }
        }
      }
    }
  ]
}

IMPORTANT:
- Always return valid JSON
- Provide confidence scores (0.0-1.0)
- Include clear reasoning for each suggestion
- Generate practical, implementable configurations
- Focus on common business patterns and requirements
"""

    def _build_transformation_user_prompt(
            self,
            source_columns: Dict[str, List[str]],
            output_schema: Dict[str, Any],
            transformation_context: Dict[str, Any] = None
    ) -> str:
        """Build user prompt for transformation analysis"""

        context_info = ""
        if transformation_context:
            context_info = f"""
TRANSFORMATION CONTEXT:
- Name: {transformation_context.get('name', 'N/A')}
- Description: {transformation_context.get('description', 'N/A')}
- Industry/Domain: {transformation_context.get('industry', 'General')}
"""

        return f"""
Please analyze the following data structure and suggest intelligent transformation rules:

SOURCE DATA STRUCTURE:
{json.dumps(source_columns, indent=2)}

TARGET OUTPUT SCHEMA:
{json.dumps(output_schema, indent=2)}

{context_info}

ANALYSIS REQUEST:
1. Analyze column name patterns and data types
2. Identify potential business logic requirements
3. Suggest row generation rules for data expansion/normalization
4. Recommend column mapping strategies
5. Provide confidence scores and clear reasoning

Focus on practical, commonly-used transformation patterns that would be valuable for business data processing.
"""

    def _parse_transformation_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate transformation suggestions response"""
        try:
            # Clean response if wrapped in markdown
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()

            parsed_data = json.loads(response)

            # Validate structure
            suggestions = {
                "row_generation": parsed_data.get("row_generation", []),
                "column_mappings": parsed_data.get("column_mappings", [])
            }

            # Validate each suggestion has required fields
            for suggestion in suggestions["row_generation"]:
                if not all(key in suggestion for key in ["confidence", "title", "rule_type"]):
                    logger.warning(f"Invalid row generation suggestion: {suggestion}")

            for suggestion in suggestions["column_mappings"]:
                if not all(key in suggestion for key in ["confidence", "title", "mapping_type"]):
                    logger.warning(f"Invalid column mapping suggestion: {suggestion}")

            return suggestions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse transformation response as JSON: {str(e)}")
            logger.error(f"Raw response: {response}")
            return {"row_generation": [], "column_mappings": []}

        except Exception as e:
            logger.error(f"Error parsing transformation response: {str(e)}")
            return {"row_generation": [], "column_mappings": []}

    def _build_system_prompt(self) -> str:
        """Build the system prompt for financial data extraction"""
        return """
You are an expert financial data extraction specialist. Your task is to extract structured financial information from unstructured text data.

FINANCIAL DATA TYPES:
- ISIN: 12-character international securities identifier (e.g., US0378331005)
- CUSIP: 9-character US securities identifier (e.g., 037833100)
- SEDOL: 7-character UK securities identifier (e.g., 2000019)
- Ticker: Stock ticker symbol (e.g., AAPL, MSFT)
- Amount: Monetary values with or without currency symbols
- Currency: 3-letter ISO currency codes (USD, EUR, GBP, etc.)
- Date: Transaction or settlement dates in various formats
- Trade ID: Transaction reference numbers
- Account ID: Account identifiers
- Counterparty: Trading counterparty names
- Description: Additional trade details

EXTRACTION RULES:
1. Extract ONLY the requested fields from the user prompt
2. Return null for missing or unclear values
3. Validate financial identifiers using standard formats
4. Provide confidence scores (0.0-1.0) for each extraction
5. Return results as valid JSON array

OUTPUT FORMAT:
Return a JSON array where each object represents one input text with extracted fields:
[
  {
    "row_index": 0,
    "extracted_fields": [
      {
        "field_name": "ISIN",
        "field_value": "US0378331005",
        "confidence": 0.95,
        "extraction_method": "llm"
      }
    ],
    "error_message": null
  }
]

IMPORTANT: 
- Always return valid JSON
- Include row_index for each input text
- Set confidence based on certainty of extraction
- Use null for missing values, not empty strings
"""

    def _build_user_prompt(
            self,
            text_batch: List[str],
            extraction_prompt: str,
            source_column: str
    ) -> str:
        """Build the user prompt with specific extraction instructions"""

        # Format the text batch with indices
        formatted_texts = []
        for i, text in enumerate(text_batch):
            formatted_texts.append(f"Index {i}: {text}")

        texts_str = "\n".join(formatted_texts)

        return f"""
EXTRACTION REQUEST:
{extraction_prompt}

SOURCE COLUMN: {source_column}

INPUT DATA:
{texts_str}

Please extract the requested financial data from each text entry and return as JSON array following the specified format.
"""

    async def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make async call to LLM API"""
        try:
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt)
            ]
            
            response = self.llm_service.generate_text(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )

            if not response.success:
                raise ValueError(f"LLM generation failed: {response.error}")

            return response.content

        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            raise

    def _parse_openai_response(
            self,
            response: str,
            original_texts: List[str],
            processing_time: float
    ) -> List[ExtractionRow]:
        """Parse OpenAI response into ExtractionRow objects"""
        try:
            # Try to parse as JSON
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()

            parsed_data = json.loads(response)

            # Handle if response is wrapped in an object
            if isinstance(parsed_data, dict) and 'results' in parsed_data:
                parsed_data = parsed_data['results']
            elif isinstance(parsed_data, dict) and 'extractions' in parsed_data:
                parsed_data = parsed_data['extractions']

            if not isinstance(parsed_data, list):
                raise ValueError("Response is not a list")

            extraction_rows = []

            for item in parsed_data:
                row_index = item.get('row_index', 0)

                # Ensure we don't exceed original texts length
                if row_index >= len(original_texts):
                    continue

                extracted_fields = []
                for field_data in item.get('extracted_fields', []):
                    extracted_fields.append(
                        ExtractedField(
                            field_name=field_data.get('field_name'),
                            field_value=field_data.get('field_value'),
                            confidence=field_data.get('confidence', 0.8),
                            extraction_method=field_data.get('extraction_method', 'llm')
                        )
                    )

                extraction_rows.append(
                    ExtractionRow(
                        row_index=row_index,
                        original_text=original_texts[row_index],
                        extracted_fields=extracted_fields,
                        processing_time=processing_time / len(original_texts),
                        error_message=item.get('error_message')
                    )
                )

            return extraction_rows

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {str(e)}")
            logger.error(f"Raw response: {response}")
            return self._create_failed_rows(original_texts, f"JSON parsing error: {str(e)}")

        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {str(e)}")
            return self._create_failed_rows(original_texts, str(e))

    def _create_failed_rows(self, original_texts: List[str], error_message: str) -> List[ExtractionRow]:
        """Create failed extraction rows for error cases"""
        failed_rows = []
        for i, text in enumerate(original_texts):
            failed_rows.append(
                ExtractionRow(
                    row_index=i,
                    original_text=text,
                    extracted_fields=[],
                    processing_time=0.0,
                    error_message=error_message
                )
            )
        return failed_rows

    async def test_connection(self) -> bool:
        """Test LLM API connection"""
        try:
            messages = [
                LLMMessage(role="system", content="You are a test assistant."),
                LLMMessage(role="user", content="Respond with 'OK' if you can hear me.")
            ]
            
            response = self.llm_service.generate_text(
                messages=messages,
                max_tokens=10,
                temperature=0
            )
            
            return response.success and "OK" in response.content
        except Exception as e:
            logger.error(f"LLM connection test failed: {str(e)}")
            return False


def get_openai_client():
    """Get LLM service for transformation routes and other operations"""
    try:
        llm_service = get_llm_service()
        if not llm_service.is_available():
            logger.warning(f"LLM service ({llm_service.get_provider_name()}) not available")
            return None
        
        return llm_service
    except Exception as e:
        logger.error(f"Failed to get LLM service: {str(e)}")
        return None


def get_async_openai_client():
    """Get LLM service for async operations (compatibility function)"""
    return get_openai_client()


# Singleton instance
openai_service = OpenAIService()
