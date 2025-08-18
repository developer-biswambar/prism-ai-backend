# backend/app/routes/regex_routes.py - AI-powered regex generation routes
import json
import logging
import os
import re
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llm_service import get_llm_service, LLMMessage

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/regex", tags=["regex"])


# Pydantic models
class RegexGenerationRequest(BaseModel):
    description: str = Field(..., description="Description of what to extract")
    sample_text: Optional[str] = Field(None, description="Sample text to test against")
    column_name: Optional[str] = Field(None, description="Name of the column")
    context: Optional[Dict] = Field(None, description="Additional context")


class RegexGenerationResponse(BaseModel):
    success: bool
    regex: str
    explanation: str
    test_cases: List[str] = []
    is_fallback: bool = False
    error: Optional[str] = None


class RegexTestRequest(BaseModel):
    regex: str
    test_text: str


class RegexTestMatch(BaseModel):
    match: str
    index: int
    groups: List[str] = []
    length: int


class RegexTestResponse(BaseModel):
    success: bool
    matches: List[RegexTestMatch] = []
    is_valid: bool = True
    error: Optional[str] = None


# Common regex patterns for fallback
COMMON_PATTERNS = {
    'dollar': {
        'regex': r'\$?([0-9,]+(?:\.[0-9]{2})?)',
        'explanation': 'Matches dollar amounts with optional currency symbol, commas, and decimal places',
        'test_cases': ['$1,234.56', '$99.99', '1000', '123.45']
    },
    'email': {
        'regex': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'explanation': 'Matches standard email addresses',
        'test_cases': ['user@example.com', 'test.email+tag@domain.org']
    },
    'date': {
        'regex': r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
        'explanation': 'Matches dates in MM/DD/YYYY or MM-DD-YYYY format',
        'test_cases': ['12/31/2023', '01-15-2024', '3/5/2023']
    },
    'phone': {
        'regex': r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
        'explanation': 'Matches US phone numbers in various formats',
        'test_cases': ['(123) 456-7890', '123-456-7890', '123.456.7890']
    },
    'isin': {
        'regex': r'[A-Z]{2}[A-Z0-9]{9}[0-9]',
        'explanation': 'Matches ISIN codes (12 character international securities identification)',
        'test_cases': ['US0378331005', 'GB0002162385', 'DE0007164600']
    },
    'percentage': {
        'regex': r'\d+(?:\.\d+)?%',
        'explanation': 'Matches percentage values',
        'test_cases': ['5.25%', '10%', '0.5%', '100%']
    },
    'transaction_id': {
        'regex': r'TXN\d{6,}',
        'explanation': 'Matches transaction IDs starting with TXN followed by digits',
        'test_cases': ['TXN123456', 'TXN7890123', 'TXN001234567']
    },
    'account_number': {
        'regex': r'\b\d{8,12}\b',
        'explanation': 'Matches account numbers (8-12 digits)',
        'test_cases': ['12345678', '123456789012']
    },
    'routing_number': {
        'regex': r'\b\d{9}\b',
        'explanation': 'Matches US routing numbers (9 digits)',
        'test_cases': ['021000021', '111000025']
    },
    'credit_card': {
        'regex': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'explanation': 'Matches credit card numbers with optional separators',
        'test_cases': ['1234-5678-9012-3456', '1234 5678 9012 3456']
    }
}


def get_llm_client():
    """Get LLM service instance"""
    llm_service = get_llm_service()
    if not llm_service.is_available():
        raise HTTPException(status_code=500, detail=f"LLM service ({llm_service.get_provider_name()}) not available")
    return llm_service


def suggest_fallback_pattern(description: str) -> Optional[Dict]:
    """Suggest a fallback pattern based on description keywords"""
    desc_lower = description.lower()

    # Check for keyword matches
    if any(word in desc_lower for word in ['dollar', 'amount', 'money', 'currency', '$']):
        return COMMON_PATTERNS['dollar']
    elif 'email' in desc_lower:
        return COMMON_PATTERNS['email']
    elif any(word in desc_lower for word in ['date', 'time']):
        return COMMON_PATTERNS['date']
    elif any(word in desc_lower for word in ['phone', 'telephone', 'mobile']):
        return COMMON_PATTERNS['phone']
    elif any(word in desc_lower for word in ['isin', 'security', 'instrument']):
        return COMMON_PATTERNS['isin']
    elif any(word in desc_lower for word in ['percent', '%', 'percentage']):
        return COMMON_PATTERNS['percentage']
    elif any(word in desc_lower for word in ['transaction', 'txn', 'reference']):
        return COMMON_PATTERNS['transaction_id']
    elif any(word in desc_lower for word in ['account', 'acct']):
        return COMMON_PATTERNS['account_number']
    elif any(word in desc_lower for word in ['routing', 'aba']):
        return COMMON_PATTERNS['routing_number']
    elif any(word in desc_lower for word in ['card', 'credit', 'debit']):
        return COMMON_PATTERNS['credit_card']

    return None


async def generate_regex_with_llm(description: str, sample_text: str = "", column_name: str = "") -> Dict:
    """Generate regex using LLM service"""
    llm_service = get_llm_client()

    user_prompt = f"""You are a regex expert. Generate a JavaScript-compatible regular expression based on this description:

Description: "{description}"
{f'Sample text: "{sample_text}"' if sample_text else ''}
{f'Column name: "{column_name}"' if column_name else ''}

Create a regex that:
1. Is JavaScript-compatible (escape backslashes properly)
2. Uses appropriate capture groups for extraction
3. Handles common variations and edge cases
4. Is as specific as possible to avoid false matches

Respond ONLY with a valid JSON object in this exact format:
{{
  "regex": "the regex pattern with proper escaping",
  "explanation": "clear explanation of what the regex does",
  "testCases": ["example1", "example2", "example3"]
}}

Example response for "Extract dollar amounts":
{{"regex": "\\\\\\$([\\\\d,]+(?:\\\\\\.\\\\d{{2}})?)", "explanation": "Matches dollar signs followed by numbers with optional commas and decimal places. Captures the numeric part.", "testCases": ["$1,234.56", "$99.99", "$1000"]}}"""

    try:
        messages = [
            LLMMessage(role="system", content="You are a regex expert that generates JavaScript-compatible regular expressions. Always respond with valid JSON containing regex, explanation, and testCases fields."),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        response = llm_service.generate_text(
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        if not response.success:
            raise ValueError(f"LLM generation failed: {response.error}")
            
        content = response.content.strip()

        # Parse JSON response
        try:
            result = json.loads(content)

            # Validate required fields
            if not all(key in result for key in ['regex', 'explanation', 'testCases']):
                raise ValueError("Missing required fields in LLM response")

            # Validate regex
            try:
                re.compile(result['regex'])
            except re.error as e:
                raise ValueError(f"Generated regex is invalid: {str(e)}")

            return result

        except json.JSONDecodeError:
            # Try to extract regex pattern from response if JSON parsing fails
            regex_match = re.search(r'"regex":\s*"([^"]+)"', content)
            if regex_match:
                regex_pattern = regex_match.group(1)
                return {
                    "regex": regex_pattern,
                    "explanation": "LLM-generated regex pattern",
                    "testCases": []
                }
            else:
                raise ValueError("Unable to parse LLM response")

    except Exception as e:
        logger.error(f"LLM service error: {str(e)}")
        raise ValueError(f"LLM service error: {str(e)}")


@router.post("/generate", response_model=RegexGenerationResponse)
async def generate_regex(request: RegexGenerationRequest):
    """Generate a regex pattern using AI"""
    try:
        logger.info(f"Generating regex for description: {request.description}")

        # Try AI generation first
        try:
            ai_result = await generate_regex_with_llm(
                request.description,
                request.sample_text or "",
                request.column_name or ""
            )

            return RegexGenerationResponse(
                success=True,
                regex=ai_result['regex'],
                explanation=ai_result['explanation'],
                test_cases=ai_result.get('testCases', []),
                is_fallback=False
            )

        except Exception as ai_error:
            logger.warning(f"AI generation failed: {str(ai_error)}")

            # Fall back to common patterns
            fallback_pattern = suggest_fallback_pattern(request.description)

            if fallback_pattern:
                logger.info(f"Using fallback pattern for: {request.description}")
                return RegexGenerationResponse(
                    success=True,
                    regex=fallback_pattern['regex'],
                    explanation=f"Fallback pattern: {fallback_pattern['explanation']}",
                    test_cases=fallback_pattern['test_cases'],
                    is_fallback=True
                )
            else:
                # No suitable fallback found
                return RegexGenerationResponse(
                    success=False,
                    regex="",
                    explanation="",
                    test_cases=[],
                    is_fallback=False,
                    error=f"Could not generate regex: {str(ai_error)}"
                )

    except Exception as e:
        logger.error(f"Regex generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Regex generation failed: {str(e)}")


@router.post("/test", response_model=RegexTestResponse)
async def test_regex(request: RegexTestRequest):
    """Test a regex pattern against sample text"""
    try:
        # Validate regex first
        try:
            regex_obj = re.compile(request.regex, re.IGNORECASE)
        except re.error as e:
            return RegexTestResponse(
                success=False,
                matches=[],
                is_valid=False,
                error=f"Invalid regex pattern: {str(e)}"
            )

        # Find matches
        matches = []
        for match in regex_obj.finditer(request.test_text):
            matches.append(RegexTestMatch(
                match=match.group(0),
                index=match.start(),
                groups=list(match.groups()),
                length=len(match.group(0))
            ))

            # Limit to prevent too many matches
            if len(matches) >= 20:
                break

        return RegexTestResponse(
            success=True,
            matches=matches,
            is_valid=True
        )

    except Exception as e:
        logger.error(f"Regex test error: {str(e)}")
        return RegexTestResponse(
            success=False,
            matches=[],
            is_valid=False,
            error=f"Test failed: {str(e)}"
        )


@router.get("/patterns", response_model=Dict)
async def get_common_patterns():
    """Get list of common regex patterns"""
    return {
        "success": True,
        "patterns": COMMON_PATTERNS
    }


@router.get("/suggestions")
async def get_pattern_suggestions(description: str):
    """Get pattern suggestions based on description"""
    fallback = suggest_fallback_pattern(description)

    return {
        "success": True,
        "suggestion": fallback,
        "has_suggestion": fallback is not None
    }
