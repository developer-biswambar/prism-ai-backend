#!/usr/bin/env python3
"""
Simple test to debug the evaluate_expression function
"""

import re
import logging

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def evaluate_expression(expression: str, row_data: dict) -> str:
    """Evaluate expressions with variable substitution like {column_name} or calculations"""
    if not expression or not isinstance(expression, str):
        return expression
    
    print(f"Input expression: '{expression}'")
    print(f"Row data: {row_data}")
    
    try:
        # Check if it's an expression with variables (contains curly braces)
        if '{' in expression and '}' in expression:
            
            # Find all variables in curly braces
            variables = re.findall(r'\{([^}]+)\}', expression)
            print(f"Found variables: {variables}")
            result_expression = expression
            
            # Replace each variable with its value
            for var in variables:
                if var in row_data:
                    value = row_data[var]
                    print(f"Replacing {{{var}}} with '{value}' (type: {type(value)})")
                    # Handle different types of values
                    if isinstance(value, (int, float)):
                        result_expression = result_expression.replace(f'{{{var}}}', str(value))
                    elif isinstance(value, str):
                        # For string values, wrap in quotes for safe evaluation
                        result_expression = result_expression.replace(f'{{{var}}}', f'"{value}"')
                    else:
                        result_expression = result_expression.replace(f'{{{var}}}', f'"{str(value)}"')
                else:
                    logger.warning(f"Variable '{var}' not found in row data")
                    return expression  # Return original if variable not found
            
            print(f"After variable substitution: '{result_expression}'")
            
            # Try to evaluate as mathematical expression first
            try:
                # Create safe evaluation context
                safe_context = {
                    '__builtins__': {},
                    'abs': abs,
                    'round': round,
                    'min': min, 
                    'max': max,
                    'int': int,
                    'float': float,
                    'str': str
                }
                
                # If it looks like a math expression, evaluate it
                if any(op in result_expression for op in ['+', '-', '*', '/', '(', ')', '%']):
                    print("Detected as mathematical expression")
                    result = eval(result_expression, safe_context)
                    print(f"Math evaluation result: '{result}'")
                    return result
                else:
                    # For string concatenation, just remove quotes and return as string
                    # Handle patterns like "John" "Doe" -> "John Doe"
                    print("Processing as string concatenation")
                    result = result_expression.replace('" "', ' ').replace('"', '')
                    print(f"String concatenation result: '{result}'")
                    return result.strip()
                    
            except Exception as e:
                logger.warning(f"Could not evaluate expression '{result_expression}': {e}")
                # Fallback: simple string concatenation
                result = result_expression.replace('" "', ' ').replace('"', '')
                print(f"Fallback result: '{result}'")
                return result.strip()
        
        else:
            # No variables, return as-is
            print("No variables found, returning as-is")
            return expression
            
    except Exception as e:
        print(f"Error in evaluate_expression: {e}")
        return expression

def test_account_summary():
    """Test the account_summary field generation"""
    
    print("=== Testing account_summary generation ===\n")
    
    # Sample row data from customer CSV
    sample_row_data = {
        'Customer_ID': 'CUST001',
        'First_Name': 'John',
        'Last_Name': 'Smith',
        'Account_Type': 'Premium',
        'Balance': 15000.50,
        'Status': 'Active'
    }
    
    # Test the static value expression
    static_expression = "{Account_Type} account with balance ${Balance}"
    print(f"Testing expression: '{static_expression}'")
    print()
    
    # Test evaluate_expression directly
    result = evaluate_expression(static_expression, sample_row_data)
    print(f"\nFinal result: '{result}'")
    print(f"Result type: {type(result)}")
    
    # Expected result
    expected = "Premium account with balance $15000.5"
    print(f"Expected result: '{expected}'")
    print(f"Match: {result == expected}")
    
    # Test with another sample
    print("\n" + "="*50)
    sample_row_data2 = {
        'Account_Type': 'Standard',
        'Balance': 2500.75
    }
    
    result2 = evaluate_expression(static_expression, sample_row_data2)
    print(f"\nSecond test result: '{result2}'")
    expected2 = "Standard account with balance $2500.75"
    print(f"Expected: '{expected2}'")
    print(f"Match: {result2 == expected2}")

if __name__ == "__main__":
    test_account_summary()