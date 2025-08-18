#!/usr/bin/env python3
"""
Test script to debug and fix the account_summary field generation issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.routes.transformation_routes import evaluate_expression, apply_column_mapping

def test_account_summary_generation():
    """Test the account_summary field generation"""
    
    print("=== Testing account_summary generation ===\n")
    
    # Sample row data from customer CSV
    sample_row_data = {
        'Customer_ID': 'CUST001',
        'First_Name': 'John',
        'Last_Name': 'Smith',
        'Email': 'john.smith@email.com',
        'Phone': '555-0101',
        'Date_Joined': '2024-01-15',
        'Account_Type': 'Premium',
        'Balance': 15000.50,
        'Status': 'Active',
        'Region': 'North',
        'City': 'New York',
        'Age': 35,
        'Gender': 'M',
        'Membership_Level': 'Gold',
        'Last_Login': '2024-01-20'
    }
    
    print("Sample row data:")
    for key, value in sample_row_data.items():
        print(f"  {key}: {value}")
    print()
    
    # Test the static value expression
    static_expression = "{Account_Type} account with balance ${Balance}"
    print(f"Testing expression: '{static_expression}'")
    
    # Test evaluate_expression directly
    result = evaluate_expression(static_expression, sample_row_data)
    print(f"Direct evaluate_expression result: '{result}'")
    print(f"Result type: {type(result)}")
    print()
    
    # Test through apply_column_mapping
    mapping_config = {
        'name': 'account_summary',
        'mapping_type': 'static',
        'static_value': static_expression
    }
    
    print(f"Mapping config: {mapping_config}")
    
    result2 = apply_column_mapping(mapping_config, sample_row_data, {})
    print(f"apply_column_mapping result: '{result2}'")
    print(f"Result type: {type(result2)}")
    print()
    
    # Expected result
    expected = "Premium account with balance $15000.5"
    print(f"Expected result: '{expected}'")
    print(f"Match: {result == expected or result2 == expected}")

if __name__ == "__main__":
    test_account_summary_generation()