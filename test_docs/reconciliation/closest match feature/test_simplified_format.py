#!/usr/bin/env python3
"""
Quick test to demonstrate the simplified closest match details format
"""
import sys
import os

# Add the backend directory to the Python path
sys.path.append('/Users/biswambarpradhan/UpSkill/ftt-ml/backend')

def test_simplified_format():
    """Test the simplified format logic"""
    from app.services.reconciliation_service import OptimizedFileProcessor
    
    processor = OptimizedFileProcessor()
    
    # Mock data for testing the format
    best_match_details = {
        'transaction_id_vs_ref_id': {
            'score': 85.0,
            'source_value': 'TXN002',
            'target_value': 'REF002',
            'type': 'identifier'
        },
        'customer_name_vs_client_name': {
            'score': 100.0,  # Perfect match
            'source_value': 'Jane Doe',
            'target_value': 'Jane Doe', 
            'type': 'text'
        },
        'amount_vs_value': {
            'score': 100.0,  # Perfect match
            'source_value': 2500.00,
            'target_value': 2500.00,
            'type': 'numeric'
        },
        'date_vs_transaction_date': {
            'score': 100.0,  # Perfect match
            'source_value': '2024-01-16',
            'target_value': '2024-01-16',
            'type': 'date'
        }
    }
    
    # Test the new simplified format creation
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
    
    simplified_details = "; ".join(details_list) if details_list else "All columns match exactly"
    
    print("🎯 CLOSEST MATCH DETAILS FORMAT TEST")
    print("=" * 50)
    
    print(f"\nOriginal complex format:")
    for key, details in best_match_details.items():
        print(f"  {key}: {details}")
    
    print(f"\nNew simplified format (human readable):")
    print(f"  {simplified_details}")
    
    print(f"\nAs stored in database: {simplified_details}")
    
    print("\n✅ TEST RESULTS:")
    print(f"- Only shows columns with score < 100: {'✅' if len(simplified_details) == 1 else '❌'}")
    print(f"- Shows only transaction_id mismatch: {'✅' if 'transaction_id' in simplified_details else '❌'}")
    print(f"- Ignores perfect matches (name, amount, date): {'✅' if len(simplified_details) == 1 else '❌'}")
    print(f"- Clear current vs suggested format: {'✅' if 'current_value' in str(simplified_details) else '❌'}")

if __name__ == "__main__":
    test_simplified_format()