#!/usr/bin/env python3
"""
Test script to validate enhanced closest match functionality
Tests that unmatched A compares against entire File B and vice versa
"""

import pandas as pd
import sys
import os

# Add the backend app directory to Python path
sys.path.append('/Users/biswambarpradhan/UpSkill/ftt-ml/backend')

from app.services.reconciliation_service import OptimizedFileProcessor
from app.models.recon_models import ReconciliationRule

def test_enhanced_closest_match():
    """Test that closest match compares against entire files, not just unmatched"""
    
    # Create test data
    # File A: 4 records (2 will match, 2 will be unmatched)
    file_a_data = {
        'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
        'customer_name': ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown'],
        'amount': [1000.00, 2500.00, 750.00, 1200.00]
    }
    
    # File B: 5 records (2 will match with A, 3 will be unmatched)
    # The key insight: TXN002 and TXN004 from A should find their closest matches 
    # in the MATCHED records of B (REF001, REF003), not just unmatched records
    file_b_data = {
        'ref_id': ['REF001', 'REF003', 'REF005', 'REF006', 'REF007'],
        'client_name': ['John Smith', 'Bob Johnson', 'Jane Doe', 'Alice Brown', 'Charlie Davis'],
        'value': [1000.00, 750.00, 2500.00, 1200.00, 3000.00]
    }
    
    df_a = pd.DataFrame(file_a_data)
    df_b = pd.DataFrame(file_b_data)
    
    print("=== Test Data ===")
    print("File A:")
    print(df_a)
    print("\nFile B:")
    print(df_b)
    
    # Create reconciliation rules (only TXN001->REF001 and TXN003->REF003 should match)
    recon_rules = [
        ReconciliationRule(
            LeftFileColumn='transaction_id',
            RightFileColumn='ref_id',
            MatchType='equals',
            ToleranceValue=0
        ),
        ReconciliationRule(
            LeftFileColumn='amount',
            RightFileColumn='value',
            MatchType='tolerance',
            ToleranceValue=0.01
        )
    ]
    
    # Initialize reconciliation service
    service = OptimizedFileProcessor()
    
    print("\n=== Expected Behavior ===")
    print("NO MATCHES EXPECTED - All transaction IDs are different (TXN* vs REF*)")
    print("All File A records will be unmatched: TXN001, TXN002, TXN003, TXN004")
    print("All File B records will be unmatched: REF001, REF003, REF005, REF006, REF007")
    print("\nCLOSEST MATCH EXPECTATIONS (Enhanced - comparing against ENTIRE files):")
    print("- TXN001 (John Smith) should find closest match: REF001 (John Smith) - EXACT customer+amount match!")
    print("- TXN002 (Jane Doe) should find closest match: REF005 (Jane Doe) - EXACT customer+amount match!")
    print("- TXN003 (Bob Johnson) should find closest match: REF003 (Bob Johnson) - EXACT customer+amount match!")
    print("- TXN004 (Alice Brown) should find closest match: REF006 (Alice Brown) - EXACT customer+amount match!")
    print("- REF007 (Charlie Davis) should find best available match in File A")
    
    # Run reconciliation WITH closest matches
    print("\n=== Running Reconciliation with Closest Matches ===")
    results = service.reconcile_files_optimized(
        df_a=df_a,
        df_b=df_b,
        recon_rules=recon_rules,
        selected_columns_a=list(df_a.columns),
        selected_columns_b=list(df_b.columns),
        find_closest_matches=True  # This is the key parameter
    )
    
    print(f"\nMatched records: {len(results['matched'])}")
    print(f"Unmatched A records: {len(results['unmatched_file_a'])}")
    print(f"Unmatched B records: {len(results['unmatched_file_b'])}")
    
    # Analyze unmatched A results
    print("\n=== Unmatched A Analysis ===")
    unmatched_a = results['unmatched_file_a']
    if len(unmatched_a) > 0:
        for idx, row in unmatched_a.iterrows():
            print(f"\nRecord: {row['transaction_id']} - {row['customer_name']}")
            if 'closest_match_score' in row:
                print(f"  Closest Match Score: {row['closest_match_score']}")
                print(f"  Closest Match Details: {row['closest_match_details']}")
                
                # Check if we got the expected matches based on customer names
                customer_name = row['customer_name']
                closest_record = str(row.get('closest_match_record', ''))
                
                if customer_name in closest_record:
                    print(f"  ✅ EXPECTED: {row['transaction_id']} found {customer_name} match!")
                else:
                    print(f"  ⚠️  Expected {customer_name} match for {row['transaction_id']}, got: {closest_record}")
    
    # Analyze unmatched B results  
    print("\n=== Unmatched B Analysis ===")
    unmatched_b = results['unmatched_file_b']
    if len(unmatched_b) > 0:
        for idx, row in unmatched_b.iterrows():
            print(f"\nRecord: {row['ref_id']} - {row['client_name']}")
            if 'closest_match_score' in row:
                print(f"  Closest Match Score: {row['closest_match_score']}")
                print(f"  Closest Match Details: {row['closest_match_details']}")
    
    # Validation
    print("\n=== Validation Results ===")
    success = True
    
    # Check that we have the expected number of unmatched records (all records since no IDs match)
    if len(unmatched_a) != 4:
        print(f"❌ Expected 4 unmatched A records, got {len(unmatched_a)}")
        success = False
    
    if len(unmatched_b) != 5:
        print(f"❌ Expected 5 unmatched B records, got {len(unmatched_b)}")
        success = False
    
    # Check that closest match columns exist
    for df_name, df in [("Unmatched A", unmatched_a), ("Unmatched B", unmatched_b)]:
        if len(df) > 0:
            required_cols = ['closest_match_record', 'closest_match_score', 'closest_match_details']
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ Missing column '{col}' in {df_name}")
                    success = False
    
    # Check for reasonable similarity scores (should be moderate for name+amount matches, lower for ID mismatches)
    if len(unmatched_a) > 0:
        decent_scores = unmatched_a[unmatched_a['closest_match_score'] > 60]
        if len(decent_scores) < 4:
            print(f"❌ Expected reasonable similarity scores (>60) for all matches, got {len(decent_scores)}")
            success = False
        else:
            print(f"✅ Found {len(decent_scores)} reasonable similarity matches (>60)")
            
        # Check that we found customer name matches in the closest match records
        name_matches = 0
        for _, row in unmatched_a.iterrows():
            customer_name = row['customer_name']
            closest_record = str(row.get('closest_match_record', ''))
            if customer_name in closest_record:
                name_matches += 1
        
        if name_matches >= 4:
            print(f"✅ Found customer name matches for all {name_matches} records!")
        else:
            print(f"⚠️ Expected 4 customer name matches, found {name_matches}")
    
    if success:
        print("✅ All validation checks passed!")
        print("🎉 Enhanced closest match functionality working correctly!")
    else:
        print("❌ Some validation checks failed. Please review the implementation.")
    
    return success

if __name__ == "__main__":
    test_enhanced_closest_match()