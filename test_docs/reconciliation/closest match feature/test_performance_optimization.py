#!/usr/bin/env python3
"""
Performance test for optimized closest match functionality
Tests different dataset sizes and optimization techniques
"""

import pandas as pd
import numpy as np
import time
import sys
import os

# Add the backend app directory to Python path
sys.path.append('/Users/biswambarpradhan/UpSkill/ftt-ml/backend')

from app.services.reconciliation_service import OptimizedFileProcessor
from app.models.recon_models import ReconciliationRule

def generate_test_data(size_a: int, size_b: int, similarity_ratio: float = 0.3):
    """
    Generate test data with controlled similarity patterns
    
    Args:
        size_a: Number of records in file A
        size_b: Number of records in file B  
        similarity_ratio: Fraction of records that should have high similarity matches
    """
    
    print(f"🏭 Generating test data: {size_a:,} vs {size_b:,} records...")
    
    # Generate File A data
    np.random.seed(42)  # For reproducible results
    
    file_a_data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(size_a)],
        'customer_name': [f'Customer_{i % 1000:03d}' for i in range(size_a)],  # Reuse names for similarity
        'amount': np.random.uniform(100, 10000, size_a).round(2),
        'date': pd.date_range('2024-01-01', periods=size_a, freq='1H').strftime('%Y-%m-%d'),
        'account_number': [f'ACC{i % 500:03d}' for i in range(size_a)]  # Reuse account numbers
    }
    
    # Generate File B data with controlled similarity
    similar_count = int(size_b * similarity_ratio)
    different_count = size_b - similar_count
    
    # Similar records (modify transaction_id but keep other fields similar)
    similar_indices = np.random.choice(size_a, similar_count, replace=True)
    similar_data = {
        'ref_id': [f'REF{i:06d}' for i in similar_indices],  # Different ID format
        'client_name': [file_a_data['customer_name'][i] for i in similar_indices],  # Same names
        'value': [file_a_data['amount'][i] + np.random.uniform(-10, 10) for i in similar_indices],  # Similar amounts
        'transaction_date': [file_a_data['date'][i] for i in similar_indices],  # Same dates
        'acc_no': [file_a_data['account_number'][i] for i in similar_indices]  # Same account numbers
    }
    
    # Different records
    different_data = {
        'ref_id': [f'NEW{i:06d}' for i in range(different_count)],
        'client_name': [f'NewCustomer_{i:03d}' for i in range(different_count)],
        'value': np.random.uniform(100, 10000, different_count).round(2),
        'transaction_date': pd.date_range('2024-06-01', periods=different_count, freq='2H').strftime('%Y-%m-%d'),
        'acc_no': [f'NEWACC{i:03d}' for i in range(different_count)]
    }
    
    # Combine similar and different data
    file_b_data = {}
    for key in similar_data.keys():
        file_b_data[key] = list(similar_data[key]) + list(different_data[key])
    
    df_a = pd.DataFrame(file_a_data)
    df_b = pd.DataFrame(file_b_data)
    
    print(f"✅ Generated data with {similar_count:,} similar patterns and {different_count:,} different records")
    
    return df_a, df_b

def run_performance_test(size_a: int, size_b: int, test_name: str):
    """Run a performance test with given dataset sizes"""
    
    print(f"\n{'='*60}")
    print(f"🧪 PERFORMANCE TEST: {test_name}")
    print(f"📊 Dataset: {size_a:,} vs {size_b:,} records ({size_a*size_b:,} total comparisons)")
    print(f"{'='*60}")
    
    # Generate test data
    start_time = time.time()
    df_a, df_b = generate_test_data(size_a, size_b)
    data_gen_time = time.time() - start_time
    print(f"⏱️ Data generation: {data_gen_time:.2f}s")
    
    # Create reconciliation rules
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
    
    # Initialize processor
    processor = OptimizedFileProcessor()
    
    # Run reconciliation WITH closest matches
    print(f"🚀 Starting optimized closest match processing...")
    start_time = time.time()
    
    try:
        results = processor.reconcile_files_optimized(
            df_a=df_a,
            df_b=df_b,
            recon_rules=recon_rules,
            selected_columns_a=list(df_a.columns),
            selected_columns_b=list(df_b.columns),
            find_closest_matches=True
        )
        
        processing_time = time.time() - start_time
        
        # Analyze results
        matched_count = len(results['matched'])
        unmatched_a_count = len(results['unmatched_file_a'])
        unmatched_b_count = len(results['unmatched_file_b'])
        
        # Check if closest match columns exist
        has_closest_match_cols = all(col in results['unmatched_file_a'].columns 
                                   for col in ['closest_match_record', 'closest_match_score', 'closest_match_details'])
        
        # Calculate performance metrics
        records_per_second = (size_a + size_b) / processing_time if processing_time > 0 else 0
        comparisons_per_second = (size_a * size_b) / processing_time if processing_time > 0 else 0
        
        print(f"\n📈 RESULTS:")
        print(f"⏱️ Processing time: {processing_time:.2f}s")
        print(f"⚡ Records/second: {records_per_second:,.0f}")
        print(f"🔥 Comparisons/second: {comparisons_per_second:,.0f}")
        print(f"✅ Matched: {matched_count:,}")
        print(f"🔍 Unmatched A: {unmatched_a_count:,}")
        print(f"🔍 Unmatched B: {unmatched_b_count:,}")
        print(f"🎯 Closest match columns: {'✅ Present' if has_closest_match_cols else '❌ Missing'}")
        
        # Analyze closest match quality
        if has_closest_match_cols and unmatched_a_count > 0:
            scores = results['unmatched_file_a']['closest_match_score'].dropna()
            if len(scores) > 0:
                avg_score = scores.mean()
                high_scores = len(scores[scores > 80])
                print(f"📊 Avg similarity: {avg_score:.1f}%")
                print(f"🎯 High matches (>80%): {high_scores:,} / {len(scores):,}")
        
        # Performance rating
        if processing_time < 30:
            rating = "🚀 EXCELLENT"
        elif processing_time < 120:
            rating = "✅ GOOD"  
        elif processing_time < 300:
            rating = "⚠️ ACCEPTABLE"
        else:
            rating = "❌ NEEDS OPTIMIZATION"
        
        print(f"🏆 Performance: {rating}")
        
        return {
            'success': True,
            'processing_time': processing_time,
            'records_per_second': records_per_second,
            'matched_count': matched_count,
            'unmatched_a_count': unmatched_a_count,
            'closest_match_available': has_closest_match_cols,
            'rating': rating
        }
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        return {'success': False, 'error': str(e)}

def main():
    """Run a suite of performance tests"""
    
    print("🧪 CLOSEST MATCH OPTIMIZATION PERFORMANCE TESTS")
    print("=" * 60)
    
    test_cases = [
        # (size_a, size_b, test_name)
        (100, 100, "Small Dataset Baseline"),
        (1_000, 1_000, "Medium Dataset"),  
        (5_000, 5_000, "Large Dataset"),
        (10_000, 10_000, "Very Large Dataset"),
        (50_000, 20_000, "Asymmetric Large Dataset"),
        # (100_000, 100_000, "Extreme Dataset - 10B Comparisons"),  # Uncomment to test your actual use case
    ]
    
    results = []
    
    for size_a, size_b, test_name in test_cases:
        result = run_performance_test(size_a, size_b, test_name)
        results.append({
            'test_name': test_name,
            'size_a': size_a,
            'size_b': size_b,
            **result
        })
        
        # Break early if a test fails dramatically
        if not result['success']:
            print(f"⚠️ Stopping tests due to failure in {test_name}")
            break
    
    # Summary report
    print(f"\n{'='*60}")
    print("📋 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        if result['success']:
            print(f"🧪 {result['test_name']}")
            print(f"   📊 Size: {result['size_a']:,} × {result['size_b']:,}")
            print(f"   ⏱️ Time: {result['processing_time']:.1f}s")
            print(f"   ⚡ Rate: {result['records_per_second']:,.0f} records/sec")
            print(f"   🏆 {result['rating']}")
        else:
            print(f"❌ {result['test_name']}: FAILED")
    
    print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
    print(f"• For datasets > 50k×50k: Use batch processing automatically kicks in")
    print(f"• For datasets > 100k×100k: Consider sampling or filtering strategies")
    print(f"• Monitor memory usage during large dataset processing")
    print(f"• Use early termination thresholds to skip poor matches")

if __name__ == "__main__":
    main()