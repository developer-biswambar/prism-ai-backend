#!/usr/bin/env python3
# Test Excel date normalization comprehensive

import pandas as pd
import io
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.date_utils import normalize_date_value

def simulate_normalize_datetime_columns(df):
    """Simulate the enhanced normalize_datetime_columns function"""
    
    print(f"🔍 Testing normalize_datetime_columns simulation...")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input column types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Simulate the enhanced logic
    all_columns = list(df.columns)
    datetime_columns_detected = df.select_dtypes(include=['datetime64[ns]']).columns
    converted_date_columns = []
    
    print(f"\nChecking all {len(all_columns)} columns for date content...")
    if len(datetime_columns_detected) > 0:
        print(f"  - Pandas auto-detected {len(datetime_columns_detected)} datetime columns: {list(datetime_columns_detected)}")

    for col in all_columns:
        # Sample some non-null values to check if they look like dates
        non_null_values = df[col].dropna()
        if len(non_null_values) == 0:
            print(f"  - Skipping empty column '{col}'")
            continue

        # Test a sample of values using the robust date parser
        sample_size = min(20, len(non_null_values))
        sample_values = non_null_values.head(sample_size).tolist()

        date_like_count = 0
        print(f"  - Testing column '{col}' ({df[col].dtype}):")
        for i, value in enumerate(sample_values[:5]):  # Show first 5 for debugging
            parsed_date_str = normalize_date_value(value)
            success = parsed_date_str is not None
            if success:
                date_like_count += 1
            print(f"    [{i+1}] {value} -> {parsed_date_str} {'✅' if success else '❌'}")
        
        # Count all samples
        for value in sample_values:
            parsed_date_str = normalize_date_value(value)
            if parsed_date_str is not None:
                date_like_count += 1

        # Use 70% threshold for conservative date detection
        detection_threshold = 0.7
        success_rate = date_like_count / sample_size
        print(f"    Success rate: {success_rate*100:.1f}% ({date_like_count}/{sample_size})")
        
        if success_rate >= detection_threshold:
            try:
                original_dtype = str(df[col].dtype)
                print(f"📅 Converting column '{col}' to normalized dates")

                # Apply the robust date parser to the entire column
                def convert_to_date_string(value):
                    if pd.isna(value):
                        return None
                    parsed_date_str = normalize_date_value(value)
                    if parsed_date_str is not None:
                        return parsed_date_str  # Already in YYYY-MM-DD format
                    return str(value)  # Convert to string if not parseable as date
                
                df[col] = df[col].apply(convert_to_date_string)
                converted_date_columns.append(col)
                print(f"  ✅ Successfully converted '{col}' from {original_dtype} to YYYY-MM-DD strings")

            except Exception as e:
                print(f"  ❌ Failed to convert column '{col}' to dates: {str(e)}")
        else:
            print(f"  - Skipping '{col}': only {success_rate*100:.1f}% samples are date-like")

    if converted_date_columns:
        print(f"\n🎉 Successfully normalized {len(converted_date_columns)} columns: {converted_date_columns}")
    else:
        print(f"\nℹ️  No date columns detected for normalization")
    
    # Final validation - check if any datetime columns still exist
    remaining_datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    if len(remaining_datetime_cols) > 0:
        print(f"\n⚠️  WARNING: {len(remaining_datetime_cols)} datetime columns still exist: {list(remaining_datetime_cols)}")
        # Force convert any remaining datetime columns
        for col in remaining_datetime_cols:
            df[col] = df[col].apply(lambda x: normalize_date_value(x) if pd.notna(x) else None)
            print(f"  🔧 Force-converted remaining datetime column: {col}")
    else:
        print(f"\n✅ SUCCESS: All datetime objects converted to YYYY-MM-DD strings")
    
    return df

def test_comprehensive_excel_scenarios():
    """Test various Excel date scenarios"""
    
    print("=" * 60)
    print("COMPREHENSIVE EXCEL DATE NORMALIZATION TEST")
    print("=" * 60)
    
    # Create comprehensive test data that simulates Excel file scenarios
    test_data = {
        # Scenario 1: String dates in various formats
        'string_iso_dates': ['2025-01-15', '2025-02-20', '2025-03-25', None],
        'string_slash_dates': ['15/01/2025', '20/02/2025', '25/03/2025', ''],
        'string_dash_dates': ['15-Jan-2025', '20-Feb-2025', '25-Mar-2025', 'invalid'],
        
        # Scenario 2: Excel serial numbers (common in Excel exports)
        'excel_serials': [45672, 45708, 45741, 0],  # 0 should not be treated as date
        'excel_serials_float': [45672.5, 45708.25, 45741.75, None],
        
        # Scenario 3: Pandas datetime objects (auto-detected by pandas)
        'pandas_datetime': [
            pd.Timestamp('2025-01-15 10:30:00'),
            pd.Timestamp('2025-02-20 14:15:00'), 
            pd.Timestamp('2025-03-25 18:45:00'),
            pd.NaT
        ],
        
        # Scenario 4: Mixed types that shouldn't be dates
        'mixed_non_dates': ['Product A', 'Product B', 'Product C', 'Product D'],
        'numeric_ids': [1001, 1002, 1003, 1004],
        'amounts': [100.50, 200.75, 300.25, 400.00],
        
        # Scenario 5: Edge cases
        'mostly_dates_some_text': ['2025-01-15', '2025-02-20', 'N/A', '2025-03-25'],
        'edge_case_numbers': [1, 2, 45672, 45708]  # Mix of small and Excel serial numbers
    }
    
    df = pd.DataFrame(test_data)
    
    print(f"BEFORE normalization:")
    print(f"Shape: {df.shape}")
    print(df)
    print(f"\nColumn types BEFORE:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
        sample_vals = df[col].dropna().head(3).tolist()
        print(f"    Samples: {sample_vals}")
    
    # Apply normalization
    print(f"\n" + "="*60)
    df_normalized = simulate_normalize_datetime_columns(df.copy())
    print("="*60)
    
    print(f"\nAFTER normalization:")
    print(df_normalized)
    print(f"\nColumn types AFTER:")
    for col in df_normalized.columns:
        print(f"  {col}: {df_normalized[col].dtype}")
        sample_vals = df_normalized[col].dropna().head(3).tolist()
        print(f"    Samples: {sample_vals}")
    
    # Validate results
    print(f"\n" + "="*60)
    print("VALIDATION RESULTS:")
    print("="*60)
    
    datetime_cols_remaining = df_normalized.select_dtypes(include=['datetime64[ns]']).columns
    if len(datetime_cols_remaining) == 0:
        print("✅ SUCCESS: No datetime64 columns remaining")
    else:
        print(f"❌ FAILURE: {len(datetime_cols_remaining)} datetime64 columns still exist: {list(datetime_cols_remaining)}")
    
    # Check which columns were successfully converted to date strings
    date_string_columns = []
    for col in df_normalized.columns:
        non_null_vals = df_normalized[col].dropna()
        if len(non_null_vals) > 0:
            # Check if values look like YYYY-MM-DD format
            sample_val = str(non_null_vals.iloc[0])
            if len(sample_val) == 10 and sample_val.count('-') == 2:
                try:
                    datetime.strptime(sample_val, '%Y-%m-%d')
                    date_string_columns.append(col)
                except:
                    pass
    
    if date_string_columns:
        print(f"✅ CONVERTED TO YYYY-MM-DD: {date_string_columns}")
    
    # Test specific scenarios
    scenarios_to_check = [
        ('string_iso_dates', 'Should remain YYYY-MM-DD'),
        ('string_slash_dates', 'Should convert DD/MM/YYYY to YYYY-MM-DD'),
        ('string_dash_dates', 'Should convert DD-Mon-YYYY to YYYY-MM-DD'),
        ('excel_serials', 'Should convert Excel serial numbers to YYYY-MM-DD'),
        ('pandas_datetime', 'Should convert datetime objects to YYYY-MM-DD'),
        ('mixed_non_dates', 'Should NOT be converted (not dates)'),
        ('numeric_ids', 'Should NOT be converted (not in Excel serial range)')
    ]
    
    for col_name, expectation in scenarios_to_check:
        if col_name in df_normalized.columns:
            sample_vals = df_normalized[col_name].dropna().head(2).tolist()
            print(f"📋 {col_name}: {sample_vals} - {expectation}")

if __name__ == "__main__":
    test_comprehensive_excel_scenarios()