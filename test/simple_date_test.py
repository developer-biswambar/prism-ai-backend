#!/usr/bin/env python3
# Simple date testing

import pandas as pd
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.date_utils import normalize_date_value

def test_sample_data():
    """Test date normalization on sample data"""
    
    print("Testing date normalization...")
    
    # Create sample data that might appear in Excel
    sample_data = {
        'date_strings': [
            '2025-01-15', '2025-02-20', '2025-03-25'
        ],
        'date_slash': [
            '15/01/2025', '20/02/2025', '25/03/2025'  
        ],
        'excel_serial': [
            45672, 45708, 45741  # Excel serials
        ],
        'mixed_dates': [
            '15-Jan-2025', '20-Feb-2025', '25-Mar-2025'
        ],
        'datetime_objects': [
            pd.Timestamp('2025-01-15 10:30:00'),
            pd.Timestamp('2025-02-20 14:15:00'), 
            pd.Timestamp('2025-03-25 18:45:00')
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal dtypes:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
        
    print("\nTesting normalize_date_value on individual values:")
    for col in df.columns:
        print(f"\n{col}:")
        for val in df[col]:
            result = normalize_date_value(val)
            print(f"  {val} -> {result}")
    
    # Test conversion of each column
    print("\nConverting columns:")
    for col in df.columns:
        print(f"\n{col}:")
        converted_series = df[col].apply(lambda x: normalize_date_value(x) if pd.notna(x) else None)
        print(f"  Converted: {converted_series.tolist()}")
        
        # Check if all values were successfully converted to date format
        non_null_values = converted_series.dropna()
        if len(non_null_values) > 0:
            successful_conversions = sum(1 for val in non_null_values if val and len(str(val)) == 10 and str(val).count('-') == 2)
            success_rate = successful_conversions / len(non_null_values) * 100
            print(f"  Success rate: {success_rate:.1f}% ({successful_conversions}/{len(non_null_values)})")

if __name__ == "__main__":
    test_sample_data()