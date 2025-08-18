#!/usr/bin/env python3
# Debug Excel file upload date detection

import sys
import io
import pandas as pd
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.date_utils import normalize_date_value
from app.routes.file_routes import normalize_datetime_columns

def debug_excel_columns(file_path=None):
    """Debug Excel file column types and date detection"""
    
    if not file_path:
        print("Usage: python debug_excel_upload.py [path_to_excel_file]")
        return
    
    try:
        # Read Excel file
        print(f"Reading Excel file: {file_path}")
        df = pd.read_excel(file_path, engine='openpyxl')
        
        print(f"\nOriginal DataFrame info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\nColumn types BEFORE normalization:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().head(3).tolist()
            print(f"  {col}: {dtype}")
            print(f"    Sample values: {sample_values}")
            
            # Test each sample with our date parser
            for i, val in enumerate(sample_values):
                parsed = normalize_date_value(val)
                print(f"      {val} -> {parsed}")
            print()
        
        # Apply our normalization
        print("Applying normalize_datetime_columns...")
        df_normalized = normalize_datetime_columns(df.copy())
        
        print(f"\nColumn types AFTER normalization:")
        for col in df_normalized.columns:
            dtype = str(df_normalized[col].dtype)
            sample_values = df_normalized[col].dropna().head(3).tolist()
            print(f"  {col}: {dtype}")
            print(f"    Sample values: {sample_values}")
            print()
        
        # Check for datetime columns that weren't converted
        datetime_cols = df_normalized.select_dtypes(include=['datetime64[ns]']).columns
        if len(datetime_cols) > 0:
            print(f"❌ WARNING: These columns still contain datetime objects:")
            for col in datetime_cols:
                print(f"   {col}: {df_normalized[col].dtype}")
                print(f"   Sample: {df_normalized[col].dropna().head(2).tolist()}")
        else:
            print("✅ SUCCESS: No datetime columns remain - all dates converted to strings")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_sample_excel_data():
    """Create and test sample Excel data with different date formats"""
    
    print("Creating sample Excel data for testing...")
    
    # Create sample data with various date formats
    sample_data = {
        'date_col_1': [
            '2025-01-15', '2025-02-20', '2025-03-25'
        ],
        'date_col_2': [
            '15/01/2025', '20/02/2025', '25/03/2025'  
        ],
        'excel_serial': [
            45672, 45708, 45741  # Excel serials for the dates above
        ],
        'mixed_dates': [
            '15-Jan-2025', '20-Feb-2025', '25-Mar-2025'
        ],
        'text_col': [
            'not a date', 'also not date', 'definitely not'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"Sample DataFrame before normalization:")
    print(df)
    print(f"\nColumn types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Apply normalization
    df_normalized = normalize_datetime_columns(df.copy())
    
    print(f"\nSample DataFrame after normalization:")
    print(df_normalized)
    print(f"\nColumn types after:")
    for col in df_normalized.columns:
        print(f"  {col}: {df_normalized[col].dtype}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_excel_columns(sys.argv[1])
    else:
        test_sample_excel_data()