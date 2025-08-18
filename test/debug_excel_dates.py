#!/usr/bin/env python3
# Debug Excel serial date conversions

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.date_utils import normalize_date_value

def test_excel_serials():
    """Test Excel serial date conversions to find correct mappings"""
    
    test_serials = [
        1, 2, 59, 60, 61,  # Early dates around Excel bug
        43831,  # Should be around 2020
        44927,  # Should be around 2022
        45658,  # Should be around 2025
    ]
    
    print("Excel Serial Date Testing:")
    print("=" * 40)
    
    for serial in test_serials:
        result = normalize_date_value(serial)
        print(f"Serial {serial:5d} -> {result}")
    
    # Also test some known date conversions
    print("\nKnown Date Conversions:")
    print("=" * 40)
    
    known_dates = [
        "2020-01-15",
        "2022-12-31", 
        "2023-01-01",
        "2025-01-15",
    ]
    
    for date_str in known_dates:
        # Convert to datetime first
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Calculate Excel serial manually
        excel_epoch = datetime(1899, 12, 30)  # Excel's epoch accounting for bug
        days_diff = (dt - excel_epoch).days
        
        print(f"Date {date_str} should be Excel serial: {days_diff}")
        
        # Test our function
        result = normalize_date_value(days_diff)
        print(f"  Our function {days_diff} -> {result}")
        
        # Verify round trip
        parsed = normalize_date_value(date_str)
        print(f"  String parse '{date_str}' -> {parsed}")
        print()

if __name__ == "__main__":
    test_excel_serials()