#!/usr/bin/env python3
# Calculate correct Excel serial dates

from datetime import datetime

def calculate_excel_serial(year, month, day):
    """Calculate Excel serial date for a given date"""
    target_date = datetime(year, month, day)
    
    # Excel epoch accounting for leap year bug
    excel_epoch = datetime(1899, 12, 30)  # Excel's actual epoch
    
    days_diff = (target_date - excel_epoch).days
    return days_diff

def main():
    """Calculate correct Excel serials for test cases"""
    
    test_dates = [
        (1900, 1, 1),   # Excel serial 1
        (1900, 2, 28),  # Before Excel bug (59)
        (1900, 3, 1),   # After Excel bug (61)
        (2020, 1, 15),  # Should be around 43831
        (2022, 12, 31), # Should be around 44927
        (2025, 1, 15),  # Should be around 45658
        (2025, 7, 15),  # For format variation tests
    ]
    
    print("Excel Serial Date Calculations:")
    print("=" * 40)
    
    for year, month, day in test_dates:
        serial = calculate_excel_serial(year, month, day)
        print(f"{year}-{month:02d}-{day:02d} -> Excel serial {serial}")
    
    # Also show the reverse - what dates correspond to test serials
    print("\nReverse Lookup:")
    print("=" * 40)
    
    test_serials = [1, 59, 60, 61, 43831, 44927, 45658]
    excel_epoch = datetime(1899, 12, 30)
    
    for serial in test_serials:
        calculated_date = excel_epoch + pd.Timedelta(days=serial)
        print(f"Excel serial {serial} -> {calculated_date.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    import pandas as pd
    main()