#!/usr/bin/env python3
# test/generate_date_test_data.py - Generate comprehensive test data for date utilities

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_comprehensive_test_data():
    """Generate comprehensive test data covering all edge cases"""
    
    test_data = {
        "valid_dates": [],
        "invalid_dates": [],
        "edge_cases": [],
        "format_variations": [],
        "performance_test_data": []
    }
    
    # Generate valid dates in various formats
    base_dates = [
        datetime(2025, 1, 15),
        datetime(2024, 2, 29),  # Leap year
        datetime(2023, 12, 31), # Year end
        datetime(2020, 1, 1),   # Year start
        datetime(1900, 1, 1),   # Min boundary
        datetime(9999, 12, 31), # Max boundary
    ]
    
    formats = [
        "%Y-%m-%d",           # ISO format
        "%d/%m/%Y",           # DD/MM/YYYY
        "%m/%d/%Y",           # MM/DD/YYYY
        "%d-%m-%Y",           # DD-MM-YYYY
        "%d.%m.%Y",           # DD.MM.YYYY
        "%d %b %Y",           # DD Mon YYYY
        "%d-%b-%Y",           # DD-Mon-YYYY
        "%b %d, %Y",          # Mon DD, YYYY
        "%B %d, %Y",          # Month DD, YYYY
        "%Y%m%d",             # Compact YYYYMMDD
        "%d/%m/%y",           # 2-digit year
        "%Y-%m-%d %H:%M:%S",  # With time
        "%d/%m/%Y %H:%M",     # With time
    ]
    
    # Generate valid date strings
    for date_obj in base_dates:
        for fmt in formats:
            try:
                date_str = date_obj.strftime(fmt)
                test_data["valid_dates"].append({
                    "input": date_str,
                    "expected_date": date_obj.strftime("%Y-%m-%d"),
                    "format": fmt,
                    "description": f"Standard format {fmt}"
                })
            except:
                pass  # Some formats may not work with all dates
    
    # Add Excel serial dates
    excel_dates = [
        (1, "1900-01-01"),
        (60, "1900-02-28"),  # Before Excel bug
        (61, "1900-03-01"),  # After Excel bug
        (43831, "2020-01-15"),
        (44927, "2022-12-31"),
        (45658, "2025-01-15"),
        (2958465, "9999-12-31"),
    ]
    
    for serial, expected in excel_dates:
        test_data["valid_dates"].append({
            "input": serial,
            "expected_date": expected,
            "format": "excel_serial",
            "description": f"Excel serial date {serial}"
        })
    
    # Generate invalid dates
    invalid_cases = [
        # Format issues
        ("32/01/2025", "invalid_day", "Day 32 doesn't exist"),
        ("01/32/2025", "invalid_day", "Day 32 doesn't exist"),
        ("2025-13-01", "invalid_month", "Month 13 doesn't exist"),
        ("2025-02-30", "invalid_day_for_month", "Feb 30 doesn't exist"),
        ("2023-02-29", "invalid_leap_day", "2023 is not leap year"),
        
        # Wrong formats
        ("not-a-date", "wrong_format", "Not a date string"),
        ("2025_01_15", "wrong_separator", "Wrong separator"),
        ("hello world", "random_text", "Random text"),
        ("12345abc", "mixed_text_number", "Mixed text and numbers"),
        
        # Out of range numbers
        (0, "excel_serial_too_low", "Excel serial below valid range"),
        (-1, "negative_number", "Negative number"),
        (2958466, "excel_serial_too_high", "Excel serial above valid range"),
        
        # Special values
        (None, "none_value", "None value"),
        ("", "empty_string", "Empty string"),
        ("   ", "whitespace_only", "Whitespace only"),
        
        # Unicode and special characters
        ("२०२५-०१-१५", "non_latin_numerals", "Non-Latin numerals"),
        ("2025‑01‑15", "unicode_hyphen", "Unicode hyphen"),
        ("2025*01*15", "wrong_separator_asterisk", "Wrong separator"),
    ]
    
    for input_val, error_type, description in invalid_cases:
        test_data["invalid_dates"].append({
            "input": input_val,
            "error_type": error_type,
            "description": description
        })
    
    # Generate edge cases
    edge_cases = [
        # Leap year cases
        ("2024-02-29", "leap_year_valid", "Valid leap day"),
        ("2023-02-29", "leap_year_invalid", "Invalid leap day"),
        ("2000-02-29", "century_leap_year", "Century leap year"),
        ("1900-02-29", "century_non_leap_year", "Century non-leap year"),
        
        # Boundary dates
        ("1900-01-01", "min_year", "Minimum year"),
        ("9999-12-31", "max_year", "Maximum year"),
        ("2025-01-01", "year_start", "Start of year"),
        ("2025-12-31", "year_end", "End of year"),
        
        # Month boundaries
        ("2025-01-31", "jan_end", "End of January"),
        ("2025-02-28", "feb_end_non_leap", "End of February (non-leap)"),
        ("2024-02-29", "feb_end_leap", "End of February (leap)"),
        ("2025-04-30", "april_end", "End of April (30 days)"),
        
        # Time components (should be ignored)
        ("2025-01-15 00:00:00", "midnight", "Midnight time"),
        ("2025-01-15 23:59:59", "almost_midnight", "Almost midnight"),
        ("2025-01-15 12:30:45.123", "with_milliseconds", "With milliseconds"),
        
        # Different number types
        (45658.0, "float_excel_serial", "Float Excel serial"),
        (45658.5, "float_with_time", "Float with time component"),
    ]
    
    for input_val, case_type, description in edge_cases:
        test_data["edge_cases"].append({
            "input": input_val,
            "case_type": case_type,
            "description": description
        })
    
    # Generate format variations for same date
    test_date = datetime(2025, 7, 15)  # Use July to avoid MM/DD vs DD/MM confusion
    
    format_variations = [
        test_date.strftime("%Y-%m-%d"),      # 2025-07-15
        test_date.strftime("%d/%m/%Y"),      # 15/07/2025
        test_date.strftime("%m/%d/%Y"),      # 07/15/2025
        test_date.strftime("%d-%b-%Y"),      # 15-Jul-2025
        test_date.strftime("%b %d, %Y"),     # Jul 15, 2025
        test_date.strftime("%B %d, %Y"),     # July 15, 2025
        test_date.strftime("%d %B %Y"),      # 15 July 2025
        test_date.strftime("%Y%m%d"),        # 20250715
        test_date.strftime("%d.%m.%Y"),      # 15.07.2025
        test_date.strftime("%Y-%m-%d %H:%M:%S").replace("00:00:00", "14:30:45"),  # With time
        45671,  # Excel serial for 2025-07-15
    ]
    
    test_data["format_variations"] = {
        "expected_date": "2025-07-15",
        "variations": [
            {"input": var, "description": f"Format variation {i+1}"} 
            for i, var in enumerate(format_variations)
        ]
    }
    
    # Generate performance test data
    performance_data = []
    for i in range(1000):
        # Random valid dates
        random_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1825))  # 5 years
        formats_to_use = random.sample(formats[:8], 3)  # Pick 3 random formats
        
        for fmt in formats_to_use:
            try:
                performance_data.append(random_date.strftime(fmt))
            except:
                pass
        
        # Add some Excel serials
        if i % 10 == 0:
            excel_serial = random.randint(43831, 47482)  # 2020-2030 range
            performance_data.append(excel_serial)
        
        # Add some invalid data
        if i % 20 == 0:
            invalid_samples = ["invalid", None, "not-date", random.randint(0, 100)]
            performance_data.append(random.choice(invalid_samples))
    
    test_data["performance_test_data"] = performance_data
    
    return test_data

def save_test_data():
    """Save generated test data to files"""
    
    test_data = generate_comprehensive_test_data()
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = test_dir / "date_test_data.json"
    with open(json_file, 'w') as f:
        json.dump(test_data, f, indent=2, default=str)
    
    # Save valid dates as CSV
    csv_file = test_dir / "valid_dates.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["input", "expected_date", "format", "description"])
        writer.writeheader()
        writer.writerows(test_data["valid_dates"])
    
    # Save invalid dates as CSV
    csv_file = test_dir / "invalid_dates.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["input", "error_type", "description"])
        writer.writeheader()
        writer.writerows(test_data["invalid_dates"])
    
    # Save performance data as text
    perf_file = test_dir / "performance_data.txt"
    with open(perf_file, 'w') as f:
        for item in test_data["performance_test_data"]:
            f.write(f"{item}\n")
    
    print(f"✅ Test data generated and saved to {test_dir}")
    print(f"   - {len(test_data['valid_dates'])} valid date test cases")
    print(f"   - {len(test_data['invalid_dates'])} invalid date test cases")
    print(f"   - {len(test_data['edge_cases'])} edge case test cases")
    print(f"   - {len(test_data['format_variations']['variations'])} format variations")
    print(f"   - {len(test_data['performance_test_data'])} performance test cases")
    
    return test_data

def print_test_summary():
    """Print a summary of test data for manual verification"""
    
    test_data = generate_comprehensive_test_data()
    
    print("📋 DATE UTILS TEST DATA SUMMARY")
    print("=" * 50)
    
    print(f"\n✅ Valid Dates ({len(test_data['valid_dates'])} cases):")
    for i, case in enumerate(test_data['valid_dates'][:5]):  # Show first 5
        print(f"   {i+1}. {case['input']} -> {case['expected_date']} ({case['format']})")
    if len(test_data['valid_dates']) > 5:
        print(f"   ... and {len(test_data['valid_dates']) - 5} more")
    
    print(f"\n❌ Invalid Dates ({len(test_data['invalid_dates'])} cases):")
    for i, case in enumerate(test_data['invalid_dates'][:5]):
        print(f"   {i+1}. {case['input']} -> {case['error_type']} ({case['description']})")
    if len(test_data['invalid_dates']) > 5:
        print(f"   ... and {len(test_data['invalid_dates']) - 5} more")
    
    print(f"\n🔍 Edge Cases ({len(test_data['edge_cases'])} cases):")
    for i, case in enumerate(test_data['edge_cases'][:5]):
        print(f"   {i+1}. {case['input']} -> {case['case_type']} ({case['description']})")
    if len(test_data['edge_cases']) > 5:
        print(f"   ... and {len(test_data['edge_cases']) - 5} more")
    
    print(f"\n🔄 Format Variations (same date: {test_data['format_variations']['expected_date']}):")
    for i, var in enumerate(test_data['format_variations']['variations'][:5]):
        print(f"   {i+1}. {var['input']}")
    if len(test_data['format_variations']['variations']) > 5:
        print(f"   ... and {len(test_data['format_variations']['variations']) - 5} more")
    
    print(f"\n⚡ Performance Data: {len(test_data['performance_test_data'])} mixed cases for stress testing")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "summary":
            print_test_summary()
        elif sys.argv[1] == "generate":
            save_test_data()
        else:
            print("Usage: python generate_date_test_data.py [summary|generate]")
    else:
        # Default: show summary and save data
        print_test_summary()
        print("\n" + "="*50)
        save_test_data()