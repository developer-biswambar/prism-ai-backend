#!/usr/bin/env python3
# test/run_date_utils_tests.py - Test runner for comprehensive date utilities tests

import sys
import os
import subprocess
import time
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def run_tests():
    """Run the comprehensive date utilities tests"""
    
    print("🧪 Running Comprehensive Date Utils Tests")
    print("=" * 50)
    
    test_file = Path(__file__).parent / "test_date_utils_comprehensive.py"
    
    # Test categories to run
    test_categories = [
        ("Basic Functionality", [
            "test_none_and_null_values",
            "test_datetime_and_timestamp_objects", 
            "test_excel_serial_dates",
            "test_invalid_excel_serial_numbers"
        ]),
        ("Date Format Parsing", [
            "test_standard_date_formats",
            "test_month_name_formats", 
            "test_two_digit_years",
            "test_datetime_with_time_components"
        ]),
        ("Edge Cases & Boundaries", [
            "test_leap_years",
            "test_edge_date_boundaries",
            "test_invalid_dates",
            "test_caching_behavior"
        ]),
        ("Date Detection", [
            "test_is_date_value_positive_cases",
            "test_is_date_value_negative_cases"
        ]),
        ("Date Comparison", [
            "test_date_equals_match_positive_cases", 
            "test_date_equals_match_negative_cases"
        ]),
        ("Performance & Stress", [
            "test_performance_with_large_dataset",
            "test_unicode_and_special_characters",
            "test_extremely_large_and_small_values",
            "test_mixed_data_types"
        ]),
        ("Real-World Scenarios", [
            "test_real_world_excel_dates",
            "test_csv_import_scenarios", 
            "test_database_date_scenarios"
        ])
    ]
    
    total_passed = 0
    total_failed = 0
    failed_tests = []
    
    for category_name, test_methods in test_categories:
        print(f"\n📂 {category_name}")
        print("-" * 30)
        
        category_passed = 0
        category_failed = 0
        
        for test_method in test_methods:
            try:
                # Run individual test
                cmd = [
                    sys.executable, "-m", "pytest", 
                    f"{test_file}::TestDateUtilsComprehensive::{test_method}",
                    "-v", "--tb=short", "--no-header", "-q"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=backend_dir)
                
                if result.returncode == 0:
                    print(f"  ✅ {test_method}")
                    category_passed += 1
                    total_passed += 1
                else:
                    print(f"  ❌ {test_method}")
                    print(f"     Error: {result.stdout.strip()}")
                    if result.stderr:
                        print(f"     Stderr: {result.stderr.strip()}")
                    category_failed += 1
                    total_failed += 1
                    failed_tests.append(f"{category_name}::{test_method}")
                    
            except Exception as e:
                print(f"  💥 {test_method} - Exception: {str(e)}")
                category_failed += 1
                total_failed += 1
                failed_tests.append(f"{category_name}::{test_method}")
        
        print(f"  📊 Category: {category_passed} passed, {category_failed} failed")
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 FINAL RESULTS")
    print("=" * 50)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"🎯 Success Rate: {(total_passed/(total_passed + total_failed)*100):.1f}%")
    
    if failed_tests:
        print(f"\n💥 Failed Tests:")
        for failed_test in failed_tests:
            print(f"   - {failed_test}")
        
        print(f"\n🔧 To debug a specific test, run:")
        print(f"   pytest {test_file}::TestDateUtilsComprehensive::test_name -v -s")
    
    return total_failed == 0

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    
    print("🚀 Running Quick Smoke Test")
    print("-" * 30)
    
    try:
        # Import the modules
        from app.utils.date_utils import normalize_date_value, is_date_value, check_date_equals_match
        from datetime import datetime
        
        # Test basic functionality
        test_cases = [
            # normalize_date_value tests
            ("2025-01-15", datetime(2025, 1, 15, 0, 0, 0)),
            ("15-Jan-2025", datetime(2025, 1, 15, 0, 0, 0)),
            (45658, datetime(2025, 1, 15, 0, 0, 0)),  # Excel serial
            ("invalid", None),
        ]
        
        print("Testing normalize_date_value:")
        for input_val, expected in test_cases:
            result = normalize_date_value(input_val)
            status = "✅" if result == expected else "❌"
            print(f"  {status} {input_val} -> {result}")
        
        # is_date_value tests
        print("\nTesting is_date_value:")
        date_detection_cases = [
            ("2025-01-15", True),
            ("15-Jan-2025", True), 
            (45658, True),
            ("invalid", False),
            (None, False),
        ]
        
        for input_val, expected in date_detection_cases:
            result = is_date_value(input_val)
            status = "✅" if result == expected else "❌"
            print(f"  {status} {input_val} -> {result}")
        
        # check_date_equals_match tests
        print("\nTesting check_date_equals_match:")
        match_cases = [
            ("2025-01-15", "15-Jan-2025", True),
            ("2025-01-15", "2025-01-16", False),
            (None, None, True),
            ("2025-01-15", None, False),
        ]
        
        for val_a, val_b, expected in match_cases:
            result = check_date_equals_match(val_a, val_b)
            status = "✅" if result == expected else "❌"
            print(f"  {status} {val_a} == {val_b} -> {result}")
        
        print("\n🎉 Smoke test completed!")
        return True
        
    except Exception as e:
        print(f"💥 Smoke test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Run quick smoke test only
        success = run_quick_smoke_test()
    else:
        # Run smoke test first
        print("Step 1: Quick smoke test")
        smoke_success = run_quick_smoke_test()
        
        if not smoke_success:
            print("❌ Smoke test failed. Skipping comprehensive tests.")
            return False
        
        print(f"\nStep 2: Comprehensive test suite")
        success = run_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)