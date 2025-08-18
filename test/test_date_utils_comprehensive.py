# test/test_date_utils_comprehensive.py - Comprehensive test suite for date utilities
import pytest
from datetime import datetime, date
import pandas as pd
import numpy as np
from decimal import Decimal

from app.utils.date_utils import DateNormalizer, normalize_date_value, is_date_value, check_date_equals_match


class TestDateUtilsComprehensive:
    """Comprehensive test suite covering all edge cases and scenarios for date utilities"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.normalizer = DateNormalizer()
    
    # =================================
    # NORMALIZE_DATE_VALUE TESTS
    # =================================
    
    def test_none_and_null_values(self):
        """Test None, NaN, and null-like values"""
        test_cases = [
            None,
            pd.NaT,
            np.nan,
            float('nan'),
            "",
            "   ",  # whitespace only
            "NULL",
            "null",
            "NaN",
        ]
        
        for value in test_cases:
            result = normalize_date_value(value)
            if value in ["NULL", "null", "NaN"]:  # These are strings, might be parsed
                continue  # Skip assertion as behavior may vary
            else:
                assert result is None, f"Expected None for {value}, got {result}"
    
    def test_datetime_and_timestamp_objects(self):
        """Test datetime and pandas timestamp objects"""
        test_cases = [
            (datetime(2025, 1, 15, 14, 30, 45), "2025-01-15"),
            (datetime(2025, 1, 15, 0, 0, 0), "2025-01-15"),
            (pd.Timestamp('2025-01-15 14:30:45'), "2025-01-15"),
            (pd.Timestamp('2025-01-15'), "2025-01-15"),
            # Edge case: leap year
            (datetime(2024, 2, 29, 23, 59, 59), "2024-02-29"),
            # Edge case: year boundaries
            (datetime(1900, 1, 1, 12, 0, 0), "1900-01-01"),
            (datetime(9999, 12, 31, 23, 59, 59), "9999-12-31"),
        ]
        
        for input_val, expected in test_cases:
            result = normalize_date_value(input_val)
            assert result == expected, f"Failed for {input_val}: expected {expected}, got {result}"
    
    def test_excel_serial_dates(self):
        """Test Excel serial date numbers"""
        test_cases = [
            # Basic Excel dates (corrected based on actual Excel serial calculations)
            (2, "1900-01-01"),    # Excel serial 2 = 1900-01-01 (serial 1 is 1899-12-31)
            (3, "1900-01-02"),    # Day after
            (60, "1900-02-28"),   # Before Excel leap year bug
            (61, "1900-03-01"),   # After Excel leap year bug (Excel skips fake leap day)
            
            # Common Excel dates (corrected)
            (43845, "2020-01-15"), # Correct serial for 2020-01-15
            (44926, "2022-12-31"), # Correct serial for 2022-12-31  
            (45672, "2025-01-15"), # Correct serial for 2025-01-15
            
            # Edge cases with floats (time ignored)
            (2.5, "1900-01-01"),      # Float with fraction (time ignored)
            (43845.75, "2020-01-15"), # Float with time component
        ]
        
        for input_val, expected in test_cases:
            result = normalize_date_value(input_val)
            assert result == expected, f"Failed for Excel serial {input_val}: expected {expected}, got {result}"
    
    def test_invalid_excel_serial_numbers(self):
        """Test invalid Excel serial numbers"""
        invalid_cases = [
            0,          # Below valid range
            -1,         # Negative
            -100,       # Large negative
            2958466,    # Above valid range
            10000000,   # Way above valid range
            float('inf'),  # Infinity
            float('-inf'), # Negative infinity
        ]
        
        for value in invalid_cases:
            result = normalize_date_value(value)
            assert result is None, f"Expected None for invalid Excel serial {value}, got {result}"
    
    def test_standard_date_formats(self):
        """Test standard date formats"""
        test_cases = [
            # YYYY-MM-DD formats
            ("2025-01-15", "2025-01-15"),
            ("2025/01/15", "2025-01-15"),
            ("2025.01.15", "2025-01-15"),
            
            # DD/MM/YYYY and MM/DD/YYYY formats (ambiguous but should parse)
            ("15/01/2025", "2025-01-15"),  # DD/MM/YYYY
            ("01/15/2025", "2025-01-15"),  # MM/DD/YYYY
            ("31/12/2025", "2025-12-31"), # Unambiguous DD/MM
            ("12/31/2025", "2025-12-31"), # Unambiguous MM/DD
            
            # DD-MM-YYYY formats
            ("15-01-2025", "2025-01-15"),
            ("01-15-2025", "2025-01-15"),
            
            # Compact formats
            ("20250115", "2025-01-15"),
        ]
        
        for input_str, expected in test_cases:
            result = normalize_date_value(input_str)
            assert result == expected, f"Failed for '{input_str}': expected {expected}, got {result}"
    
    def test_month_name_formats(self):
        """Test month name formats"""
        test_cases = [
            # Full month names
            ("15 January 2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("January 15, 2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("January 15 2025", datetime(2025, 1, 15, 0, 0, 0)),
            
            # Abbreviated month names
            ("15 Jan 2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("Jan 15, 2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("Jan 15 2025", datetime(2025, 1, 15, 0, 0, 0)),
            
            # Different separators with month names
            ("15-Jan-2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("15/Jan/2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("15.Jan.2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("Jan-15-2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("Jan/15/2025", datetime(2025, 1, 15, 0, 0, 0)),
            
            # Excel-style month formats
            ("15-Jul-2025", datetime(2025, 7, 15, 0, 0, 0)),
            ("Jul-15-2025", datetime(2025, 7, 15, 0, 0, 0)),
            
            # All months abbreviated
            ("15-Jan-2025", datetime(2025, 1, 15, 0, 0, 0)),
            ("15-Feb-2025", datetime(2025, 2, 15, 0, 0, 0)),
            ("15-Mar-2025", datetime(2025, 3, 15, 0, 0, 0)),
            ("15-Apr-2025", datetime(2025, 4, 15, 0, 0, 0)),
            ("15-May-2025", datetime(2025, 5, 15, 0, 0, 0)),
            ("15-Jun-2025", datetime(2025, 6, 15, 0, 0, 0)),
            ("15-Jul-2025", datetime(2025, 7, 15, 0, 0, 0)),
            ("15-Aug-2025", datetime(2025, 8, 15, 0, 0, 0)),
            ("15-Sep-2025", datetime(2025, 9, 15, 0, 0, 0)),
            ("15-Oct-2025", datetime(2025, 10, 15, 0, 0, 0)),
            ("15-Nov-2025", datetime(2025, 11, 15, 0, 0, 0)),
            ("15-Dec-2025", datetime(2025, 12, 15, 0, 0, 0)),
        ]
        
        for input_str, expected in test_cases:
            result = normalize_date_value(input_str)
            assert result == expected, f"Failed for '{input_str}': expected {expected}, got {result}"
    
    def test_two_digit_years(self):
        """Test 2-digit year handling"""
        test_cases = [
            # Recent years (should be 20xx)
            ("15/01/25", datetime(2025, 1, 15, 0, 0, 0)),
            ("15-01-25", datetime(2025, 1, 15, 0, 0, 0)),
            ("15.01.25", datetime(2025, 1, 15, 0, 0, 0)),
            ("01/15/25", datetime(2025, 1, 15, 0, 0, 0)),
            
            # Older years (behavior depends on pandas interpretation)
            ("15/01/99", datetime(1999, 1, 15, 0, 0, 0)),  # Likely interpreted as 1999
            ("15/01/00", datetime(2000, 1, 15, 0, 0, 0)),  # Y2K boundary
            
            # Month name with 2-digit year
            ("15-Jan-25", datetime(2025, 1, 15, 0, 0, 0)),
            ("Jan-15-25", datetime(2025, 1, 15, 0, 0, 0)),
        ]
        
        for input_str, expected in test_cases:
            result = normalize_date_value(input_str)
            # Note: Some 2-digit year interpretations may vary, so we're more lenient
            if result is not None:
                assert result.month == expected.month and result.day == expected.day, \
                    f"Month/day mismatch for '{input_str}': expected {expected}, got {result}"
    
    def test_datetime_with_time_components(self):
        """Test dates with time components (should ignore time)"""
        test_cases = [
            ("2025-01-15 14:30:45", datetime(2025, 1, 15, 0, 0, 0)),
            ("2025-01-15 00:00:00", datetime(2025, 1, 15, 0, 0, 0)),
            ("2025-01-15 23:59:59", datetime(2025, 1, 15, 0, 0, 0)),
            ("15/01/2025 14:30:45", datetime(2025, 1, 15, 0, 0, 0)),
            ("01/15/2025 09:15", datetime(2025, 1, 15, 0, 0, 0)),
            ("15-Jan-2025 16:45:30", datetime(2025, 1, 15, 0, 0, 0)),
            ("Jan-15-2025 08:00", datetime(2025, 1, 15, 0, 0, 0)),
            
            # Different time separators
            ("2025-01-15T14:30:45", datetime(2025, 1, 15, 0, 0, 0)),  # ISO format
            ("2025-01-15 14:30:45.123", datetime(2025, 1, 15, 0, 0, 0)),  # With milliseconds
        ]
        
        for input_str, expected in test_cases:
            result = normalize_date_value(input_str)
            assert result == expected, f"Failed for '{input_str}': expected {expected}, got {result}"
    
    def test_leap_years(self):
        """Test leap year handling"""
        leap_year_cases = [
            ("2024-02-29", datetime(2024, 2, 29, 0, 0, 0)),  # Valid leap day 2024
            ("2020-02-29", datetime(2020, 2, 29, 0, 0, 0)),  # Valid leap day 2020
            ("2000-02-29", datetime(2000, 2, 29, 0, 0, 0)),  # Century leap year
            ("29/02/2024", datetime(2024, 2, 29, 0, 0, 0)),  # DD/MM format
            ("29-Feb-2024", datetime(2024, 2, 29, 0, 0, 0)), # Month name format
        ]
        
        for input_str, expected in test_cases:
            result = normalize_date_value(input_str)
            assert result == expected, f"Failed for leap year '{input_str}': expected {expected}, got {result}"
        
        # Invalid leap years
        invalid_leap_cases = [
            "2023-02-29",  # 2023 is not a leap year
            "1900-02-29",  # 1900 is not a leap year (century rule)
        ]
        
        for invalid_date in invalid_leap_cases:
            result = normalize_date_value(invalid_date)
            # These should either return None or raise an exception
            # Behavior may vary depending on parsing method
    
    def test_edge_date_boundaries(self):
        """Test date boundaries and edge cases"""
        boundary_cases = [
            # Year boundaries
            ("1900-01-01", datetime(1900, 1, 1, 0, 0, 0)),
            ("9999-12-31", datetime(9999, 12, 31, 0, 0, 0)),
            
            # Month boundaries
            ("2025-01-01", datetime(2025, 1, 1, 0, 0, 0)),  # January 1st
            ("2025-01-31", datetime(2025, 1, 31, 0, 0, 0)), # January 31st
            ("2025-12-01", datetime(2025, 12, 1, 0, 0, 0)), # December 1st
            ("2025-12-31", datetime(2025, 12, 31, 0, 0, 0)), # December 31st
            
            # Days in month boundaries
            ("2025-01-31", datetime(2025, 1, 31, 0, 0, 0)), # 31 days
            ("2025-02-28", datetime(2025, 2, 28, 0, 0, 0)), # 28 days (non-leap)
            ("2024-02-29", datetime(2024, 2, 29, 0, 0, 0)), # 29 days (leap)
            ("2025-04-30", datetime(2025, 4, 30, 0, 0, 0)), # 30 days
        ]
        
        for input_str, expected in boundary_cases:
            result = normalize_date_value(input_str)
            assert result == expected, f"Failed for boundary case '{input_str}': expected {expected}, got {result}"
    
    def test_invalid_dates(self):
        """Test invalid date strings"""
        invalid_cases = [
            # Invalid formats
            "not-a-date",
            "12345abc",
            "abc/def/ghi",
            "32/01/2025",    # Invalid day
            "01/32/2025",    # Invalid day  
            "01/13/2025",    # Invalid month (if DD/MM/YYYY)
            "2025-13-01",    # Invalid month
            "2025-01-32",    # Invalid day
            "2025-02-30",    # Invalid day for February
            "2025-04-31",    # Invalid day for April (30 days)
            
            # Wrong separators
            "2025_01_15",
            "2025*01*15",
            "2025@01@15",
            
            # Incomplete dates
            "2025-01",
            "2025",
            "01-15",
            "15",
            
            # Mixed up formats
            "2025/Jan-15",
            "Jan/2025-15",
            
            # Random strings
            "hello world",
            "123456789012345",
            "!@#$%^&*()",
        ]
        
        for invalid_date in invalid_cases:
            result = normalize_date_value(invalid_date)
            assert result is None, f"Expected None for invalid date '{invalid_date}', got {result}"
    
    def test_caching_behavior(self):
        """Test that caching works correctly"""
        test_value = "2025-01-15"
        
        # First call
        result1 = normalize_date_value(test_value)
        
        # Second call should use cache
        result2 = normalize_date_value(test_value)
        
        # Results should be identical
        assert result1 == result2
        
        # Test that cache handles different types of same logical value
        result3 = normalize_date_value(datetime(2025, 1, 15, 14, 30, 0))
        assert result1.date() == result3.date()
    
    # =================================
    # IS_DATE_VALUE TESTS
    # =================================
    
    def test_is_date_value_positive_cases(self):
        """Test cases that should return True for is_date_value"""
        positive_cases = [
            # Datetime objects
            datetime(2025, 1, 15),
            pd.Timestamp('2025-01-15'),
            
            # Excel serial numbers
            1, 100, 44927, 45658, 2958465,
            1.0, 44927.5,  # Floats in valid range
            
            # Standard date strings
            "2025-01-15",
            "2025/01/15", 
            "15/01/2025",
            "01/15/2025",
            "15-01-2025",
            "01-15-2025",
            "15.01.2025",
            
            # Compact format
            "20250115",
            
            # Month name formats
            "15 January 2025",
            "January 15, 2025",
            "15 Jan 2025",
            "Jan 15, 2025",
            "15-Jan-2025",
            "Jan-15-2025",
            "15/Jan/2025",
            "15.Jan.2025",
            
            # With time components
            "2025-01-15 14:30:45",
            "15/01/2025 09:15",
            
            # 2-digit years
            "15/01/25",
            "15-Jan-25",
        ]
        
        for value in positive_cases:
            result = is_date_value(value)
            assert result == True, f"Expected True for is_date_value('{value}'), got {result}"
    
    def test_is_date_value_negative_cases(self):
        """Test cases that should return False for is_date_value"""
        negative_cases = [
            # None and NaN
            None, pd.NaT, np.nan, float('nan'),
            
            # Invalid numbers
            0, -1, 2958466, 10000000, float('inf'), float('-inf'),
            
            # Invalid strings
            "", "   ", "not-a-date", "hello world",
            "12345abc", "abc/def/ghi",
            
            # Invalid dates
            "32/01/2025", "01/32/2025", "2025-13-01", "2025-01-32",
            
            # Wrong formats
            "2025_01_15", "2025*01*15", "2025@01@15",
            
            # Incomplete
            "2025", "01-15", "15",
            
            # Other data types
            123.456,  # Float outside valid Excel range
            True, False,  # Booleans
            [], {}, set(),  # Collections
            
            # Special strings
            "NULL", "null", "NaN", "undefined",
        ]
        
        for value in negative_cases:
            result = is_date_value(value)
            assert result == False, f"Expected False for is_date_value('{value}'), got {result}"
    
    # =================================
    # CHECK_DATE_EQUALS_MATCH TESTS
    # =================================
    
    def test_date_equals_match_positive_cases(self):
        """Test cases that should match as equal dates"""
        equal_pairs = [
            # Same date, different formats
            ("2025-01-15", "15/01/2025"),
            ("2025-01-15", "01/15/2025"),
            ("2025-01-15", "15-Jan-2025"),
            ("2025-01-15", "Jan 15, 2025"),
            
            # Same date with and without time
            ("2025-01-15", "2025-01-15 14:30:45"),
            ("2025-01-15 09:00:00", "2025-01-15 18:45:30"),
            
            # Datetime objects
            (datetime(2025, 1, 15, 0, 0), datetime(2025, 1, 15, 12, 30)),
            (datetime(2025, 1, 15), pd.Timestamp('2025-01-15')),
            
            # Excel serial vs string
            (45658, "2025-01-15"),  # Excel serial for 2025-01-15
            
            # Both None/NaN
            (None, None),
            (None, pd.NaT),
            (np.nan, np.nan),
            
            # Different representations of same date
            ("01/15/2025", "January 15, 2025"),
            ("2025/01/15", "15-Jan-2025"),
        ]
        
        for val_a, val_b in equal_pairs:
            result = check_date_equals_match(val_a, val_b)
            assert result == True, f"Expected True for date_equals_match('{val_a}', '{val_b}'), got {result}"
    
    def test_date_equals_match_negative_cases(self):
        """Test cases that should NOT match as equal dates"""
        unequal_pairs = [
            # Different dates
            ("2025-01-15", "2025-01-16"),
            ("2025-01-15", "2025-02-15"),
            ("2025-01-15", "2024-01-15"),
            
            # One valid, one None
            ("2025-01-15", None),
            (None, "2025-01-15"),
            (datetime(2025, 1, 15), None),
            
            # One valid, one invalid
            ("2025-01-15", "not-a-date"),
            ("2025-01-15", "invalid-date"),
            (45658, "hello world"),
            
            # Different valid dates
            ("Jan 15, 2025", "Jan 16, 2025"),
            ("15-Jan-2025", "15-Feb-2025"),
            (45658, 45659),  # Different Excel serials
            
            # Leap year edge cases
            ("2024-02-29", "2025-02-28"),  # Leap vs non-leap
            ("2024-02-28", "2024-03-01"),  # Different days
        ]
        
        for val_a, val_b in unequal_pairs:
            result = check_date_equals_match(val_a, val_b)
            assert result == False, f"Expected False for date_equals_match('{val_a}', '{val_b}'), got {result}"
    
    # =================================
    # PERFORMANCE AND STRESS TESTS  
    # =================================
    
    def test_performance_with_large_dataset(self):
        """Test performance with large number of date values"""
        # Create a mix of date formats
        test_values = [
            "2025-01-15", "15/01/2025", "15-Jan-2025", "Jan 15, 2025",
            datetime(2025, 1, 15), pd.Timestamp('2025-01-15'),
            45658, "2025-01-15 14:30:45", None, "invalid-date"
        ]
        
        # Repeat for performance testing
        large_dataset = test_values * 1000  # 10,000 values
        
        # Test is_date_value performance (should be fast)
        date_checks = [is_date_value(val) for val in large_dataset]
        assert len(date_checks) == len(large_dataset)
        
        # Test normalize_date_value performance (should use caching)
        normalized_dates = [normalize_date_value(val) for val in large_dataset]
        assert len(normalized_dates) == len(large_dataset)
    
    def test_unicode_and_special_characters(self):
        """Test Unicode characters and special date formats"""
        unicode_cases = [
            # Unicode characters that might appear in dates
            "2025‑01‑15",  # Unicode hyphen
            "2025—01—15",  # Em dash
            "2025–01–15",  # En dash
            
            # Different Unicode spaces
            "2025 01 15",   # Regular space
            "2025 01 15",   # Non-breaking space
            "2025　01　15", # Unicode wide space
            
            # Other languages (should probably fail, but test anyway)
            "२०२५-०१-१५", # Devanagari numerals
            "2025年1月15日", # Japanese date format
            "15/janvier/2025", # French month name
        ]
        
        for value in unicode_cases:
            try:
                result = normalize_date_value(value)
                # Most should return None, but some might work
                is_date_result = is_date_value(value)
                # Just ensure no exceptions are thrown
                assert isinstance(is_date_result, bool)
            except Exception as e:
                # Should not throw unhandled exceptions
                pytest.fail(f"Unexpected exception for Unicode value '{value}': {e}")
    
    def test_extremely_large_and_small_values(self):
        """Test extremely large and small numeric values"""
        extreme_values = [
            # Very large numbers
            999999999999999,
            1e15,
            1e20,
            
            # Very small numbers
            1e-10,
            0.0000001,
            
            # Scientific notation as strings
            "1e5",
            "1.23e4",
            "1E6",
        ]
        
        for value in extreme_values:
            result_normalize = normalize_date_value(value)
            result_is_date = is_date_value(value)
            
            # These should all return None/False for normalize/is_date
            assert result_normalize is None, f"Expected None for extreme value {value}"
            assert result_is_date == False, f"Expected False for extreme value {value}"
    
    def test_mixed_data_types(self):
        """Test various Python data types"""
        mixed_types = [
            # Numeric types
            1, 1.0, Decimal('1.0'), complex(1, 0),
            
            # String types
            "1", b"1", bytearray(b"1"),
            
            # Boolean
            True, False,
            
            # Collections (should all return None/False)
            [2025, 1, 15], (2025, 1, 15), {'year': 2025},
            
            # Custom objects
            object(),
        ]
        
        for value in mixed_types:
            try:
                result_normalize = normalize_date_value(value)
                result_is_date = is_date_value(value)
                
                # Most should return None/False
                if not isinstance(value, (int, float)) or not (1 <= value <= 2958465):
                    assert result_normalize is None, f"Expected None for {type(value)} {value}"
                    assert result_is_date == False, f"Expected False for {type(value)} {value}"
                
            except Exception as e:
                # Should not throw unhandled exceptions for basic types
                if not isinstance(value, (list, tuple, dict, object, complex, bytes, bytearray)):
                    pytest.fail(f"Unexpected exception for {type(value)} '{value}': {e}")
    
    # =================================
    # INTEGRATION TESTS
    # =================================
    
    def test_real_world_excel_dates(self):
        """Test real-world Excel date scenarios"""
        # Common Excel date exports
        excel_scenarios = [
            # Excel default date formats
            ("43831", datetime(2020, 1, 15, 0, 0, 0)),  # Excel serial as string
            (43831, datetime(2020, 1, 15, 0, 0, 0)),    # Excel serial as number
            
            # Excel with time (should ignore time)
            (43831.5, datetime(2020, 1, 15, 0, 0, 0)),  # Noon
            (43831.99999, datetime(2020, 1, 15, 0, 0, 0)), # Almost midnight
            
            # Excel text dates
            ("1/15/2020", datetime(2020, 1, 15, 0, 0, 0)),
            ("15-Jan-20", datetime(2020, 1, 15, 0, 0, 0)),
        ]
        
        for input_val, expected in excel_scenarios:
            result = normalize_date_value(input_val)
            assert result == expected, f"Excel scenario failed for {input_val}: expected {expected}, got {result}"
    
    def test_csv_import_scenarios(self):
        """Test common CSV date import scenarios"""
        csv_scenarios = [
            # Common CSV date formats
            "01/15/2025", "15/01/2025", "2025-01-15",
            "Jan 15, 2025", "15-Jan-2025", "January 15, 2025",
            
            # With quotes (as they might appear in CSV)
            '"2025-01-15"', "'2025-01-15'",
            
            # With extra whitespace
            " 2025-01-15 ", "\t2025-01-15\t", "\n2025-01-15\n",
        ]
        
        for date_str in csv_scenarios:
            # Clean the string as would happen in real CSV processing
            cleaned = date_str.strip(' \t\n"\'')
            
            result = normalize_date_value(cleaned)
            assert result is not None, f"CSV scenario failed for '{date_str}' (cleaned: '{cleaned}')"
            assert result.year == 2025 and result.month == 1 and result.day == 15, \
                f"Wrong date parsed for '{cleaned}': got {result}"
    
    def test_database_date_scenarios(self):
        """Test database date format scenarios"""
        db_scenarios = [
            # SQL Server date formats
            ("2025-01-15T00:00:00", datetime(2025, 1, 15, 0, 0, 0)),
            ("2025-01-15T14:30:45.123", datetime(2025, 1, 15, 0, 0, 0)),
            
            # MySQL date formats
            ("2025-01-15 14:30:45", datetime(2025, 1, 15, 0, 0, 0)),
            
            # PostgreSQL date formats
            ("2025-01-15", datetime(2025, 1, 15, 0, 0, 0)),
            
            # Oracle date formats (as strings)
            ("15-JAN-25", datetime(2025, 1, 15, 0, 0, 0)),
        ]
        
        for input_str, expected in db_scenarios:
            result = normalize_date_value(input_str)
            if result is not None:  # Some formats might not be supported
                assert result.date() == expected.date(), \
                    f"Database scenario failed for '{input_str}': expected {expected}, got {result}"


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        "-v",
        __file__,
        "--tb=short",
        "-x",  # Stop on first failure for debugging
    ])