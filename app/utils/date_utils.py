# backend/app/utils/date_utils.py - Shared date utilities
import re
from datetime import datetime
from functools import lru_cache
from typing import Optional

import pandas as pd


class DateNormalizer:
    """
    Shared date normalization utility class.
    Provides comprehensive date parsing and normalization functionality.
    """
    
    def __init__(self):
        self._date_cache = {}  # Cache parsed dates for performance
    
    def normalize_date_value(self, value) -> Optional[str]:
        """
        Normalize date value to universal YYYY-MM-DD string format.
        Handles all Excel date formats and returns consistent format for entire repo.
        Returns only the date part (ignoring time components).
        """
        if pd.isna(value) or value is None:
            return None

        # Convert to string for caching key
        cache_key = str(value)
        if cache_key in self._date_cache:
            return self._date_cache[cache_key]

        parsed_date = None

        try:
            # Handle different input types
            if isinstance(value, (datetime, pd.Timestamp)):
                # Already a datetime, just extract date part
                parsed_date = value.replace(hour=0, minute=0, second=0, microsecond=0)
            elif isinstance(value, (int, float)):
                # Skip Excel serial number conversion entirely
                # Only numeric values that are already datetime objects should be handled
                # All other numeric values (IDs, amounts, counts, etc.) should remain as numbers
                pass  # Do not convert any numeric values to dates
            else:
                # String parsing with comprehensive format support
                value_str = str(value).strip()

                # Try pandas date parsing - be more explicit about format detection
                try:
                    # First try common unambiguous formats (YYYY-MM-DD, etc.)
                    if '/' in value_str and len(value_str.split('/')[0]) <= 2:
                        # Likely DD/MM/YYYY format, use dayfirst=True
                        parsed_date = pd.to_datetime(value_str, dayfirst=True, errors='raise')
                    else:
                        # For other formats, use dayfirst=False (default)
                        parsed_date = pd.to_datetime(value_str, dayfirst=False, errors='raise')
                    
                    if isinstance(parsed_date, pd.Timestamp):
                        parsed_date = parsed_date.to_pydatetime()
                    parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                except:
                    # Fallback: try the opposite dayfirst setting
                    try:
                        if '/' in value_str and len(value_str.split('/')[0]) <= 2:
                            # Already tried dayfirst=True above, try dayfirst=False
                            parsed_date = pd.to_datetime(value_str, dayfirst=False, errors='raise')
                        else:
                            # Try dayfirst=True as fallback
                            parsed_date = pd.to_datetime(value_str, dayfirst=True, errors='raise')
                        
                        if isinstance(parsed_date, pd.Timestamp):
                            parsed_date = parsed_date.to_pydatetime()
                        parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    except:
                        # Manual parsing for specific Excel formats
                        date_formats = [
                            # Standard numeric formats
                            '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%Y/%m/%d',
                            '%d-%m-%Y', '%m-%d-%Y', '%d.%m.%Y', '%m.%d.%Y',

                            # Month name formats (covers "10-Jul-2025" style)
                            '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y',
                            '%d-%b-%Y', '%d-%B-%Y', '%b-%d-%Y', '%B-%d-%Y',
                            '%d.%b.%Y', '%d.%B.%Y', '%b.%d.%Y', '%B.%d.%Y',
                            '%d/%b/%Y', '%d/%B/%Y', '%b/%d/%Y', '%B/%d/%Y',

                            # Additional month name variations
                            '%b %d %Y', '%B %d %Y', '%d %b, %Y', '%d %B, %Y',
                            '%b-%d-%Y', '%B-%d-%Y', '%b.%d.%Y', '%B.%d.%Y',
                            '%b/%d/%Y', '%B/%d/%Y',

                            # Compact formats
                            '%Y%m%d', '%d%m%Y', '%m%d%Y',

                            # 2-digit year formats
                            '%d/%m/%y', '%m/%d/%y', '%y-%m-%d', '%y/%m/%d',
                            '%d-%m-%y', '%m-%d-%y', '%d.%m.%y', '%m.%d.%y',
                            '%d-%b-%y', '%d-%B-%y', '%b-%d-%y', '%B-%d-%y',

                            # With time components (will be ignored)
                            '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
                            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                            '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
                            '%d-%b-%Y %H:%M:%S', '%d-%B-%Y %H:%M:%S',
                            '%b-%d-%Y %H:%M:%S', '%B-%d-%Y %H:%M:%S',
                            '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M',
                            '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M',
                            '%d-%m-%Y %H:%M', '%m-%d-%Y %H:%M',
                            '%d-%b-%Y %H:%M', '%d-%B-%Y %H:%M',
                            '%b-%d-%Y %H:%M', '%B-%d-%Y %H:%M',
                        ]

                        for fmt in date_formats:
                            try:
                                parsed_date = datetime.strptime(value_str, fmt)
                                # Always set time to 00:00:00 for date-only comparison
                                parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                                break
                            except ValueError:
                                continue

        except Exception:
            # If all parsing fails, return None
            parsed_date = None

        # Convert to universal string format before caching
        if parsed_date is not None and pd.notna(parsed_date):
            try:
                universal_date_str = parsed_date.strftime('%Y-%m-%d')
                self._date_cache[cache_key] = universal_date_str
                return universal_date_str
            except (ValueError, AttributeError):
                # Handle pandas NaT or other edge cases
                self._date_cache[cache_key] = None
                return None
        else:
            self._date_cache[cache_key] = None
            return None

    def is_date_value(self, value) -> bool:
        """Check if a value appears to be a date"""
        if pd.isna(value) or value is None:
            return False

        # Quick check for obvious date types
        if isinstance(value, (datetime, pd.Timestamp)):
            return True

        # Never treat numeric values as dates - no Excel serial conversion
        if isinstance(value, (int, float)):
            return False  # Never convert numeric values to dates

        # Check string patterns
        value_str = str(value).strip()

        # Common date patterns
        date_patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # DD/MM/YYYY or MM/DD/YYYY
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',  # YYYY-MM-DD or YYYY/MM/DD
            r'^\d{1,2}\.\d{1,2}\.\d{2,4}$',  # DD.MM.YYYY
            r'^\d{8}$',  # YYYYMMDD
            r'^\d{1,2}\s+\w+\s+\d{2,4}$',  # DD Month YYYY
            r'^\w+\s+\d{1,2},?\s+\d{2,4}$',  # Month DD, YYYY
            r'^\d{1,2}[-/\.]\w+[-/\.]\d{2,4}$',  # DD-MMM-YYYY (like "10-Jul-2025")
            r'^\w+[-/\.]\d{1,2}[-/\.]\d{2,4}$',  # MMM-DD-YYYY (like "Jul-10-2025")
            r'^\d{1,2}\s+\w+,?\s+\d{2,4}$',  # DD MMM YYYY (like "10 Jul 2025")
            r'^\w+\s+\d{1,2}[,\s]+\d{2,4}$',  # MMM DD, YYYY (like "Jul 10, 2025")
        ]

        for pattern in date_patterns:
            if re.match(pattern, value_str):
                return True

        return False

    def check_date_equals_match(self, val_a, val_b) -> bool:
        """Check if two values match as dates (exact date comparison, ignoring time)"""
        date_a = self.normalize_date_value(val_a)
        date_b = self.normalize_date_value(val_b)

        # Both must be valid dates or both None/NaN
        if date_a is None and date_b is None:
            return True
        if date_a is None or date_b is None:
            return False

        # Compare the normalized date strings directly
        return date_a == date_b


# Global instance for reuse
_date_normalizer = DateNormalizer()


def normalize_date_value(value) -> Optional[str]:
    """
    Global function to normalize date values to universal YYYY-MM-DD format.
    Uses shared DateNormalizer instance for caching benefits.
    """
    return _date_normalizer.normalize_date_value(value)


def is_date_value(value) -> bool:
    """
    Global function to check if value appears to be a date.
    Uses shared DateNormalizer instance.
    """
    return _date_normalizer.is_date_value(value)


def check_date_equals_match(val_a, val_b) -> bool:
    """
    Global function to check date equality.
    Uses shared DateNormalizer instance.
    """
    return _date_normalizer.check_date_equals_match(val_a, val_b)