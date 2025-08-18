# backend/app/utils/__init__.py
"""
Utility functions and helper classes for the Financial Data Extraction System.
"""

try:
    # Try relative import (when imported as package)
    from .financial_validators import FinancialValidators
except ImportError:
    # Try absolute import (when run directly or in development)
    try:
        from financial_validators import FinancialValidators
    except ImportError:
        # If both fail, define a placeholder
        print("⚠️  FinancialValidators not found. Make sure financial_validators.py exists.")
        FinancialValidators = None

__all__ = ["FinancialValidators"]

# Allow file to be run directly for testing
if __name__ == "__main__":
    print("📁 Utils package")
    if FinancialValidators:
        print("✅ FinancialValidators loaded successfully")
    else:
        print("❌ FinancialValidators failed to load")
