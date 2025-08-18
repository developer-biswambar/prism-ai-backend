#!/usr/bin/env python3
"""
Simple Test Runner for Reconciliation Tests
Place this file in your project root directory

Usage:
    python run_reconciliation_tests.py [test_type]

Test Types:
    all         - Run all tests (default)
    unit        - Run unit tests only
    integration - Run integration tests only
    smoke       - Run smoke tests only
    performance - Run performance tests only
    reconciliation - Run reconciliation-specific tests

Examples:
    python run_reconciliation_tests.py
    python run_reconciliation_tests.py smoke
    python run_reconciliation_tests.py unit
"""

import os
import subprocess
import sys


def run_tests(test_type="all"):
    """Run tests based on type"""

    # Ensure we're in the project root
    if not os.path.exists("test") and not os.path.exists("tests"):
        print("❌ Error: No 'test' or 'tests' directory found.")
        print("Please run this script from the project root directory.")
        return 1

    # Determine test directory
    test_dir = "test" if os.path.exists("test") else "tests"

    # Build pytest command based on test type
    if test_type == "all":
        cmd = ["pytest", test_dir, "-v"]
        description = "All Tests"
    elif test_type == "unit":
        cmd = ["pytest", "-m", "unit", test_dir, "-v"]
        description = "Unit Tests"
    elif test_type == "integration":
        cmd = ["pytest", "-m", "integration", test_dir, "-v"]
        description = "Integration Tests"
    elif test_type == "smoke":
        cmd = ["pytest", "-m", "smoke", test_dir, "-v"]
        description = "Smoke Tests"
    elif test_type == "performance":
        cmd = ["pytest", "-m", "slow", test_dir, "-v", "--durations=10"]
        description = "Performance Tests"
    elif test_type == "reconciliation":
        # Run the specific reconciliation test file
        reconciliation_file = os.path.join(test_dir, "test_reconciliation_routes.py")
        if os.path.exists(reconciliation_file):
            cmd = ["pytest", reconciliation_file, "-v"]
        else:
            cmd = ["pytest", "-m", "reconciliation", test_dir, "-v"]
        description = "Reconciliation Tests"
    else:
        print(f"❌ Unknown test type: {test_type}")
        print_usage()
        return 1

    # Print what we're about to run
    print(f"\n🧪 Running {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)

        # Print result
        if result.returncode == 0:
            print(f"\n✅ {description} - PASSED")
        else:
            print(f"\n❌ {description} - FAILED")

        return result.returncode

    except FileNotFoundError:
        print("❌ Error: pytest not found. Please install pytest:")
        print("pip install pytest")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 1


def print_usage():
    """Print usage information"""
    print(__doc__)


def main():
    """Main entry point"""

    # Check for help
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_usage()
        return 0

    # Get test type
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Validate test type
    valid_types = ["all", "unit", "integration", "smoke", "performance", "reconciliation"]
    if test_type not in valid_types:
        print(f"❌ Invalid test type: {test_type}")
        print(f"Valid types: {', '.join(valid_types)}")
        return 1

    # Run the tests
    return run_tests(test_type)


if __name__ == "__main__":
    sys.exit(main())
