#!/usr/bin/env python3
"""
Test runner script for the file management API
Usage: python run_tests.py [options]
"""

import argparse
import subprocess
import sys


def run_command(cmd):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run tests for the file management API')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--marker', '-m', help='Run tests with specific marker')
    parser.add_argument('--file', '-f', help='Run specific test file')
    parser.add_argument('--test', '-t', help='Run specific test')
    parser.add_argument('--parallel', '-p', action='store_true', help='Run tests in parallel')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML coverage report')

    args = parser.parse_args()

    # Base pytest command
    cmd = ['pytest']

    # Add verbose flag
    if args.verbose:
        cmd.append('-v')

    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=app', '--cov-report=term-missing'])
        if args.html_report:
            cmd.append('--cov-report=html')

    # Add marker filter
    if args.marker:
        cmd.extend(['-m', args.marker])

    # Add specific file
    if args.file:
        cmd.append(f'tests/{args.file}')

    # Add specific test
    if args.test:
        cmd.extend(['-k', args.test])

    # Add parallel execution
    if args.parallel:
        cmd.extend(['-n', 'auto'])

    # Run the tests
    success = run_command(cmd)

    if success:
        print("\n✅ All tests passed!")
        if args.coverage and args.html_report:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    # Quick test commands
    if len(sys.argv) == 1:
        print("🧪 File Management API Test Runner")
        print("\nQuick commands:")
        print("  python run_tests.py --coverage          # Run all tests with coverage")
        print("  python run_tests.py -m file_upload      # Run file upload tests only")
        print("  python run_tests.py -m viewer           # Run viewer tests only")
        print("  python run_tests.py -m error            # Run error handling tests")
        print("  python run_tests.py -f test_file_routes.py  # Run specific file")
        print("  python run_tests.py -t upload           # Run tests matching 'upload'")
        print("  python run_tests.py --parallel          # Run tests in parallel")
        print("\nFor more options: python run_tests.py --help")

        # Ask user what they want to run
        choice = input("\nWhat would you like to run? (all/file_upload/viewer/coverage): ").strip().lower()

        if choice == 'all':
            run_command(['pytest', '-v'])
        elif choice == 'file_upload':
            run_command(['pytest', '-v', '-m', 'file_upload'])
        elif choice == 'viewer':
            run_command(['pytest', '-v', '-m', 'viewer'])
        elif choice == 'coverage':
            run_command(['pytest', '-v', '--cov=app', '--cov-report=html', '--cov-report=term-missing'])
            print("📊 Open htmlcov/index.html to view detailed coverage report")
        else:
            print("Running all tests...")
            run_command(['pytest', '-v'])
    else:
        main()
