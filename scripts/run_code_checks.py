#!/usr/bin/env python
"""
Script to run code quality checks for STOCKER Pro.

This script runs the following checks:
1. Black code formatting
2. isort import sorting
3. Flake8 linting
4. MyPy type checking
5. Bandit security scanning
6. Pytest unit tests with coverage

Usage:
    python scripts/run_code_checks.py [options]

Options:
    --format     Format code with black and isort
    --lint       Run linting with flake8
    --typecheck  Run type checking with mypy
    --security   Run security checks with bandit
    --test       Run tests with pytest
    --all        Run all checks (default)
    --ci         Run checks in CI mode (non-interactive)
"""

import argparse
import os
import subprocess
import sys
from typing import List


def run_command(command: List[str], description: str) -> bool:
    """Run a shell command and return True if it succeeds."""
    print(f"\n\n{'=' * 80}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(command)}")
    print('-' * 80)
    
    result = subprocess.run(command)
    success = result.returncode == 0
    
    if success:
        print(f"\n✅ {description} passed!")
    else:
        print(f"\n❌ {description} failed!")
    
    return success


def format_code(fix: bool = True) -> bool:
    """Format code with Black and isort."""
    src_dirs = ["src", "tests", "scripts"]
    
    # Run Black
    black_command = ["black"]
    if not fix:
        black_command.append("--check")
    black_command.extend(src_dirs)
    black_success = run_command(black_command, "Black code formatting")
    
    # Run isort
    isort_command = ["isort"]
    if not fix:
        isort_command.append("--check-only")
    isort_command.extend(src_dirs)
    isort_success = run_command(isort_command, "isort import sorting")
    
    return black_success and isort_success


def run_linting() -> bool:
    """Run linting with Flake8."""
    return run_command(["flake8", "src", "tests", "scripts"], "Flake8 linting")


def run_type_checking() -> bool:
    """Run type checking with MyPy."""
    return run_command(["mypy", "src"], "MyPy type checking")


def run_security_checks() -> bool:
    """Run security checks with Bandit."""
    return run_command(
        ["bandit", "-r", "src", "-x", "tests", "-ll"], "Bandit security scanning"
    )


def run_tests(with_coverage: bool = True) -> bool:
    """Run tests with pytest."""
    command = ["pytest", "tests/"]
    
    if with_coverage:
        command.extend([
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html:coverage_html",
            "--cov-report=xml:coverage.xml"
        ])
    
    return run_command(command, "Pytest unit tests")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument("--format", action="store_true", help="Format code with black and isort")
    parser.add_argument("--lint", action="store_true", help="Run linting with flake8")
    parser.add_argument("--typecheck", action="store_true", help="Run type checking with mypy")
    parser.add_argument("--security", action="store_true", help="Run security checks with bandit")
    parser.add_argument("--test", action="store_true", help="Run tests with pytest")
    parser.add_argument("--all", action="store_true", help="Run all checks (default)")
    parser.add_argument("--ci", action="store_true", help="Run in CI mode (no formatting fixes)")
    return parser.parse_args()


def main():
    """Run the code quality checks based on command-line arguments."""
    args = parse_args()
    
    # If no specific checks are requested, run all checks
    run_all = args.all or not (args.format or args.lint or args.typecheck or args.security or args.test)
    
    success = True
    
    try:
        # Determine if we should fix formatting issues
        fix_formatting = not args.ci
        
        # Format code
        if args.format or run_all:
            if not format_code(fix=fix_formatting):
                success = False
        
        # Lint code
        if args.lint or run_all:
            if not run_linting():
                success = False
        
        # Type check
        if args.typecheck or run_all:
            if not run_type_checking():
                success = False
        
        # Security check
        if args.security or run_all:
            if not run_security_checks():
                success = False
        
        # Run tests
        if args.test or run_all:
            if not run_tests():
                success = False
        
        if success:
            print("\n\n✅ All checks passed!")
            return 0
        else:
            print("\n\n❌ Some checks failed!")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Checks interrupted!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 