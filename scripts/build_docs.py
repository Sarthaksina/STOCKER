#!/usr/bin/env python
"""
Script to build and check documentation for STOCKER Pro.

This script:
1. Builds HTML documentation using Sphinx
2. Generates coverage report for undocumented code
3. Checks for broken links in the documentation

Usage:
    python scripts/build_docs.py [--check] [--serve]

Options:
    --check    Only check for undocumented code and broken links, don't build docs
    --serve    Start a local server to view the documentation after building
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path


def run_command(command, description, exit_on_error=True):
    """Run a shell command and print its output."""
    print(f"\n\n{'-' * 80}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(command)}")
    print('-' * 80)
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error running {description}:")
        print(result.stderr)
        if exit_on_error:
            sys.exit(result.returncode)
        return False
    
    print(f"\n✅ {description} completed successfully!")
    return True


def build_docs():
    """Build HTML documentation using Sphinx."""
    docs_dir = Path("docs")
    build_dir = docs_dir / "_build" / "html"
    
    # Make sure _static directory exists
    static_dir = docs_dir / "_static"
    static_dir.mkdir(exist_ok=True)
    
    # Clean the build directory first
    if build_dir.exists():
        run_command(
            ["sphinx-build", "-M", "clean", "docs", "docs/_build"],
            "Cleaning documentation build directory"
        )
    
    # Build HTML documentation
    return run_command(
        ["sphinx-build", "-M", "html", "docs", "docs/_build", "-W"],
        "Building HTML documentation"
    )


def check_docs_coverage():
    """Generate coverage report for undocumented code."""
    return run_command(
        ["sphinx-build", "-b", "coverage", "docs", "docs/_build/coverage"],
        "Checking documentation coverage",
        exit_on_error=False
    )


def check_links():
    """Check for broken links in the documentation."""
    return run_command(
        ["sphinx-build", "-b", "linkcheck", "docs", "docs/_build/linkcheck"],
        "Checking documentation links",
        exit_on_error=False
    )


def serve_docs(port=8000):
    """Start a local server to view the documentation."""
    build_dir = os.path.join("docs", "_build", "html")
    
    if not os.path.exists(build_dir):
        print("Documentation build directory does not exist. Run with --build first.")
        return False
    
    os.chdir(build_dir)
    
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    
    url = f"http://localhost:{port}"
    print(f"\n\n{'-' * 80}")
    print(f"Serving documentation at {url}")
    print(f"Press Ctrl+C to stop the server")
    print('-' * 80)
    
    # Open the docs in the default browser
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build and check documentation")
    parser.add_argument("--check", action="store_true", help="Only check docs, don't build")
    parser.add_argument("--serve", action="store_true", help="Serve docs after building")
    return parser.parse_args()


def main():
    """Run the documentation build and check process."""
    args = parse_args()
    success = True
    
    if args.check:
        # Only run checks
        success = check_docs_coverage() and success
        success = check_links() and success
    else:
        # Build and check
        success = build_docs() and success
        success = check_docs_coverage() and success
        success = check_links() and success
        
        if args.serve and success:
            serve_docs()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 