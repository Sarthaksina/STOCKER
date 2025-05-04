# Code Quality Guidelines

This document outlines the code quality standards and tools used in the STOCKER Pro project.

## Code Quality Tools

STOCKER Pro uses the following code quality tools:

### Formatting

- **Black**: An opinionated code formatter that ensures consistent code style.
- **isort**: A utility to sort imports alphabetically and automatically separated into sections.

### Linting

- **Flake8**: A wrapper around PyFlakes, pycodestyle, and McCabe complexity checker.
  - **flake8-docstrings**: Flake8 plugin to check docstrings.
  - **flake8-bugbear**: Flake8 plugin to find bugs and design problems.
  - **pep8-naming**: Flake8 plugin to check PEP 8 naming conventions.

### Type Checking

- **MyPy**: Static type checker for Python.

### Security

- **Bandit**: A tool to find common security issues in Python code.

### Testing

- **Pytest**: A framework for writing and running tests.
- **pytest-cov**: A pytest plugin to measure code coverage.

## Running Code Quality Checks

You can run code quality checks using the provided script:

```bash
# Run all checks
python scripts/run_code_checks.py

# Format code only
python scripts/run_code_checks.py --format

# Run linting only
python scripts/run_code_checks.py --lint

# Run type checking only
python scripts/run_code_checks.py --typecheck

# Run security checks only
python scripts/run_code_checks.py --security

# Run tests only
python scripts/run_code_checks.py --test
```

## Pre-commit Hooks

This project uses pre-commit hooks to automatically run code quality checks before each commit. To install the pre-commit hooks, run:

```bash
pip install pre-commit
pre-commit install
```

## Continuous Integration

Code quality checks are also run in CI using GitHub Actions. The workflow is defined in `.github/workflows/ci.yml`.

## Coding Standards

### Python Version

All code should be compatible with Python 3.8+.

### Docstrings

All modules, classes, and functions should have docstrings following the Google style:

```python
def example_function(param1: str, param2: int) -> bool:
    """Short description of the function.
    
    Longer description explaining details about the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When the function encounters an error
    """
```

### Type Hints

Use type hints for all function parameters and return values:

```python
def calculate_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # Implementation
```

### Imports

Organize imports into sections:
1. Standard library imports
2. Third-party imports
3. Local application imports

Within each section, imports should be alphabetically sorted.

### Variable Names

- Use descriptive variable names
- Use snake_case for variables and functions
- Use PascalCase for classes
- Use UPPER_CASE for constants

### Exception Handling

- Be specific about the exceptions you catch
- Provide meaningful error messages
- Avoid catching `Exception` generally unless you re-raise it

### Comments

- Use comments sparingly and only when necessary
- Focus on explaining **why** rather than **what**
- Use TODO, FIXME, and NOTE comments with a name or issue number when appropriate

## Best Practices

- Follow the DRY (Don't Repeat Yourself) principle
- Write modular, testable code
- Keep functions and methods small and focused on a single task
- Limit function arguments (aim for 3 or fewer if possible)
- Use default parameter values instead of creating multiple similar functions
- Use context managers for resource management
- Follow the principle of least surprise 