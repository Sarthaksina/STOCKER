# Contributing to STOCKER Pro

Thank you for your interest in contributing to STOCKER Pro! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of other contributors. We strive to maintain a welcoming and inclusive community.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/stocker-pro.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: 
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Install development dependencies: `pip install -r requirements-dev.txt`
7. Install pre-commit hooks: `pre-commit install`

## Project Structure

Please follow the project structure as outlined in the README.md file. In particular:

- Place core functionality in `src/core/`
- Place data-related code in `src/data/`
- Place feature engineering code in `src/features/`
- Place machine learning models in `src/ml/`
- Place API endpoints in `src/api/`
- Place business logic services in `src/services/`
- Place UI components in `src/ui/`
- Place CLI commands in `src/cli/`
- Place RAG/LLM intelligence in `src/intelligence/`

## Code Style

We use the following tools to maintain code quality:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For linting
- **mypy**: For type checking

All code should:

- Be formatted with Black (line length: 88)
- Have proper type annotations
- Include docstrings (Google style)
- Follow PEP8 guidelines

## Testing

All new features should include tests. We use pytest for testing.

To run tests:

```bash
pytest
```

To check test coverage:

```bash
pytest --cov=src
```

## Pull Request Process

1. Create a new branch for your feature/fix: `git checkout -b feature/your-feature-name`
2. Make your changes and commit them with descriptive commit messages
3. Run tests to ensure your changes don't break existing functionality
4. Push to your fork: `git push origin feature/your-feature-name`
5. Submit a pull request to the main repository

Please include in your PR:
- A description of the changes
- Any relevant issue numbers (e.g., "Fixes #123")
- Screenshots or examples if applicable

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
