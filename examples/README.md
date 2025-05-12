# STOCKER Examples

This directory contains example scripts demonstrating how to use various features of the STOCKER platform.

## Available Examples

### Technical Indicators

- `using_indicators.py`: Demonstrates how to use the consolidated technical indicators module with both class-based and function-based interfaces.

### Feature Engineering

- `using_feature_engineering.py`: Demonstrates how to use the FeatureEngineer class to create various features for financial time series data, including lag features, rolling window features, return features, and date-based features.

## Running the Examples

To run an example, navigate to the project root directory and execute the example script:

```bash
# From the project root directory
python -m examples.using_indicators
```

Or navigate to the examples directory and run:

```bash
# From the examples directory
python using_indicators.py
```

## Example Output

The examples will generate output in the console showing the results of various operations. Some examples may also generate visualizations or save output files to the `examples/output` directory.

## Creating New Examples

When creating new examples, please follow these guidelines:

1. Include a detailed docstring at the top of the file explaining the purpose of the example
2. Add proper error handling to make the example robust
3. Include comments explaining key steps
4. Update this README.md to include your new example
5. Follow the project's coding standards (PEP8, type hints, etc.)

## Dependencies

All examples should work with the dependencies listed in the project's `requirements.txt` file. If an example requires additional dependencies, please document them clearly in the example file's docstring.
