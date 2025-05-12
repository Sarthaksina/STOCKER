# STOCKER Pro Configuration Guide

## Overview

STOCKER Pro uses a unified configuration system that supports multiple sources:

1. Default values (hardcoded in the configuration classes)
2. Configuration files (YAML or JSON)
3. Environment variables
4. Direct code modification

The configuration system follows a hierarchical approach, where each source can override the previous one in the order listed above.

## Configuration Structure

The configuration is organized into several sections:

- **Environment**: Development, testing, or production
- **Project Paths**: Data directory, log directory
- **Database Settings**: MongoDB URI, database name
- **Data Sources**: Settings for Yahoo Finance, Alpha Vantage, etc.
- **Models**: Settings for LSTM, XGBoost, LightGBM, and ensemble models
- **API**: Host, port, CORS settings, rate limiting
- **Portfolio**: Risk-free rate, benchmark symbol, optimization methods

## Using Configuration Files

STOCKER Pro supports both YAML and JSON configuration files. By default, it looks for a `config.yaml` file in the current working directory or in the user's home directory under `.stocker/config.yaml`.

### Example: Loading from a Configuration File

```python
from src.unified_config import get_config

# Load from default locations
config = get_config()

# Or specify a custom path
config = get_config("path/to/custom/config.yaml")

# Access configuration values
data_dir = config.data_dir
risk_free_rate = config.portfolio_config.risk_free_rate
default_model = config.model_config.default_model
```

## Using Environment Variables

Environment variables can be used to override configuration values. The system uses the prefix `STOCKER_` followed by the configuration key in uppercase. For nested configurations, use double underscores (`__`) as separators.

### Example: Setting Environment Variables

```bash
# Set environment
STOCKER_ENVIRONMENT=production

# Set data directory
STOCKER_DATA_DIR=/app/data

# Set nested configuration values
STOCKER_PORTFOLIO_CONFIG__RISK_FREE_RATE=0.05
STOCKER_MODEL_CONFIG__DEFAULT_MODEL=xgboost
STOCKER_DATA_SOURCE_CONFIG__ALPHA_VANTAGE_ENABLED=true
STOCKER_DATA_SOURCE_CONFIG__ALPHA_VANTAGE_API_KEY=your_api_key
```

## Programmatic Configuration

You can also modify the configuration directly in code:

```python
from src.unified_config import get_config, Environment

# Get the configuration instance
config = get_config()

# Modify configuration values
config.environment = Environment.PRODUCTION
config.data_dir = "/app/data"

# Modify nested configurations
config.portfolio_config.risk_free_rate = 0.05
config.model_config.default_model = "xgboost"
config.data_source_config.alpha_vantage_enabled = True
config.data_source_config.alpha_vantage_api_key = "your_api_key"

# Save configuration to file
config.save_to_file("config.yaml")
```

## Configuration Sections

### Environment

```yaml
environment: development  # Options: development, testing, production
```

### Project Paths

```yaml
data_dir: data
log_dir: logs
```

### Database Settings

```yaml
mongodb_uri: mongodb://localhost:27017
db_name: stocker_db
```

### Data Source Configuration

```yaml
data_source_config:
  yahoo_finance_enabled: true
  yahoo_cache_days: 7
  alpha_vantage_enabled: false
  alpha_vantage_api_key: "YOUR_API_KEY_HERE"
  alpha_vantage_requests_per_minute: 5
  default_start_date: "2010-01-01"
  default_end_date: ""  # Empty means current date
  data_cache_dir: cache
```

### Model Configuration

```yaml
model_config:
  default_model: ensemble
  model_save_dir: models
  
  # LSTM settings
  lstm_units: 50
  lstm_dropout: 0.2
  lstm_recurrent_dropout: 0.2
  lstm_epochs: 100
  lstm_batch_size: 32
  
  # XGBoost settings
  xgboost_max_depth: 6
  xgboost_learning_rate: 0.01
  xgboost_n_estimators: 1000
  xgboost_objective: reg:squarederror
  
  # LightGBM settings
  lightgbm_max_depth: 6
  lightgbm_learning_rate: 0.01
  lightgbm_n_estimators: 1000
  lightgbm_objective: regression
  
  # Ensemble settings
  ensemble_models:
    - lstm
    - xgboost
    - lightgbm
  ensemble_weights:
    - 0.4
    - 0.3
    - 0.3
```

### API Configuration

```yaml
api_config:
  host: 127.0.0.1
  port: 8000
  debug: false
  enable_docs: true
  cors_origins:
    - "*"
  rate_limit_enabled: false
  rate_limit_requests: 100
  rate_limit_period_seconds: 3600
```

### Portfolio Configuration

```yaml
portfolio_config:
  risk_free_rate: 0.04
  benchmark_symbol: SPY
  rebalance_frequency: monthly
  max_portfolio_size: 20
  optimization_method: efficient_frontier
  risk_tolerance: 0.5
```

## Best Practices

1. **Environment-Specific Configurations**: Create separate configuration files for different environments (development, testing, production).

2. **Sensitive Information**: Store sensitive information like API keys in environment variables rather than configuration files.

3. **Version Control**: Include a sample configuration file in version control with placeholder values, but exclude the actual configuration file with real values.

4. **Validation**: Validate configuration values when loading to ensure they meet requirements.

5. **Documentation**: Document all configuration options and their effects on the system.