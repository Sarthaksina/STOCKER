"""Example script demonstrating the unified configuration system.

This script shows how to use the STOCKER Pro configuration system in practice,
including loading from files, environment variables, and programmatic modification.
"""

import os
import pathlib
from src.unified_config import get_config, Environment, StockerConfig


def print_config_section(title, config_dict):
    """Helper function to print a configuration section"""
    print(f"\n{title}:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")


def main():
    """Main function demonstrating configuration usage"""
    print("STOCKER Pro Configuration Example\n")
    
    # 1. Load default configuration
    print("1. Loading default configuration...")
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Risk-Free Rate: {config.portfolio_config.risk_free_rate}")
    print(f"Default Model: {config.model_config.default_model}")
    
    # 2. Load from file
    print("\n2. Loading from config.yaml file...")
    config_path = pathlib.Path(os.getcwd()) / "config.yaml"
    if config_path.exists():
        # Reset the global instance to force reloading
        import src.unified_config
        src.unified_config._config_instance = None
        
        config = get_config(config_path)
        print(f"Loaded configuration from {config_path}")
        print(f"Environment: {config.environment}")
        print(f"Data Directory: {config.data_dir}")
    else:
        print(f"Config file {config_path} not found. Skipping this step.")
    
    # 3. Override with environment variables
    print("\n3. Overriding with environment variables...")
    print("Setting STOCKER_DATA_DIR=/env/data")
    print("Setting STOCKER_PORTFOLIO_CONFIG__RISK_FREE_RATE=0.05")
    
    os.environ["STOCKER_DATA_DIR"] = "/env/data"
    os.environ["STOCKER_PORTFOLIO_CONFIG__RISK_FREE_RATE"] = "0.05"
    
    # Reset the global instance to force reloading
    import src.unified_config
    src.unified_config._config_instance = None
    
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Risk-Free Rate: {config.portfolio_config.risk_free_rate}")
    
    # Clean up environment variables
    del os.environ["STOCKER_DATA_DIR"]
    del os.environ["STOCKER_PORTFOLIO_CONFIG__RISK_FREE_RATE"]
    
    # 4. Programmatic configuration
    print("\n4. Programmatic configuration...")
    # Create a new configuration instance
    custom_config = StockerConfig()
    custom_config.environment = Environment.PRODUCTION
    custom_config.data_dir = "/app/data"
    custom_config.mongodb_uri = "mongodb://user:password@mongodb:27017"
    
    # Configure data sources
    custom_config.data_source_config.alpha_vantage_enabled = True
    custom_config.data_source_config.alpha_vantage_api_key = "demo_api_key"
    
    # Configure models
    custom_config.model_config.default_model = "xgboost"
    custom_config.model_config.xgboost_learning_rate = 0.005
    
    # Configure portfolio
    custom_config.portfolio_config.risk_free_rate = 0.03
    custom_config.portfolio_config.optimization_method = "min_variance"
    
    # Print the configuration
    print(f"Environment: {custom_config.environment}")
    print(f"Data Directory: {custom_config.data_dir}")
    print(f"MongoDB URI: {custom_config.mongodb_uri}")
    
    # Print data source configuration
    print_config_section("Data Source Config", custom_config.data_source_config.to_dict())
    
    # Print model configuration
    print_config_section("Model Config", custom_config.model_config.to_dict())
    
    # Print portfolio configuration
    print_config_section("Portfolio Config", custom_config.portfolio_config.to_dict())
    
    # 5. Save configuration to file
    print("\n5. Saving configuration to file...")
    save_path = pathlib.Path(os.getcwd()) / "custom_config.yaml"
    custom_config.save_to_file(save_path)
    print(f"Configuration saved to {save_path}")
    
    # 6. Load the saved configuration
    print("\n6. Loading the saved configuration...")
    loaded_config = get_config(save_path)
    print(f"Environment: {loaded_config.environment}")
    print(f"Data Directory: {loaded_config.data_dir}")
    print(f"Alpha Vantage Enabled: {loaded_config.data_source_config.alpha_vantage_enabled}")
    print(f"Default Model: {loaded_config.model_config.default_model}")
    
    print("\nConfiguration example completed.")


if __name__ == "__main__":
    main()