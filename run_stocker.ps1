# STOCKER Pro Runner Script
# This script sets the Alpha Vantage API key and runs the STOCKER Pro application

# Activate the virtual environment
& .\venv\Scripts\Activate.ps1

# Set the Alpha Vantage API key
$env:ALPHA_VANTAGE_API_KEY="3SYPIGG1DSL1NN3Y"

# Check if a command was provided
if ($args.Count -eq 0) {
    Write-Host "Usage: .\run_stocker.ps1 [command] [args...]"
    Write-Host "Example: .\run_stocker.ps1 data AAPL"
    exit 1
}

# Run the STOCKER Pro application with the provided arguments
python -m src.app $args
