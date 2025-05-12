@echo off
REM STOCKER Pro Runner Script
REM This script sets the Alpha Vantage API key and runs the STOCKER Pro application

REM Activate the virtual environment
call .\venv\Scripts\activate.bat

REM Load the Alpha Vantage API key from .env file
if exist .env (
    for /F "tokens=*" %%A in (.env) do (
        set %%A
    )
)

REM Check if API key is set
if not defined ALPHA_VANTAGE_API_KEY (
    echo WARNING: ALPHA_VANTAGE_API_KEY not found in .env file.
    echo Please create a .env file with your API key in the format: ALPHA_VANTAGE_API_KEY=your_key_here
)

REM Check if a command was provided
if "%1"=="" (
    echo Usage: run_stocker.bat [command] [args...]
    echo Example: run_stocker.bat data AAPL
    exit /b 1
)

REM Run the STOCKER Pro application with the provided arguments
python -m src.app %*
