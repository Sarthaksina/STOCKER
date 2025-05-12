# STOCKER Pro: Enhanced Financial Market Intelligence Platform

STOCKER Pro is an advanced financial market intelligence platform that combines real-time market data analysis, ensemble machine learning predictions, and RAG-powered insights.

## Project Structure

```
stocker/
├── data/                 # Raw and processed data
├── docs/                 # Documentation files
├── examples/             # Example usage scripts
├── logs/                 # Log files 
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── api/              # API endpoints
│   │   ├── routes/       # API route modules
│   │   │   ├── portfolio.py # Portfolio-related endpoints
│   │   │   ├── analysis.py  # Financial analysis endpoints
│   │   │   ├── market_data.py # Market data endpoints
│   │   │   └── agent.py    # Agent interaction endpoints
│   │   ├── dependencies.py # FastAPI dependencies
│   │   └── server.py     # FastAPI application
│   ├── cli/              # Command-line interface
│   │   └── commands.py   # CLI commands implementation
│   ├── core/             # Core functionality
│   │   ├── config.py     # Configuration management
│   │   ├── logging.py    # Logging utilities
│   │   ├── exceptions.py # Exception definitions
│   │   ├── utils.py      # General utilities
│   │   └── artifacts.py  # Artifact management
│   ├── data/             # Data ingestion and management
│   │   ├── clients/      # Data source clients
│   │   ├── ingestion.py  # Data ingestion logic
│   │   ├── manager.py    # Data management utilities
│   │   └── schemas.py    # Data validation schemas
│   ├── db/               # Database models and connections
│   │   ├── models.py     # Database models
│   │   └── session.py    # Database connection management
│   ├── features/         # Feature engineering and analysis
│   │   ├── engineering.py # Feature engineering utilities
│   │   ├── indicators.py # Technical indicators
│   │   ├── analytics.py  # Analytics functionality
│   │   └── portfolio/    # Portfolio optimization
│   │       ├── core.py   # Core portfolio functionality
│   │       ├── optimization.py # Portfolio optimization
│   │       ├── risk.py   # Risk analysis
│   │       └── visualization.py # Portfolio visualization
│   ├── intelligence/     # Intelligence layer (RAG, LLM)
│   │   ├── llm.py        # LLM utilities
│   │   ├── news.py       # News processing
│   │   ├── rag.py        # RAG pipeline
│   │   └── vector_store.py # Vector database interactions
│   ├── ml/               # Machine learning models
│   │   ├── base.py       # Base model abstract class
│   │   ├── models.py     # Model implementations
│   │   ├── evaluation.py # Model evaluation utilities
│   │   └── pipelines.py  # ML pipelines
│   ├── services/         # Business logic services
│   │   ├── auth.py       # Authentication service
│   │   ├── portfolio.py  # Portfolio service
│   │   ├── prediction.py # Prediction service
│   │   └── training.py   # Model training service
│   └── ui/               # User interface components
│       ├── components.py # UI components
│       └── dashboard.py  # Main dashboard
├── tests/                # Test files
├── .gitignore            # Git ignore file
├── pyproject.toml        # Project metadata and dependencies
├── requirements.txt      # Package dependencies
└── README.md             # Project overview
```

## Recent Reorganization and Consolidation

We have recently completed a comprehensive reorganization of the STOCKER codebase to improve maintainability, readability, and testability. Key improvements include:

### 1. Module Consolidation

* **Core Module**: Consolidated configuration, logging, exceptions, and utilities into a clean core module
* **Features Module**: Consolidated technical indicators and feature engineering functionality
* **API Layer**: Organized API endpoints into domain-specific route modules
* **Data Layer**: Structured data access clients and ingestion logic

### 2. Improved Structure

* Eliminated redundant code and duplicate functionality
* Maintained backward compatibility through careful refactoring
* Added comprehensive unit tests for all consolidated modules
* Standardized naming conventions and file organization

### 3. Technical Indicators

The technical indicators functionality has been significantly improved:

* Consolidated multiple indicator implementations into a single, comprehensive module
* Added class-based and function-based interfaces for flexibility
* Expanded indicator coverage with volatility and trend indicators
* Improved error handling and validation

## Getting Started

1. **Installation**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/stocker-pro.git
   cd stocker-pro
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**:
   - Copy `config.yaml.example` to `config.yaml`
   - Add your API keys and configure database settings

3. **Running the application**:
   ```bash
   # Run the API server
   python -m src.app --api
   
   # Run the dashboard UI
   python -m src.app --ui
   
   # Run both API and UI
   python -m src.app --all
   ```

4. **Using the CLI**:
   ```bash
   # Get stock data
   python -m src.cli.commands data get AAPL --start-date 2023-01-01 --end-date 2023-12-31
   
   # Get company info
   python -m src.cli.commands data company MSFT
   
   # Make a prediction
   python -m src.cli.commands predict stock TSLA --model-type xgboost --horizon 5d
   
   # Optimize a portfolio
   python -m src.cli.commands portfolio optimize AAPL MSFT GOOG --method efficient_frontier
   ```

5. **Using the API**:
   ```bash
   # Example: Get stock data
   curl http://localhost:8000/data/stock/AAPL?start_date=2023-01-01&end_date=2023-12-31
   
   # Example: Make a prediction
   curl -X POST http://localhost:8000/predict/stock \
     -H "Content-Type: application/json" \
     -d '{"symbol": "TSLA", "model_type": "xgboost", "prediction_horizon": "5d"}'
   ```

## Key Features

- Multi-source financial data collection with Alpha Vantage integration
- Advanced feature engineering for financial time series
- Ensemble forecasting with LSTM, XGBoost, and LightGBM
- Portfolio optimization and risk assessment
- RAG-powered financial insights
- Interactive dashboard with advanced financial visualizations
- RESTful API for programmatic access
- Command-line interface for automation

## Dependencies

- pandas, numpy: Data manipulation
- pymongo: Database connectivity
- ta-lib: Technical indicators
- scikit-learn: Machine learning utilities
- keras, tensorflow: Deep learning
- xgboost, lightgbm: Gradient boosting
- fastapi: API framework
- plotly: Interactive visualizations
- pydantic: Data validation
- pandera: DataFrame validation
- langchain, chromadb: RAG implementation

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request