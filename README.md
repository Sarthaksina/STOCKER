# STOCKER Pro: Enhanced Financial Market Intelligence Platform

STOCKER Pro is an advanced financial market intelligence platform that combines real-time market data analysis, ensemble machine learning predictions, and RAG-powered insights. The platform integrates gradient boosting machines (XGBoost, LightGBM) alongside LSTM models to provide accurate market forecasts while leveraging cloud-based infrastructure (ThunderCompute) to overcome local computing limitations.

## Features

- **Advanced Machine Learning Pipeline**: Ensemble models combining LSTM, XGBoost, and LightGBM for superior prediction accuracy
- **Cloud-Based Training**: Integration with ThunderCompute for high-performance model training
- **Financial-Specific Metrics**: Directional accuracy, Sharpe ratio, Sortino ratio, maximum drawdown analysis
- **Portfolio Optimization**: Modern Portfolio Theory implementation with efficient frontier analysis
- **Comprehensive Technical Indicators**: MACD, RSI, Bollinger Bands, and more
- **Interactive Visualizations**: Price predictions with confidence intervals, feature importance, portfolio composition
- **RAG System**: Financial news and reports integration for contextual insights

## Installation

```bash
# Clone the repository
git clone https://github.com/stockerpro/stocker.git
cd stocker

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```
