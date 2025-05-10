"""
STOCKER Pro: Enhanced Financial Market Intelligence Platform
"""

__version__ = "1.0.0"

from src.core.config import config
from src.core.logging import logger, setup_logging
from src.data.manager import DataManager
from src.features.engineering import FeatureEngineering
from src.db.models import StockDataResponse, CompanyInfo
