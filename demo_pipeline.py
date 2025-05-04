# demo_pipeline.py
"""
Demo entry point for running the pipeline end-to-end.
"""
from src.pipeline.config_entity import DataIngestionConfig, NewsAgentConfig
from src.pipeline.pipeline import StockerPipeline
from src.ingestion.data_ingestion import get_all_nse_symbols

if __name__ == "__main__":
    symbols = get_all_nse_symbols()[:5]  # For demo, limit to 5 stocks
    stock_config = DataIngestionConfig(symbols=symbols, start_date="2023-01-01", end_date="2025-01-01")
    news_config = NewsAgentConfig(max_articles=3, summarize=True, sentiment=True)

    pipeline = StockerPipeline(stock_config, news_config)
    pipeline.run()
