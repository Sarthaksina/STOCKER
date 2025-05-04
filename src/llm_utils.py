from src.utils.logger import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

import os
from typing import List
from src.constant.constants import DEFAULT_SUMMARIZATION_MODEL, DEFAULT_SENTIMENT_MODEL

try:
    from transformers import pipeline
except ImportError:
    pipeline = None
    logger.error("transformers not installed. Please add to requirements.txt if you want LLM summarization.")

# Use Hugging Face transformers for summarization and sentiment

def summarize_text(text: str, max_length: int = 60) -> str:
    """
    Summarize the given text using a Hugging Face summarization pipeline.
    """
    if pipeline is None:
        logger.warning("Summarization pipeline not available.")
        return text
    try:
        summarizer = pipeline("summarization", model=DEFAULT_SUMMARIZATION_MODEL)
        summary = summarizer(text, max_length=max_length, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return text

def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of the given text using a Hugging Face sentiment-analysis pipeline.
    Returns 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.
    """
    if pipeline is None:
        logger.warning("Sentiment pipeline not available.")
        return "NEUTRAL"
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model=DEFAULT_SENTIMENT_MODEL)
        result = sentiment_analyzer(text)
        return result[0]['label'].upper()
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "NEUTRAL"
