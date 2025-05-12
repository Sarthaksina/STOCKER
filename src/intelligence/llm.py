"""
LLM utilities for STOCKER Pro.

This module provides functionality for interacting with Large Language Models (LLMs)
for text generation, summarization, and other NLP tasks.
"""

import os
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import requests

from src.core.config import Config
from src.core.exceptions import LLMError
from src.core.logging import logger

class LLMClient:
    """
    LLM client for text generation and processing.
    
    Provides a unified interface to various LLM providers (OpenAI, HuggingFace, etc.)
    for text generation, summarization, and other NLP tasks.
    """
    
    def __init__(self, config=None):
        """
        Initialize the LLM client.
        
        Args:
            config: Optional configuration for the LLM client
        """
        self.config = config or {}
        self.provider = self.config.get('llm_provider', 'openai').lower()
        self.model = self.config.get('llm_model', 'gpt-3.5-turbo')
        self.api_key = self._get_api_key()
        self.client = None  # Will be initialized on demand
    
    def _get_api_key(self) -> str:
        """Get API key for the LLM provider."""
        # First try from config
        api_key = self.config.get('api_key', None)
        
        # Then try from environment variables
        if not api_key:
            if self.provider == 'openai':
                api_key = os.environ.get('OPENAI_API_KEY', '')
            elif self.provider == 'huggingface':
                api_key = os.environ.get('HUGGINGFACE_API_KEY', '')
            elif self.provider == 'anthropic':
                api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        
        if not api_key:
            logger.warning(f"No API key found for {self.provider}")
        
        return api_key
    
    def initialize(self):
        """Initialize the LLM client if not already initialized."""
        if self.client is not None:
            return
        
        try:
            if self.provider == 'openai':
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            elif self.provider == 'huggingface':
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=self.api_key)
            elif self.provider == 'anthropic':
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                raise LLMError(f"Unsupported LLM provider: {self.provider}")
                
        except ImportError as e:
            raise LLMError(f"Failed to import {self.provider} dependencies: {e}")
        except Exception as e:
            raise LLMError(f"Failed to initialize {self.provider} client: {e}")
    
    def complete(self, prompt: str, 
               max_tokens: int = 500,
               temperature: float = 0.7,
               stop: Optional[List[str]] = None) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: Text prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stop: Optional list of strings that stop generation
            
        Returns:
            Generated text
        """
        self.initialize()
        
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop
                )
                return response.choices[0].message.content
                
            elif self.provider == 'huggingface':
                response = self.client.text_generation(
                    prompt,
                    model=self.model,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop if stop else []
                )
                return response
                
            elif self.provider == 'anthropic':
                response = self.client.completions.create(
                    model=self.model,
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop if stop else []
                )
                return response.completion
                
            else:
                raise LLMError(f"Unsupported LLM provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating text with {self.provider}: {e}")
            raise LLMError(f"Failed to generate text: {e}")
    
    def chat(self, messages: List[Dict[str, str]], 
           max_tokens: int = 500,
           temperature: float = 0.7) -> str:
        """
        Chat with the LLM using a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Generated response text
        """
        self.initialize()
        
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            elif self.provider == 'huggingface':
                # Convert messages to prompt
                prompt = ""
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    if role == 'system':
                        prompt += f"System: {content}\n\n"
                    elif role == 'user':
                        prompt += f"User: {content}\n\n"
                    elif role == 'assistant':
                        prompt += f"Assistant: {content}\n\n"
                
                prompt += "Assistant: "
                
                response = self.client.text_generation(
                    prompt,
                    model=self.model,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                return response
                
            elif self.provider == 'anthropic':
                # Convert messages to prompt
                prompt = ""
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    if role == 'system':
                        prompt += f"System: {content}\n\n"
                    elif role == 'user':
                        prompt += f"Human: {content}\n\n"
                    elif role == 'assistant':
                        prompt += f"Assistant: {content}\n\n"
                
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt + "Assistant:",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                return response.completion
                
            else:
                raise LLMError(f"Unsupported LLM provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating chat response with {self.provider}: {e}")
            raise LLMError(f"Failed to generate chat response: {e}")
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text using the LLM.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary
            
        Returns:
            Generated summary
        """
        if len(text) < max_length:
            return text
        
        prompt = f"""
        Please summarize the following text in a concise manner, capturing the key points:
        
        {text}
        
        Summary:
        """
        
        return self.complete(prompt, max_tokens=max_length)
    
    def extract_insights(self, text: str, entity: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Extract insights from text.
        
        Args:
            text: Text to extract insights from
            entity: Optional entity to focus on (company, market, etc.)
            
        Returns:
            List of extracted insights
        """
        entity_prefix = f"about {entity}" if entity else ""
        
        prompt = f"""
        Please extract the key insights {entity_prefix} from the following text.
        Return a list of insights, each with a brief explanation.
        Focus on actionable information, market trends, and important facts.
        
        {text}
        
        Insights:
        """
        
        response = self.complete(prompt, max_tokens=500)
        
        # Parse response into list of insights
        insights = []
        lines = response.strip().split('\n')
        current_insight = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new insight (assumes numbered or bulleted list)
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '•', '-', '*')):
                if current_insight:
                    insights.append(current_insight)
                
                # Remove the bullet or number
                for prefix in ('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '•', '-', '*'):
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                
                current_insight = {'insight': line, 'explanation': ''}
            elif current_insight:
                if current_insight['explanation']:
                    current_insight['explanation'] += ' ' + line
                else:
                    current_insight['explanation'] = line
        
        # Add the last insight
        if current_insight:
            insights.append(current_insight)
            
        return insights


def get_embedding_model(config: Dict[str, Any] = None) -> Callable:
    """
    Get an embedding model function based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Function that takes texts and returns embeddings
    """
    config = config or {}
    provider = config.get('embedding_provider', 'openai').lower()
    model_name = config.get('embedding_model', 'text-embedding-ada-002')
    
    try:
        if provider == 'openai':
            from openai import OpenAI
            
            # Get API key
            api_key = config.get('api_key', os.environ.get('OPENAI_API_KEY', ''))
            if not api_key:
                raise LLMError("No OpenAI API key found")
            
            # Initialize client
            client = OpenAI(api_key=api_key)
            
            # Define embedding function
            def get_embeddings(texts: List[str]) -> List[List[float]]:
                response = client.embeddings.create(
                    model=model_name,
                    input=texts
                )
                return [item.embedding for item in response.data]
            
            return get_embeddings
            
        elif provider == 'huggingface':
            from sentence_transformers import SentenceTransformer
            
            # Initialize model
            model = SentenceTransformer(model_name)
            
            # Define embedding function
            def get_embeddings(texts: List[str]) -> List[List[float]]:
                embeddings = model.encode(texts)
                return embeddings.tolist()
            
            return get_embeddings
            
        else:
            raise LLMError(f"Unsupported embedding provider: {provider}")
            
    except ImportError as e:
        raise LLMError(f"Failed to import {provider} embedding dependencies: {e}")
    except Exception as e:
        raise LLMError(f"Failed to initialize {provider} embedding model: {e}") 


# Hugging Face transformers utilities
try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None
    logger.warning("transformers not installed. Please add to requirements.txt if you want LLM summarization.")


def summarize_text(text: str, max_length: int = 60) -> str:
    """
    Summarize the given text using a Hugging Face summarization pipeline.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of the summary
        
    Returns:
        Summarized text
    """
    if hf_pipeline is None:
        logger.warning("Summarization pipeline not available.")
        return text
    try:
        # Default to a lightweight model for summarization
        model_name = os.environ.get('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn')
        summarizer = hf_pipeline("summarization", model=model_name)
        summary = summarizer(text, max_length=max_length, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return text


def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of the given text using a Hugging Face sentiment-analysis pipeline.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment label (POSITIVE, NEGATIVE, or NEUTRAL)
    """
    if hf_pipeline is None:
        logger.warning("Sentiment pipeline not available.")
        return "NEUTRAL"
    try:
        # Default to a lightweight model for sentiment analysis
        model_name = os.environ.get('SENTIMENT_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english')
        sentiment_analyzer = hf_pipeline("sentiment-analysis", model=model_name)
        result = sentiment_analyzer(text)
        return result[0]['label'].upper()
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "NEUTRAL"