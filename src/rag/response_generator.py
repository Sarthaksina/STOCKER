"""
Response generation system for the RAG pipeline.
Generates insightful answers based on retrieved financial information.
"""
import logging
import json
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.rag.query_formulation import optimize_query
from src.rag.news_collector import get_news_collector
from src.configuration.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Builds relevant context from retrieved documents for response generation.
    """
    def __init__(self, 
                max_context_length: int = 4000,
                relevance_threshold: float = 0.5):
        """
        Initialize context builder with configuration.
        
        Args:
            max_context_length: Maximum context length in characters
            relevance_threshold: Minimum relevance score for inclusion
        """
        self.max_context_length = max_context_length
        self.relevance_threshold = relevance_threshold
    
    def build_context(self, articles: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved articles.
        
        Args:
            articles: List of retrieved article dictionaries
        
        Returns:
            Formatted context string
        """
        if not articles:
            return "No relevant information found."
        
        # Filter by relevance threshold
        relevant_articles = [
            article for article in articles 
            if article.get("relevance", 0) >= self.relevance_threshold
        ]
        
        if not relevant_articles:
            # Fall back to using all articles if none pass threshold
            relevant_articles = articles
        
        # Sort by relevance (highest first)
        relevant_articles = sorted(
            relevant_articles, 
            key=lambda x: x.get("relevance", 0),
            reverse=True
        )
        
        # Build context with formatting
        context_parts = []
        total_length = 0
        
        for i, article in enumerate(relevant_articles):
            # Extract metadata
            source = article["metadata"].get("source", "Unknown Source")
            published = article["metadata"].get("published", "Unknown Date")
            title = article["metadata"].get("title", "Untitled")
            
            # Format date for readability
            try:
                date_obj = datetime.fromisoformat(published)
                published = date_obj.strftime("%B %d, %Y")
            except (ValueError, TypeError):
                pass
            
            # Create article reference heading
            article_ref = f"[Article {i+1} - {source} - {published}]"
            
            # Get text content
            content = article["text"]
            
            # Limit content length if needed
            max_article_length = min(1500, self.max_context_length // 2)
            if len(content) > max_article_length:
                content = content[:max_article_length] + "..."
            
            # Create formatted article entry
            article_entry = f"{article_ref}\nTitle: {title}\n{content}\n\n"
            
            # Check if adding this would exceed max context length
            if total_length + len(article_entry) > self.max_context_length:
                # If this is the first article, add a truncated version
                if i == 0:
                    truncate_to = self.max_context_length - len(article_ref) - 20
                    if truncate_to > 100:
                        content = content[:truncate_to] + "..."
                        article_entry = f"{article_ref}\nTitle: {title}\n{content}\n\n"
                        context_parts.append(article_entry)
                break
            
            # Add article to context
            context_parts.append(article_entry)
            total_length += len(article_entry)
        
        # Combine context parts
        if context_parts:
            return "".join(context_parts)
        else:
            return "Context exceeds maximum length. Unable to provide relevant information."


class TemplateManager:
    """
    Manages response templates for various types of financial queries.
    """
    def __init__(self):
        """Initialize template manager with predefined templates."""
        self.templates = {
            "stock_performance": (
                "Based on the retrieved information, {ticker} ({company_name}) "
                "has shown {performance_trend} performance recently. "
                "{key_points_about_stock} "
                "The recent news indicates {news_sentiment}. "
                "{additional_insights}"
            ),
            
            "market_analysis": (
                "Market Analysis:\n"
                "Current Trend: {market_trend}\n"
                "Key Factors: {key_factors}\n"
                "Recent Developments: {recent_developments}\n"
                "Outlook: {market_outlook}\n"
                "{additional_insights}"
            ),
            
            "company_news": (
                "Recent news about {company_name} ({ticker}):\n"
                "{news_summary}\n"
                "Potential Impact: {potential_impact}\n"
                "{additional_insights}"
            ),
            
            "economic_indicator": (
                "Analysis of {indicator_name}:\n"
                "Current Value: {current_value}\n"
                "Trend: {indicator_trend}\n"
                "Implications: {implications}\n"
                "{additional_insights}"
            ),
            
            "investment_strategy": (
                "Investment Strategy Insights:\n"
                "Current Market Environment: {market_environment}\n"
                "Recommended Approach: {recommended_approach}\n"
                "Key Considerations: {key_considerations}\n"
                "Risk Factors: {risk_factors}\n"
                "{additional_insights}"
            ),
            
            "general": (
                "{main_insight}\n\n"
                "Based on the available information:\n"
                "• {point_1}\n"
                "• {point_2}\n"
                "• {point_3}\n"
                "{additional_insights}"
            )
        }
    
    def get_template(self, query_type: str) -> str:
        """
        Get appropriate template for the query type.
        
        Args:
            query_type: Type of query
        
        Returns:
            Template string
        """
        return self.templates.get(query_type, self.templates["general"])


class ResponseGenerator:
    """
    Generates insightful responses to financial queries using retrieved context.
    """
    def __init__(self):
        """Initialize response generator components."""
        self.news_collector = get_news_collector()
        self.context_builder = ContextBuilder()
        self.template_manager = TemplateManager()
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine the type of financial query.
        
        Args:
            query: User query string
        
        Returns:
            Query type identifier
        """
        query_lower = query.lower()
        
        # Check for stock ticker pattern (1-5 uppercase letters)
        if re.search(r'\b[A-Z]{1,5}\b', query):
            return "stock_performance"
        
        # Check for market analysis queries
        if re.search(r'market|index|trends|s&p|dow|nasdaq', query_lower):
            return "market_analysis"
        
        # Check for company news queries
        if re.search(r'news|announcement|press release|report', query_lower):
            return "company_news"
        
        # Check for economic indicator queries
        if re.search(r'gdp|inflation|unemployment|interest rate|fed|fomc|cpi', query_lower):
            return "economic_indicator"
        
        # Check for investment strategy queries
        if re.search(r'invest|portfolio|strategy|allocation|risk|dividend', query_lower):
            return "investment_strategy"
        
        # Default to general
        return "general"
    
    def _extract_template_variables(self, 
                                   query: str, 
                                   context: str, 
                                   query_type: str) -> Dict[str, str]:
        """
        Extract variables to fill response template.
        
        Args:
            query: User query string
            context: Retrieved context
            query_type: Type of query
        
        Returns:
            Dictionary of template variables
        """
        variables = {}
        
        # Extract stock ticker and company name if relevant
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', query)
        if ticker_match:
            variables["ticker"] = ticker_match.group(0)
            # Try to find company name in context
            company_pattern = rf'{variables["ticker"]}\s+\(([^)]+)\)'
            company_match = re.search(company_pattern, context)
            if company_match:
                variables["company_name"] = company_match.group(1)
            else:
                variables["company_name"] = "the company"
        
        # Set defaults based on query type
        if query_type == "stock_performance":
            variables.setdefault("ticker", "the stock")
            variables.setdefault("company_name", "the company")
            
            # Try to determine performance trend from context
            if re.search(r'up|gain|rise|grow|positive|increase', context, re.IGNORECASE):
                variables["performance_trend"] = "positive"
            elif re.search(r'down|loss|fall|drop|negative|decrease', context, re.IGNORECASE):
                variables["performance_trend"] = "negative"
            else:
                variables["performance_trend"] = "mixed"
            
            # Extract key points
            variables["key_points_about_stock"] = self._extract_key_points(context)
            
            # Determine news sentiment
            if re.search(r'positive|optimistic|bullish', context, re.IGNORECASE):
                variables["news_sentiment"] = "a positive outlook"
            elif re.search(r'negative|pessimistic|bearish', context, re.IGNORECASE):
                variables["news_sentiment"] = "concerns about future performance"
            else:
                variables["news_sentiment"] = "mixed sentiment among analysts"
                
        elif query_type == "market_analysis":
            # Try to determine market trend from context
            if re.search(r'bull|up|gain|rise|grow|positive', context, re.IGNORECASE):
                variables["market_trend"] = "Bullish"
            elif re.search(r'bear|down|loss|fall|drop|negative', context, re.IGNORECASE):
                variables["market_trend"] = "Bearish"
            else:
                variables["market_trend"] = "Mixed/Neutral"
                
            # Extract key factors
            variables["key_factors"] = self._extract_key_points(context, "factors")
            
            # Recent developments
            variables["recent_developments"] = self._extract_recent_developments(context)
            
            # Market outlook
            if re.search(r'optimistic|positive outlook|growth expected', context, re.IGNORECASE):
                variables["market_outlook"] = "Positive with potential for continued growth"
            elif re.search(r'cautious|uncertain|potential risks', context, re.IGNORECASE):
                variables["market_outlook"] = "Cautious with notable uncertainties ahead"
            else:
                variables["market_outlook"] = "Mixed signals with both opportunities and challenges"
                
        # Default values for other templates
        variables.setdefault("main_insight", self._extract_main_insight(context))
        variables.setdefault("point_1", self._extract_key_points(context, count=1))
        variables.setdefault("point_2", self._extract_secondary_point(context))
        variables.setdefault("point_3", self._extract_additional_point(context))
        
        # Additional insights for all templates
        variables["additional_insights"] = self._generate_additional_insights(context, query)
        
        return variables
    
    def _extract_key_points(self, context: str, topic: str = "performance", count: int = 1) -> str:
        """
        Extract key points from context about a specific topic.
        
        Args:
            context: Retrieved context
            topic: Topic to extract points about
            count: Number of points to extract
        
        Returns:
            String with key points
        """
        # Extract sentences containing key terms related to the topic
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        topic_terms = {
            "performance": ["price", "stock", "shares", "value", "market cap", "growth", "decline"],
            "factors": ["factors", "due to", "because", "influenced by", "result of", "driven by"],
            "outlook": ["outlook", "expect", "future", "forecast", "predict", "anticipate"],
            "general": ["important", "significant", "key", "critical", "notable", "major"]
        }
        
        terms = topic_terms.get(topic, topic_terms["general"])
        
        # Find sentences containing topic terms
        relevant_sentences = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in terms):
                relevant_sentences.append(sentence.strip())
        
        if not relevant_sentences:
            if topic == "performance":
                return "Specific performance details are not available in the current information."
            return "Specific information on this topic is not available in the current context."
        
        # Return specified number of points
        if count == 1:
            return relevant_sentences[0]
        
        # Return multiple points as bulleted list
        points = relevant_sentences[:min(count, len(relevant_sentences))]
        return "\n".join([f"• {point}" for point in points])
    
    def _extract_main_insight(self, context: str) -> str:
        """
        Extract the main insight from the context.
        
        Args:
            context: Retrieved context
        
        Returns:
            Main insight as a string
        """
        # Look for sentences with strong indicator words
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        indicator_words = [
            "important", "significant", "key", "critical", "notable",
            "major", "essential", "crucial", "primary", "fundamental"
        ]
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in indicator_words):
                return sentence.strip()
        
        # If no strong indicators, return the first substantial sentence
        for sentence in sentences:
            if len(sentence.split()) >= 8:  # Avoid short sentences
                return sentence.strip()
        
        return "Based on the available information, a clear insight could not be determined."
    
    def _extract_secondary_point(self, context: str) -> str:
        """
        Extract a secondary point from the context.
        
        Args:
            context: Retrieved context
        
        Returns:
            Secondary point as a string
        """
        # Look for sentences with quantitative information
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Look for sentences with numbers or percentages
        for sentence in sentences:
            if re.search(r'\d+%|\d+\.\d+|\$\d+', sentence):
                return sentence.strip()
        
        # If no quantitative information, find a sentence with financial terms
        financial_terms = [
            "growth", "revenue", "earnings", "profit", "margin",
            "quarter", "year", "forecast", "guidance", "market"
        ]
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in financial_terms):
                return sentence.strip()
        
        return "Additional specific details are not available in the current information."
    
    def _extract_additional_point(self, context: str) -> str:
        """
        Extract an additional point from the context.
        
        Args:
            context: Retrieved context
        
        Returns:
            Additional point as a string
        """
        # Look for sentences with future indicators
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        future_indicators = [
            "will", "future", "expect", "anticipate", "forecast",
            "predict", "outlook", "guidance", "plan", "strategy"
        ]
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in future_indicators):
                return sentence.strip()
        
        # If no future indicators, find a sentence with analyst mentions
        analyst_indicators = [
            "analyst", "expert", "estimate", "recommend", "target",
            "consensus", "rating", "upgrade", "downgrade", "research"
        ]
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in analyst_indicators):
                return sentence.strip()
        
        return "Further analysis would be needed for additional insights."
    
    def _extract_recent_developments(self, context: str) -> str:
        """
        Extract recent developments from context.
        
        Args:
            context: Retrieved context
        
        Returns:
            Recent developments as a string
        """
        time_indicators = [
            "recently", "last week", "this week", "yesterday", "today",
            "this month", "last month", "last quarter", "this quarter"
        ]
        
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Find sentences with time indicators
        relevant_sentences = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for time in time_indicators for indicator in time):
                relevant_sentences.append(sentence.strip())
        
        if not relevant_sentences:
            return "No specific recent developments were identified in the current information."
        
        # Return up to 3 developments
        developments = relevant_sentences[:min(3, len(relevant_sentences))]
        return "\n".join([f"• {dev}" for dev in developments])
    
    def _generate_additional_insights(self, context: str, query: str) -> str:
        """
        Generate additional insights based on context and query.
        
        Args:
            context: Retrieved context
            query: User query
        
        Returns:
            Additional insights as a string
        """
        # Determine what's missing from the context
        has_quantitative = bool(re.search(r'\d+%|\d+\.\d+|\$\d+', context))
        has_comparison = bool(re.search(r'compared to|versus|vs\.|higher than|lower than', context, re.IGNORECASE))
        has_market_context = bool(re.search(r'market|index|industry|sector|peer', context, re.IGNORECASE))
        has_future_outlook = bool(re.search(r'outlook|future|expect|forecast|predict|anticipate', context, re.IGNORECASE))
        
        insights = []
        
        # Add missing context as needed
        if not has_quantitative:
            insights.append("Note: Specific quantitative data is not available in the current information.")
        
        if not has_comparison and ("compared" in query.lower() or "vs" in query.lower()):
            insights.append("Note: Direct comparison data is not available in the current information.")
        
        if not has_market_context and ("market" in query.lower() or "industry" in query.lower()):
            insights.append("Note: Broader market context is not fully available in the current information.")
        
        if not has_future_outlook and "outlook" in query.lower():
            insights.append("Note: Future outlook details are limited in the current information.")
        
        # Add disclaimer for investment decisions
        insights.append("This information is based on available news and should not be considered as investment advice.")
        
        if insights:
            return "\n\n" + "\n".join(insights)
        
        return ""
    
    def search_relevant_articles(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles relevant to the query.
        
        Args:
            query: User query
            n_results: Number of results to return
        
        Returns:
            List of relevant articles
        """
        # Optimize query for search
        optimized_query = optimize_query(query)
        
        # Handle if optimize_query returns a list
        if isinstance(optimized_query, list):
            # Use the first query in the list
            search_query = optimized_query[0]
        else:
            search_query = optimized_query
        
        # Search for articles
        articles = self.news_collector.search_articles(
            query=search_query,
            n_results=n_results,
            max_age_days=30  # Use relatively recent news
        )
        
        return articles
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a financial query.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with generated response and metadata
        """
        start_time = datetime.now()
        
        # Search for relevant articles
        articles = self.search_relevant_articles(query, n_results=5)
        
        # Build context from articles
        context = self.context_builder.build_context(articles)
        
        # Determine query type
        query_type = self._determine_query_type(query)
        
        # Get appropriate template
        template = self.template_manager.get_template(query_type)
        
        # Extract template variables
        variables = self._extract_template_variables(query, context, query_type)
        
        # Fill template with variables
        try:
            response_text = template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            # Fall back to general template
            template = self.template_manager.get_template("general")
            response_text = template.format(**variables)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response dictionary
        response = {
            "query": query,
            "response": response_text,
            "query_type": query_type,
            "sources": [
                {
                    "title": article["metadata"].get("title", "Untitled"),
                    "source": article["metadata"].get("source", "Unknown"),
                    "link": article["metadata"].get("link", "#"),
                    "published": article["metadata"].get("published", "Unknown")
                }
                for article in articles[:3]  # Include top 3 sources
            ],
            "processing_time": processing_time,
            "generated_at": datetime.now().isoformat()
        }
        
        return response


def generate_financial_insight(query: str) -> Dict[str, Any]:
    """
    Generate financial insight for a user query.
    
    Args:
        query: User query string
    
    Returns:
        Dictionary with generated insight and metadata
    """
    generator = ResponseGenerator()
    return generator.generate_response(query)


if __name__ == "__main__":
    # Set up basic logging for standalone usage
    logging.basicConfig(level=logging.INFO)
    
    # Example queries
    example_queries = [
        "What is the current performance of AAPL?",
        "How has the market responded to recent Fed announcements?",
        "What are the growth prospects for Tesla in 2023?",
        "Explain the impact of inflation on tech stocks"
    ]
    
    # Generate insights for example queries
    for query in example_queries:
        print(f"\n\n=== Query: {query} ===\n")
        response = generate_financial_insight(query)
        print(response["response"])
        print("\nSources:")
        for source in response["sources"]:
            print(f"- {source['title']} ({source['source']})")
        print(f"\nProcessing time: {response['processing_time']:.2f} seconds") 