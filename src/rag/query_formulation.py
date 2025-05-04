"""
Query formulation system for the RAG pipeline.
Generates optimized queries for retrieving relevant financial knowledge.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from src.configuration.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

class QueryFormatter:
    """
    Formats user queries for optimal retrieval from the RAG system.
    Applies domain-specific expansions and reformulations for finance queries.
    """
    def __init__(self, 
                stock_symbols_path: Optional[str] = None,
                financial_terms_path: Optional[str] = None):
        """
        Initialize query formatter with financial knowledge.
        
        Args:
            stock_symbols_path: Path to stock symbols dictionary file
            financial_terms_path: Path to financial terms dictionary file
        """
        self.stock_symbols = self._load_symbols(stock_symbols_path)
        self.financial_terms = self._load_terms(financial_terms_path)
    
    def _load_symbols(self, path: Optional[str]) -> Dict[str, str]:
        """
        Load stock symbols dictionary from file.
        
        Args:
            path: Path to stock symbols file
        
        Returns:
            Dictionary mapping symbols to company names
        """
        symbols = {}
        
        # Try to load from file if provided
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',', 1)
                        if len(parts) == 2:
                            symbol, name = parts
                            symbols[symbol.upper()] = name.strip()
                logger.info(f"Loaded {len(symbols)} stock symbols from {path}")
            except Exception as e:
                logger.warning(f"Failed to load stock symbols from {path}: {e}")
        
        # Default well-known symbols if file not available or empty
        if not symbols:
            default_symbols = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation",
                "GOOGL": "Alphabet Inc.",
                "AMZN": "Amazon.com, Inc.",
                "META": "Meta Platforms, Inc.",
                "TSLA": "Tesla, Inc.",
                "NVDA": "NVIDIA Corporation",
                "JPM": "JPMorgan Chase & Co.",
                "BAC": "Bank of America Corporation",
                "WMT": "Walmart Inc.",
                "JNJ": "Johnson & Johnson",
                "V": "Visa Inc.",
                "PG": "Procter & Gamble Company",
                "XOM": "Exxon Mobil Corporation",
                "NFLX": "Netflix, Inc."
            }
            symbols.update(default_symbols)
            logger.info(f"Using {len(default_symbols)} default stock symbols")
        
        return symbols
    
    def _load_terms(self, path: Optional[str]) -> Dict[str, str]:
        """
        Load financial terms dictionary from file.
        
        Args:
            path: Path to financial terms file
        
        Returns:
            Dictionary mapping terms to definitions/expansions
        """
        terms = {}
        
        # Try to load from file if provided
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(':', 1)
                        if len(parts) == 2:
                            term, definition = parts
                            terms[term.lower()] = definition.strip()
                logger.info(f"Loaded {len(terms)} financial terms from {path}")
            except Exception as e:
                logger.warning(f"Failed to load financial terms from {path}: {e}")
        
        # Default financial terms if file not available or empty
        if not terms:
            default_terms = {
                "eps": "earnings per share",
                "pe": "price-to-earnings ratio",
                "ebitda": "earnings before interest, taxes, depreciation, and amortization",
                "roi": "return on investment",
                "ipo": "initial public offering",
                "ytd": "year-to-date",
                "cagr": "compound annual growth rate",
                "roa": "return on assets",
                "roe": "return on equity",
                "gdp": "gross domestic product",
                "cpi": "consumer price index",
                "fomc": "Federal Open Market Committee",
                "fed": "Federal Reserve",
                "sec": "Securities and Exchange Commission",
                "etf": "exchange-traded fund",
                "bull market": "upward trend in stock prices",
                "bear market": "downward trend in stock prices"
            }
            terms.update(default_terms)
            logger.info(f"Using {len(default_terms)} default financial terms")
        
        return terms
    
    def identify_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Identify financial entities in the query.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary of identified entities by type
        """
        entities = {
            "stock_symbols": [],
            "company_names": [],
            "financial_terms": []
        }
        
        # Extract stock symbols (assumed to be uppercase 1-5 letter words)
        for word in re.findall(r'\b[A-Z]{1,5}\b', query):
            if word in self.stock_symbols:
                entities["stock_symbols"].append(word)
                entities["company_names"].append(self.stock_symbols[word])
        
        # Extract financial terms
        lowercase_query = query.lower()
        for term in self.financial_terms:
            if term in lowercase_query:
                entities["financial_terms"].append(term)
        
        return entities
    
    def expand_query(self, query: str, max_expansion_length: int = 300) -> str:
        """
        Expand query with relevant financial context.
        
        Args:
            query: Original user query
            max_expansion_length: Maximum length of expanded additions
        
        Returns:
            Expanded query with financial context
        """
        entities = self.identify_entities(query)
        expansions = []
        
        # Add company name for stock symbols
        for i, symbol in enumerate(entities["stock_symbols"]):
            if i < len(entities["company_names"]):
                company = entities["company_names"][i]
                expansions.append(f"{symbol} ({company})")
        
        # Add definitions for financial terms
        for term in entities["financial_terms"]:
            definition = self.financial_terms.get(term)
            if definition:
                expansions.append(f"{term} ({definition})")
        
        # Combine original query with expansions, if any
        if expansions:
            # Limit expansion length
            expansion_str = ", ".join(expansions)
            if len(expansion_str) > max_expansion_length:
                expansion_str = expansion_str[:max_expansion_length] + "..."
                
            expanded_query = f"{query} | Context: {expansion_str}"
            logger.info(f"Expanded query with {len(expansions)} entities")
            return expanded_query
        
        return query
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query to be more effective for retrieval.
        
        Args:
            query: Original user query
        
        Returns:
            Rewritten query optimized for retrieval
        """
        # Expand acronyms and add context
        expanded = self.expand_query(query)
        
        # Remove question words and common filler terms for better embedding matching
        filler_words = ["what", "when", "how", "is", "are", "the", "a", "an", "of", "for", "about"]
        words = expanded.split()
        
        # Remove common filler words only if they're not part of a company name or important term
        rewritten_words = []
        for word in words:
            word_lower = word.lower()
            skip = False
            
            # Skip filler words at beginning and not part of entities
            if word_lower in filler_words:
                entities = self.identify_entities(expanded)
                for entity_list in entities.values():
                    if any(word_lower in entity.lower() for entity in entity_list):
                        skip = False
                        break
            
            if not skip:
                rewritten_words.append(word)
        
        rewritten = " ".join(rewritten_words)
        
        # If significant reduction, log and return new query
        if len(rewritten) < len(expanded) * 0.8:
            logger.info(f"Rewrote query: {query} → {rewritten}")
            return rewritten
        
        # If no significant change, return expanded version
        return expanded


class QueryGenerator:
    """
    Generates multiple alternative queries from an original query
    to improve RAG retrieval performance.
    """
    def __init__(self, formatter: Optional[QueryFormatter] = None):
        """
        Initialize query generator with formatter.
        
        Args:
            formatter: Query formatter instance
        """
        self.formatter = formatter or QueryFormatter()
    
    def generate_variations(self, query: str, count: int = 3) -> List[str]:
        """
        Generate variations of the query for improved retrieval.
        
        Args:
            query: Original user query
            count: Number of variations to generate
        
        Returns:
            List of query variations
        """
        variations = [query]  # Start with original query
        
        # Standard rewrite from formatter
        rewritten = self.formatter.rewrite_query(query)
        if rewritten != query:
            variations.append(rewritten)
            
        # Additional variations
        
        # 1. Focus on financial implications
        finance_focused = f"financial implications of {query}"
        variations.append(finance_focused)
        
        # 2. Focus on market impact
        market_focused = f"market impact of {query}"
        variations.append(market_focused)
        
        # 3. Focus on stock performance
        stock_focused = f"stock performance related to {query}"
        variations.append(stock_focused)
        
        # 4. Focus on investor perspective
        investor_focused = f"investor perspective on {query}"
        variations.append(investor_focused)
        
        # Return only the requested number of variations
        return variations[:count]
    
    def generate_decomposed_queries(self, query: str) -> List[str]:
        """
        Decompose complex query into multiple simpler queries.
        
        Args:
            query: Original user query
        
        Returns:
            List of decomposed queries
        """
        decomposed = []
        
        # Add original query
        decomposed.append(query)
        
        # Look for common financial patterns to decompose
        if "compared to" in query.lower() or "vs" in query.lower():
            # Split comparison queries
            parts = re.split(r'\s+compared\s+to\s+|\s+vs\.?\s+', query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                decomposed.append(parts[0].strip())
                decomposed.append(parts[1].strip())
        
        # Look for multi-part financial questions
        if "and" in query.lower():
            if re.search(r'(what|how|why).+?\band\b.+?(what|how|why)', query, re.IGNORECASE):
                # Split multi-question query
                parts = re.split(r'\band\b', query, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip()
                    if part and len(part.split()) >= 3:  # Avoid too short parts
                        decomposed.append(part)
        
        return decomposed


def optimize_query(query: str) -> Union[str, List[str]]:
    """
    Optimize a query for financial RAG retrieval.
    
    Args:
        query: User query string
    
    Returns:
        Optimized query or list of queries
    """
    formatter = QueryFormatter()
    generator = QueryGenerator(formatter)
    
    # Check query complexity
    word_count = len(query.split())
    
    if word_count <= 5:
        # Simple query, just expand it
        return formatter.expand_query(query)
    elif word_count <= 15:
        # Medium query, rewrite it
        return formatter.rewrite_query(query)
    else:
        # Complex query, generate variations
        variations = generator.generate_decomposed_queries(query)
        if len(variations) > 1:
            return variations
        else:
            # Couldn't decompose, generate alternatives
            return generator.generate_variations(query, count=3) 