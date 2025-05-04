"""
FastAPI endpoints for the financial RAG system.
Provides API access to news search, query processing, and insight generation.
"""
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from src.rag.news_collector import get_news_collector
from src.rag.response_generator import generate_financial_insight
from src.rag.query_formulation import optimize_query
from src.configuration.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Create FastAPI app
rag_app = FastAPI(
    title="STOCKER Financial RAG API",
    description="API for financial insights using RAG technology",
    version="1.0.0"
)

# Define request and response models
class InsightRequest(BaseModel):
    """Request model for financial insights."""
    query: str = Field(..., 
                      description="User query about financial topics", 
                      min_length=3, 
                      max_length=500,
                      example="What is the current performance of AAPL?")
    max_results: Optional[int] = Field(5, 
                                     description="Maximum number of results to consider",
                                     ge=1, 
                                     le=10)

class NewsSource(BaseModel):
    """Model for news sources in responses."""
    title: str
    source: str
    link: str
    published: str

class InsightResponse(BaseModel):
    """Response model for financial insights."""
    query: str
    response: str
    query_type: str
    sources: List[NewsSource]
    processing_time: float
    generated_at: str

class NewsSearchRequest(BaseModel):
    """Request model for news search."""
    query: str = Field(..., 
                     description="Search query", 
                     min_length=3, 
                     max_length=500,
                     example="Tesla stock performance")
    max_results: Optional[int] = Field(5, 
                                     description="Maximum number of results",
                                     ge=1, 
                                     le=20)
    filter_by_source: Optional[str] = Field(None,
                                         description="Filter by news source")
    max_age_days: Optional[int] = Field(30,
                                      description="Maximum age of articles in days",
                                      ge=1,
                                      le=365)

class NewsArticle(BaseModel):
    """Model for news articles in search responses."""
    id: str
    text: str
    metadata: Dict[str, Any]
    relevance: float

class NewsSearchResponse(BaseModel):
    """Response model for news search."""
    query: str
    results: List[NewsArticle]
    result_count: int
    processing_time: float

class QueryOptimizationRequest(BaseModel):
    """Request model for query optimization."""
    query: str = Field(..., 
                     description="Query to optimize", 
                     min_length=3, 
                     max_length=500,
                     example="How will interest rates affect tech stocks?")

class QueryOptimizationResponse(BaseModel):
    """Response model for query optimization."""
    original_query: str
    optimized_queries: List[str]

# Define API endpoints
@rag_app.post("/insights", response_model=InsightResponse)
async def get_financial_insight(request: InsightRequest):
    """
    Generate financial insights for a user query.
    
    The system will:
    1. Optimize the query for better retrieval
    2. Search for relevant financial news
    3. Generate an insightful response using templates
    4. Return the response with source attribution
    """
    try:
        # Log the incoming request
        logger.info(f"Received insight request: {request.query}")
        
        # Generate insight
        response = generate_financial_insight(request.query)
        
        return response
    except Exception as e:
        logger.error(f"Error generating insight: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating insight: {str(e)}")

@rag_app.post("/news/search", response_model=NewsSearchResponse)
async def search_news(request: NewsSearchRequest):
    """
    Search for financial news articles.
    
    The system will:
    1. Optimize the search query
    2. Retrieve relevant news articles
    3. Return the articles with metadata and relevance scores
    """
    try:
        # Log the incoming request
        logger.info(f"Received news search request: {request.query}")
        
        # Get news collector
        news_collector = get_news_collector()
        
        # Search for articles
        import time
        start_time = time.time()
        
        articles = news_collector.search_articles(
            query=request.query,
            n_results=request.max_results,
            filter_by_source=request.filter_by_source,
            max_age_days=request.max_age_days
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = {
            "query": request.query,
            "results": articles,
            "result_count": len(articles),
            "processing_time": processing_time
        }
        
        return response
    except Exception as e:
        logger.error(f"Error searching news: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching news: {str(e)}")

@rag_app.post("/query/optimize", response_model=QueryOptimizationResponse)
async def optimize_user_query(request: QueryOptimizationRequest):
    """
    Optimize a user query for better retrieval performance.
    
    The system will:
    1. Analyze the query for financial entities
    2. Expand financial acronyms and terms
    3. Potentially decompose complex queries
    4. Return optimized query or queries
    """
    try:
        # Log the incoming request
        logger.info(f"Received query optimization request: {request.query}")
        
        # Optimize query
        optimized = optimize_query(request.query)
        
        # Handle different return types from optimize_query
        if isinstance(optimized, list):
            optimized_queries = optimized
        else:
            optimized_queries = [optimized]
        
        # Build response
        response = {
            "original_query": request.query,
            "optimized_queries": optimized_queries
        }
        
        return response
    except Exception as e:
        logger.error(f"Error optimizing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error optimizing query: {str(e)}")

@rag_app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        Status of the API and key components
    """
    try:
        # Check news collector
        news_collector = get_news_collector()
        
        # Get ChromaDB stats
        chroma_stats = news_collector.chroma_db.get_collection_stats()
        
        return {
            "status": "healthy",
            "components": {
                "api": "up",
                "news_collector": "up",
                "vector_db": "up",
                "document_count": chroma_stats["document_count"]
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the app
    uvicorn.run(
        "src.rag.api:rag_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    ) 