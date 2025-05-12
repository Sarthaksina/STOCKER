# STOCKER Pro Financial RAG System

The RAG (Retrieval-Augmented Generation) system provides context-aware financial insights by combining vector-based retrieval of financial news with structured template-based response generation.

## Overview

The RAG system consists of several key components:

1. **ChromaDB Vector Database** - Stores and indexes financial news and information for semantic search
2. **News Collector** - Fetches and processes financial news from various sources
3. **Chunking Pipeline** - Splits documents into manageable chunks for embedding
4. **Query Formulation** - Optimizes user queries for better retrieval
5. **Response Generator** - Creates insightful responses based on retrieved information

## Component Architecture

```
User Query → Query Formulation → ChromaDB Search → Context Building → Response Generation → Insight Response
                    ↑                    ↑
                    |                    |
                    |                    ↓
               News Collection → Chunking → Embedding → ChromaDB Storage
```

## Installation

The RAG system is included in the main STOCKER Pro installation. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Key Dependencies

- `chromadb`: Vector store for embedding storage and retrieval
- `sentence-transformers`: For generating text embeddings
- `feedparser`: For parsing RSS feeds when collecting news
- `beautifulsoup4`: For HTML parsing of news content
- `fastapi`: For the API layer
- `nltk`: For text processing and sentence tokenization

## Usage Examples

### Running the Full RAG Pipeline

```python
from src.rag.run_rag_pipeline import run_pipeline

# Run the complete pipeline (collection, processing, querying)
run_pipeline()
```

### Generating Financial Insights

```python
from src.rag.response_generator import generate_financial_insight

# Generate insight for a specific query
insight = generate_financial_insight("What is the impact of rising interest rates on technology stocks?")
print(insight["response"])
```

### Using the API

Start the FastAPI server:

```bash
python -m src.rag.api
```

Then make requests to:
- `POST /insights` - Generate financial insights
- `POST /news/search` - Search for financial news articles
- `POST /query/optimize` - Optimize financial queries

## Key Files

- `chroma_db.py` - ChromaDB vector store implementation
- `news_collector.py` - Financial news collection system
- `chunking.py` - Document chunking and embedding pipeline
- `query_formulation.py` - Financial query optimization
- `response_generator.py` - Template-based response generation
- `api.py` - FastAPI endpoints

## Running Tests

```bash
pytest tests/rag/test_rag_components.py
```

## Customization

### Adding New News Sources

Edit `src.rag.news_collector.NewsCollector` to add new RSS feeds or web sources:

```python
self.rss_feeds = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "your_new_source": "https://example.com/feed.rss"
}
```

### Creating Custom Response Templates

Edit `src.rag.response_generator.TemplateManager` to customize response templates:

```python
self.templates["custom_template"] = (
    "Custom response for {custom_variable}\n"
    "Based on {another_variable}\n"
)
```

## Architecture Details

### News Collection Process

1. RSS feeds are parsed to extract article metadata and content
2. HTML content is cleaned and processed
3. Articles are chunked into manageable segments
4. Each chunk is embedded using a sentence transformer model
5. Embeddings and metadata are stored in ChromaDB

### Query Processing Pipeline

1. User query is analyzed for financial entities (stocks, terms)
2. Query is optimized for retrieval (expanded, rewritten)
3. Vector search is performed on the ChromaDB collection
4. Retrieved articles are ranked by relevance
5. Context is built from the most relevant articles
6. Response is generated using appropriate templates

### Response Generation

The system identifies the query type and selects an appropriate template:
- Stock performance
- Market analysis
- Company news
- Economic indicators
- Investment strategies

The template is filled with variables extracted from the retrieved context.

## Performance Considerations

- **Vector Index**: ChromaDB uses an efficient vector index for fast similarity search
- **Caching**: Frequently accessed articles are cached
- **Chunking Strategies**: Different chunking strategies are available depending on content type
- **Batch Processing**: Document embedding is done in batches for efficiency

## Further Development

Future enhancements planned:
- Integration with real-time market data APIs
- Improved sentiment analysis for news articles
- User feedback incorporation for response quality improvement
- Expanded financial template library 