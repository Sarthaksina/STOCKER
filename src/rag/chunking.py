"""
Document chunking and embedding pipeline for the RAG system.
Splits long documents into manageable chunks and prepares them for embedding.
"""
import re
import logging
import uuid
from typing import List, Dict, Any, Callable, Optional, Tuple

import nltk
from nltk.tokenize import sent_tokenize

# Initialize logger
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ChunkingPipeline:
    """
    Pipeline for chunking documents into smaller pieces for efficient embedding and retrieval.
    Supports various chunking strategies optimized for financial text.
    """
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 strategy: str = "sentence"):
        """
        Initialize chunking pipeline with specified parameters.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            strategy: Chunking strategy ('fixed', 'sentence', 'paragraph')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if strategy not in ["fixed", "sentence", "paragraph"]:
            logger.warning(f"Unknown chunking strategy '{strategy}', defaulting to 'sentence'")
            strategy = "sentence"
        
        self.strategy = strategy
        logger.info(f"Initialized chunking pipeline with {strategy} strategy")
    
    def _split_by_fixed_size(self, text: str) -> List[str]:
        """
        Split text into chunks of fixed size with overlap.
        
        Args:
            text: Text to split
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Get chunk with specified size
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            
            # Avoid infinite loop for small texts
            if start >= text_len or end == text_len:
                break
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences, respecting chunk size.
        
        Args:
            text: Text to split
        
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk size and we have content,
            # finalize current chunk and start a new one
            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # For overlap, keep some sentences from the previous chunk
                overlap_size = 0
                overlap_chunks = []
                
                # Add sentences from the end until we reach desired overlap
                for prev_sentence in reversed(current_chunk):
                    if overlap_size + len(prev_sentence) <= self.chunk_overlap:
                        overlap_chunks.insert(0, prev_sentence)
                        overlap_size += len(prev_sentence)
                    else:
                        break
                
                current_chunk = overlap_chunks
                current_size = overlap_size
            
            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_size += sentence_len
        
        # Add the last chunk if we have one
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into chunks by paragraphs, respecting chunk size.
        
        Args:
            text: Text to split
        
        Returns:
            List of text chunks
        """
        # Split text into paragraphs using double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            # Clean up paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_len = len(paragraph)
            
            # If this paragraph is already larger than chunk size, split it by sentences
            if paragraph_len > self.chunk_size:
                # Split large paragraph using sentence strategy
                paragraph_chunks = self._split_by_sentences(paragraph)
                for p_chunk in paragraph_chunks:
                    chunks.append(p_chunk)
                continue
            
            # If adding this paragraph would exceed chunk size and we have content,
            # finalize current chunk and start a new one
            if current_size + paragraph_len > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Add current paragraph to chunk
            current_chunk.append(paragraph)
            current_size += paragraph_len
        
        # Add the last chunk if we have one
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks based on the configured strategy.
        
        Args:
            text: Text to split
        
        Returns:
            List of chunk dictionaries with ID, text, and metadata
        """
        # Choose chunking strategy
        if self.strategy == "fixed":
            chunk_texts = self._split_by_fixed_size(text)
        elif self.strategy == "paragraph":
            chunk_texts = self._split_by_paragraphs(text)
        else:  # Default to sentence
            chunk_texts = self._split_by_sentences(text)
        
        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = str(uuid.uuid4())
            chunk = {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "chunk_index": i,
                    "chunk_count": len(chunk_texts),
                    "chunk_strategy": self.strategy,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "char_length": len(chunk_text),
                    "word_count": len(chunk_text.split())
                }
            }
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks using {self.strategy} strategy")
        return chunks
    
    def process_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an article into chunks with metadata.
        
        Args:
            article: Article dictionary with 'content', 'title', etc.
        
        Returns:
            List of chunk dictionaries with ID, text, and metadata
        """
        # Prepare text for chunking
        title = article.get("title", "")
        content = article.get("content", "")
        
        # Add title to the beginning of each chunk for context
        full_text = f"{title}\n\n{content}" if title else content
        
        # Create basic chunks
        chunks = self.create_chunks(full_text)
        
        # Add article metadata to each chunk
        for chunk in chunks:
            # Add article metadata
            for key, value in article.items():
                if key not in ["content", "text"]:
                    chunk["metadata"][f"article_{key}"] = value
        
        return chunks


class EmbeddingPipeline:
    """
    Pipeline for embedding text chunks using various embedding models.
    """
    def __init__(self, 
                 embed_func: Optional[Callable[[str], List[float]]] = None,
                 embed_dim: int = 384):
        """
        Initialize embedding pipeline with the specified embedding function.
        
        Args:
            embed_func: Function to convert text to embedding vector
            embed_dim: Dimension of embedding vectors
        """
        self.embed_func = embed_func
        self.embed_dim = embed_dim
        
        # Placeholder for initialized embedding function
        self._initialized_embed_func = None
    
    def _get_embed_func(self) -> Callable[[str], List[float]]:
        """
        Get or initialize embedding function.
        
        Returns:
            Function to convert text to embedding vector
        """
        if self._initialized_embed_func is not None:
            return self._initialized_embed_func
            
        if self.embed_func is not None:
            self._initialized_embed_func = self.embed_func
            return self.embed_func
        
        # Default embedding function using HuggingFace embeddings
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            def hf_embed_func(text: str) -> List[float]:
                return model.encode(text).tolist()
                
            self._initialized_embed_func = hf_embed_func
            logger.info("Initialized default HuggingFace embedding function")
            return hf_embed_func
            
        except ImportError:
            logger.error("Failed to initialize default embedding function, falling back to dummy embeddings")
            
            # Return dummy embedding function that returns zeros
            import numpy as np
            def dummy_embed_func(text: str) -> List[float]:
                return [0.0] * self.embed_dim
                
            self._initialized_embed_func = dummy_embed_func
            return dummy_embed_func
    
    def embed_text(self, text: str) -> List[float]:
        """
        Convert text to embedding vector.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        embed_func = self._get_embed_func()
        return embed_func(text)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed multiple text chunks.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            List of chunk dictionaries with added embeddings
        """
        embed_func = self._get_embed_func()
        
        # Process in batches for efficiency
        for chunk in chunks:
            try:
                embedding = embed_func(chunk["text"])
                chunk["embedding"] = embedding
            except Exception as e:
                logger.error(f"Error embedding chunk {chunk['id']}: {e}")
                # Add dummy embedding
                chunk["embedding"] = [0.0] * self.embed_dim
        
        logger.info(f"Embedded {len(chunks)} chunks")
        return chunks


def process_and_embed_article(article: Dict[str, Any], 
                             chunk_strategy: str = "sentence",
                             chunk_size: int = 512,
                             chunk_overlap: int = 128) -> List[Dict[str, Any]]:
    """
    Process an article into chunks and embed them.
    
    Args:
        article: Article dictionary
        chunk_strategy: Chunking strategy to use
        chunk_size: Target size of chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked and embedded document segments
    """
    # Initialize pipelines
    chunking_pipeline = ChunkingPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy
    )
    
    embedding_pipeline = EmbeddingPipeline()
    
    # Process article into chunks
    chunks = chunking_pipeline.process_article(article)
    
    # Embed chunks
    embedded_chunks = embedding_pipeline.embed_chunks(chunks)
    
    return embedded_chunks 