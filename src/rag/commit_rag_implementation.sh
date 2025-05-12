#!/bin/bash
# Shell script to commit the RAG system implementation

# Make sure we're in the repository root
cd "$(git rev-parse --show-toplevel)" || exit 1

echo "Adding RAG system files to git..."

# Add all RAG implementation files
git add src/rag/
git add tests/rag/
git add task.md.md

# Commit the changes with a descriptive message
git commit -m "Implement RAG system for financial insights

This commit implements the complete Retrieval-Augmented Generation (RAG) system for STOCKER Pro.
The system provides context-aware financial insights by combining vector-based retrieval
of financial news with structured template-based response generation.

Key components:
- ChromaDB vector database integration
- Financial news collector and processor
- Document chunking and embedding pipeline 
- Query formulation system for financial insights
- Context-aware response generation
- FastAPI endpoints for the RAG system
- Unit tests for all RAG components

Closes Task 7: RAG System Implementation"

echo "RAG implementation successfully committed!"
echo "Use 'git push' to push the changes to the remote repository." 