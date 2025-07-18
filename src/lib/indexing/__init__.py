"""
NextJS Codebase Indexing System

This package implements vector-based indexing for efficient context retrieval,
replacing the manual file analysis approach with semantic search.

Core Components:
- CodebaseIndexer: Main indexing orchestrator
- CodeChunker: Semantic code chunking with Tree-Sitter
- EmbeddingService: Vector embedding generation
- ContextRetriever: Semantic search and context assembly
- ChunkStore: Vector storage and retrieval
"""

from .indexer import CodebaseIndexer
from .chunker import CodeChunker, Chunk
from .embeddings import EmbeddingService
from .retriever import ContextRetriever
from .storage import ChunkStore

__version__ = "1.0.0"
__all__ = [
    "CodebaseIndexer",
    "CodeChunker", 
    "Chunk",
    "EmbeddingService",
    "ContextRetriever",
    "ChunkStore"
] 