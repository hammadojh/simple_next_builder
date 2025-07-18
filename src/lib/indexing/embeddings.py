"""
Embedding Service for Code Chunks

This module handles vector embedding generation for code chunks using OpenAI's
text-embedding-3-small model, optimized for code and technical content.
"""

import os
import time
import asyncio
from typing import List, Dict, Optional, Tuple
import requests
import numpy as np
from dataclasses import dataclass

from .chunker import Chunk


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    chunk_id: str
    embedding: List[float]
    token_count: int
    success: bool
    error_message: Optional[str] = None


class EmbeddingService:
    """Service for generating embeddings from code chunks."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
        self.max_tokens = 8192
        self.batch_size = 100  # Process chunks in batches
        self.rate_limit_delay = 0.1  # seconds between requests
        
        if not self.api_key:
            raise ValueError("OpenAI API key required for embedding service")
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple chunks with rate limiting."""
        print(f"🔤 Generating embeddings for {len(chunks)} chunks...")
        
        results = []
        
        # Process in batches to respect rate limits
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            print(f"📦 Processing batch {i//self.batch_size + 1}/{(len(chunks) + self.batch_size - 1)//self.batch_size}")
            
            batch_results = self._embed_batch(batch)
            results.extend(batch_results)
            
            # Rate limiting
            if i + self.batch_size < len(chunks):
                time.sleep(self.rate_limit_delay)
        
        success_count = sum(1 for r in results if r.success)
        print(f"✅ Successfully generated {success_count}/{len(results)} embeddings")
        
        return results
    
    def embed_single_chunk(self, chunk: Chunk) -> EmbeddingResult:
        """Generate embedding for a single chunk."""
        return self._embed_batch([chunk])[0]
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        try:
            # Prepare the text with context
            embedding_text = self._prepare_query_text(query)
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": embedding_text,
                    "dimensions": self.dimensions
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data["data"][0]["embedding"]
            
        except Exception as e:
            print(f"❌ Failed to generate query embedding: {str(e)}")
            raise
    
    def _embed_batch(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of chunks."""
        # Prepare texts for embedding
        texts = []
        chunk_ids = []
        
        for chunk in chunks:
            text = self._prepare_chunk_text(chunk)
            texts.append(text)
            chunk_ids.append(chunk.chunk_hash)
        
        try:
            # Call OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": texts,
                    "dimensions": self.dimensions
                },
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Process results
            results = []
            for i, chunk in enumerate(chunks):
                embedding_data = data["data"][i]
                
                result = EmbeddingResult(
                    chunk_id=chunk.chunk_hash,
                    embedding=embedding_data["embedding"],
                    token_count=data["usage"]["total_tokens"] // len(chunks),  # Approximate
                    success=True
                )
                results.append(result)
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {str(e)}")
            # Return failed results for all chunks
            return [
                EmbeddingResult(
                    chunk_id=chunk.chunk_hash,
                    embedding=[],
                    token_count=0,
                    success=False,
                    error_message=str(e)
                )
                for chunk in chunks
            ]
        except Exception as e:
            print(f"❌ Unexpected error during embedding: {str(e)}")
            return [
                EmbeddingResult(
                    chunk_id=chunk.chunk_hash,
                    embedding=[],
                    token_count=0,
                    success=False,
                    error_message=str(e)
                )
                for chunk in chunks
            ]
    
    def _prepare_chunk_text(self, chunk: Chunk) -> str:
        """Prepare chunk text for embedding with context."""
        # Add metadata as context for better embeddings
        context_parts = []
        
        # File context
        context_parts.append(f"File: {chunk.file_path}")
        context_parts.append(f"Language: {chunk.language}")
        context_parts.append(f"Type: {chunk.chunk_type.value}")
        
        # Add metadata context
        if chunk.metadata.function_name:
            context_parts.append(f"Function: {chunk.metadata.function_name}")
        if chunk.metadata.component_name:
            context_parts.append(f"Component: {chunk.metadata.component_name}")
        if chunk.metadata.interface_name:
            context_parts.append(f"Interface: {chunk.metadata.interface_name}")
        if chunk.metadata.imports:
            context_parts.append(f"Imports: {', '.join(chunk.metadata.imports[:5])}")  # Limit imports
        
        # Combine context with content
        context = " | ".join(context_parts)
        full_text = f"{context}\n\n{chunk.content}"
        
        # Truncate if too long (leave room for context)
        max_content_length = 7000  # Conservative limit
        if len(full_text) > max_content_length:
            content_limit = max_content_length - len(context) - 10
            truncated_content = chunk.content[:content_limit] + "..."
            full_text = f"{context}\n\n{truncated_content}"
        
        return full_text
    
    def _prepare_query_text(self, query: str) -> str:
        """Prepare query text for embedding."""
        # Add context to help match with code chunks
        query_context = "NextJS TypeScript React code search query: "
        return f"{query_context}{query}"
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, float]:
        """Get statistics about embeddings."""
        if not embeddings:
            return {}
        
        # Convert to numpy array
        embed_array = np.array(embeddings)
        
        return {
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "mean_magnitude": float(np.mean(np.linalg.norm(embed_array, axis=1))),
            "std_magnitude": float(np.std(np.linalg.norm(embed_array, axis=1))),
            "mean_value": float(np.mean(embed_array)),
            "std_value": float(np.std(embed_array))
        }


class EmbeddingCache:
    """Simple file-based cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = ".embeddings_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cached_embedding(self, chunk_hash: str) -> Optional[List[float]]:
        """Get cached embedding for a chunk."""
        cache_file = os.path.join(self.cache_dir, f"{chunk_hash}.npy")
        
        if os.path.exists(cache_file):
            try:
                embedding = np.load(cache_file).tolist()
                return embedding
            except Exception:
                # Remove corrupted cache file
                os.remove(cache_file)
        
        return None
    
    def cache_embedding(self, chunk_hash: str, embedding: List[float]):
        """Cache an embedding."""
        cache_file = os.path.join(self.cache_dir, f"{chunk_hash}.npy")
        
        try:
            np.save(cache_file, np.array(embedding))
        except Exception as e:
            print(f"⚠️ Failed to cache embedding for {chunk_hash}: {e}")
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npy')]
        
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in cache_files
        )
        
        return {
            "cached_embeddings": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        } 