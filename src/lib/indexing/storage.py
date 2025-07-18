"""
Vector Storage System for Code Chunks

This module implements vector storage and retrieval for code chunks,
supporting both in-memory and persistent storage options.
"""

import os
import json
import pickle
import sqlite3
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

from .chunker import Chunk, ChunkType
from .embeddings import EmbeddingService


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk: Chunk
    similarity: float
    rank: int


@dataclass
class StorageStats:
    """Statistics about the vector storage."""
    total_chunks: int
    total_embeddings: int
    index_size_mb: float
    file_types: Dict[str, int]
    chunk_types: Dict[str, int]


class ChunkStore:
    """Vector storage for code chunks with search capabilities."""
    
    def __init__(self, storage_dir: str = ".chunk_store", project_id: str = "default"):
        self.storage_dir = Path(storage_dir)
        self.project_id = project_id
        self.embedding_service = EmbeddingService()
        
        # Create storage directories
        self.storage_dir.mkdir(exist_ok=True)
        (self.storage_dir / project_id).mkdir(exist_ok=True)
        
        # Database paths
        self.db_path = self.storage_dir / project_id / "chunks.db"
        self.embeddings_path = self.storage_dir / project_id / "embeddings.npy"
        self.metadata_path = self.storage_dir / project_id / "metadata.json"
        
        # Initialize database
        self._init_database()
        
        # In-memory index for fast search
        self._chunks_cache: Dict[str, Chunk] = {}
        self._embeddings_cache: Optional[np.ndarray] = None
        self._chunk_id_to_index: Dict[str, int] = {}
        
        # Load existing data
        self._load_index()
    
    def _init_database(self):
        """Initialize SQLite database for chunk metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                language TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_language ON chunks(language)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_project_id ON chunks(project_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Store chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        print(f"💾 Storing {len(chunks)} chunks with embeddings...")
        
        # Store chunk metadata in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk in chunks:
            cursor.execute('''
                INSERT OR REPLACE INTO chunks 
                (chunk_hash, file_path, start_line, end_line, chunk_type, language, 
                 file_hash, project_id, content, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                chunk.chunk_hash,
                chunk.file_path,
                chunk.start_line,
                chunk.end_line,
                chunk.chunk_type.value,
                chunk.language,
                chunk.file_hash,
                chunk.project_id,
                chunk.content,
                json.dumps(asdict(chunk.metadata))
            ))
        
        conn.commit()
        conn.close()
        
        # Store embeddings
        self._store_embeddings(chunks, embeddings)
        
        # Update in-memory cache
        self._update_cache(chunks, embeddings)
        
        print("✅ Chunks and embeddings stored successfully")
    
    def _store_embeddings(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Store embeddings in numpy format."""
        # Load existing embeddings
        existing_embeddings = []
        existing_chunk_ids = []
        
        if self.embeddings_path.exists():
            try:
                existing_data = np.load(self.embeddings_path, allow_pickle=True).item()
                existing_embeddings = existing_data.get('embeddings', [])
                existing_chunk_ids = existing_data.get('chunk_ids', [])
            except Exception as e:
                print(f"⚠️ Failed to load existing embeddings: {e}")
        
        # Merge with new embeddings
        chunk_id_to_embedding = dict(zip(existing_chunk_ids, existing_embeddings))
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id_to_embedding[chunk.chunk_hash] = embedding
        
        # Save merged embeddings
        final_chunk_ids = list(chunk_id_to_embedding.keys())
        final_embeddings = [chunk_id_to_embedding[cid] for cid in final_chunk_ids]
        
        data = {
            'embeddings': final_embeddings,
            'chunk_ids': final_chunk_ids
        }
        
        np.save(self.embeddings_path, data)
        
        # Update metadata
        metadata = {
            'total_chunks': len(final_chunk_ids),
            'embedding_dimensions': len(final_embeddings[0]) if final_embeddings else 0,
            'project_id': self.project_id
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_cache(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Update in-memory cache."""
        # Update chunks cache
        for chunk in chunks:
            self._chunks_cache[chunk.chunk_hash] = chunk
        
        # Reload embeddings cache
        self._load_embeddings_cache()
    
    def _load_index(self):
        """Load existing index into memory."""
        print("📚 Loading chunk index...")
        
        # Load chunks from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE project_id = ?', (self.project_id,))
        rows = cursor.fetchall()
        
        for row in rows:
            chunk_hash, file_path, start_line, end_line, chunk_type, language, file_hash, project_id, content, metadata_json, created_at, updated_at = row
            
            # Parse metadata
            try:
                metadata_dict = json.loads(metadata_json)
                from .chunker import ChunkMetadata
                metadata = ChunkMetadata(**metadata_dict)
            except Exception:
                metadata = ChunkMetadata()
            
            # Create chunk
            chunk = Chunk(
                content=content,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type=ChunkType(chunk_type),
                language=language,
                chunk_hash=chunk_hash,
                metadata=metadata,
                file_hash=file_hash,
                project_id=project_id
            )
            
            self._chunks_cache[chunk_hash] = chunk
        
        conn.close()
        
        # Load embeddings
        self._load_embeddings_cache()
        
        print(f"✅ Loaded {len(self._chunks_cache)} chunks")
    
    def _load_embeddings_cache(self):
        """Load embeddings into memory for fast search."""
        if not self.embeddings_path.exists():
            self._embeddings_cache = None
            return
        
        try:
            data = np.load(self.embeddings_path, allow_pickle=True).item()
            embeddings = data.get('embeddings', [])
            chunk_ids = data.get('chunk_ids', [])
            
            if embeddings:
                self._embeddings_cache = np.array(embeddings)
                self._chunk_id_to_index = {cid: i for i, cid in enumerate(chunk_ids)}
            else:
                self._embeddings_cache = None
                
        except Exception as e:
            print(f"⚠️ Failed to load embeddings cache: {e}")
            self._embeddings_cache = None
    
    def search(self, query: str, k: int = 30, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for relevant chunks using vector similarity."""
        if self._embeddings_cache is None or len(self._chunks_cache) == 0:
            return []
        
        # Get query embedding
        try:
            query_embedding = self.embedding_service.embed_query(query)
        except Exception as e:
            print(f"❌ Failed to embed query: {e}")
            return []
        
        # Calculate similarities
        query_vec = np.array(query_embedding)
        similarities = np.dot(self._embeddings_cache, query_vec) / (
            np.linalg.norm(self._embeddings_cache, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k * 2]  # Get more for filtering
        
        # Convert to results with filtering
        results = []
        chunk_ids = list(self._chunk_id_to_index.keys())
        
        for rank, idx in enumerate(top_indices):
            if len(results) >= k:
                break
                
            chunk_id = chunk_ids[idx]
            chunk = self._chunks_cache.get(chunk_id)
            
            if not chunk:
                continue
            
            # Apply filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            
            similarity = float(similarities[idx])
            
            result = SearchResult(
                chunk=chunk,
                similarity=similarity,
                rank=rank + 1
            )
            results.append(result)
        
        return results
    
    def _matches_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches search filters."""
        # File type filter
        if 'file_types' in filters:
            file_ext = Path(chunk.file_path).suffix.lower()
            if file_ext not in filters['file_types']:
                return False
        
        # Chunk type filter
        if 'chunk_types' in filters:
            if chunk.chunk_type.value not in filters['chunk_types']:
                return False
        
        # Language filter
        if 'languages' in filters:
            if chunk.language not in filters['languages']:
                return False
        
        # File path filter
        if 'file_paths' in filters:
            if not any(path in chunk.file_path for path in filters['file_paths']):
                return False
        
        return True
    
    def get_chunk(self, chunk_hash: str) -> Optional[Chunk]:
        """Get a specific chunk by hash."""
        return self._chunks_cache.get(chunk_hash)
    
    def get_chunks_by_file(self, file_path: str) -> List[Chunk]:
        """Get all chunks for a specific file."""
        return [
            chunk for chunk in self._chunks_cache.values()
            if chunk.file_path == file_path
        ]
    
    def remove_file_chunks(self, file_path: str):
        """Remove all chunks for a specific file."""
        print(f"🗑️ Removing chunks for file: {file_path}")
        
        chunks_to_remove = [
            chunk_hash for chunk_hash, chunk in self._chunks_cache.items()
            if chunk.file_path == file_path
        ]
        
        # Remove from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk_hash in chunks_to_remove:
            cursor.execute('DELETE FROM chunks WHERE chunk_hash = ?', (chunk_hash,))
            del self._chunks_cache[chunk_hash]
        
        conn.commit()
        conn.close()
        
        # Rebuild embeddings cache
        self._rebuild_embeddings_cache()
        
        print(f"✅ Removed {len(chunks_to_remove)} chunks")
    
    def _rebuild_embeddings_cache(self):
        """Rebuild embeddings cache after chunk removal."""
        if not self._chunks_cache:
            self._embeddings_cache = None
            self._chunk_id_to_index = {}
            return
        
        # Load all embeddings
        if not self.embeddings_path.exists():
            return
        
        try:
            data = np.load(self.embeddings_path, allow_pickle=True).item()
            all_embeddings = data.get('embeddings', [])
            all_chunk_ids = data.get('chunk_ids', [])
            
            # Filter to only existing chunks
            valid_embeddings = []
            valid_chunk_ids = []
            
            for chunk_id, embedding in zip(all_chunk_ids, all_embeddings):
                if chunk_id in self._chunks_cache:
                    valid_embeddings.append(embedding)
                    valid_chunk_ids.append(chunk_id)
            
            # Save filtered embeddings
            if valid_embeddings:
                filtered_data = {
                    'embeddings': valid_embeddings,
                    'chunk_ids': valid_chunk_ids
                }
                np.save(self.embeddings_path, filtered_data)
                
                self._embeddings_cache = np.array(valid_embeddings)
                self._chunk_id_to_index = {cid: i for i, cid in enumerate(valid_chunk_ids)}
            else:
                self._embeddings_cache = None
                self._chunk_id_to_index = {}
                
        except Exception as e:
            print(f"⚠️ Failed to rebuild embeddings cache: {e}")
    
    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        # Count file types
        file_types = {}
        chunk_types = {}
        
        for chunk in self._chunks_cache.values():
            file_ext = Path(chunk.file_path).suffix.lower()
            file_types[file_ext] = file_types.get(file_ext, 0) + 1
            
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Calculate index size
        index_size_mb = 0
        if self.embeddings_path.exists():
            index_size_mb = self.embeddings_path.stat().st_size / (1024 * 1024)
        
        return StorageStats(
            total_chunks=len(self._chunks_cache),
            total_embeddings=len(self._embeddings_cache) if self._embeddings_cache is not None else 0,
            index_size_mb=index_size_mb,
            file_types=file_types,
            chunk_types=chunk_types
        )
    
    def clear(self):
        """Clear all stored data."""
        print("🗑️ Clearing all stored chunks and embeddings...")
        
        # Clear database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chunks WHERE project_id = ?', (self.project_id,))
        conn.commit()
        conn.close()
        
        # Clear files
        if self.embeddings_path.exists():
            self.embeddings_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        
        # Clear cache
        self._chunks_cache.clear()
        self._embeddings_cache = None
        self._chunk_id_to_index.clear()
        
        print("✅ All data cleared")


class InMemoryChunkStore:
    """Simple in-memory chunk store for testing and small projects."""
    
    def __init__(self, project_id: str = "default"):
        self.project_id = project_id
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.embedding_service = EmbeddingService()
    
    def store_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Store chunks with embeddings in memory."""
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks[chunk.chunk_hash] = chunk
            self.embeddings[chunk.chunk_hash] = embedding
    
    def search(self, query: str, k: int = 30, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Simple in-memory search."""
        if not self.chunks:
            return []
        
        # Get query embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for chunk_hash, chunk in self.chunks.items():
            if filters and not self._matches_filters(chunk, filters):
                continue
                
            chunk_embedding = self.embeddings.get(chunk_hash)
            if chunk_embedding:
                similarity = self.embedding_service.calculate_similarity(query_embedding, chunk_embedding)
                similarities.append((chunk, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (chunk, similarity) in enumerate(similarities[:k]):
            result = SearchResult(
                chunk=chunk,
                similarity=similarity,
                rank=rank + 1
            )
            results.append(result)
        
        return results
    
    def _matches_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches filters."""
        if 'file_types' in filters:
            file_ext = Path(chunk.file_path).suffix.lower()
            if file_ext not in filters['file_types']:
                return False
        return True
    
    def get_chunk(self, chunk_hash: str) -> Optional[Chunk]:
        """Get chunk by hash."""
        return self.chunks.get(chunk_hash)
    
    def clear(self):
        """Clear all data."""
        self.chunks.clear()
        self.embeddings.clear() 