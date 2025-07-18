"""
Main Codebase Indexer

This module provides the main interface for indexing NextJS codebases,
orchestrating chunking, embedding generation, and storage.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

from .chunker import CodeChunker, Chunk
from .embeddings import EmbeddingService, EmbeddingCache
from .storage import ChunkStore, StorageStats
from .retriever import ContextRetriever, RetrievalConfig


@dataclass
class IndexingStats:
    """Statistics from indexing process."""
    total_files_scanned: int
    total_files_indexed: int
    total_chunks_created: int
    total_embeddings_generated: int
    indexing_time_seconds: float
    storage_size_mb: float
    files_by_type: Dict[str, int]
    chunks_by_type: Dict[str, int]
    errors: List[str]


class CodebaseIndexer:
    """Main indexer for NextJS codebases with semantic search capabilities."""
    
    def __init__(self, 
                 project_path: str,
                 project_id: Optional[str] = None,
                 storage_dir: str = ".chunk_store",
                 cache_dir: str = ".embeddings_cache"):
        
        self.project_path = Path(project_path)
        self.project_id = project_id or self._generate_project_id()
        
        # Initialize components
        self.chunker = CodeChunker(self.project_id)
        self.embedding_service = EmbeddingService()
        self.embedding_cache = EmbeddingCache(cache_dir)
        self.chunk_store = ChunkStore(storage_dir, self.project_id)
        self.retriever = ContextRetriever(self.chunk_store)
        
        # Configuration
        self.excluded_patterns = {
            'node_modules', '.git', '.next', 'dist', 'build',
            '*.min.js', '*.map', '*.d.ts', '.DS_Store'
        }
        
        self.included_extensions = {
            '.ts', '.tsx', '.js', '.jsx', '.json', '.md', '.css', '.scss'
        }
        
        print(f"🏗️ Initialized indexer for project: {self.project_id}")
        print(f"📁 Project path: {self.project_path}")
    
    def _generate_project_id(self) -> str:
        """Generate a project ID based on project path."""
        path_str = str(self.project_path.absolute())
        hash_obj = hashlib.md5(path_str.encode())
        return hash_obj.hexdigest()[:12]
    
    def index_project(self, force_reindex: bool = False) -> IndexingStats:
        """
        Index the entire project.
        
        Args:
            force_reindex: If True, reindex all files even if they haven't changed
            
        Returns:
            IndexingStats with indexing results
        """
        print(f"🚀 Starting codebase indexing...")
        start_time = time.time()
        
        stats = IndexingStats(
            total_files_scanned=0,
            total_files_indexed=0,
            total_chunks_created=0,
            total_embeddings_generated=0,
            indexing_time_seconds=0,
            storage_size_mb=0,
            files_by_type={},
            chunks_by_type={},
            errors=[]
        )
        
        try:
            # Discover files to index
            files_to_index = self._discover_files()
            stats.total_files_scanned = len(files_to_index)
            
            print(f"📂 Found {len(files_to_index)} files to scan")
            
            # Filter files that need indexing
            if not force_reindex:
                files_to_index = self._filter_changed_files(files_to_index)
            
            if not files_to_index:
                print("✅ No files need indexing")
                stats.indexing_time_seconds = time.time() - start_time
                return stats
            
            print(f"📝 Indexing {len(files_to_index)} files...")
            stats.total_files_indexed = len(files_to_index)
            
            # Process files in batches
            batch_size = 10
            all_chunks = []
            all_embeddings = []
            
            for i in range(0, len(files_to_index), batch_size):
                batch_files = files_to_index[i:i + batch_size]
                print(f"📦 Processing batch {i//batch_size + 1}/{(len(files_to_index) + batch_size - 1)//batch_size}")
                
                batch_chunks, batch_embeddings, batch_errors = self._process_file_batch(batch_files)
                
                all_chunks.extend(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                stats.errors.extend(batch_errors)
                
                # Update file type stats
                for file_path in batch_files:
                    file_ext = file_path.suffix.lower()
                    stats.files_by_type[file_ext] = stats.files_by_type.get(file_ext, 0) + 1
            
            # Store all chunks and embeddings
            if all_chunks:
                print(f"💾 Storing {len(all_chunks)} chunks with embeddings...")
                self.chunk_store.store_chunks(all_chunks, all_embeddings)
                
                stats.total_chunks_created = len(all_chunks)
                stats.total_embeddings_generated = len(all_embeddings)
                
                # Update chunk type stats
                for chunk in all_chunks:
                    chunk_type = chunk.chunk_type.value
                    stats.chunks_by_type[chunk_type] = stats.chunks_by_type.get(chunk_type, 0) + 1
            
            # Get storage stats
            storage_stats = self.chunk_store.get_stats()
            stats.storage_size_mb = storage_stats.index_size_mb
            
            stats.indexing_time_seconds = time.time() - start_time
            
            print(f"✅ Indexing complete!")
            print(f"📊 Results: {stats.total_chunks_created} chunks, {stats.total_embeddings_generated} embeddings")
            print(f"⏱️ Time: {stats.indexing_time_seconds:.1f}s")
            
            if stats.errors:
                print(f"⚠️ {len(stats.errors)} errors occurred during indexing")
            
            return stats
            
        except Exception as e:
            stats.errors.append(f"Indexing failed: {str(e)}")
            stats.indexing_time_seconds = time.time() - start_time
            print(f"❌ Indexing failed: {str(e)}")
            return stats
    
    def _discover_files(self) -> List[Path]:
        """Discover all files that should be indexed."""
        files = []
        
        for file_path in self.project_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Check if file should be excluded
            if self._should_exclude_file(file_path):
                continue
            
            # Check file extension
            if file_path.suffix.lower() not in self.included_extensions:
                continue
            
            files.append(file_path)
        
        return files
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if a file should be excluded from indexing."""
        # Check against excluded patterns
        for part in file_path.parts:
            if any(pattern in part for pattern in self.excluded_patterns):
                return True
        
        # Check file name patterns
        filename = file_path.name
        for pattern in self.excluded_patterns:
            if '*' in pattern:
                # Simple glob pattern matching
                pattern_parts = pattern.split('*')
                if len(pattern_parts) == 2 and filename.endswith(pattern_parts[1]):
                    return True
            elif pattern in filename:
                return True
        
        return False
    
    def _filter_changed_files(self, files: List[Path]) -> List[Path]:
        """Filter files to only those that have changed."""
        # For now, return all files
        # In a production system, you'd compare file mtimes with last index time
        return files
    
    def _process_file_batch(self, files: List[Path]) -> Tuple[List[Chunk], List[List[float]], List[str]]:
        """Process a batch of files into chunks and embeddings."""
        chunks = []
        embeddings = []
        errors = []
        
        for file_path in files:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert to relative path
                relative_path = str(file_path.relative_to(self.project_path))
                
                # Chunk the file
                file_chunks = self.chunker.chunk_file(relative_path, content)
                
                if not file_chunks:
                    continue
                
                # Generate embeddings with caching
                chunk_embeddings = []
                for chunk in file_chunks:
                    # Check cache first
                    cached_embedding = self.embedding_cache.get_cached_embedding(chunk.chunk_hash)
                    
                    if cached_embedding:
                        chunk_embeddings.append(cached_embedding)
                    else:
                        # Generate new embedding
                        embedding_result = self.embedding_service.embed_single_chunk(chunk)
                        
                        if embedding_result.success:
                            chunk_embeddings.append(embedding_result.embedding)
                            # Cache the embedding
                            self.embedding_cache.cache_embedding(chunk.chunk_hash, embedding_result.embedding)
                        else:
                            errors.append(f"Failed to embed chunk in {relative_path}: {embedding_result.error_message}")
                            continue
                
                # Add to results if we got embeddings for all chunks
                if len(chunk_embeddings) == len(file_chunks):
                    chunks.extend(file_chunks)
                    embeddings.extend(chunk_embeddings)
                else:
                    errors.append(f"Partial embedding failure for {relative_path}")
                
            except UnicodeDecodeError:
                errors.append(f"Could not decode file: {file_path}")
            except Exception as e:
                errors.append(f"Error processing {file_path}: {str(e)}")
        
        return chunks, embeddings, errors
    
    def update_file(self, file_path: str) -> bool:
        """
        Update index for a single file.
        
        Args:
            file_path: Path to the file (relative to project root)
            
        Returns:
            True if update was successful
        """
        print(f"🔄 Updating index for: {file_path}")
        
        try:
            full_path = self.project_path / file_path
            
            if not full_path.exists():
                # File was deleted, remove from index
                self.chunk_store.remove_file_chunks(file_path)
                print(f"🗑️ Removed chunks for deleted file: {file_path}")
                return True
            
            # Remove existing chunks for this file
            self.chunk_store.remove_file_chunks(file_path)
            
            # Process the updated file
            chunks, embeddings, errors = self._process_file_batch([full_path])
            
            if errors:
                print(f"⚠️ Errors updating {file_path}: {errors}")
                return False
            
            if chunks:
                self.chunk_store.store_chunks(chunks, embeddings)
                print(f"✅ Updated {len(chunks)} chunks for {file_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to update {file_path}: {str(e)}")
            return False
    
    def search_code(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search the codebase for relevant chunks.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with chunk info
        """
        results = self.chunk_store.search(query, k=max_results)
        
        formatted_results = []
        for result in results:
            chunk = result.chunk
            formatted_result = {
                'file_path': chunk.file_path,
                'content': chunk.content,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'chunk_type': chunk.chunk_type.value,
                'similarity': result.similarity,
                'metadata': {
                    'function_name': chunk.metadata.function_name,
                    'component_name': chunk.metadata.component_name,
                    'interface_name': chunk.metadata.interface_name,
                }
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_context_for_request(self, 
                               user_request: str,
                               current_file: Optional[str] = None,
                               recent_files: Optional[List[str]] = None) -> str:
        """
        Get relevant context for a user request - this replaces analyze_app_structure_enhanced.
        
        Args:
            user_request: The user's request or query
            current_file: Currently active file
            recent_files: Recently edited files
            
        Returns:
            Formatted context string for LLM
        """
        return self.retriever.get_relevant_context(
            user_request=user_request,
            current_file=current_file,
            recent_files=recent_files
        )
    
    def get_file_context(self, file_path: str) -> str:
        """Get context for a specific file."""
        return self.retriever.get_file_context(file_path)
    
    def get_stats(self) -> Dict:
        """Get comprehensive indexing statistics."""
        storage_stats = self.chunk_store.get_stats()
        cache_stats = self.embedding_cache.get_cache_stats()
        
        return {
            'project_id': self.project_id,
            'project_path': str(self.project_path),
            'storage': {
                'total_chunks': storage_stats.total_chunks,
                'total_embeddings': storage_stats.total_embeddings,
                'index_size_mb': storage_stats.index_size_mb,
                'file_types': storage_stats.file_types,
                'chunk_types': storage_stats.chunk_types
            },
            'cache': cache_stats,
            'retriever_config': {
                'max_chunks': self.retriever.config.max_chunks,
                'max_total_tokens': self.retriever.config.max_total_tokens,
                'min_similarity': self.retriever.config.min_similarity
            }
        }
    
    def clear_index(self):
        """Clear all indexed data."""
        print("🗑️ Clearing index...")
        self.chunk_store.clear()
        self.embedding_cache.clear_cache()
        print("✅ Index cleared")
    
    def configure_retrieval(self, config: RetrievalConfig):
        """Update retrieval configuration."""
        self.retriever.config = config
        print("⚙️ Retrieval configuration updated")


class SmartIndexer(CodebaseIndexer):
    """Enhanced indexer with smart features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file_hashes: Dict[str, str] = {}
        self._last_index_time: float = 0
    
    def smart_update(self) -> IndexingStats:
        """Smart update that only processes changed files."""
        print("🧠 Starting smart incremental update...")
        
        # Get current file states
        current_files = self._discover_files()
        current_hashes = {}
        
        for file_path in current_files:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    file_hash = hashlib.md5(content).hexdigest()
                    relative_path = str(file_path.relative_to(self.project_path))
                    current_hashes[relative_path] = file_hash
            except Exception:
                continue
        
        # Find changed files
        changed_files = []
        for file_path, file_hash in current_hashes.items():
            old_hash = self._file_hashes.get(file_path)
            if old_hash != file_hash:
                changed_files.append(self.project_path / file_path)
        
        # Find deleted files
        deleted_files = set(self._file_hashes.keys()) - set(current_hashes.keys())
        
        print(f"📊 Smart update: {len(changed_files)} changed, {len(deleted_files)} deleted")
        
        # Remove deleted files from index
        for deleted_file in deleted_files:
            self.chunk_store.remove_file_chunks(deleted_file)
        
        # Process changed files
        if changed_files:
            start_time = time.time()
            
            # Use existing batch processing
            all_chunks = []
            all_embeddings = []
            errors = []
            
            # Remove old chunks for changed files
            for file_path in changed_files:
                relative_path = str(file_path.relative_to(self.project_path))
                self.chunk_store.remove_file_chunks(relative_path)
            
            # Process changed files
            batch_chunks, batch_embeddings, batch_errors = self._process_file_batch(changed_files)
            all_chunks.extend(batch_chunks)
            all_embeddings.extend(batch_embeddings)
            errors.extend(batch_errors)
            
            # Store updates
            if all_chunks:
                self.chunk_store.store_chunks(all_chunks, all_embeddings)
            
            # Update tracking
            self._file_hashes = current_hashes
            self._last_index_time = time.time()
            
            # Build stats
            stats = IndexingStats(
                total_files_scanned=len(current_files),
                total_files_indexed=len(changed_files),
                total_chunks_created=len(all_chunks),
                total_embeddings_generated=len(all_embeddings),
                indexing_time_seconds=time.time() - start_time,
                storage_size_mb=self.chunk_store.get_stats().index_size_mb,
                files_by_type={},
                chunks_by_type={},
                errors=errors
            )
            
            print(f"✅ Smart update complete: {len(all_chunks)} chunks updated")
            return stats
        else:
            print("✅ No changes detected")
            return IndexingStats(
                total_files_scanned=len(current_files),
                total_files_indexed=0,
                total_chunks_created=0,
                total_embeddings_generated=0,
                indexing_time_seconds=0,
                storage_size_mb=self.chunk_store.get_stats().index_size_mb,
                files_by_type={},
                chunks_by_type={},
                errors=[]
            )
    
    def watch_and_update(self, poll_interval: int = 10):
        """Watch for file changes and update index automatically."""
        print(f"👀 Starting file watcher (polling every {poll_interval}s)...")
        
        try:
            while True:
                time.sleep(poll_interval)
                self.smart_update()
        except KeyboardInterrupt:
            print("\n⏹️ File watcher stopped")


# Helper function for integration with existing code
def create_project_indexer(project_path: str) -> CodebaseIndexer:
    """Create and initialize a project indexer."""
    indexer = CodebaseIndexer(project_path)
    
    # Check if index exists, if not create it
    stats = indexer.get_stats()
    if stats['storage']['total_chunks'] == 0:
        print("🆕 No existing index found, creating initial index...")
        indexer.index_project()
    else:
        print(f"📚 Found existing index with {stats['storage']['total_chunks']} chunks")
    
    return indexer 