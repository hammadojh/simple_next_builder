"""
Context Retrieval System

This module handles semantic search and context assembly for LLM prompts,
replacing manual file analysis with intelligent chunk retrieval.
"""

import os
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass

from .storage import ChunkStore, SearchResult
from .chunker import Chunk, ChunkType


@dataclass
class RetrievalConfig:
    """Configuration for context retrieval."""
    max_chunks: int = 25
    min_similarity: float = 0.1
    max_total_tokens: int = 12000
    boost_current_file: float = 1.6
    boost_recent_files: float = 1.2
    boost_imports: float = 1.3
    boost_components: float = 1.4
    boost_interfaces: float = 1.2
    overlap_dedup_threshold: float = 0.85
    
    # File type priorities
    priority_file_types: List[str] = None
    excluded_file_types: List[str] = None
    
    def __post_init__(self):
        if self.priority_file_types is None:
            self.priority_file_types = ['.tsx', '.ts', '.jsx', '.js']
        if self.excluded_file_types is None:
            self.excluded_file_types = ['.min.js', '.d.ts']


@dataclass
class ContextChunk:
    """Context chunk with enhanced metadata for prompt building."""
    chunk: Chunk
    similarity: float
    boost_reason: str
    final_score: float
    token_estimate: int


class ContextRetriever:
    """Retrieves and assembles relevant context for LLM prompts."""
    
    def __init__(self, chunk_store: ChunkStore, config: Optional[RetrievalConfig] = None):
        self.chunk_store = chunk_store
        self.config = config or RetrievalConfig()
        
    def get_relevant_context(self, 
                           user_request: str,
                           current_file: Optional[str] = None,
                           recent_files: Optional[List[str]] = None,
                           target_files: Optional[List[str]] = None) -> str:
        """
        Get relevant context for a user request.
        
        Args:
            user_request: The user's query or change request
            current_file: Currently open/active file (gets boost)
            recent_files: Recently edited files (get boost)
            target_files: Specific files the user mentioned (get high boost)
            
        Returns:
            Formatted context string for LLM prompt
        """
        print(f"🔍 Retrieving context for: {user_request[:100]}...")
        
        # Perform semantic search
        search_results = self._search_with_filters(user_request)
        
        if not search_results:
            print("⚠️ No relevant chunks found")
            return "No relevant code context found."
        
        # Apply heuristic boosts
        boosted_chunks = self._apply_heuristic_boosts(
            search_results, 
            current_file, 
            recent_files,
            target_files
        )
        
        # Deduplicate and rank
        final_chunks = self._deduplicate_and_rank(boosted_chunks)
        
        # Limit by token count
        selected_chunks = self._select_by_token_limit(final_chunks)
        
        # Format for LLM
        context = self._format_context_for_llm(selected_chunks, user_request)
        
        print(f"✅ Retrieved {len(selected_chunks)} relevant chunks")
        
        return context
    
    def _search_with_filters(self, query: str) -> List[SearchResult]:
        """Perform filtered semantic search."""
        # Create filters for NextJS-specific content
        filters = {
            'file_types': self.config.priority_file_types,
            'languages': ['typescript', 'javascript']
        }
        
        # Search with higher k to allow for filtering
        results = self.chunk_store.search(
            query=query,
            k=self.config.max_chunks * 2,
            filters=filters
        )
        
        # Filter out very low similarity results
        filtered_results = [
            result for result in results 
            if result.similarity >= self.config.min_similarity
        ]
        
        return filtered_results
    
    def _apply_heuristic_boosts(self, 
                               search_results: List[SearchResult],
                               current_file: Optional[str],
                               recent_files: Optional[List[str]],
                               target_files: Optional[List[str]]) -> List[ContextChunk]:
        """Apply heuristic boosts to search results."""
        recent_files = recent_files or []
        target_files = target_files or []
        
        boosted_chunks = []
        
        for result in search_results:
            chunk = result.chunk
            base_similarity = result.similarity
            boost_reasons = []
            total_boost = 1.0
            
            # Current file boost
            if current_file and chunk.file_path == current_file:
                total_boost *= self.config.boost_current_file
                boost_reasons.append("current_file")
            
            # Recent files boost
            if chunk.file_path in recent_files:
                total_boost *= self.config.boost_recent_files
                boost_reasons.append("recent_file")
            
            # Target files boost (highest priority)
            if chunk.file_path in target_files:
                total_boost *= 2.0  # Strong boost for explicitly mentioned files
                boost_reasons.append("target_file")
            
            # Chunk type boosts
            if chunk.chunk_type == ChunkType.IMPORT_BLOCK:
                total_boost *= self.config.boost_imports
                boost_reasons.append("imports")
            elif chunk.chunk_type == ChunkType.COMPONENT:
                total_boost *= self.config.boost_components
                boost_reasons.append("component")
            elif chunk.chunk_type == ChunkType.INTERFACE:
                total_boost *= self.config.boost_interfaces
                boost_reasons.append("interface")
            
            # Function name matching boost
            if chunk.metadata.function_name:
                query_words = set(word.lower() for word in user_request.split())
                function_words = set(word.lower() for word in chunk.metadata.function_name.split('_'))
                if query_words & function_words:  # Word overlap
                    total_boost *= 1.3
                    boost_reasons.append("name_match")
            
            final_score = base_similarity * total_boost
            boost_reason = ", ".join(boost_reasons) if boost_reasons else "semantic"
            
            context_chunk = ContextChunk(
                chunk=chunk,
                similarity=base_similarity,
                boost_reason=boost_reason,
                final_score=final_score,
                token_estimate=self._estimate_tokens(chunk.content)
            )
            
            boosted_chunks.append(context_chunk)
        
        return boosted_chunks
    
    def _deduplicate_and_rank(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Deduplicate overlapping chunks and rank by final score."""
        # Sort by final score
        chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        # Deduplicate overlapping chunks
        deduplicated = []
        seen_content_hashes = set()
        
        for chunk in chunks:
            # Create content hash for overlap detection
            content_hash = hash(chunk.chunk.content[:500])  # Use first 500 chars
            
            # Check for high overlap with existing chunks
            is_duplicate = False
            for existing_chunk in deduplicated:
                if self._chunks_overlap(chunk.chunk, existing_chunk.chunk):
                    is_duplicate = True
                    break
            
            if not is_duplicate and content_hash not in seen_content_hashes:
                deduplicated.append(chunk)
                seen_content_hashes.add(content_hash)
        
        return deduplicated
    
    def _chunks_overlap(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """Check if two chunks overlap significantly."""
        # Same file overlap check
        if chunk1.file_path == chunk2.file_path:
            # Check line overlap
            lines1 = set(range(chunk1.start_line, chunk1.end_line + 1))
            lines2 = set(range(chunk2.start_line, chunk2.end_line + 1))
            overlap = len(lines1 & lines2)
            total_lines = len(lines1 | lines2)
            
            overlap_ratio = overlap / total_lines if total_lines > 0 else 0
            
            if overlap_ratio > self.config.overlap_dedup_threshold:
                return True
        
        # Content similarity check for different files
        content1_words = set(chunk1.content.lower().split())
        content2_words = set(chunk2.content.lower().split())
        
        if len(content1_words) > 0 and len(content2_words) > 0:
            similarity = len(content1_words & content2_words) / len(content1_words | content2_words)
            if similarity > self.config.overlap_dedup_threshold:
                return True
        
        return False
    
    def _select_by_token_limit(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Select chunks within token limit."""
        selected = []
        total_tokens = 0
        
        for chunk in chunks:
            if total_tokens + chunk.token_estimate <= self.config.max_total_tokens:
                selected.append(chunk)
                total_tokens += chunk.token_estimate
            else:
                break
        
        return selected
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        # Rough estimation: 1 token ≈ 4 characters for code
        return len(content) // 4
    
    def _format_context_for_llm(self, chunks: List[ContextChunk], user_request: str) -> str:
        """Format retrieved chunks into LLM prompt context."""
        if not chunks:
            return "No relevant code context found."
        
        context_parts = []
        
        # Add header
        context_parts.append("RELEVANT CODE CONTEXT:")
        context_parts.append("=" * 60)
        context_parts.append(f"Query: {user_request}")
        context_parts.append(f"Retrieved {len(chunks)} relevant code chunks:")
        context_parts.append("")
        
        # Group chunks by file for better organization
        chunks_by_file = {}
        for chunk in chunks:
            file_path = chunk.chunk.file_path
            if file_path not in chunks_by_file:
                chunks_by_file[file_path] = []
            chunks_by_file[file_path].append(chunk)
        
        # Format each file's chunks
        for file_path, file_chunks in chunks_by_file.items():
            context_parts.append(f"📁 FILE: {file_path}")
            context_parts.append("-" * 50)
            
            for chunk_ctx in file_chunks:
                chunk = chunk_ctx.chunk
                
                # Add chunk header with metadata
                chunk_header = (
                    f"<chunk lines=\"{chunk.start_line}-{chunk.end_line}\" "
                    f"type=\"{chunk.chunk_type.value}\" "
                    f"similarity=\"{chunk_ctx.similarity:.3f}\" "
                    f"boost=\"{chunk_ctx.boost_reason}\">"
                )
                
                context_parts.append(chunk_header)
                context_parts.append(chunk.content)
                context_parts.append("</chunk>")
                context_parts.append("")
        
        # Add footer with retrieval stats
        total_similarity = sum(c.similarity for c in chunks)
        avg_similarity = total_similarity / len(chunks)
        
        context_parts.append("=" * 60)
        context_parts.append(f"RETRIEVAL STATS:")
        context_parts.append(f"• Total chunks: {len(chunks)}")
        context_parts.append(f"• Average similarity: {avg_similarity:.3f}")
        context_parts.append(f"• Files covered: {len(chunks_by_file)}")
        context_parts.append(f"• Estimated tokens: {sum(c.token_estimate for c in chunks)}")
        
        # Add chunk type breakdown
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        context_parts.append(f"• Chunk types: {dict(chunk_types)}")
        context_parts.append("")
        
        return '\n'.join(context_parts)
    
    def get_file_context(self, file_path: str, max_chunks: int = 10) -> str:
        """Get context for a specific file."""
        chunks = self.chunk_store.get_chunks_by_file(file_path)
        
        if not chunks:
            return f"No chunks found for file: {file_path}"
        
        # Sort chunks by line number
        chunks.sort(key=lambda c: c.start_line)
        
        # Limit chunks
        selected_chunks = chunks[:max_chunks]
        
        context_parts = []
        context_parts.append(f"FILE CONTEXT: {file_path}")
        context_parts.append("=" * 60)
        
        for chunk in selected_chunks:
            context_parts.append(f"<chunk lines=\"{chunk.start_line}-{chunk.end_line}\" type=\"{chunk.chunk_type.value}\">")
            context_parts.append(chunk.content)
            context_parts.append("</chunk>")
            context_parts.append("")
        
        return '\n'.join(context_parts)
    
    def find_related_chunks(self, chunk: Chunk, max_related: int = 5) -> List[Chunk]:
        """Find chunks related to the given chunk."""
        # Search using chunk content as query
        query = f"{chunk.metadata.function_name or ''} {chunk.metadata.component_name or ''} {chunk.chunk_type.value}"
        
        results = self.chunk_store.search(query, k=max_related + 5)  # Get extra for filtering
        
        # Filter out the original chunk and return related ones
        related = []
        for result in results:
            if result.chunk.chunk_hash != chunk.chunk_hash:
                related.append(result.chunk)
            
            if len(related) >= max_related:
                break
        
        return related
    
    def get_import_context(self, imports: List[str]) -> List[Chunk]:
        """Get context for specific imports."""
        import_chunks = []
        
        for imp in imports:
            # Search for chunks related to this import
            results = self.chunk_store.search(
                query=f"import {imp} export {imp}",
                k=3,
                filters={'chunk_types': ['import_block', 'export', 'function', 'component']}
            )
            
            for result in results:
                if result.chunk not in import_chunks:
                    import_chunks.append(result.chunk)
        
        return import_chunks
    
    def get_component_hierarchy(self, component_name: str) -> Dict[str, List[Chunk]]:
        """Get component and its related components (parent/child relationships)."""
        # Search for the main component
        results = self.chunk_store.search(
            query=f"component {component_name}",
            k=10,
            filters={'chunk_types': ['component', 'function']}
        )
        
        hierarchy = {
            'main': [],
            'children': [],
            'parents': []
        }
        
        for result in results:
            chunk = result.chunk
            
            # Main component (exact name match)
            if chunk.metadata.component_name == component_name:
                hierarchy['main'].append(chunk)
            
            # Child components (uses this component)
            elif component_name.lower() in chunk.content.lower():
                hierarchy['children'].append(chunk)
            
            # Parent components (this component uses them)
            elif chunk.metadata.component_name and chunk.metadata.component_name.lower() in component_name.lower():
                hierarchy['parents'].append(chunk)
        
        return hierarchy


class AdvancedRetriever(ContextRetriever):
    """Advanced retriever with enhanced features."""
    
    def __init__(self, chunk_store: ChunkStore, config: Optional[RetrievalConfig] = None):
        super().__init__(chunk_store, config)
        self._query_history: List[str] = []
        self._session_context: Set[str] = set()  # Track chunks used in session
    
    def get_context_with_memory(self, 
                               user_request: str,
                               conversation_history: Optional[List[str]] = None,
                               **kwargs) -> str:
        """Get context with conversation memory."""
        # Add to query history
        self._query_history.append(user_request)
        
        # Enhance query with conversation context
        if conversation_history:
            # Look for patterns in conversation
            enhanced_query = self._enhance_query_with_history(user_request, conversation_history)
        else:
            enhanced_query = user_request
        
        # Get regular context
        context = self.get_relevant_context(enhanced_query, **kwargs)
        
        # Track chunks used
        self._update_session_context(context)
        
        return context
    
    def _enhance_query_with_history(self, query: str, history: List[str]) -> str:
        """Enhance query based on conversation history."""
        # Extract key terms from recent conversation
        recent_terms = set()
        for msg in history[-3:]:  # Last 3 messages
            words = msg.lower().split()
            # Extract potential component/function names (camelCase, PascalCase)
            for word in words:
                if len(word) > 3 and (word[0].isupper() or '_' in word):
                    recent_terms.add(word)
        
        # Combine with current query
        if recent_terms:
            enhanced_query = f"{query} {' '.join(list(recent_terms)[:3])}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _update_session_context(self, context: str):
        """Track chunks used in this session."""
        # Extract chunk hashes from context (simplified)
        # In a real implementation, you'd pass chunk metadata through
        pass
    
    def get_session_stats(self) -> Dict[str, any]:
        """Get statistics about the current session."""
        return {
            'queries_made': len(self._query_history),
            'unique_chunks_accessed': len(self._session_context),
            'recent_queries': self._query_history[-5:] if self._query_history else []
        } 