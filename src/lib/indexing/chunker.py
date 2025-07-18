"""
Semantic Code Chunking for NextJS Projects

This module implements intelligent code chunking using AST parsing to split code
into semantically meaningful units for vector indexing.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum

class ChunkType(Enum):
    """Types of code chunks for semantic classification."""
    IMPORT_BLOCK = "import_block"
    INTERFACE = "interface" 
    TYPE_DEFINITION = "type_definition"
    FUNCTION = "function"
    COMPONENT = "component"
    CLASS = "class"
    VARIABLE = "variable"
    EXPORT = "export"
    CSS_RULE = "css_rule"
    COMMENT_BLOCK = "comment_block"
    CONFIG = "config"
    UNKNOWN = "unknown"

@dataclass
class ChunkMetadata:
    """Metadata associated with each code chunk."""
    function_name: Optional[str] = None
    component_name: Optional[str] = None
    interface_name: Optional[str] = None
    imports: List[str] = None
    exports: List[str] = None
    dependencies: List[str] = None
    complexity_score: int = 0
    
    def __post_init__(self):
        if self.imports is None:
            self.imports = []
        if self.exports is None:
            self.exports = []
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class Chunk:
    """Represents a semantic chunk of code."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    language: str
    chunk_hash: str
    metadata: ChunkMetadata
    file_hash: str
    project_id: str
    
    @classmethod
    def create(cls, content: str, file_path: str, start_line: int, end_line: int, 
               chunk_type: ChunkType, language: str, metadata: ChunkMetadata,
               file_hash: str, project_id: str) -> 'Chunk':
        """Create a chunk with auto-generated hash."""
        chunk_hash = hashlib.sha256(
            f"{file_path}:{start_line}:{end_line}:{content}".encode()
        ).hexdigest()[:16]
        
        return cls(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            language=language,
            chunk_hash=chunk_hash,
            metadata=metadata,
            file_hash=file_hash,
            project_id=project_id
        )

class CodeChunker:
    """Semantic code chunker for NextJS projects."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.max_chunk_size = 6000  # characters
        self.min_chunk_size = 50
        
    def chunk_file(self, file_path: str, content: str) -> List[Chunk]:
        """Chunk a file based on its type and content."""
        file_ext = Path(file_path).suffix.lower()
        
        # Calculate file hash
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        if file_ext in ['.ts', '.tsx', '.js', '.jsx']:
            return self._chunk_typescript_file(file_path, content, file_hash)
        elif file_ext in ['.css', '.scss']:
            return self._chunk_css_file(file_path, content, file_hash)
        elif file_ext == '.json':
            return self._chunk_json_file(file_path, content, file_hash)
        elif file_ext == '.md':
            return self._chunk_markdown_file(file_path, content, file_hash)
        else:
            # Fallback to simple chunking
            return self._chunk_generic_file(file_path, content, file_hash)
    
    def _chunk_typescript_file(self, file_path: str, content: str, file_hash: str) -> List[Chunk]:
        """Chunk TypeScript/JavaScript files using AST-like parsing."""
        lines = content.split('\n')
        chunks = []
        
        # First, extract imports as a single chunk
        import_chunk = self._extract_imports(file_path, lines, file_hash)
        if import_chunk:
            chunks.append(import_chunk)
        
        # Then parse the rest of the file
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments at top level
            if not line or line.startswith('//'):
                i += 1
                continue
                
            # Skip imports (already handled)
            if line.startswith('import ') or line.startswith('export ') and 'from' in line:
                i += 1
                continue
            
            chunk, next_i = self._parse_typescript_construct(file_path, lines, i, file_hash)
            if chunk:
                chunks.append(chunk)
                i = next_i
            else:
                i += 1
        
        return self._validate_and_merge_chunks(chunks)
    
    def _extract_imports(self, file_path: str, lines: List[str], file_hash: str) -> Optional[Chunk]:
        """Extract import statements as a single chunk."""
        import_lines = []
        start_line = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or (stripped.startswith('export ') and 'from' in stripped):
                if start_line is None:
                    start_line = i + 1
                import_lines.append(line)
            elif stripped.startswith('"use client"') or stripped.startswith("'use client'"):
                if start_line is None:
                    start_line = i + 1
                import_lines.append(line)
            elif import_lines and stripped:  # Non-empty line after imports
                break
        
        if not import_lines:
            return None
        
        content = '\n'.join(import_lines)
        imports = self._extract_import_names(content)
        
        metadata = ChunkMetadata(
            imports=imports,
            complexity_score=len(imports)
        )
        
        return Chunk.create(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line + len(import_lines) - 1,
            chunk_type=ChunkType.IMPORT_BLOCK,
            language=self._get_language(file_path),
            metadata=metadata,
            file_hash=file_hash,
            project_id=self.project_id
        )
    
    def _parse_typescript_construct(self, file_path: str, lines: List[str], start_i: int, file_hash: str) -> Tuple[Optional[Chunk], int]:
        """Parse a TypeScript construct (function, component, interface, etc.)."""
        start_line = lines[start_i].strip()
        
        # Interface or type definition
        if start_line.startswith('interface ') or start_line.startswith('type '):
            return self._parse_interface_or_type(file_path, lines, start_i, file_hash)
        
        # Function or component
        elif ('function ' in start_line or 
              start_line.startswith('const ') or 
              start_line.startswith('export default function') or
              start_line.startswith('export function')):
            return self._parse_function_or_component(file_path, lines, start_i, file_hash)
        
        # Class definition
        elif 'class ' in start_line:
            return self._parse_class(file_path, lines, start_i, file_hash)
        
        # Variable or constant
        elif start_line.startswith(('const ', 'let ', 'var ')):
            return self._parse_variable(file_path, lines, start_i, file_hash)
        
        # Export statement
        elif start_line.startswith('export '):
            return self._parse_export(file_path, lines, start_i, file_hash)
        
        return None, start_i + 1
    
    def _parse_function_or_component(self, file_path: str, lines: List[str], start_i: int, file_hash: str) -> Tuple[Optional[Chunk], int]:
        """Parse function or React component."""
        start_line_num = start_i + 1
        brace_count = 0
        paren_count = 0
        in_string = False
        string_char = None
        i = start_i
        
        # Find the complete function/component
        while i < len(lines):
            line = lines[i]
            
            for char in line:
                if not in_string:
                    if char in ['"', "'", '`']:
                        in_string = True
                        string_char = char
                    elif char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    elif char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                else:
                    if char == string_char and (i == 0 or line[i-1] != '\\'):
                        in_string = False
                        string_char = None
            
            i += 1
            
            # Function/component is complete when braces are balanced
            if brace_count == 0 and i > start_i + 1:
                break
                
            # Safety check to prevent infinite loops
            if i - start_i > 1000:
                break
        
        end_line_num = i
        content = '\n'.join(lines[start_i:i])
        
        # Determine if it's a component or function
        is_component = self._is_react_component(content)
        chunk_type = ChunkType.COMPONENT if is_component else ChunkType.FUNCTION
        
        # Extract metadata
        function_name = self._extract_function_name(content)
        component_name = function_name if is_component else None
        
        metadata = ChunkMetadata(
            function_name=function_name,
            component_name=component_name,
            complexity_score=self._calculate_complexity(content)
        )
        
        chunk = Chunk.create(
            content=content,
            file_path=file_path,
            start_line=start_line_num,
            end_line=end_line_num,
            chunk_type=chunk_type,
            language=self._get_language(file_path),
            metadata=metadata,
            file_hash=file_hash,
            project_id=self.project_id
        )
        
        return chunk, i
    
    def _parse_interface_or_type(self, file_path: str, lines: List[str], start_i: int, file_hash: str) -> Tuple[Optional[Chunk], int]:
        """Parse interface or type definition."""
        start_line_num = start_i + 1
        brace_count = 0
        i = start_i
        
        # Find the complete interface/type
        while i < len(lines):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            i += 1
            
            if brace_count == 0 and i > start_i + 1:
                break
                
            # Safety check
            if i - start_i > 100:
                break
        
        end_line_num = i
        content = '\n'.join(lines[start_i:i])
        
        # Extract interface/type name
        interface_name = self._extract_interface_name(content)
        chunk_type = ChunkType.INTERFACE if content.strip().startswith('interface') else ChunkType.TYPE_DEFINITION
        
        metadata = ChunkMetadata(
            interface_name=interface_name,
            complexity_score=len(content.split('\n'))
        )
        
        chunk = Chunk.create(
            content=content,
            file_path=file_path,
            start_line=start_line_num,
            end_line=end_line_num,
            chunk_type=chunk_type,
            language=self._get_language(file_path),
            metadata=metadata,
            file_hash=file_hash,
            project_id=self.project_id
        )
        
        return chunk, i
    
    def _parse_class(self, file_path: str, lines: List[str], start_i: int, file_hash: str) -> Tuple[Optional[Chunk], int]:
        """Parse class definition."""
        # Similar logic to function parsing but for classes
        return self._parse_function_or_component(file_path, lines, start_i, file_hash)
    
    def _parse_variable(self, file_path: str, lines: List[str], start_i: int, file_hash: str) -> Tuple[Optional[Chunk], int]:
        """Parse variable declaration."""
        start_line_num = start_i + 1
        content = lines[start_i]
        
        # Handle multi-line variable declarations
        if '{' in content or '[' in content:
            brace_count = content.count('{') - content.count('}')
            bracket_count = content.count('[') - content.count(']')
            i = start_i + 1
            
            while i < len(lines) and (brace_count > 0 or bracket_count > 0):
                line = lines[i]
                content += '\n' + line
                brace_count += line.count('{') - line.count('}')
                bracket_count += line.count('[') - line.count(']')
                i += 1
        else:
            i = start_i + 1
        
        metadata = ChunkMetadata(
            complexity_score=1
        )
        
        chunk = Chunk.create(
            content=content,
            file_path=file_path,
            start_line=start_line_num,
            end_line=i,
            chunk_type=ChunkType.VARIABLE,
            language=self._get_language(file_path),
            metadata=metadata,
            file_hash=file_hash,
            project_id=self.project_id
        )
        
        return chunk, i
    
    def _parse_export(self, file_path: str, lines: List[str], start_i: int, file_hash: str) -> Tuple[Optional[Chunk], int]:
        """Parse export statement."""
        content = lines[start_i]
        exports = self._extract_export_names(content)
        
        metadata = ChunkMetadata(
            exports=exports,
            complexity_score=1
        )
        
        chunk = Chunk.create(
            content=content,
            file_path=file_path,
            start_line=start_i + 1,
            end_line=start_i + 1,
            chunk_type=ChunkType.EXPORT,
            language=self._get_language(file_path),
            metadata=metadata,
            file_hash=file_hash,
            project_id=self.project_id
        )
        
        return chunk, start_i + 1
    
    def _chunk_css_file(self, file_path: str, content: str, file_hash: str) -> List[Chunk]:
        """Chunk CSS files by rules."""
        chunks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('/*'):
                i += 1
                continue
            
            # Find CSS rule
            if '{' in line or (i + 1 < len(lines) and '{' in lines[i + 1]):
                start_i = i
                brace_count = 0
                
                while i < len(lines):
                    brace_count += lines[i].count('{') - lines[i].count('}')
                    i += 1
                    if brace_count == 0:
                        break
                
                rule_content = '\n'.join(lines[start_i:i])
                
                metadata = ChunkMetadata(
                    complexity_score=rule_content.count(';')
                )
                
                chunk = Chunk.create(
                    content=rule_content,
                    file_path=file_path,
                    start_line=start_i + 1,
                    end_line=i,
                    chunk_type=ChunkType.CSS_RULE,
                    language='css',
                    metadata=metadata,
                    file_hash=file_hash,
                    project_id=self.project_id
                )
                chunks.append(chunk)
            else:
                i += 1
        
        return chunks
    
    def _chunk_json_file(self, file_path: str, content: str, file_hash: str) -> List[Chunk]:
        """Chunk JSON configuration files."""
        metadata = ChunkMetadata(complexity_score=content.count('{'))
        
        chunk = Chunk.create(
            content=content,
            file_path=file_path,
            start_line=1,
            end_line=len(content.split('\n')),
            chunk_type=ChunkType.CONFIG,
            language='json',
            metadata=metadata,
            file_hash=file_hash,
            project_id=self.project_id
        )
        
        return [chunk]
    
    def _chunk_markdown_file(self, file_path: str, content: str, file_hash: str) -> List[Chunk]:
        """Chunk markdown files by sections."""
        chunks = []
        lines = content.split('\n')
        current_section = []
        start_line = 1
        
        for i, line in enumerate(lines):
            if line.startswith('#') and current_section:
                # Save previous section
                section_content = '\n'.join(current_section)
                chunk = Chunk.create(
                    content=section_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=i,
                    chunk_type=ChunkType.COMMENT_BLOCK,
                    language='markdown',
                    metadata=ChunkMetadata(complexity_score=len(current_section)),
                    file_hash=file_hash,
                    project_id=self.project_id
                )
                chunks.append(chunk)
                
                # Start new section
                current_section = [line]
                start_line = i + 1
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_content = '\n'.join(current_section)
            chunk = Chunk.create(
                content=section_content,
                file_path=file_path,
                start_line=start_line,
                end_line=len(lines),
                chunk_type=ChunkType.COMMENT_BLOCK,
                language='markdown',
                metadata=ChunkMetadata(complexity_score=len(current_section)),
                file_hash=file_hash,
                project_id=self.project_id
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_generic_file(self, file_path: str, content: str, file_hash: str) -> List[Chunk]:
        """Generic chunking for unknown file types."""
        chunk = Chunk.create(
            content=content,
            file_path=file_path,
            start_line=1,
            end_line=len(content.split('\n')),
            chunk_type=ChunkType.UNKNOWN,
            language='text',
            metadata=ChunkMetadata(complexity_score=len(content) // 100),
            file_hash=file_hash,
            project_id=self.project_id
        )
        
        return [chunk]
    
    def _validate_and_merge_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Validate chunks and merge small adjacent chunks."""
        validated_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small or too large
            if len(chunk.content) < self.min_chunk_size:
                continue
            
            if len(chunk.content) > self.max_chunk_size:
                # Split large chunks
                split_chunks = self._split_large_chunk(chunk)
                validated_chunks.extend(split_chunks)
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a chunk that's too large."""
        lines = chunk.content.split('\n')
        sub_chunks = []
        
        chunk_size = len(lines) // 2  # Split in half
        for i in range(0, len(lines), chunk_size):
            sub_content = '\n'.join(lines[i:i + chunk_size])
            
            sub_chunk = Chunk.create(
                content=sub_content,
                file_path=chunk.file_path,
                start_line=chunk.start_line + i,
                end_line=chunk.start_line + i + len(lines[i:i + chunk_size]) - 1,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                metadata=chunk.metadata,
                file_hash=chunk.file_hash,
                project_id=chunk.project_id
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    # Helper methods
    def _is_react_component(self, content: str) -> bool:
        """Determine if a function is a React component."""
        return (
            'return (' in content and 
            ('<' in content and '>' in content) and
            ('jsx' in content.lower() or 'tsx' in content.lower() or 
             content.count('<') > 2)
        )
    
    def _extract_function_name(self, content: str) -> Optional[str]:
        """Extract function name from content."""
        # Try different patterns
        patterns = [
            r'function\s+(\w+)',
            r'const\s+(\w+)\s*=',
            r'export\s+default\s+function\s+(\w+)',
            r'export\s+function\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_interface_name(self, content: str) -> Optional[str]:
        """Extract interface name from content."""
        match = re.search(r'(?:interface|type)\s+(\w+)', content)
        return match.group(1) if match else None
    
    def _extract_import_names(self, content: str) -> List[str]:
        """Extract imported module names."""
        imports = []
        for line in content.split('\n'):
            if 'from' in line:
                match = re.search(r'from\s+[\'"]([^\'"]+)[\'"]', line)
                if match:
                    imports.append(match.group(1))
        return imports
    
    def _extract_export_names(self, content: str) -> List[str]:
        """Extract exported names."""
        exports = []
        if 'export {' in content:
            match = re.search(r'export\s*\{\s*([^}]+)\s*\}', content)
            if match:
                exports.extend([name.strip() for name in match.group(1).split(',')])
        elif 'export default' in content:
            exports.append('default')
        return exports
    
    def _calculate_complexity(self, content: str) -> int:
        """Calculate a simple complexity score."""
        # Count various complexity indicators
        score = 0
        score += content.count('if ') * 2
        score += content.count('for ') * 2
        score += content.count('while ') * 2
        score += content.count('switch ') * 3
        score += content.count('catch ') * 2
        score += content.count('useState') * 1
        score += content.count('useEffect') * 2
        score += len(content.split('\n'))  # Lines of code
        return score
    
    def _get_language(self, file_path: str) -> str:
        """Get language from file extension."""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.css': 'css',
            '.scss': 'scss',
            '.json': 'json',
            '.md': 'markdown'
        }
        return lang_map.get(ext, 'text') 