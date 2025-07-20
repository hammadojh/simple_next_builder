#!/usr/bin/env python3
"""
Intelligent Context Selector

Uses an LLM to intelligently select which files and context are most relevant
for a given edit request, preventing context overflow and improving relevance.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class ContextFile:
    """Represents a file with relevance metadata."""
    path: str
    content: str
    size: int
    relevance_score: float
    reasons: List[str]
    dependencies: List[str]


@dataclass
class ContextSelection:
    """Result of context selection process."""
    selected_files: List[ContextFile]
    total_tokens: int
    selection_reasoning: str
    excluded_files: List[str]
    dependency_map: Dict[str, List[str]]


class IntelligentContextSelector:
    """
    Intelligently selects the most relevant files and context for LLM requests.
    
    Prevents context overflow while ensuring the LLM has all necessary information
    to complete the task effectively.
    """
    
    def __init__(self, max_tokens: int = 150000):
        """Initialize the context selector."""
        self.max_tokens = max_tokens
        self.tokens_per_char = 0.25  # Rough estimate: 4 chars per token
        
        # Initialize OpenAI client for context selection
        self.openai_client = None
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
            except ImportError:
                print("⚠️ OpenAI package not installed")
        
        print("🧠 Intelligent Context Selector initialized")
    
    def select_context(self, 
                      request: str, 
                      app_path: str, 
                      operation_type: str = "edit") -> ContextSelection:
        """
        Select the most relevant files and context for a given request.
        
        Args:
            request: The user's edit request or app idea
            app_path: Path to the app directory
            operation_type: Type of operation (edit, create, fix)
            
        Returns:
            ContextSelection with optimal files and reasoning
        """
        print(f"🧠 Selecting intelligent context for: {request}")
        print(f"📁 App path: {app_path}")
        
        # Get all available files
        all_files = self._scan_app_files(app_path)
        
        # Use LLM to analyze and select most relevant files
        if self.openai_client and len(all_files) > 3:
            selection = self._llm_select_context(request, all_files, operation_type)
        else:
            selection = self._fallback_select_context(request, all_files)
        
        print(f"✅ Selected {len(selection.selected_files)} files ({selection.total_tokens:,} tokens)")
        print(f"🎯 Selection reasoning: {selection.selection_reasoning}")
        
        return selection
    
    def _scan_app_files(self, app_path: str) -> List[Dict[str, Any]]:
        """Scan app directory and create file metadata."""
        files = []
        app_dir = Path(app_path)
        
        # File patterns to include
        include_patterns = ['.tsx', '.ts', '.jsx', '.js', '.json', '.css']
        exclude_dirs = {'node_modules', '.next', 'dist', 'build', '.git'}
        
        for file_path in app_dir.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix in include_patterns and
                not any(excluded in file_path.parts for excluded in exclude_dirs)):
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    relative_path = str(file_path.relative_to(app_dir))
                    
                    files.append({
                        'path': relative_path,
                        'content': content,
                        'size': len(content),
                        'tokens': int(len(content) * self.tokens_per_char),
                        'type': self._get_file_type(relative_path),
                        'imports': self._extract_imports(content),
                        'exports': self._extract_exports(content)
                    })
                except Exception as e:
                    print(f"⚠️ Could not read {file_path}: {e}")
        
        # Sort by potential relevance (components first, then pages, etc.)
        files.sort(key=lambda f: self._get_file_priority(f['type']))
        
        return files
    
    def _llm_select_context(self, 
                           request: str, 
                           all_files: List[Dict[str, Any]], 
                           operation_type: str) -> ContextSelection:
        """Use LLM to intelligently select most relevant files."""
        
        # Create file summary for LLM analysis
        file_summaries = []
        for file in all_files:
            summary = {
                'path': file['path'],
                'type': file['type'],
                'size': file['size'],
                'tokens': file['tokens'],
                'imports': file['imports'][:10],  # Limit for brevity
                'exports': file['exports'][:10],  # Limit for brevity
                'preview': file['content'][:200] + "..." if len(file['content']) > 200 else file['content']
            }
            file_summaries.append(summary)
        
        # Create LLM prompt for context selection
        selection_prompt = f"""You are an expert code analyst. Given a user request and a list of files in a NextJS app, select files needed to fulfill the request effectively.

USER REQUEST: {request}
OPERATION TYPE: {operation_type}
MAX TOKENS ALLOWED: {self.max_tokens}

AVAILABLE FILES:
{json.dumps(file_summaries, indent=2)}

INSTRUCTIONS:
Think step by step through this analysis:

1. **UNDERSTAND THE REQUEST**: What is the user trying to accomplish?
2. **IDENTIFY PRIMARY FILES**: Which files will be directly modified?
3. **FIND DEPENDENCIES**: What files are imported by or related to the primary files?
4. **CONSIDER CONTEXT**: What additional files help understand the overall structure?
5. **ERR ON THE SIDE OF INCLUSION**: It's better to include more files than miss important context.

Select files that are:
1. DIRECTLY needed to fulfill the request
2. Dependencies of the files being modified  
3. Related components or utilities that provide context
4. Type definitions and interfaces used
5. Configuration files if relevant

Return your analysis as JSON:
{{
  "selected_files": [
    {{
      "path": "file/path.tsx",
      "relevance_score": 0.95,
      "reasons": ["Primary component being modified", "Contains main logic"],
      "dependencies": ["@/components/ui/button", "@/types/todo"]
    }}
  ],
  "reasoning": "Brief explanation of selection strategy",
  "excluded_files": ["file1.tsx", "file2.tsx"],
  "exclusion_reasoning": "Why certain files were excluded"
}}

Focus on RELEVANCE over completeness. It's better to have precise context than overwhelming context."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use smaller model for analysis
                messages=[
                    {"role": "system", "content": "You are an expert code analyst specializing in context selection for LLM requests."},
                    {"role": "user", "content": selection_prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            analysis = self._parse_llm_selection(response_text)
            
            # Build final selection
            selected_files = []
            total_tokens = 0
            
            for selection in analysis.get('selected_files', []):
                file_path = selection['path']
                file_data = next((f for f in all_files if f['path'] == file_path), None)
                
                if file_data and total_tokens + file_data['tokens'] <= self.max_tokens:
                    context_file = ContextFile(
                        path=file_data['path'],
                        content=file_data['content'],
                        size=file_data['size'],
                        relevance_score=selection.get('relevance_score', 0.5),
                        reasons=selection.get('reasons', []),
                        dependencies=selection.get('dependencies', [])
                    )
                    selected_files.append(context_file)
                    total_tokens += file_data['tokens']
            
            return ContextSelection(
                selected_files=selected_files,
                total_tokens=total_tokens,
                selection_reasoning=analysis.get('reasoning', 'LLM-based selection'),
                excluded_files=analysis.get('excluded_files', []),
                dependency_map=self._build_dependency_map(selected_files)
            )
            
        except Exception as e:
            print(f"❌ LLM context selection failed: {e}")
            return self._fallback_select_context(request, all_files)
    
    def _fallback_select_context(self, 
                                request: str, 
                                all_files: List[Dict[str, Any]]) -> ContextSelection:
        """Fallback context selection using heuristics."""
        
        selected_files = []
        total_tokens = 0
        
        # More inclusive heuristic: include more files, prioritize by type and relevance
        priority_order = ['component', 'page', 'type', 'util', 'config', 'style']
        
        # Group files by type
        files_by_type = {}
        for file in all_files:
            file_type = file['type']
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(file)
        
        # Select files in priority order, but be more generous
        for file_type in priority_order:
            if file_type in files_by_type:
                # Sort files within type by size (smaller first for better coverage)
                sorted_files = sorted(files_by_type[file_type], key=lambda f: f['size'])
                
                for file_data in sorted_files:
                    if total_tokens + file_data['tokens'] <= self.max_tokens:
                        context_file = ContextFile(
                            path=file_data['path'],
                            content=file_data['content'],
                            size=file_data['size'],
                            relevance_score=0.5,
                            reasons=[f"Included by {file_type} priority", "Conservative inclusion strategy"],
                            dependencies=file_data.get('imports', [])
                        )
                        selected_files.append(context_file)
                        total_tokens += file_data['tokens']
        
        # Also include any remaining small files if we have token budget
        for file in all_files:
            if file['tokens'] < 500 and total_tokens + file['tokens'] <= self.max_tokens:
                if not any(sf.path == file['path'] for sf in selected_files):
                    context_file = ContextFile(
                        path=file['path'],
                        content=file['content'],
                        size=file['size'],
                        relevance_score=0.3,
                        reasons=["Small file inclusion", "Additional context"],
                        dependencies=file.get('imports', [])
                    )
                    selected_files.append(context_file)
                    total_tokens += file['tokens']
        
        return ContextSelection(
            selected_files=selected_files,
            total_tokens=total_tokens,
            selection_reasoning="Heuristic-based selection by file type priority",
            excluded_files=[f['path'] for f in all_files if f not in [sf.path for sf in selected_files]],
            dependency_map=self._build_dependency_map(selected_files)
        )
    
    def _parse_llm_selection(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response for file selection."""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            print(f"⚠️ Could not parse LLM selection response: {e}")
        
        return {'selected_files': [], 'reasoning': 'Parse error', 'excluded_files': []}
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type based on path and extension."""
        path = file_path.lower()
        
        if 'components/' in path:
            return 'component'
        elif 'app/' in path and path.endswith('.tsx'):
            return 'page'
        elif 'types/' in path:
            return 'type'
        elif 'lib/' in path or 'utils/' in path:
            return 'util'
        elif path.endswith('.json'):
            return 'config'
        elif path.endswith(('.css', '.scss')):
            return 'style'
        else:
            return 'other'
    
    def _get_file_priority(self, file_type: str) -> int:
        """Get priority score for file type (lower = higher priority)."""
        priorities = {
            'component': 1,
            'page': 2,
            'type': 3,
            'util': 4,
            'config': 5,
            'style': 6,
            'other': 7
        }
        return priorities.get(file_type, 10)
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from file content."""
        import re
        
        # Match various import patterns
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        imports = []
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)
        
        return imports[:20]  # Limit to prevent overflow
    
    def _extract_exports(self, content: str) -> List[str]:
        """Extract export statements from file content."""
        import re
        
        # Match export patterns
        export_patterns = [
            r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)',
            r'export\s+\{\s*([^}]+)\s*\}',
            r'export\s+default\s+(\w+)'
        ]
        
        exports = []
        for pattern in export_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    exports.extend([m.strip() for m in match if m.strip()])
                else:
                    exports.append(match.strip())
        
        return exports[:10]  # Limit to prevent overflow
    
    def _build_dependency_map(self, selected_files: List[ContextFile]) -> Dict[str, List[str]]:
        """Build a dependency map from selected files."""
        dependency_map = {}
        
        for file in selected_files:
            dependency_map[file.path] = file.dependencies
        
        return dependency_map
    
    def format_context_for_llm(self, selection: ContextSelection) -> str:
        """Format selected context for LLM consumption."""
        
        context_parts = [
            f"📁 SELECTED CONTEXT ({selection.total_tokens:,} tokens)",
            f"🎯 Selection Strategy: {selection.selection_reasoning}",
            f"📄 Files Included: {len(selection.selected_files)}",
            ""
        ]
        
        for file in selection.selected_files:
            context_parts.extend([
                f"📄 FILE: {file.path}",
                f"🎯 Relevance: {file.relevance_score:.2f} | Reasons: {', '.join(file.reasons)}",
                f"📦 Dependencies: {', '.join(file.dependencies[:5])}" + ("..." if len(file.dependencies) > 5 else ""),
                "```",
                file.content,
                "```",
                ""
            ])
        
        if selection.excluded_files:
            context_parts.extend([
                f"📋 EXCLUDED FILES ({len(selection.excluded_files)}): {', '.join(selection.excluded_files[:10])}" + ("..." if len(selection.excluded_files) > 10 else ""),
                ""
            ])
        
        return "\n".join(context_parts) 