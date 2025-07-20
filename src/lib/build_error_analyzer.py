"""
Smart Build Error Analyzer

Takes build errors and uses LLM to generate structured task lists for fixing them.
This replaces the problematic diff-based approach with intelligent task planning.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    EDIT_FILE = "edit_file"
    CREATE_FILE = "create_file" 
    DELETE_FILE = "delete_file"
    RUN_COMMAND = "run_command"
    INSTALL_DEPENDENCY = "install_dependency"
    ADD_IMPORT = "add_import"
    FIX_SYNTAX = "fix_syntax"

@dataclass
class FixTask:
    """Represents a single task to fix a build error"""
    id: str
    type: TaskType
    description: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    line_number: Optional[int] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class BuildErrorAnalysis:
    """Result of analyzing build errors"""
    error_summary: str
    root_cause: str
    confidence: float  # 0-1 confidence in the analysis
    tasks: List[FixTask]
    estimated_complexity: str  # "simple", "medium", "complex"
    success_probability: float  # 0-1 probability of success

class SmartBuildErrorAnalyzer:
    """
    Analyzes build errors and generates intelligent task lists to fix them.
    
    This replaces the problematic diff-based approach with structured task planning.
    """
    
    def __init__(self, app_path: str):
        self.app_path = app_path
        
    def analyze_build_error(self, error_output: str, failed_files: List[str] = None) -> BuildErrorAnalysis:
        """
        Analyze build error and generate structured fix tasks.
        
        Args:
            error_output: Raw build error output
            failed_files: List of files that failed to compile
            
        Returns:
            BuildErrorAnalysis with structured tasks to fix the error
        """
        print("🧠 Analyzing build error with smart LLM...")
        
        # Get codebase context
        codebase_structure = self._get_codebase_structure()
        related_files = self._get_related_files(error_output, failed_files)
        
        # Prepare context for LLM
        analysis_context = {
            "error_output": error_output,
            "failed_files": failed_files or [],
            "codebase_structure": codebase_structure,
            "related_files": related_files
        }
        
        # Get LLM analysis
        analysis_result = self._request_llm_analysis(analysis_context)
        
        if analysis_result:
            return self._parse_analysis_result(analysis_result)
        else:
            # Fallback to basic analysis
            return self._fallback_analysis(error_output, failed_files)
    
    def _get_codebase_structure(self) -> Dict[str, Any]:
        """Get high-level codebase structure for context"""
        structure = {
            "directories": [],
            "key_files": [],
            "package_json": None,
            "framework": "nextjs"
        }
        
        try:
            # Get directory structure
            for root, dirs, files in os.walk(self.app_path):
                # Skip node_modules and .next
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.next', '.git']]
                
                rel_root = os.path.relpath(root, self.app_path)
                if rel_root != '.':
                    structure["directories"].append(rel_root)
                
                # Collect key files
                for file in files:
                    if file.endswith(('.tsx', '.ts', '.js', '.jsx', 'package.json', 'next.config.js')):
                        file_path = os.path.join(rel_root, file) if rel_root != '.' else file
                        structure["key_files"].append(file_path)
            
            # Read package.json if exists
            package_json_path = os.path.join(self.app_path, 'package.json')
            if os.path.exists(package_json_path):
                with open(package_json_path, 'r') as f:
                    structure["package_json"] = json.load(f)
                    
        except Exception as e:
            print(f"  ⚠️ Error getting codebase structure: {e}")
            
        return structure
    
    def _get_related_files(self, error_output: str, failed_files: List[str] = None) -> Dict[str, str]:
        """Get content of files related to the error"""
        related_files = {}
        
        # Start with failed files
        files_to_check = set(failed_files or [])
        
        # Extract file paths from error output
        import re
        file_patterns = [
            r'\.\/([^:\s]+\.(tsx?|jsx?))[:]\d+',  # ./file.tsx:10
            r'([^:\s]+\.(tsx?|jsx?))[:]\d+',      # file.tsx:10
            r'Error in ([^:\s]+\.(tsx?|jsx?))',   # Error in file.tsx
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, error_output)
            for match in matches:
                if isinstance(match, tuple):
                    files_to_check.add(match[0])
                else:
                    files_to_check.add(match)
        
        # Read file contents
        for file_path in files_to_check:
            full_path = os.path.join(self.app_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Limit content size for LLM
                        if len(content) > 5000:
                            content = content[:5000] + "\n... (truncated)"
                        related_files[file_path] = content
                except Exception as e:
                    related_files[file_path] = f"Error reading file: {e}"
        
        return related_files
    
    def _request_llm_analysis(self, context: Dict[str, Any]) -> Optional[str]:
        """Request LLM analysis of the build error"""
        try:
            import openai
            import os
            
            prompt = self._build_analysis_prompt(context)
            
            # Use OpenAI directly for JSON response (bypass legacy diff validation)
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("  ⚠️ No OpenAI API key found")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert developer analyzing build errors. Return ONLY valid JSON as specified in the prompt."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"  ⚠️ Direct LLM analysis failed: {e}")
            # Fallback to app_builder method
            try:
                from .app_builder import MultiLLMAppBuilder
                
                app_builder = MultiLLMAppBuilder()
                
                # Use a simpler prompt that doesn't expect diff format
                simple_prompt = f"""
Analyze this build error and return a simple JSON response:

ERROR:
{context['error_output']}

Return JSON with: {{"error_type": "...", "fix_description": "...", "confidence": 0.8}}
"""
                
                success, response = app_builder.make_openai_request(simple_prompt, context="error_analysis")
                
                if success:
                    return response
                else:
                    print(f"  ⚠️ Fallback LLM request failed: {response}")
                    return None
                    
            except Exception as fallback_e:
                print(f"  ⚠️ All LLM methods failed: {fallback_e}")
                return None
    
    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for LLM analysis"""
        prompt = f"""
SMART BUILD ERROR ANALYSIS

You are a expert developer analyzing a build error. Your task is to analyze the error and return a structured JSON response with specific tasks to fix it.

ERROR OUTPUT:
{context['error_output']}

FAILED FILES:
{', '.join(context['failed_files'])}

CODEBASE STRUCTURE:
{json.dumps(context['codebase_structure'], indent=2)}

RELATED FILE CONTENTS:
{json.dumps(context['related_files'], indent=2)}

Please analyze this build error and return ONLY a JSON response with this exact structure:

```json
{{
    "error_summary": "Brief summary of the error",
    "root_cause": "Detailed explanation of what's causing the error",
    "confidence": 0.9,
    "estimated_complexity": "simple|medium|complex",
    "success_probability": 0.8,
    "tasks": [
        {{
            "id": "task_1",
            "type": "edit_file|create_file|delete_file|run_command|install_dependency|add_import|fix_syntax",
            "description": "What this task does",
            "file_path": "path/to/file.tsx",
            "content": "new file content or specific edit",
            "command": "npm install package-name",
            "line_number": 42,
            "dependencies": ["task_id_this_depends_on"]
        }}
    ]
}}
```

TASK TYPES:
- edit_file: Modify existing file content
- create_file: Create new file with content
- delete_file: Delete a file
- run_command: Execute shell command
- install_dependency: Install npm package
- add_import: Add import statement to file
- fix_syntax: Fix syntax error in file

Focus on:
1. The EXACT cause of the error
2. Minimal, targeted fixes
3. Proper task ordering with dependencies
4. Complete file content when editing
5. Specific commands to run

Return ONLY the JSON, no other text.
"""
        return prompt
    
    def _parse_analysis_result(self, llm_response: str) -> BuildErrorAnalysis:
        """Parse LLM response into BuildErrorAnalysis"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if not json_match:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{.*\})', llm_response, re.DOTALL)
            
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group(1))
            
            # Parse tasks
            tasks = []
            for task_data in data.get('tasks', []):
                task = FixTask(
                    id=task_data.get('id', f"task_{len(tasks)}"),
                    type=TaskType(task_data.get('type', 'edit_file')),
                    description=task_data.get('description', ''),
                    file_path=task_data.get('file_path'),
                    content=task_data.get('content'),
                    command=task_data.get('command'),
                    line_number=task_data.get('line_number'),
                    dependencies=task_data.get('dependencies', [])
                )
                tasks.append(task)
            
            return BuildErrorAnalysis(
                error_summary=data.get('error_summary', 'Unknown error'),
                root_cause=data.get('root_cause', 'Unknown cause'),
                confidence=data.get('confidence', 0.5),
                tasks=tasks,
                estimated_complexity=data.get('estimated_complexity', 'medium'),
                success_probability=data.get('success_probability', 0.5)
            )
            
        except Exception as e:
            print(f"  ⚠️ Failed to parse LLM analysis: {e}")
            return self._fallback_analysis(llm_response, [])
    
    def _fallback_analysis(self, error_output: str, failed_files: List[str] = None) -> BuildErrorAnalysis:
        """Fallback analysis when LLM fails"""
        print("  🔄 Using fallback analysis...")
        
        # Simple pattern-based analysis
        tasks = []
        
        if "Cannot resolve module" in error_output or "Module not found" in error_output:
            # Missing dependency
            import re
            module_match = re.search(r"Can't resolve '([^']+)'", error_output)
            if module_match:
                module_name = module_match.group(1)
                tasks.append(FixTask(
                    id="install_missing_dep",
                    type=TaskType.INSTALL_DEPENDENCY,
                    description=f"Install missing dependency: {module_name}",
                    command=f"npm install {module_name}"
                ))
        
        elif "unexpected eof" in error_output.lower():
            # Syntax error - likely missing closing brace
            # Extract file path from error
            import re
            file_match = re.search(r'([^/\s]+\.tsx?)', error_output)
            files_to_fix = failed_files or []
            if file_match and file_match.group(1) not in files_to_fix:
                files_to_fix.append(file_match.group(1))
            
            for file_path in files_to_fix:
                tasks.append(FixTask(
                    id=f"fix_syntax_{file_path.replace('/', '_').replace('.', '_')}",
                    type=TaskType.FIX_SYNTAX,
                    description=f"Fix unexpected EOF syntax error in {file_path}",
                    file_path=file_path
                ))
        
        return BuildErrorAnalysis(
            error_summary="Build error detected",
            root_cause="Unable to determine specific cause",
            confidence=0.3,
            tasks=tasks,
            estimated_complexity="medium",
            success_probability=0.4
        ) 