"""
Intent-based code editor that minimizes AI's role in diff generation.

Instead of asking AI to generate diffs, we ask for structured editing intent
and then generate the diffs programmatically. This is more reliable because:
1. AI describes WHAT to change, not HOW to change it  
2. We control the diff format generation
3. Less chance of malformed diffs
4. More robust and predictable
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import difflib


@dataclass
class EditIntent:
    """Represents a structured editing intent from AI."""
    file_path: str
    action: str  # 'replace', 'insert', 'delete', 'modify'
    target: str  # What to find/target
    replacement: str  # What to replace it with (for replace/modify)
    context: str  # Additional context for matching
    line_number: Optional[int] = None  # Specific line if known


class IntentBasedEditor:
    """
    Editor that uses structured AI intent to make precise code changes.
    
    This reduces AI's role from generating complex diffs to just describing
    what changes need to be made in a structured format.
    """
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
    
    def apply_intent_list(self, intents: List[EditIntent]) -> List[Tuple[bool, str]]:
        """
        Apply a list of editing intents to files.
        
        Args:
            intents: List of structured editing intents
            
        Returns:
            List of (success, error_message) tuples
        """
        results = []
        
        # Group intents by file for efficient processing
        intents_by_file = {}
        for intent in intents:
            if intent.file_path not in intents_by_file:
                intents_by_file[intent.file_path] = []
            intents_by_file[intent.file_path].append(intent)
        
        # Process each file
        for file_path, file_intents in intents_by_file.items():
            try:
                result = self._apply_file_intents(file_path, file_intents)
                results.append(result)
            except Exception as e:
                results.append((False, f"Error processing {file_path}: {str(e)}"))
        
        return results
    
    def _apply_file_intents(self, file_path: str, intents: List[EditIntent]) -> Tuple[bool, str]:
        """Apply all intents for a single file."""
        target_file = self.repo_root / file_path
        
        if not target_file.exists():
            return False, f"File not found: {file_path}"
        
        try:
            # Read current content
            with open(target_file, 'r') as f:
                original_content = f.read()
            
            current_content = original_content
            
            # Apply each intent in order
            for intent in intents:
                current_content = self._apply_single_intent(current_content, intent)
            
            # Write back if changed
            if current_content != original_content:
                with open(target_file, 'w') as f:
                    f.write(current_content)
                
                # Generate diff for debugging
                self._save_debug_diff(file_path, original_content, current_content)
                
                return True, f"Applied {len(intents)} intents to {file_path}"
            else:
                return True, f"No changes needed for {file_path}"
                
        except Exception as e:
            return False, f"Error applying intents to {file_path}: {str(e)}"
    
    def _apply_single_intent(self, content: str, intent: EditIntent) -> str:
        """Apply a single editing intent to content."""
        if intent.action == 'replace':
            return self._apply_replace_intent(content, intent)
        elif intent.action == 'insert':
            return self._apply_insert_intent(content, intent)
        elif intent.action == 'delete':
            return self._apply_delete_intent(content, intent)
        elif intent.action == 'modify':
            return self._apply_modify_intent(content, intent)
        else:
            print(f"⚠️ Unknown intent action: {intent.action}")
            return content
    
    def _apply_replace_intent(self, content: str, intent: EditIntent) -> str:
        """Replace target text with replacement text."""
        # Try exact match first
        if intent.target in content:
            return content.replace(intent.target, intent.replacement, 1)
        
        # Try fuzzy matching with context
        return self._fuzzy_replace(content, intent.target, intent.replacement)
    
    def _apply_insert_intent(self, content: str, intent: EditIntent) -> str:
        """Insert text at specified location."""
        if intent.line_number is not None:
            lines = content.split('\n')
            lines.insert(intent.line_number - 1, intent.replacement)
            return '\n'.join(lines)
        
        # Insert after target
        if intent.target in content:
            return content.replace(intent.target, intent.target + '\n' + intent.replacement, 1)
        
        return content
    
    def _apply_delete_intent(self, content: str, intent: EditIntent) -> str:
        """Delete specified text."""
        return content.replace(intent.target, '', 1)
    
    def _apply_modify_intent(self, content: str, intent: EditIntent) -> str:
        """Modify text using pattern-based replacement."""
        # This is similar to replace but with more intelligent matching
        return self._fuzzy_replace(content, intent.target, intent.replacement)
    
    def _fuzzy_replace(self, content: str, target: str, replacement: str) -> str:
        """Perform fuzzy text replacement with whitespace normalization."""
        # Normalize whitespace for matching
        normalized_target = re.sub(r'\s+', ' ', target.strip())
        
        # Try to find similar text in content
        lines = content.split('\n')
        for i, line in enumerate(lines):
            normalized_line = re.sub(r'\s+', ' ', line.strip())
            if normalized_target in normalized_line:
                # Replace in this line
                lines[i] = line.replace(target.strip(), replacement.strip())
                return '\n'.join(lines)
        
        # If not found, fall back to simple replacement
        return content.replace(target, replacement, 1)
    
    def _save_debug_diff(self, file_path: str, old_content: str, new_content: str):
        """Save a debug diff for manual inspection."""
        try:
            diff_lines = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=''
            ))
            
            if diff_lines:
                debug_dir = self.repo_root / "debug_responses"
                debug_dir.mkdir(exist_ok=True)
                
                import time
                timestamp = int(time.time())
                debug_file = debug_dir / f"intent_diff_{file_path.replace('/', '_')}_{timestamp}.patch"
                
                with open(debug_file, 'w') as f:
                    f.write(''.join(diff_lines))
                
                print(f"🔍 Debug diff saved: {debug_file}")
        except Exception:
            pass  # Debug file save is not critical


def parse_ai_intent_response(ai_response: str) -> List[EditIntent]:
    """
    Parse AI response containing structured editing intents.
    
    Expected AI response format:
    ```json
    {
        "intents": [
            {
                "file_path": "app/page.tsx",
                "action": "replace",
                "target": "className=\"bg-blue-500\"",
                "replacement": "className=\"bg-blue-700\"",
                "context": "button styling"
            }
        ]
    }
    ```
    """
    intents = []
    
    try:
        # Extract JSON from AI response
        json_match = re.search(r'```json\s*(.*?)\s*```', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_str = ai_response.strip()
        
        data = json.loads(json_str)
        
        if 'intents' in data:
            for intent_data in data['intents']:
                intent = EditIntent(
                    file_path=intent_data['file_path'],
                    action=intent_data['action'],
                    target=intent_data['target'],
                    replacement=intent_data.get('replacement', ''),
                    context=intent_data.get('context', ''),
                    line_number=intent_data.get('line_number')
                )
                intents.append(intent)
    
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse AI intent JSON: {e}")
        print(f"Raw response: {ai_response[:200]}...")
    except Exception as e:
        print(f"❌ Error parsing AI intent: {e}")
    
    return intents


def get_intent_based_prompt() -> str:
    """
    Get the prompt for AI to generate structured editing intents
    instead of raw diffs.
    """
    return """
You are editing a NextJS application. Instead of generating diffs, please provide structured editing intents in JSON format.

RESPOND WITH JSON IN THIS EXACT FORMAT:

```json
{
    "intents": [
        {
            "file_path": "app/page.tsx",
            "action": "replace",
            "target": "className=\"bg-blue-500 hover:bg-blue-600\"",
            "replacement": "className=\"bg-blue-700 hover:bg-blue-800\"",
            "context": "making button darker blue"
        },
        {
            "file_path": "app/layout.tsx", 
            "action": "replace",
            "target": "className=\"bg-gray-50\"",
            "replacement": "className=\"bg-gray-900 text-white\"",
            "context": "changing to dark theme"
        }
    ]
}
```

AVAILABLE ACTIONS:
- "replace": Replace exact text with new text
- "insert": Insert text at specific location or after target
- "delete": Remove specific text
- "modify": Modify text with pattern matching

ADVANTAGES OF THIS APPROACH:
✅ No malformed diffs from AI
✅ Precise, targeted changes
✅ Easy to validate and debug
✅ Programmatic diff generation
✅ Less chance of syntax errors

Focus on making precise, minimal changes. Always include context to explain the purpose of each change.
""" 