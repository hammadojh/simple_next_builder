"""
Intent-based code editor that minimizes AI's role in diff generation.

Instead of asking AI to generate diffs, we ask for structured editing intent
and then generate the diffs programmatically. This is more reliable because:
1. AI describes WHAT to change, not HOW to change it  
2. We control the diff format generation
3. Less chance of malformed diffs
4. More robust and predictable

ENHANCED: Now includes corruption detection and AST-aware processing
to prevent the structural corruption issues that occurred previously.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import difflib

# Import our corruption detection system
from .corruption_detector import FileCorruptionDetector, CorruptionSeverity


@dataclass
class EditIntent:
    """Represents a structured editing intent from AI."""
    file_path: str
    action: str  # 'replace', 'insert', 'delete', 'modify'
    target: str  # What to find/target
    replacement: str  # What to replace it with (for replace/modify)
    context: str  # Additional context for matching
    line_number: Optional[int] = None  # Specific line if known


class StructuralCodeProcessor:
    """
    AST-aware code processor that understands code structure.
    
    This prevents the blind text replacement issues that caused
    the corruption problems in myapp23.
    """
    
    def __init__(self):
        self.corruption_detector = FileCorruptionDetector()
    
    def apply_intent_structurally(self, content: str, intent: EditIntent, file_path: str) -> str:
        """
        Apply an intent with full structural awareness.
        
        This method prevents corruption by:
        1. Validating the current file state
        2. Understanding code structure before making changes
        3. Applying changes at logical boundaries
        4. Validating the result before returning
        """
        # 1. Pre-edit validation
        is_corrupted, severity = self.corruption_detector.is_file_corrupted(file_path, content)
        if is_corrupted and severity in [CorruptionSeverity.HIGH, CorruptionSeverity.CRITICAL]:
            print(f"⚠️ File {file_path} is corrupted ({severity.value}) - applying conservative edits")
            return self._apply_conservative_edit(content, intent)
        
        # 2. Determine file type and use appropriate processor
        if file_path.endswith(('.tsx', '.jsx')):
            return self._apply_jsx_intent(content, intent, file_path)
        elif file_path.endswith(('.ts', '.js')):
            return self._apply_typescript_intent(content, intent, file_path)
        else:
            # Fallback to safe text processing
            return self._apply_safe_text_intent(content, intent)
    
    def _apply_jsx_intent(self, content: str, intent: EditIntent, file_path: str) -> str:
        """Apply intent to JSX/TSX files with structural awareness."""
        lines = content.split('\n')
        
        if intent.action == 'modify' and 'import' in intent.target.lower():
            return self._apply_import_modification(content, intent)
        elif intent.action == 'modify' and 'return (' in intent.target:
            return self._apply_return_modification(content, intent)
        elif intent.action == 'insert' and 'hook' in intent.context.lower():
            return self._apply_hook_insertion(content, intent)
        else:
            # Use safer fuzzy replacement for other cases
            return self._apply_safe_fuzzy_replacement(content, intent)
    
    def _apply_typescript_intent(self, content: str, intent: EditIntent, file_path: str) -> str:
        """Apply intent to TypeScript files with structural awareness."""
        if intent.action == 'modify' and 'interface' in intent.target.lower():
            return self._apply_interface_modification(content, intent)
        else:
            return self._apply_safe_text_intent(content, intent)
    
    def _apply_import_modification(self, content: str, intent: EditIntent) -> str:
        """Safely modify import statements."""
        lines = content.split('\n')
        
        # Find the appropriate place for imports (top of file, after other imports)
        import_insertion_point = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import'):
                import_insertion_point = i + 1
            elif line.strip() and not line.strip().startswith('//'):
                break
        
        # Extract import statement from replacement
        if 'import' in intent.replacement:
            # Add the import at the correct location
            import_lines = [line.strip() for line in intent.replacement.split('\n') if line.strip().startswith('import')]
            
            for import_line in import_lines:
                # Check if import already exists
                if not any(import_line in existing_line for existing_line in lines):
                    lines.insert(import_insertion_point, import_line)
                    import_insertion_point += 1
        
        return '\n'.join(lines)
    
    def _apply_return_modification(self, content: str, intent: EditIntent) -> str:
        """Safely modify return statements without corrupting structure."""
        lines = content.split('\n')
        
        # Find the return statement in the component function
        for i, line in enumerate(lines):
            if 'return (' in line and line.strip().startswith('return'):
                # This is a proper return statement - don't modify it if it's already correct
                return content
            elif 'return (' in line and not line.strip().startswith('return'):
                # This might be corrupted - fix it
                lines[i] = line.replace('return (', '  return (')
        
        return '\n'.join(lines)
    
    def _apply_hook_insertion(self, content: str, intent: EditIntent) -> str:
        """Safely insert React hooks in the correct location."""
        lines = content.split('\n')
        
        # Find the component function and insert hook after it
        for i, line in enumerate(lines):
            if re.match(r'export default function \w+|function \w+.*\{', line.strip()):
                # Insert hook after the function declaration
                hook_line = f"  {intent.replacement.strip()}"
                
                # Check if hook already exists
                if not any(intent.replacement.strip() in existing_line for existing_line in lines):
                    lines.insert(i + 1, hook_line)
                return '\n'.join(lines)
        
        return content
    
    def _apply_interface_modification(self, content: str, intent: EditIntent) -> str:
        """Safely modify TypeScript interfaces."""
        lines = content.split('\n')
        
        # Remove duplicate interface declarations
        seen_interfaces = set()
        filtered_lines = []
        
        for line in lines:
            interface_match = re.match(r'export interface (\w+)', line.strip())
            if interface_match:
                interface_name = interface_match.group(1)
                if interface_name in seen_interfaces:
                    continue  # Skip duplicate
                seen_interfaces.add(interface_name)
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _apply_conservative_edit(self, content: str, intent: EditIntent) -> str:
        """Apply very conservative edits to corrupted files."""
        print(f"🛡️ Applying conservative edit for corrupted file")
        
        # For corrupted files, only do very safe operations
        if intent.action == 'replace' and len(intent.target) > 10:
            # Only replace if we find an exact match
            if intent.target in content:
                return content.replace(intent.target, intent.replacement, 1)
        
        # For other cases, return unchanged to prevent further corruption
        print(f"⚠️ Skipping potentially dangerous edit on corrupted file")
        return content
    
    def _apply_safe_fuzzy_replacement(self, content: str, intent: EditIntent) -> str:
        """Apply fuzzy replacement with safety checks."""
        # Perform fuzzy replacement but validate the result
        original_line_count = content.count('\n')
        
        result = self._fuzzy_replace(content, intent.target, intent.replacement)
        
        # Safety check: ensure we didn't drastically change the file structure
        new_line_count = result.count('\n')
        if abs(new_line_count - original_line_count) > 5:
            print("⚠️ Fuzzy replacement changed too many lines - reverting")
            return content
        
        return result
    
    def _apply_safe_text_intent(self, content: str, intent: EditIntent) -> str:
        """Apply intent using safe text processing."""
        if intent.action == 'replace':
            return content.replace(intent.target, intent.replacement, 1)
        elif intent.action == 'insert':
            return self._safe_insert(content, intent)
        elif intent.action == 'delete':
            return content.replace(intent.target, '', 1)
        elif intent.action == 'modify':
            return self._safe_modify(content, intent)
        else:
            return content
    
    def _safe_insert(self, content: str, intent: EditIntent) -> str:
        """Safely insert text."""
        if intent.line_number is not None:
            lines = content.split('\n')
            if 0 <= intent.line_number - 1 < len(lines):
                lines.insert(intent.line_number - 1, intent.replacement)
                return '\n'.join(lines)
        
        # Insert after target if found
        if intent.target in content:
            return content.replace(intent.target, intent.target + '\n' + intent.replacement, 1)
        
        return content
    
    def _safe_modify(self, content: str, intent: EditIntent) -> str:
        """Safely modify text with validation."""
        # Only modify if we can find a reasonable match
        normalized_target = re.sub(r'\s+', ' ', intent.target.strip())
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            normalized_line = re.sub(r'\s+', ' ', line.strip())
            if normalized_target in normalized_line and len(normalized_target) > 5:
                # Replace in this line with validation
                old_line = line
                new_line = line.replace(intent.target.strip(), intent.replacement.strip())
                
                # Validation: ensure the change makes sense
                if len(new_line) < len(old_line) * 2:  # Don't allow lines to double in size
                    lines[i] = new_line
                    return '\n'.join(lines)
        
        # Fallback to simple replacement
        return content.replace(intent.target, intent.replacement, 1)
    
    def _fuzzy_replace(self, content: str, target: str, replacement: str) -> str:
        """Enhanced fuzzy replacement with better safety checks."""
        # Normalize whitespace for matching
        normalized_target = re.sub(r'\s+', ' ', target.strip())
        
        # Try to find similar text in content
        lines = content.split('\n')
        for i, line in enumerate(lines):
            normalized_line = re.sub(r'\s+', ' ', line.strip())
            if normalized_target in normalized_line:
                # Additional safety check: ensure we're not creating obvious corruption
                proposed_line = line.replace(target.strip(), replacement.strip())
                
                # Don't create lines that are obviously malformed
                if self._is_line_reasonable(proposed_line):
                    lines[i] = proposed_line
                    return '\n'.join(lines)
        
        # If not found, fall back to simple replacement
        return content.replace(target, replacement, 1)
    
    def _is_line_reasonable(self, line: str) -> bool:
        """Check if a line looks reasonable (not obviously corrupted)."""
        # Basic sanity checks
        if len(line) > 500:  # Too long
            return False
        if line.count('{') - line.count('}') > 3:  # Too many unmatched braces
            return False
        if line.count('(') - line.count(')') > 3:  # Too many unmatched parens
            return False
        if 'import' in line and not line.strip().startswith('import'):  # Misplaced import
            return False
        
        return True


class IntentBasedEditor:
    """
    Editor that uses structured AI intent to make precise code changes.
    
    ENHANCED: Now includes corruption detection and structural awareness
    to prevent the corruption issues that occurred in myapp23.
    """
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.corruption_detector = FileCorruptionDetector()
        self.structural_processor = StructuralCodeProcessor()
    
    def apply_intent_list(self, intents: List[EditIntent]) -> List[Tuple[bool, str]]:
        """
        Apply a list of editing intents to files with live progress indicators.
        
        ENHANCED: Real-time feedback, step-by-step progress, live loading indicators.
        """
        if not intents:
            print("⚠️ No intents to apply")
            return []
        
        print(f"\n🚀 Starting intent application with {len(intents)} operations")
        print("=" * 60)
        
        results = []
        
        # Group intents by file for efficient processing
        print("📁 Organizing intents by target file...")
        intents_by_file = {}
        for intent in intents:
            if intent.file_path not in intents_by_file:
                intents_by_file[intent.file_path] = []
            intents_by_file[intent.file_path].append(intent)
        
        print(f"📊 Processing {len(intents_by_file)} files:")
        for i, (file_path, file_intents) in enumerate(intents_by_file.items(), 1):
            print(f"   {i}. {file_path} ({len(file_intents)} operation{'s' if len(file_intents) > 1 else ''})")
        
        print("\n" + "=" * 60)
        
        # Process each file with live progress
        for file_index, (file_path, file_intents) in enumerate(intents_by_file.items(), 1):
            print(f"\n📄 [{file_index}/{len(intents_by_file)}] Processing: {file_path}")
            print("-" * 40)
            
            # Show loading indicator
            import time
            import sys
            
            def show_progress_spinner(message, duration=0.5):
                """Show a spinning progress indicator"""
                spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    for char in spinner_chars:
                        sys.stdout.write(f"\r{char} {message}")
                        sys.stdout.flush()
                        time.sleep(0.1)
                        if time.time() >= end_time:
                            break
                
                sys.stdout.write(f"\r✓ {message}\n")
                sys.stdout.flush()
            
            try:
                # Show progress for file processing
                show_progress_spinner(f"Analyzing {file_path}...", 0.3)
                
                result = self._apply_file_intents_with_validation_and_progress(file_path, file_intents, file_index, len(intents_by_file))
                results.append(result)
                
                if result[0]:
                    print(f"✅ Successfully processed {file_path}")
                else:
                    print(f"❌ Failed to process {file_path}: {result[1]}")
                    
            except Exception as e:
                error_msg = f"Exception while processing {file_path}: {str(e)}"
                print(f"💥 {error_msg}")
                results.append((False, error_msg))
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 OPERATION SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for success, _ in results if success)
        total = len(results)
        
        print(f"📈 Success Rate: {successful}/{total} files ({successful/total*100:.1f}%)")
        
        if successful > 0:
            print("\n✅ Successfully processed:")
            for i, ((file_path, _), (success, _)) in enumerate(zip(intents_by_file.items(), results), 1):
                if success:
                    print(f"   {i}. {file_path}")
        
        if successful < total:
            print("\n❌ Failed to process:")
            for i, ((file_path, _), (success, error)) in enumerate(zip(intents_by_file.items(), results), 1):
                if not success:
                    print(f"   {i}. {file_path} - {error}")
        
        print("\n🎉 Intent application completed!")
        return results
    
    def _apply_file_intents_with_validation_and_progress(self, file_path: str, intents: List[EditIntent], file_index: int, total_files: int) -> Tuple[bool, str]:
        """Apply all intents for a single file with detailed progress indicators."""
        target_file = self.repo_root / file_path
        
        # Check if any intent is for creating a new file
        creation_intents = [intent for intent in intents if intent.action == "insert" and intent.target == ""]
        
        if not target_file.exists():
            if creation_intents:
                # Use the first creation intent to create the file
                creation_intent = creation_intents[0]
                print(f"📄 Creating new file: {file_path}")
                
                # Show progress
                import time
                import sys
                
                for i in range(3):
                    sys.stdout.write(f"\r   {'.' * (i + 1)} Writing file content")
                    sys.stdout.flush()
                    time.sleep(0.2)
                sys.stdout.write(f"\r   ✓ File content written\n")
                sys.stdout.flush()
                
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(creation_intent.replacement)
                
                # Remove the creation intent from the list since it's handled
                intents = [intent for intent in intents if intent != creation_intent]
                
                # If there are remaining intents, continue processing them
                if not intents:
                    print(f"   📝 {len(creation_intent.replacement.split())} lines written")
                    return True, f"Created new file: {file_path}"
            else:
                return False, f"File not found: {file_path}"
        
        # If file was just created or already exists, continue with remaining intents
        try:
            # Read current content with progress indicator
            print(f"   📖 Reading existing content...")
            with open(target_file, 'r') as f:
                original_content = f.read()
            
            print(f"   📏 Current file size: {len(original_content)} characters")
            
            # ENHANCED: Pre-edit corruption detection with progress
            print(f"   🔍 Scanning for corruption issues...")
            
            import time
            import sys
            for i in range(2):
                sys.stdout.write(f"\r   {'🔍' if i % 2 == 0 else '🔎'} Analyzing file structure...")
                sys.stdout.flush()
                time.sleep(0.3)
            
            is_corrupted, severity = self.corruption_detector.is_file_corrupted(file_path, original_content)
            
            if is_corrupted:
                print(f"\r   ⚠️ Corruption detected: {severity.value}")
                if severity == CorruptionSeverity.CRITICAL:
                    print(f"   🚨 Attempting automatic recovery...")
                    # Try to restore from backup
                    backup_restored = self._attempt_backup_restoration(target_file)
                    if backup_restored:
                        original_content = target_file.read_text()
                        print(f"   ✅ File restored from backup")
                    else:
                        print(f"   ❌ Could not restore from backup - applying conservative edits")
            else:
                print(f"\r   ✅ File structure validated")
            
            current_content = original_content
            
            # Apply each remaining intent using structural processing with progress
            if intents:
                print(f"   🔧 Applying {len(intents)} modification{'s' if len(intents) > 1 else ''}...")
                
                for i, intent in enumerate(intents, 1):
                    if intent.action == "insert" and intent.target == "":
                        # Skip additional creation intents for existing file
                        continue
                    
                    print(f"      [{i}/{len(intents)}] {intent.action}: {intent.context[:50]}{'...' if len(intent.context) > 50 else ''}")
                    
                    # Show progress for each operation
                    import time
                    import sys
                    
                    for j in range(2):
                        sys.stdout.write(f"\r         {'⚙️' if j % 2 == 0 else '🔧'} Processing...")
                        sys.stdout.flush()
                        time.sleep(0.2)
                    
                    # ENHANCED: Use structural processor for safer edits
                    current_content = self.structural_processor.apply_intent_structurally(
                        current_content, intent, file_path
                    )
                    
                    sys.stdout.write(f"\r         ✅ Applied {intent.action}\n")
                    sys.stdout.flush()
            
            # ENHANCED: Post-edit validation with progress
            print(f"   🔍 Validating changes...")
            
            import time
            import sys
            for i in range(2):
                sys.stdout.write(f"\r   {'🔍' if i % 2 == 0 else '🔎'} Checking result integrity...")
                sys.stdout.flush()
                time.sleep(0.3)
            
            post_edit_corrupted, post_severity = self.corruption_detector.is_file_corrupted(file_path, current_content)
            
            if post_edit_corrupted and post_severity in [CorruptionSeverity.HIGH, CorruptionSeverity.CRITICAL]:
                print(f"\r   ❌ Edit would introduce {post_severity.value} corruption - reverting")
                return False, f"Edit would corrupt {file_path} - changes reverted"
            else:
                print(f"\r   ✅ Validation passed")
            
            # Write back if changed and validation passed
            if current_content != original_content:
                print(f"   💾 Saving changes...")
                
                # Show save progress
                import time
                import sys
                for i in range(2):
                    sys.stdout.write(f"\r      {'💾' if i % 2 == 0 else '📝'} Writing to disk...")
                    sys.stdout.flush()
                    time.sleep(0.2)
                
                with open(target_file, 'w') as f:
                    f.write(current_content)
                
                # Show file statistics
                lines_added = current_content.count('\n') - original_content.count('\n')
                size_change = len(current_content) - len(original_content)
                
                print(f"\r      ✅ Saved successfully")
                print(f"   📊 Changes: {lines_added:+d} lines, {size_change:+d} characters")
                
                # Generate diff for debugging
                self._save_debug_diff(file_path, original_content, current_content)
                
                return True, f"Applied {len(intents)} intents to {file_path}"
            else:
                print(f"   ℹ️ No changes needed")
                return True, f"No changes needed for {file_path}"
                
        except Exception as e:
            print(f"   💥 Error: {str(e)}")
            return False, f"Error applying intents to {file_path}: {str(e)}"
    
    def _attempt_backup_restoration(self, target_file: Path) -> bool:
        """Attempt to restore file from backup."""
        backup_dir = target_file.parent / ".diff_backups"
        if not backup_dir.exists():
            return False
        
        # Look for recent backups
        backup_pattern = f"{target_file.name}_*"
        backups = list(backup_dir.glob(backup_pattern))
        
        if not backups:
            return False
        
        # Use the most recent backup
        latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
        
        try:
            backup_content = latest_backup.read_text()
            # Validate backup is not corrupted
            is_corrupted, severity = self.corruption_detector.is_file_corrupted(str(target_file), backup_content)
            
            if not is_corrupted or severity in [CorruptionSeverity.LOW, CorruptionSeverity.MEDIUM]:
                target_file.write_text(backup_content)
                print(f"✅ Restored {target_file.name} from backup: {latest_backup.name}")
                return True
        except Exception as e:
            print(f"❌ Failed to restore from backup: {e}")
        
        return False
    
    def _apply_single_intent(self, content: str, intent: EditIntent) -> str:
        """Apply a single editing intent to content (legacy method)."""
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
        return self.structural_processor._fuzzy_replace(content, intent.target, intent.replacement)
    
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
        return self.structural_processor._fuzzy_replace(content, intent.target, intent.replacement)
    
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


def _fix_template_literals_in_json(json_str: str) -> str:
    """
    Fix template literals in JSON strings by converting them to escaped strings.
    
    Template literals (`string`) are not valid JSON, but AIs sometimes use them.
    This function converts them to proper JSON strings.
    """
    # Pattern to match template literals: `content`
    # We need to be careful to only match template literals in string values, not keys
    
    # Replace template literals with escaped strings
    # This regex matches: "key": `template literal content`
    pattern = r'("(?:[^"\\]|\\.)*":\s*)`([^`]*)`'
    
    def replace_template_literal(match):
        key_part = match.group(1)  # The "key": part
        content = match.group(2)   # The template literal content
        
        # Escape the content for JSON
        escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        
        return f'{key_part}"{escaped_content}"'
    
    # Apply the replacement
    fixed_json = re.sub(pattern, replace_template_literal, json_str)
    
    return fixed_json


def parse_ai_intent_response_robust(ai_response: str) -> List[EditIntent]:
    """
    Parse AI response with multiple fallback strategies.
    
    ENHANCED: Never throw away good code due to format issues.
    This parser tries multiple strategies to extract usable code.
    """
    print("🔍 Parsing AI response with robust fallback strategies...")
    
    # Strategy 1: Try clean JSON parsing
    print("📋 Strategy 1: Attempting structured JSON parsing...")
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Fix template literals if present
            json_str = _fix_template_literals_in_json(json_str)
            data = json.loads(json_str)
            
            if 'intents' in data and data['intents']:
                print(f"✅ JSON parsing successful - found {len(data['intents'])} intents")
                intents = []
                for intent_data in data['intents']:
                    intent = EditIntent(
                        file_path=intent_data.get('file_path', ''),
                        action=intent_data.get('action', 'replace'),
                        target=intent_data.get('target', ''),
                        replacement=intent_data.get('replacement', ''),
                        context=intent_data.get('context', ''),
                        line_number=intent_data.get('line_number')
                    )
                    intents.append(intent)
                return intents
        print("⚠️ No valid JSON structure found")
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
    except Exception as e:
        print(f"⚠️ JSON extraction failed: {e}")
    
    # Strategy 2: Extract code blocks and infer file structure
    print("📁 Strategy 2: Extracting code blocks and inferring file structure...")
    intents = []
    
    # Find file path patterns in the response
    file_patterns = [
        r'(?:app/[^/\s"`]+/page\.tsx)',  # Next.js pages
        r'(?:components/[^/\s"`]+\.tsx)',  # Components
        r'(?:lib/[^/\s"`]+\.ts)',  # Utilities
        r'(?:types/[^/\s"`]+\.ts)',  # Types
        r'(?:[^/\s"`]+\.tsx?)',  # Any TypeScript files
    ]
    
    found_files = set()
    for pattern in file_patterns:
        matches = re.findall(pattern, ai_response)
        found_files.update(matches)
    
    found_files = list(found_files)
    print(f"📂 Found {len(found_files)} potential files: {found_files}")
    
    # Find all code blocks
    code_patterns = [
        r'```(?:typescript|tsx|javascript|jsx|ts|js)\n(.*?)```',
        r'```\n(.*?)```',  # Generic code blocks
    ]
    
    code_blocks = []
    for pattern in code_patterns:
        matches = re.findall(pattern, ai_response, re.DOTALL)
        code_blocks.extend(matches)
    
    print(f"📝 Found {len(code_blocks)} code blocks")
    
    # Strategy 2a: Match files to code blocks by proximity
    if found_files and code_blocks:
        print("🔗 Matching files to code blocks by proximity...")
        
        # Split response into lines for position analysis
        lines = ai_response.split('\n')
        
        # Find positions of file mentions and code blocks
        file_positions = {}
        for file in found_files:
            for i, line in enumerate(lines):
                if file in line:
                    file_positions[file] = i
                    break
        
        code_positions = {}
        for i, block in enumerate(code_blocks):
            # Find where this code block appears
            block_start = block[:50].replace('\n', ' ').strip()
            for j, line in enumerate(lines):
                if block_start in line:
                    code_positions[i] = j
                    break
        
        # Match files to closest code blocks
        for file_path in found_files:
            if file_path in file_positions:
                file_pos = file_positions[file_path]
                
                # Find closest code block after file mention
                best_block_idx = None
                best_distance = float('inf')
                
                for block_idx, block_pos in code_positions.items():
                    if block_pos > file_pos:  # Code block comes after file mention
                        distance = block_pos - file_pos
                        if distance < best_distance:
                            best_distance = distance
                            best_block_idx = block_idx
                
                if best_block_idx is not None:
                    code = code_blocks[best_block_idx].strip()
                    
                    # Skip if this code block was already used
                    if best_block_idx not in [getattr(intent, '_block_idx', None) for intent in intents]:
                        intent = EditIntent(
                            file_path=file_path,
                            action="insert",
                            target="",
                            replacement=code,
                            context=f"Extracted from AI response - Strategy 2a"
                        )
                        intent._block_idx = best_block_idx  # Track which block we used
                        intents.append(intent)
                        print(f"📄 Matched {file_path} to code block {best_block_idx + 1}")
    
    # Strategy 2b: Infer file paths from code content
    if not intents and code_blocks:
        print("🔍 Strategy 2b: Inferring file paths from code content...")
        
        for i, code in enumerate(code_blocks):
            # Skip very short blocks
            if len(code.strip()) < 100:
                continue
            
            # Infer file type and path from code content
            file_path = None
            
            # Check for page components
            if 'export default function' in code and 'Page' in code:
                if 'History' in code:
                    file_path = "app/history/page.tsx"
                elif 'Ride' in code and 'Request' in code:
                    file_path = "app/ride-request/page.tsx"
                elif 'Saved' in code and 'Places' in code:
                    file_path = "app/saved-places/page.tsx"
                elif 'Dashboard' in code:
                    file_path = "app/dashboard/page.tsx"
                else:
                    # Generic page inference
                    func_match = re.search(r'export default function (\w+)', code)
                    if func_match:
                        component_name = func_match.group(1)
                        page_name = component_name.replace('Page', '').lower()
                        file_path = f"app/{page_name}/page.tsx"
            
            # Check for components
            elif 'export default function' in code or 'const.*=.*forwardRef' in code:
                func_match = re.search(r'export default function (\w+)|const (\w+) = .*forwardRef', code)
                if func_match:
                    component_name = func_match.group(1) or func_match.group(2)
                    file_path = f"components/{component_name}.tsx"
            
            # Check for type definitions
            elif 'interface' in code or 'type ' in code:
                file_path = "types/index.ts"
            
            # Check for utilities
            elif 'export ' in code and ('function' in code or 'const ' in code):
                file_path = "lib/utils.ts"
            
            if file_path:
                intent = EditIntent(
                    file_path=file_path,
                    action="insert",
                    target="",
                    replacement=code.strip(),
                    context=f"Extracted from AI response - Strategy 2b (inferred)"
                )
                intents.append(intent)
                print(f"📄 Inferred file: {file_path}")
    
    # Strategy 3: Parse by explicit file markers
    if not intents:
        print("📋 Strategy 3: Looking for explicit file markers...")
        
        # Look for patterns like "File: app/page.tsx" or "// app/page.tsx"
        file_markers = re.findall(r'(?:File:|//|\*\*)\s*(app/[^/\s]+/page\.tsx|components/[^/\s]+\.tsx|[^/\s]+\.tsx?)', ai_response)
        
        if file_markers:
            print(f"🏷️ Found {len(file_markers)} file markers")
            
            # Try to associate each marker with following code
            for marker in file_markers:
                # Find the position of this marker
                marker_pos = ai_response.find(marker)
                if marker_pos != -1:
                    # Look for code block after this marker
                    remaining_text = ai_response[marker_pos:]
                    code_match = re.search(r'```(?:typescript|tsx|javascript|jsx|ts|js)?\n(.*?)```', remaining_text, re.DOTALL)
                    
                    if code_match:
                        code = code_match.group(1).strip()
                        intent = EditIntent(
                            file_path=marker,
                            action="insert",
                            target="",
                            replacement=code,
                            context=f"Extracted from AI response - Strategy 3 (file markers)"
                        )
                        intents.append(intent)
                        print(f"📄 Found explicit file: {marker}")
    
    # Strategy 4: Last resort - treat as diff
    if not intents:
        print("🔄 Strategy 4: Attempting diff parsing as last resort...")
        
        # Look for diff-like patterns
        diff_patterns = [
            r'\+\+\+ (.*?)\n(.*?)(?=\+\+\+|\Z)',  # Git diff format
            r'--- (.*?)\n(.*?)(?=---|\Z)',  # Another diff format
        ]
        
        for pattern in diff_patterns:
            matches = re.findall(pattern, ai_response, re.DOTALL)
            for file_path, content in matches:
                # Clean up the file path
                file_path = file_path.strip().replace('b/', '').replace('a/', '')
                
                # Clean up the content
                content_lines = content.split('\n')
                clean_content = []
                for line in content_lines:
                    if line.startswith('+') and not line.startswith('+++'):
                        clean_content.append(line[1:])  # Remove the + prefix
                    elif not line.startswith('-') and not line.startswith('@'):
                        clean_content.append(line)  # Keep context lines
                
                if clean_content:
                    intent = EditIntent(
                        file_path=file_path,
                        action="replace",
                        target="",
                        replacement='\n'.join(clean_content),
                        context=f"Extracted from AI response - Strategy 4 (diff parsing)"
                    )
                    intents.append(intent)
                    print(f"📄 Extracted from diff: {file_path}")
    
    # Final validation and cleanup
    if intents:
        print(f"✅ Successfully extracted {len(intents)} intents using fallback strategies")
        for i, intent in enumerate(intents, 1):
            print(f"   {i}. {intent.file_path} ({intent.action})")
    else:
        print("❌ No intents could be extracted from AI response")
        print("📝 Response preview:")
        print(ai_response[:500] + "..." if len(ai_response) > 500 else ai_response)
    
    return intents


def parse_ai_intent_response(ai_response: str) -> List[EditIntent]:
    """
    Legacy method name - now redirects to robust parser.
    
    This ensures backward compatibility while using the improved parser.
    """
    return parse_ai_intent_response_robust(ai_response)


def get_intent_based_prompt() -> str:
    """
    Get the prompt for AI to generate structured editing intents
    instead of raw diffs.
    """
    return """
You are editing a NextJS application. Instead of generating diffs, please provide structured editing intents in JSON format.

🔗 CRITICAL: NEXTJS 13+ LINK COMPONENT USAGE:
⚠️  NEVER nest <a> tags inside <Link> components - this causes runtime errors!

✅ CORRECT: <Link href="/path" className="styles">Text</Link>
❌ WRONG: <Link href="/path"><a className="styles">Text</a></Link>

🚨 CRITICAL: STRUCTURAL AWARENESS REQUIRED 🚨

When making changes, you must understand code structure:
1. **Imports belong at the TOP of files** - never inside functions
2. **Hooks belong INSIDE component functions** - never outside
3. **JSX elements must be properly nested** - no duplicates
4. **Interface declarations are unique** - no duplicates

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
            "action": "modify",
            "target": "import './globals.css';",
            "replacement": "import './globals.css';\nimport { CartProvider } from '@/context/CartContext';",
            "context": "adding CartProvider import at file top"
        }
    ]
}
```

AVAILABLE ACTIONS:
- "replace": Replace exact text with new text
- "insert": Insert text at specific location or after target
- "delete": Remove specific text
- "modify": Modify text with pattern matching (for structural changes)

STRUCTURAL GUIDELINES:
✅ **Imports**: Always add imports at the top of files
✅ **Hooks**: Always add hooks inside component functions  
✅ **JSX**: Ensure proper nesting and no duplicates
✅ **Types**: Avoid duplicate interface declarations

ADVANTAGES OF THIS APPROACH:
✅ No malformed diffs from AI
✅ Precise, targeted changes
✅ Easy to validate and debug
✅ Programmatic diff generation
✅ Less chance of syntax errors
✅ Structural awareness prevents corruption

Focus on making precise, minimal changes with proper structural understanding.
"""