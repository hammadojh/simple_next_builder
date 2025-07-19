"""
Diff-based code builder for safe, context-aware code editing.
Based on the technical guide for implementing diff/patch systems.
"""

import re
import os
import difflib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, NamedTuple
from dataclasses import dataclass
import shutil


@dataclass
class DiffHunk:
    """Represents a single hunk in a unified diff."""
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    context_lines: List[str]
    removed_lines: List[str]
    added_lines: List[str]
    header: str = ""


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    file_path: str
    error_message: str = ""
    failed_hunks: List[DiffHunk] = None


class DiffSanitizer:
    """Sanitizes and fixes malformed AI-generated diffs."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
    
    def sanitize_patch(self, patch_text: str) -> str:
        """
        Clean up and fix malformed AI-generated diffs.
        
        Args:
            patch_text: Raw AI-generated patch
            
        Returns:
            Sanitized patch with proper formatting
        """
        print("🧹 Sanitizing AI-generated diff...")
        
        lines = patch_text.strip().split('\n')
        sanitized_lines = []
        current_file = None
        in_hunk = False
        hunk_header_line = None
        
        for i, line in enumerate(lines):
            # Preserve sentinels and file headers
            if line.startswith('*** Begin Patch') or line.startswith('*** End Patch'):
                sanitized_lines.append(line)
                continue
                
            if line.startswith('*** Update File:'):
                current_file = line.split(':', 1)[1].strip()
                sanitized_lines.append(line)
                in_hunk = False
                continue
            
            # Handle hunk headers
            if line.startswith('@@'):
                in_hunk = True
                hunk_header_line = i
                sanitized_lines.append(line)
                
                # Try to fix the hunk content that follows
                fixed_hunk_lines = self._fix_hunk_content(
                    lines[i+1:], current_file, line
                )
                sanitized_lines.extend(fixed_hunk_lines)
                
                # Skip the original hunk content since we've replaced it
                j = i + 1
                while j < len(lines) and not lines[j].startswith('@@') and not lines[j].startswith('***'):
                    j += 1
                i = j - 1  # Will be incremented by for loop
                in_hunk = False
                continue
            
            # If we're not in a hunk, just copy the line
            if not in_hunk:
                sanitized_lines.append(line)
        
        result = '\n'.join(sanitized_lines)
        print("✅ Diff sanitization complete")
        return result
    
    def _fix_hunk_content(self, hunk_lines: List[str], file_path: str, hunk_header: str) -> List[str]:
        """
        Fix malformed hunk content by adding proper +/- prefixes.
        
        This method reads the actual file and intelligently determines
        what should be removed vs added based on the hunk header and content.
        """
        print(f"🔧 Fixing hunk content for {file_path}")
        
        # Extract hunk info from header
        match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', hunk_header)
        if not match:
            print(f"⚠️ Invalid hunk header: {hunk_header}")
            return hunk_lines
        
        old_start, old_count, new_start, new_count = map(int, match.groups())
        
        # Read the actual file to understand what should be changed
        target_file = self.repo_root / file_path
        if not target_file.exists():
            print(f"⚠️ Target file doesn't exist: {file_path}")
            return hunk_lines
        
        try:
            with open(target_file, 'r') as f:
                file_lines = f.readlines()
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            return hunk_lines
        
        # Get the relevant section from the file
        old_section_start = old_start - 1  # Convert to 0-based
        old_section_end = old_section_start + old_count
        old_section = [line.rstrip('\n') for line in file_lines[old_section_start:old_section_end]]
        
        # Process hunk lines and determine what's what
        fixed_lines = []
        hunk_content = []
        
        # Stop at next hunk or end marker
        for line in hunk_lines:
            if line.startswith('@@') or line.startswith('***'):
                break
            hunk_content.append(line)
        
        # If lines already have proper prefixes, use them as-is
        has_prefixes = any(line.startswith(('+', '-', ' ')) for line in hunk_content if line.strip())
        
        if has_prefixes:
            print("✅ Hunk already has proper prefixes")
            return hunk_content
        
        # Lines don't have prefixes - we need to fix this
        print("🔧 Adding missing +/- prefixes to hunk")
        
        # Strategy: Compare hunk content with old file section
        # Lines that match old section = context or removed
        # Lines that don't match = added
        
        for line in hunk_content:
            line_content = line.strip()
            if not line_content:
                fixed_lines.append(' ')  # Empty context line
                continue
            
            # Check if this line exists in the old section
            found_in_old = False
            for old_line in old_section:
                if old_line.strip() == line_content:
                    found_in_old = True
                    break
            
            if found_in_old:
                # This line exists in old - could be context or removed
                # For now, treat as context unless we can be more specific
                fixed_lines.append(f' {line_content}')
            else:
                # This line doesn't exist in old - it's being added
                fixed_lines.append(f'+{line_content}')
        
        # If we have counts, try to be more precise about what's removed
        if old_count > 0 and len(fixed_lines) > 0:
            # We need to mark some lines as removed
            # Strategy: lines that appear in old section but not in new should be marked as removed
            for old_line in old_section:
                old_content = old_line.strip()
                if old_content and not any(line.endswith(old_content) for line in fixed_lines):
                    # This old line is not in the new content - it's being removed
                    fixed_lines.insert(0, f'-{old_content}')
        
        print(f"✅ Fixed {len(fixed_lines)} lines in hunk")
        return fixed_lines
    
    def validate_diff_structure(self, patch_text: str) -> Tuple[bool, List[str]]:
        """
        Validate the structure of a diff and return issues found.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if '*** Begin Patch' not in patch_text:
            issues.append("Missing '*** Begin Patch' sentinel")
        
        if '*** End Patch' not in patch_text:
            issues.append("Missing '*** End Patch' sentinel")
        
        # Check for file updates
        file_updates = re.findall(r'\*\*\* Update File: (.+)', patch_text)
        if not file_updates:
            issues.append("No file updates found")
        
        # Check each hunk
        hunks = re.findall(r'@@[^@]+@@', patch_text)
        for i, hunk in enumerate(hunks):
            if not re.match(r'@@ -\d+,\d+ \+\d+,\d+ @@', hunk):
                issues.append(f"Hunk {i+1} has invalid header format: {hunk}")
        
        # Enhanced validation for context issues
        context_issues = self._validate_diff_context(patch_text)
        issues.extend(context_issues)
        
        return len(issues) == 0, issues
    
    def _validate_diff_context(self, patch_text: str) -> List[str]:
        """Validate that diff context makes sense and doesn't duplicate existing content."""
        issues = []
        
        # Extract file updates and their content
        file_sections = re.split(r'\*\*\* Update File: (.+)', patch_text)
        
        for i in range(1, len(file_sections), 2):
            if i + 1 >= len(file_sections):
                continue
                
            file_path = file_sections[i].strip()
            diff_content = file_sections[i + 1]
            
            # Check if file exists to validate context
            target_file = self.repo_root / file_path
            if not target_file.exists():
                continue
                
            try:
                existing_content = target_file.read_text()
                file_issues = self._check_file_diff_context(file_path, diff_content, existing_content)
                issues.extend(file_issues)
            except Exception:
                continue  # Skip validation if file can't be read
        
        return issues
    
    def _check_file_diff_context(self, file_path: str, diff_content: str, existing_content: str) -> List[str]:
        """Check for common AI diff generation mistakes."""
        issues = []
        
        # Find all added lines (lines starting with +)
        added_lines = []
        for line in diff_content.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:].strip())  # Remove + prefix
        
        # Check if added lines already exist in the file
        existing_lines = [line.strip() for line in existing_content.split('\n')]
        
        duplicate_count = 0
        for added_line in added_lines:
            if added_line and added_line in existing_lines:
                duplicate_count += 1
        
        # If more than 50% of added lines already exist, flag as suspicious
        if added_lines and duplicate_count / len(added_lines) > 0.5:
            issues.append(f"Suspicious diff for {file_path}: {duplicate_count}/{len(added_lines)} added lines already exist in file")
        
        # Check for common patterns that indicate AI confusion
        if any("className=" in line for line in added_lines):
            # Check if we're trying to add existing JSX structure
            jsx_elements = [line for line in added_lines if "<div" in line or "<button" in line or "<span" in line]
            existing_jsx = [line for line in existing_lines if "<div" in line or "<button" in line or "<span" in line]
            
            if jsx_elements and len(jsx_elements) > 1:
                # Check for similarity with existing JSX
                similar_count = 0
                for jsx_line in jsx_elements:
                    for existing_jsx_line in existing_jsx:
                        # Simple similarity check
                        if len(set(jsx_line.split()) & set(existing_jsx_line.split())) > 3:
                            similar_count += 1
                            break
                
                if similar_count > len(jsx_elements) * 0.5:
                    issues.append(f"Suspicious JSX duplication in {file_path}: AI may be re-adding existing elements")
        
        return issues


class DiffParser:
    """Parse unified diff format with OpenAI-style sentinels."""
    
    def __init__(self):
        self.max_patch_lines = 1000  # Security limit
        
    def parse_patch(self, patch_text: str) -> List[DiffHunk]:
        """
        Parse a unified diff with OpenAI-style sentinels.
        
        Expected format:
        *** Begin Patch
        *** Update File: src/app.ts
        @@ -65,3 +65,4 @@ function example() {
        - old line
        + new line
         context line
        *** End Patch
        """
        if not self._validate_patch(patch_text):
            raise ValueError("Invalid patch format or security check failed")
            
        lines = patch_text.strip().split('\n')
        hunks = []
        current_file = None
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for file header
            if line.startswith('*** Update File:'):
                current_file = line.split(':', 1)[1].strip()
                i += 1
                continue
                
            # Look for hunk header @@ -old_start,old_count +new_start,new_count @@
            if line.startswith('@@'):
                if not current_file:
                    raise ValueError("Hunk found without file header")
                    
                hunk_match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)$', line)
                if not hunk_match:
                    raise ValueError(f"Invalid hunk header: {line}")
                    
                old_start, old_count, new_start, new_count = map(int, hunk_match.groups()[:4])
                header = hunk_match.group(5).strip()
                
                # Parse hunk content
                hunk, next_i = self._parse_hunk_content(lines, i + 1, old_count, new_count)
                hunk.file_path = current_file
                hunk.old_start = old_start
                hunk.old_count = old_count
                hunk.new_start = new_start
                hunk.new_count = new_count
                hunk.header = header
                
                hunks.append(hunk)
                i = next_i
                continue
                
            i += 1
            
        return hunks
    
    def _validate_patch(self, patch_text: str) -> bool:
        """Validate patch format and security."""
        # Check for required sentinels
        if not ('*** Begin Patch' in patch_text and '*** End Patch' in patch_text):
            return False
            
        # Check size limit
        if len(patch_text.split('\n')) > self.max_patch_lines:
            return False
            
        # Check for binary content (NUL bytes)
        if '\x00' in patch_text:
            return False
            
        return True
    
    def _parse_hunk_content(self, lines: List[str], start_idx: int, old_count: int, new_count: int) -> Tuple[DiffHunk, int]:
        """Parse the content of a single hunk."""
        hunk = DiffHunk("", 0, 0, 0, 0, [], [], [])
        i = start_idx
        
        while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('***'):
            line = lines[i]
            
            if line.startswith('-'):
                hunk.removed_lines.append(line[1:])
            elif line.startswith('+'):
                hunk.added_lines.append(line[1:])
            elif line.startswith(' '):
                hunk.context_lines.append(line[1:])
            elif line.strip() == '':
                # Empty line - treat as context
                hunk.context_lines.append('')
            else:
                # No prefix - treat as context (some diff formats)
                hunk.context_lines.append(line)
                
            i += 1
            
        return hunk, i


class FuzzyMatcher:
    """Fuzzy context matching for robust patch application."""
    
    def __init__(self, max_offset: int = 5, max_levenshtein_dist: int = 2):
        self.max_offset = max_offset
        self.max_levenshtein_dist = max_levenshtein_dist
    
    def find_context_match(self, file_lines: List[str], context_lines: List[str], expected_line: int) -> Optional[int]:
        """
        Find the best match for context lines in the file.
        
        Args:
            file_lines: Lines of the target file
            context_lines: Context lines from the diff
            expected_line: Expected line number (0-based)
            
        Returns:
            Line number where context matches, or None if no match found
        """
        print(f"🔍 DEBUG: Fuzzy matching - expected_line={expected_line}, context_lines={len(context_lines)}")
        print(f"🔍 DEBUG: Context to match: {context_lines[:3] if context_lines else 'None'}")
        
        # Special case: New file creation (empty file)
        if not file_lines and expected_line == 0:
            print(f"🔍 DEBUG: New file creation - using line 0")
            return 0
        
        if expected_line < len(file_lines):
            print(f"🔍 DEBUG: File content around expected line {expected_line}:")
            start_preview = max(0, expected_line - 2)
            end_preview = min(len(file_lines), expected_line + 3)
            for i in range(start_preview, end_preview):
                marker = ">>> " if i == expected_line else "    "
                print(f"🔍 DEBUG: {marker}{i}: {repr(file_lines[i][:50])}")
        
        if not context_lines:
            # No context to match against - use expected line if valid or end of file
            if 0 <= expected_line <= len(file_lines):
                print(f"🔍 DEBUG: No context lines, using expected line {expected_line}")
                return expected_line
            print(f"🔍 DEBUG: No context lines and invalid expected line {expected_line}, using end of file")
            return len(file_lines)
            
        # First try sliding window search around expected location
        print(f"🔍 DEBUG: Trying sliding window search")
        match = self._sliding_window_search(file_lines, context_lines, expected_line)
        if match is not None:
            print(f"🔍 DEBUG: Sliding window found match at line {match}")
            return match
            
        # Fall back to Levenshtein distance search
        print(f"🔍 DEBUG: Sliding window failed, trying Levenshtein search")
        result = self._levenshtein_search(file_lines, context_lines)
        if result is not None:
            print(f"🔍 DEBUG: Levenshtein found match at line {result}")
        else:
            print(f"🔍 DEBUG: No match found with any method")
        return result
    
    def _sliding_window_search(self, file_lines: List[str], context_lines: List[str], expected_line: int) -> Optional[int]:
        """Search for exact match within max_offset of expected position."""
        start_search = max(0, expected_line - self.max_offset)
        end_search = min(len(file_lines) - len(context_lines) + 1, expected_line + self.max_offset)
        
        print(f"🔍 DEBUG: Sliding window search from line {start_search} to {end_search}")
        print(f"🔍 DEBUG: Looking for context: {[line.rstrip() for line in context_lines[:3]]}")
        
        for pos in range(start_search, end_search):
            file_slice = file_lines[pos:pos + len(context_lines)]
            print(f"🔍 DEBUG: Trying position {pos}, comparing:")
            print(f"🔍 DEBUG:   Context: {[line.rstrip() for line in context_lines[:2]]}")
            print(f"🔍 DEBUG:   File:    {[line.rstrip() for line in file_slice[:2]]}")
            
            if self._lines_match_exactly(file_slice, context_lines):
                print(f"🔍 DEBUG: EXACT MATCH found at position {pos}")
                return pos
            else:
                print(f"🔍 DEBUG: No match at position {pos}")
                
        print(f"🔍 DEBUG: No exact matches found in sliding window")
        return None
    
    def _levenshtein_search(self, file_lines: List[str], context_lines: List[str]) -> Optional[int]:
        """Search for best match using Levenshtein distance."""
        best_match = None
        best_distance = float('inf')
        
        for pos in range(len(file_lines) - len(context_lines) + 1):
            file_slice = file_lines[pos:pos + len(context_lines)]
            distance = self._calculate_levenshtein_distance(file_slice, context_lines)
            
            if distance <= self.max_levenshtein_dist and distance < best_distance:
                best_match = pos
                best_distance = distance
                
        return best_match
    
    def _lines_match_exactly(self, file_slice: List[str], context_lines: List[str]) -> bool:
        """Check if lines match exactly (ignoring trailing whitespace)."""
        if len(file_slice) != len(context_lines):
            return False
            
        for f_line, c_line in zip(file_slice, context_lines):
            if f_line.rstrip() != c_line.rstrip():
                return False
                
        return True
    
    def _calculate_levenshtein_distance(self, lines1: List[str], lines2: List[str]) -> int:
        """Calculate Levenshtein distance between two sequences of lines."""
        if len(lines1) != len(lines2):
            return float('inf')
            
        total_distance = 0
        for l1, l2 in zip(lines1, lines2):
            # Simple character-level distance for each line
            total_distance += self._string_levenshtein(l1.strip(), l2.strip())
            
        return total_distance
    
    def _string_levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._string_levenshtein(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]


class DiffApplier:
    """Apply diffs with transactional safety and error handling."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.fuzzy_matcher = FuzzyMatcher()
        self.backup_dir = self.repo_root / ".diff_backups"
        
    def apply_patch(self, patch_text: str) -> List[PatchResult]:
        """
        Apply a patch with transactional safety.
        
        Returns list of results for each file modified.
        """
        parser = DiffParser()
        
        try:
            hunks = parser.parse_patch(patch_text)
        except ValueError as e:
            return [PatchResult(False, "", f"Parse error: {e}")]
            
        # Group hunks by file
        hunks_by_file = {}
        for hunk in hunks:
            if hunk.file_path not in hunks_by_file:
                hunks_by_file[hunk.file_path] = []
            hunks_by_file[hunk.file_path].append(hunk)
            
        results = []
        
        # Process each file
        for file_path, file_hunks in hunks_by_file.items():
            result = self._apply_file_hunks(file_path, file_hunks)
            results.append(result)
            
        return results
    
    def _apply_file_hunks(self, file_path: str, hunks: List[DiffHunk]) -> PatchResult:
        """Apply all hunks for a single file transactionally."""
        # Resolve and validate file path
        target_file = self._resolve_file_path(file_path)
        if not target_file:
            return PatchResult(False, file_path, f"Invalid file path: {file_path}")
            
        # Read current file content
        try:
            if target_file.exists():
                current_content = target_file.read_text()
                current_lines = current_content.splitlines(keepends=True)
            else:
                current_lines = []
                # For new files, ensure parent directory exists
                target_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return PatchResult(False, file_path, f"Failed to read file: {e}")
            
        # Create backup
        backup_path = self._create_backup(target_file)
        
        try:
            # Apply hunks in ascending order
            sorted_hunks = sorted(hunks, key=lambda h: h.old_start)
            modified_lines = current_lines.copy()
            offset = 0  # Track line number offset from previous edits
            
            for hunk in sorted_hunks:
                success, new_offset = self._apply_single_hunk(modified_lines, hunk, offset)
                if not success:
                    # Rollback and create .rej file
                    self._rollback_from_backup(target_file, backup_path)
                    self._create_reject_file(target_file, [hunk])
                    return PatchResult(False, file_path, f"Failed to apply hunk at line {hunk.old_start}")
                offset = new_offset

            # CRITICAL: Validate syntax before writing modified content
            modified_content = ''.join(modified_lines)
            if file_path.endswith(('.tsx', '.ts', '.js', '.jsx')):
                syntax_valid, syntax_error = self._validate_javascript_syntax(modified_content, file_path)
                if not syntax_valid:
                    print(f"❌ Syntax validation failed: {syntax_error}")
                    self._rollback_from_backup(target_file, backup_path)
                    return PatchResult(False, file_path, f"Generated code has syntax errors: {syntax_error}")
                else:
                    print(f"✅ Syntax validation passed for {file_path}")
                
            # Write modified content
            target_file.write_text(modified_content)
            
            # Clean up backup on success
            if backup_path and backup_path.exists():
                backup_path.unlink()
                
            return PatchResult(True, file_path)
            
        except Exception as e:
            # Rollback on any error
            self._rollback_from_backup(target_file, backup_path)
            return PatchResult(False, file_path, f"Unexpected error: {e}")
    
    def _apply_single_hunk(self, file_lines: List[str], hunk: DiffHunk, offset: int) -> Tuple[bool, int]:
        """
        Apply a single hunk to the file lines.
        
        Returns (success, new_offset) where new_offset is the change in line count.
        """
        # Adjust expected position by offset from previous edits
        expected_pos = max(0, hunk.old_start - 1 + offset)  # Convert to 0-based
        
        # Build context for matching (removed lines + surrounding context)
        context_for_match = hunk.removed_lines if hunk.removed_lines else hunk.context_lines
        
        # Find actual position using fuzzy matching
        actual_pos = self.fuzzy_matcher.find_context_match(file_lines, context_for_match, expected_pos)
        
        if actual_pos is None:
            return False, offset
            
        # Apply the change
        if hunk.removed_lines and hunk.added_lines:
            # Replace lines
            del file_lines[actual_pos:actual_pos + len(hunk.removed_lines)]
            for i, new_line in enumerate(hunk.added_lines):
                file_lines.insert(actual_pos + i, new_line if new_line.endswith('\n') else new_line + '\n')
            new_offset = offset + len(hunk.added_lines) - len(hunk.removed_lines)
        elif hunk.removed_lines:
            # Delete lines
            del file_lines[actual_pos:actual_pos + len(hunk.removed_lines)]
            new_offset = offset - len(hunk.removed_lines)
        elif hunk.added_lines:
            # Insert lines
            for i, new_line in enumerate(hunk.added_lines):
                file_lines.insert(actual_pos + i, new_line if new_line.endswith('\n') else new_line + '\n')
            new_offset = offset + len(hunk.added_lines)
        else:
            # No change (context only)
            new_offset = offset
            
        return True, new_offset
    
    def _resolve_file_path(self, file_path: str) -> Optional[Path]:
        """Resolve file path safely within repo root."""
        try:
            # Remove any path prefixes (a/, b/, etc.)
            clean_path = file_path
            for prefix in ['a/', 'b/', 'src/', './']:
                if clean_path.startswith(prefix):
                    clean_path = clean_path[len(prefix):]
                    
            resolved = (self.repo_root / clean_path).resolve()
            
            # Security check: ensure path is within repo root
            if not str(resolved).startswith(str(self.repo_root.resolve())):
                return None
                
            return resolved
        except Exception:
            return None
    
    def _create_backup(self, target_file: Path) -> Optional[Path]:
        """Create backup of file before modification."""
        if not target_file.exists():
            return None
            
        try:
            self.backup_dir.mkdir(exist_ok=True)
            backup_path = self.backup_dir / f"{target_file.name}.backup"
            shutil.copy2(target_file, backup_path)
            return backup_path
        except Exception:
            return None
    
    def _rollback_from_backup(self, target_file: Path, backup_path: Optional[Path]):
        """Restore file from backup."""
        if backup_path and backup_path.exists():
            try:
                shutil.copy2(backup_path, target_file)
                backup_path.unlink()
            except Exception:
                pass
    
    def _create_reject_file(self, target_file: Path, failed_hunks: List[DiffHunk]):
        """Create .rej file with failed hunks."""
        try:
            rej_path = target_file.with_suffix(target_file.suffix + '.rej')
            rej_content = "*** REJECTED HUNKS ***\n\n"
            
            for hunk in failed_hunks:
                rej_content += f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@\n"
                for line in hunk.removed_lines:
                    rej_content += f"-{line}\n"
                for line in hunk.added_lines:
                    rej_content += f"+{line}\n"
                rej_content += "\n"
                
            rej_path.write_text(rej_content)
        except Exception:
            pass

    def _validate_javascript_syntax(self, content: str, file_path: str) -> Tuple[bool, str]:
        """
        Basic syntax validation for JavaScript/TypeScript files.
        
        Args:
            content: File content to validate
            file_path: Path for context in error messages
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            lines = content.split('\n')
            
            # Check for balanced braces, brackets, and parentheses
            brace_count = 0
            bracket_count = 0
            paren_count = 0
            in_string = False
            string_char = None
            
            for line_num, line in enumerate(lines, 1):
                i = 0
                while i < len(line):
                    char = line[i]
                    
                    # Handle string literals (skip escape sequences)
                    if char in ['"', "'", '`'] and not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char and in_string:
                        # Check if escaped
                        if i == 0 or line[i-1] != '\\':
                            in_string = False
                            string_char = None
                    
                    # Skip counting inside strings
                    if in_string:
                        i += 1
                        continue
                        
                    # Count brackets/braces/parens
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    elif char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                    elif char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        
                    i += 1
                    
                # Check for malformed template literals
                if '} ${' in line:
                    return False, f"Line {line_num}: Malformed template literal - invalid '}} ${{' pattern"
                    
                # Check for incomplete ternary operators
                stripped = line.strip()
                if stripped.endswith(':') and ('?' in line or 'else' in line):
                    next_line = lines[line_num] if line_num < len(lines) else ""
                    if next_line.strip() == '}' or next_line.strip().startswith('}'):
                        return False, f"Line {line_num}: Incomplete ternary or conditional expression"
                        
                # Check for missing else/else if patterns
                if stripped.startswith('if (') and line_num < len(lines):
                    # Look ahead for patterns like: } else { ... } if (
                    for check_line in range(line_num, min(line_num + 5, len(lines))):
                        check_content = lines[check_line].strip()
                        if check_content == '} else {' and check_line + 2 < len(lines):
                            follow_up = lines[check_line + 2].strip()
                            if follow_up.startswith('if (') and not follow_up.startswith('if ('):
                                return False, f"Line {check_line + 3}: Suspicious control flow - should probably be 'else if'"
            
            # Final balance check
            if brace_count != 0:
                return False, f"Unbalanced braces: {brace_count} extra {'opening' if brace_count > 0 else 'closing'}"
            if bracket_count != 0:
                return False, f"Unbalanced brackets: {bracket_count} extra {'opening' if bracket_count > 0 else 'closing'}"
            if paren_count != 0:
                return False, f"Unbalanced parentheses: {paren_count} extra {'opening' if paren_count > 0 else 'closing'}"
                
            return True, "Syntax validation passed"
            
        except Exception as e:
            return False, f"Syntax validation error: {str(e)}"


class DiffBuilder:
    """Main interface for diff-based code building."""
    
    def __init__(self, patch_file: str, app_dir: str):
        self.patch_file = patch_file
        self.app_dir = Path(app_dir)
        self.applier = DiffApplier(self.app_dir)
        self.sanitizer = DiffSanitizer(self.app_dir)
        self.max_retries = 2
        
    def build(self) -> bool:
        """
        Apply diff patch with retry logic.
        
        Returns True if successful, False otherwise.
        """
        if not os.path.exists(self.patch_file):
            print(f"❌ Patch file not found: {self.patch_file}")
            return False
            
        try:
            with open(self.patch_file, 'r') as f:
                raw_patch_content = f.read()
        except Exception as e:
            print(f"❌ Failed to read patch file: {e}")
            return False
        
        # Validate and sanitize the patch first
        is_valid, issues = self.sanitizer.validate_diff_structure(raw_patch_content)
        if not is_valid:
            print("⚠️ Diff validation issues found:")
            for issue in issues:
                print(f"   - {issue}")
            print("🧹 Attempting to sanitize diff...")
        
        # Sanitize the patch to fix common AI mistakes
        patch_content = self.sanitizer.sanitize_patch(raw_patch_content)
        
        # Save sanitized patch for debugging
        sanitized_file = self.patch_file.replace('.patch', '_sanitized.patch')
        try:
            with open(sanitized_file, 'w') as f:
                f.write(patch_content)
            print(f"💾 Sanitized patch saved to: {sanitized_file}")
        except Exception:
            pass  # Not critical if we can't save debug file
            
        # Apply patch with retries
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"🔄 Retry attempt {attempt}/{self.max_retries}")
                
            results = self.applier.apply_patch(patch_content)
            
            # Check results
            all_success = True
            for result in results:
                if result.success:
                    print(f"✅ Applied diff to {result.file_path}")
                else:
                    print(f"❌ Failed to apply diff to {result.file_path}: {result.error_message}")
                    all_success = False
                    
            if all_success:
                print("🎉 All diffs applied successfully!")
                return True
                
            if attempt < self.max_retries:
                print("⚠️ Some diffs failed, retrying...")
            else:
                print("💥 Max retries exceeded, some diffs failed")
                return False
                
        return False


def generate_unified_diff(old_text: str, new_text: str, file_path: str) -> str:
    """
    Generate a unified diff between old and new text.
    
    Args:
        old_text: Original file content
        new_text: Modified file content  
        file_path: Path to the file being modified
        
    Returns:
        Unified diff string with OpenAI-style sentinels
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=''
    ))
    
    if not diff_lines:
        return ""  # No changes
        
    # Wrap with OpenAI-style sentinels
    diff_content = '\n'.join(diff_lines)
    
    return f"""*** Begin Patch
*** Update File: {file_path}
{diff_content}
*** End Patch""" 