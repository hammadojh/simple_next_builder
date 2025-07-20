"""
Enhanced Diff-based Code Builder

This module provides robust diff application with:
1. Atomic transactions for multi-file operations
2. Comprehensive validation and error handling  
3. Advanced fuzzy matching for context resolution
4. Built-in corruption detection and recovery
5. Rollback capabilities for failed operations

ENHANCED: Now includes corruption detection integration to prevent
applying diffs to corrupted files and causing further damage.
"""

import os
import re
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import corruption detection system
from .corruption_detector import FileCorruptionDetector, CorruptionSeverity


@dataclass
class DiffHunk:
    """Represents a single hunk in a diff."""
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    context_lines: List[str]
    removed_lines: List[str]
    added_lines: List[str]


@dataclass
class PatchResult:
    """Result of applying a patch to a file."""
    success: bool
    file_path: str
    error_message: str = ""


class DiffParser:
    """Parse unified diff format with enhanced error handling."""
    
    def parse_patch(self, patch_text: str) -> List[DiffHunk]:
        """
        Parse a unified diff patch into hunks.
        
        ENHANCED: Better error handling and validation.
        """
        hunks = []
        current_file = None
        
        lines = patch_text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for file headers
            if line.startswith('*** Update File:'):
                current_file = line.replace('*** Update File:', '').strip()
                i += 1
                continue
            elif line.startswith('--- ') or line.startswith('+++ '):
                # Standard diff file headers
                if line.startswith('--- a/'):
                    current_file = line[6:]
                elif line.startswith('--- '):
                    current_file = line[4:]
                i += 1
                continue
            elif line.startswith('@@'):
                # Hunk header
                if current_file:
                    hunk, lines_processed = self._parse_hunk(lines[i:], current_file)
                    if hunk:
                        hunks.append(hunk)
                    i += lines_processed
                else:
                    i += 1
            else:
                i += 1
                
        return hunks
    
    def _parse_hunk(self, lines: List[str], file_path: str) -> Tuple[Optional[DiffHunk], int]:
        """Parse a single hunk from diff lines."""
        if not lines or not lines[0].startswith('@@'):
            return None, 1
        
        # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
        header = lines[0]
        match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', header)
        if not match:
            return None, 1
        
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1
        
        # Parse hunk content
        context_lines = []
        removed_lines = []
        added_lines = []
        
        i = 1
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('@@') or line.startswith('***'):
                # Next hunk or section
                break
            elif line.startswith('-'):
                removed_lines.append(line[1:])
            elif line.startswith('+'):
                added_lines.append(line[1:])
            elif line.startswith(' '):
                context_lines.append(line[1:])
            elif line.strip() == '':
                # Empty line might be context
                context_lines.append('')
            
            i += 1
        
        hunk = DiffHunk(
            file_path=file_path,
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            context_lines=context_lines,
            removed_lines=removed_lines,
            added_lines=added_lines
        )
        
        return hunk, i


class FuzzyMatcher:
    """Fuzzy context matching for robust patch application."""
    
    def __init__(self, max_offset: int = 5, max_levenshtein_dist: int = 2):
        self.max_offset = max_offset
        self.max_levenshtein_dist = max_levenshtein_dist
    
    def find_context_match(self, file_lines: List[str], context_lines: List[str], expected_line: int) -> Optional[int]:
        """
        Find the best match for context lines in the file.
        
        ENHANCED: Better handling of corrupted files and improved matching logic.
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


class CorruptionAwareDiffApplier:
    """
    Apply diffs with corruption detection and recovery capabilities.
    
    ENHANCED: Integrates corruption detection to prevent applying diffs
    to corrupted files and causing further damage.
    """
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.fuzzy_matcher = FuzzyMatcher()
        self.backup_dir = self.repo_root / ".diff_backups"
        self.corruption_detector = FileCorruptionDetector()
        self.transaction_id = None
        self.transaction_backups = {}  # file_path -> backup_path
        
    def apply_patch(self, patch_text: str) -> List[PatchResult]:
        """
        Apply a patch with corruption detection and recovery.
        
        ENHANCED: Now includes pre-patch corruption detection and
        automatic recovery for corrupted files.
        NEW: Real-time context verification to prevent staleness mismatches.
        """
        print("🔄 Starting corruption-aware patch application...")
        
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
        
        print(f"📁 Transaction will modify {len(hunks_by_file)} files")
        
        # NEW: Real-time context verification
        print("🔍 Phase 0a: Real-time context verification...")
        context_verification = self._verify_context_freshness(hunks_by_file)
        if not context_verification["all_fresh"]:
            stale_files = context_verification["stale_files"]
            print(f"⚠️ Context staleness detected in {len(stale_files)} files:")
            for file_path, mismatch_info in stale_files.items():
                print(f"   📄 {file_path}: {mismatch_info}")
            
            # Ask if we should continue or regenerate
            print("🤔 Options: (1) Continue with fuzzy matching (2) Request context refresh")
            print("   Proceeding with enhanced fuzzy matching...")
        
        # ENHANCED: Pre-patch corruption detection
        print("🔍 Phase 0b: Pre-patch corruption detection...")
        corruption_issues = self._detect_target_file_corruption(hunks_by_file.keys())
        
        if corruption_issues:
            print(f"⚠️ Corruption detected in {len(corruption_issues)} files")
            for file_path, severity in corruption_issues.items():
                if severity in [CorruptionSeverity.HIGH, CorruptionSeverity.CRITICAL]:
                    print(f"🚨 {file_path}: {severity.value} corruption - attempting restoration")
                    restored = self._attempt_corruption_recovery(file_path)
                    if not restored:
                        print(f"❌ Could not restore {file_path} - patch may fail")
        
        # Start transaction
        transaction_id = self._start_transaction()
        
        try:
            # Phase 1: Validate ALL changes before applying ANY
            print("🔍 Phase 1: Pre-validating all changes...")
            validation_results = self._validate_all_changes(hunks_by_file)
            
            failed_validations = [r for r in validation_results if not r.success]
            if failed_validations:
                print(f"❌ Pre-validation failed for {len(failed_validations)} files")
                for result in failed_validations:
                    print(f"   - {result.file_path}: {result.error_message}")
                self._abort_transaction(transaction_id)
                return validation_results
            
            print("✅ All changes pre-validated successfully")
            
            # Phase 2: Create backups for ALL files
            print("📸 Phase 2: Creating atomic backups...")
            backup_results = self._create_transaction_backups(hunks_by_file.keys())
            
            failed_backups = [r for r in backup_results if not r.success]
            if failed_backups:
                print(f"❌ Backup creation failed for {len(failed_backups)} files")
                self._abort_transaction(transaction_id)
                return backup_results
            
            print("✅ All backups created successfully")
            
            # Phase 3: Apply ALL changes
            print("⚡ Phase 3: Applying all changes atomically...")
            results = []
            
            for file_path, file_hunks in hunks_by_file.items():
                result = self._apply_file_hunks_transactional(file_path, file_hunks, transaction_id)
                results.append(result)
                
                if not result.success:
                    print(f"❌ Atomic transaction failed at {file_path}: {result.error_message}")
                    print("🔄 Rolling back all changes...")
                    self._abort_transaction(transaction_id)
                    return results
            
            # Phase 4: Commit transaction
            print("✅ Phase 4: Committing transaction...")
            self._commit_transaction(transaction_id)
            
            print(f"🎉 Successfully applied patch to {len(hunks_by_file)} files")
            return results
            
        except Exception as e:
            print(f"💥 Unexpected error during patch application: {e}")
            self._abort_transaction(transaction_id)
            return [PatchResult(False, "", f"Transaction error: {e}")]
    
    def _detect_target_file_corruption(self, file_paths: List[str]) -> Dict[str, CorruptionSeverity]:
        """Detect corruption in files that will be modified."""
        corruption_issues = {}
        
        for file_path in file_paths:
            target_file = self._resolve_file_path(file_path)
            if target_file and target_file.exists():
                try:
                    content = target_file.read_text()
                    is_corrupted, severity = self.corruption_detector.is_file_corrupted(file_path, content)
                    
                    if is_corrupted:
                        corruption_issues[file_path] = severity
                        print(f"🚨 Corruption detected in {file_path}: {severity.value}")
                        
                        # Print detailed corruption report
                        issues = self.corruption_detector.detect_corruption(file_path, content)
                        for issue in issues[:3]:  # Show first 3 issues
                            print(f"   - Line {issue.line_number}: {issue.description}")
                            
                except Exception as e:
                    print(f"⚠️ Could not check corruption in {file_path}: {e}")
        
        return corruption_issues
    
    def _attempt_corruption_recovery(self, file_path: str) -> bool:
        """Attempt to recover a corrupted file."""
        target_file = self._resolve_file_path(file_path)
        if not target_file:
            return False
        
        # Try to restore from backup
        backup_restored = self._restore_from_backup(target_file)
        if backup_restored:
            print(f"✅ Restored {file_path} from backup")
            return True
        
        # Try basic corruption fixes
        try:
            content = target_file.read_text()
            fixed_content = self._apply_basic_corruption_fixes(content, file_path)
            
            if fixed_content != content:
                # Validate the fix
                is_still_corrupted, severity = self.corruption_detector.is_file_corrupted(file_path, fixed_content)
                
                if not is_still_corrupted or severity in [CorruptionSeverity.LOW, CorruptionSeverity.MEDIUM]:
                    target_file.write_text(fixed_content)
                    print(f"✅ Applied basic corruption fixes to {file_path}")
                    return True
        except Exception as e:
            print(f"❌ Failed to apply corruption fixes to {file_path}: {e}")
        
        return False
    
    def _apply_basic_corruption_fixes(self, content: str, file_path: str) -> str:
        """Apply basic fixes to common corruption patterns."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix misplaced imports
            if 'import ' in line and not line.strip().startswith('import'):
                # Move import to be properly indented
                import_part = re.search(r'import\s+.*', line)
                if import_part:
                    fixed_lines.append(import_part.group(0))
                    continue
            
            # Fix malformed return statements
            if 'return (' in line and not line.strip().startswith('return'):
                # Extract and fix return statement
                return_part = re.search(r'return\s*\(.*', line)
                if return_part:
                    fixed_lines.append('  ' + return_part.group(0))
                    continue
            
            # Remove duplicate interface declarations
            if line.strip().startswith('export interface'):
                interface_name = re.search(r'export interface (\w+)', line)
                if interface_name:
                    # Check if we've already seen this interface
                    name = interface_name.group(1)
                    if any(f'export interface {name}' in prev_line for prev_line in fixed_lines):
                        continue  # Skip duplicate
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _restore_from_backup(self, target_file: Path) -> bool:
        """Restore file from most recent backup."""
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
                return True
        except Exception:
            pass
        
        return False
    
    def _verify_context_freshness(self, hunks_by_file: Dict[str, List]) -> Dict[str, Any]:
        """
        NEW: Verify that the expected context in diffs matches current file reality.
        
        This prevents the classic AI context staleness problem where the AI generates
        diffs based on outdated context snapshots.
        
        Args:
            hunks_by_file: Dictionary mapping file paths to their hunks
            
        Returns:
            Dictionary with verification results
        """
        verification_result = {
            "all_fresh": True,
            "stale_files": {},
            "total_checked": 0,
            "context_matches": 0
        }
        
        for file_path, file_hunks in hunks_by_file.items():
            target_file = self._resolve_file_path(file_path)
            if not target_file or not target_file.exists():
                continue
            
            try:
                # Read current file content
                current_content = target_file.read_text()
                current_lines = current_content.splitlines()
                
                verification_result["total_checked"] += 1
                
                # Check if ANY expected context from the hunks exists in current file
                context_found = False
                sample_context = []
                
                for hunk in file_hunks[:3]:  # Check first 3 hunks
                    # Look for context lines that should exist
                    for line in hunk.lines:
                        if line.startswith(' '):  # Context line (unchanged)
                            context_line = line[1:].strip()  # Remove the space prefix
                            if context_line and context_line in current_content:
                                context_found = True
                                break
                            elif context_line:
                                sample_context.append(context_line[:50])
                    
                    if context_found:
                        break
                
                if context_found:
                    verification_result["context_matches"] += 1
                else:
                    verification_result["all_fresh"] = False
                    verification_result["stale_files"][file_path] = {
                        "issue": "Expected context not found in current file",
                        "sample_expected": sample_context[:3],
                        "current_start": current_lines[:3] if current_lines else ["<empty file>"]
                    }
                    
            except Exception as e:
                verification_result["stale_files"][file_path] = {
                    "issue": f"Could not verify context: {e}",
                    "sample_expected": [],
                    "current_start": []
                }
                verification_result["all_fresh"] = False
        
        # Report summary
        fresh_count = verification_result["context_matches"]
        total_count = verification_result["total_checked"]
        
        if verification_result["all_fresh"]:
            print(f"✅ Context verification: {fresh_count}/{total_count} files have fresh context")
        else:
            stale_count = len(verification_result["stale_files"])
            print(f"⚠️ Context verification: {stale_count}/{total_count} files have stale context")
        
        return verification_result
    
    def _validate_all_changes(self, hunks_by_file: Dict[str, List]) -> List[PatchResult]:
        """Pre-validate all changes without applying them."""
        results = []
        
        for file_path, file_hunks in hunks_by_file.items():
            # Resolve and validate file path
            target_file = self._resolve_file_path(file_path)
            if not target_file:
                results.append(PatchResult(False, file_path, f"Invalid file path: {file_path}"))
                continue
            
            # Read current file content
            try:
                if target_file.exists():
                    current_content = target_file.read_text()
                    current_lines = current_content.splitlines(keepends=True)
                else:
                    current_lines = []
            except Exception as e:
                results.append(PatchResult(False, file_path, f"Failed to read file: {e}"))
                continue
            
            # Validate each hunk can be applied
            validation_success = True
            validation_errors = []
            
            # Simulate applying hunks to check for conflicts
            test_lines = current_lines.copy()
            offset = 0
            
            sorted_hunks = sorted(file_hunks, key=lambda h: h.old_start)
            for hunk in sorted_hunks:
                # Check if this hunk can be applied
                success, new_offset = self._validate_single_hunk(test_lines, hunk, offset)
                if not success:
                    validation_success = False
                    validation_errors.append(f"Hunk at line {hunk.old_start} cannot be applied")
                else:
                    offset = new_offset
            
            if validation_success:
                results.append(PatchResult(True, file_path))
            else:
                error_msg = f"Validation failed: {'; '.join(validation_errors)}"
                results.append(PatchResult(False, file_path, error_msg))
        
        return results
    
    def _validate_single_hunk(self, file_lines: List[str], hunk: DiffHunk, offset: int) -> Tuple[bool, int]:
        """Validate that a single hunk can be applied without actually applying it."""
        expected_pos = max(0, hunk.old_start - 1 + offset)
        
        # For insertion-only hunks, just check bounds
        if not hunk.removed_lines and hunk.added_lines:
            return True, len(hunk.added_lines)
        
        # For hunks with removals, validate the match
        if hunk.removed_lines:
            context_for_match = hunk.removed_lines
            actual_pos = self.fuzzy_matcher.find_context_match(file_lines, context_for_match, expected_pos)
            
            if actual_pos is None:
                return False, offset
            
            # Validate match quality
            match_quality = self._validate_match_quality(file_lines, context_for_match, actual_pos)
            if match_quality < 0.8:
                return False, offset
            
            # Check bounds
            if actual_pos + len(hunk.removed_lines) > len(file_lines):
                return False, offset
            
            # Verify removal content
            if not self._verify_removal_content(file_lines, hunk.removed_lines, actual_pos):
                return False, offset
            
            # Calculate offset change
            if hunk.added_lines:
                new_offset = offset + len(hunk.added_lines) - len(hunk.removed_lines)
            else:
                new_offset = offset - len(hunk.removed_lines)
            
            return True, new_offset
        
        return True, offset
    
    def _validate_match_quality(self, file_lines: List[str], context_lines: List[str], pos: int) -> float:
        """Calculate match quality between context and file content."""
        if pos + len(context_lines) > len(file_lines):
            return 0.0
        
        file_slice = file_lines[pos:pos + len(context_lines)]
        matches = sum(1 for f, c in zip(file_slice, context_lines) if f.rstrip() == c.rstrip())
        
        return matches / len(context_lines) if context_lines else 1.0
    
    def _verify_removal_content(self, file_lines: List[str], removed_lines: List[str], pos: int) -> bool:
        """Verify that content to be removed matches what's in the file."""
        if pos + len(removed_lines) > len(file_lines):
            return False
        
        file_slice = file_lines[pos:pos + len(removed_lines)]
        for f_line, r_line in zip(file_slice, removed_lines):
            if f_line.rstrip() != r_line.rstrip():
                return False
        
        return True
    
    def _start_transaction(self) -> str:
        """Start a new transaction and return transaction ID."""
        transaction_id = f"txn_{int(time.time())}_{id(self)}"
        self.transaction_id = transaction_id
        self.transaction_backups = {}
        
        # Create transaction backup directory
        txn_backup_dir = self.backup_dir / transaction_id
        txn_backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Started transaction: {transaction_id}")
        return transaction_id
    
    def _create_transaction_backups(self, file_paths: List[str]) -> List[PatchResult]:
        """Create backups for all files in the transaction."""
        results = []
        
        for file_path in file_paths:
            target_file = self._resolve_file_path(file_path)
            if not target_file:
                results.append(PatchResult(False, file_path, f"Invalid file path: {file_path}"))
                continue
            
            if target_file.exists():
                try:
                    # Create backup
                    backup_path = self._create_file_backup(target_file)
                    self.transaction_backups[file_path] = backup_path
                    results.append(PatchResult(True, file_path))
                except Exception as e:
                    results.append(PatchResult(False, file_path, f"Backup failed: {e}"))
            else:
                # File doesn't exist - no backup needed
                results.append(PatchResult(True, file_path))
        
        return results
    
    def _apply_file_hunks_transactional(self, file_path: str, hunks: List[DiffHunk], transaction_id: str) -> PatchResult:
        """Apply hunks to a file within a transaction."""
        target_file = self._resolve_file_path(file_path)
        if not target_file:
            return PatchResult(False, file_path, f"Invalid file path: {file_path}")
        
        try:
            # Read current content
            if target_file.exists():
                current_content = target_file.read_text()
                current_lines = current_content.splitlines(keepends=True)
            else:
                current_lines = []
            
            # Apply hunks in order
            sorted_hunks = sorted(hunks, key=lambda h: h.old_start)
            offset = 0
            
            for hunk in sorted_hunks:
                current_lines, new_offset = self._apply_single_hunk(current_lines, hunk, offset)
                offset = new_offset
            
            # Write the modified content
            new_content = ''.join(current_lines)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(new_content)
            
            # ENHANCED: Post-application corruption check
            is_corrupted, severity = self.corruption_detector.is_file_corrupted(file_path, new_content)
            if is_corrupted and severity in [CorruptionSeverity.HIGH, CorruptionSeverity.CRITICAL]:
                return PatchResult(False, file_path, f"Application would introduce {severity.value} corruption")
            
            return PatchResult(True, file_path)
            
        except Exception as e:
            return PatchResult(False, file_path, f"Error applying hunks: {e}")
    
    def _apply_single_hunk(self, file_lines: List[str], hunk: DiffHunk, offset: int) -> Tuple[List[str], int]:
        """Apply a single hunk to file lines."""
        expected_pos = max(0, hunk.old_start - 1 + offset)
        
        print(f"🔧 Applying hunk at line {hunk.old_start} (adjusted: {expected_pos + 1})")
        print(f"   Old count: {hunk.old_count}, New count: {hunk.new_count}")
        
        # Handle different hunk types
        if not hunk.removed_lines and hunk.added_lines:
            # Pure insertion
            return self._apply_insertion_hunk(file_lines, hunk, expected_pos, offset)
        elif hunk.removed_lines and not hunk.added_lines:
            # Pure deletion
            return self._apply_deletion_hunk(file_lines, hunk, expected_pos, offset)
        else:
            # Mixed or replacement
            return self._apply_replacement_hunk(file_lines, hunk, expected_pos, offset)
    
    def _apply_insertion_hunk(self, file_lines: List[str], hunk: DiffHunk, expected_pos: int, offset: int) -> Tuple[List[str], int]:
        """Apply an insertion-only hunk."""
        # For insertions, we can be more lenient about position
        insert_pos = min(expected_pos, len(file_lines))
        
        # Insert new lines
        new_lines = [line + '\n' if not line.endswith('\n') else line for line in hunk.added_lines]
        file_lines[insert_pos:insert_pos] = new_lines
        
        new_offset = offset + len(hunk.added_lines)
        print(f"✅ Inserted {len(hunk.added_lines)} lines at position {insert_pos + 1}")
        
        return file_lines, new_offset
    
    def _apply_deletion_hunk(self, file_lines: List[str], hunk: DiffHunk, expected_pos: int, offset: int) -> Tuple[List[str], int]:
        """Apply a deletion-only hunk."""
        # Find the best match for the content to be removed
        context_for_match = hunk.removed_lines
        actual_pos = self.fuzzy_matcher.find_context_match(file_lines, context_for_match, expected_pos)
        
        if actual_pos is None:
            print(f"⚠️ Could not find content to remove for deletion hunk")
            return file_lines, offset
        
        # Verify the content matches before removing
        end_pos = actual_pos + len(hunk.removed_lines)
        if end_pos > len(file_lines):
            print(f"⚠️ Deletion would exceed file bounds")
            return file_lines, offset
        
        # Remove the lines
        del file_lines[actual_pos:end_pos]
        
        new_offset = offset - len(hunk.removed_lines)
        print(f"✅ Removed {len(hunk.removed_lines)} lines at position {actual_pos + 1}")
        
        return file_lines, new_offset
    
    def _apply_replacement_hunk(self, file_lines: List[str], hunk: DiffHunk, expected_pos: int, offset: int) -> Tuple[List[str], int]:
        """Apply a replacement hunk (removes old lines and adds new ones)."""
        if hunk.removed_lines:
            # Find the content to be replaced
            context_for_match = hunk.removed_lines
            actual_pos = self.fuzzy_matcher.find_context_match(file_lines, context_for_match, expected_pos)
            
            if actual_pos is None:
                print(f"⚠️ Could not find content to replace - applying as insertion")
                return self._apply_insertion_hunk(file_lines, hunk, expected_pos, offset)
            
            # Remove old lines
            end_pos = actual_pos + len(hunk.removed_lines)
            if end_pos > len(file_lines):
                print(f"⚠️ Replacement would exceed file bounds")
                return file_lines, offset
            
            del file_lines[actual_pos:end_pos]
            
            # Insert new lines at the same position
            if hunk.added_lines:
                new_lines = [line + '\n' if not line.endswith('\n') else line for line in hunk.added_lines]
                file_lines[actual_pos:actual_pos] = new_lines
            
            new_offset = offset + len(hunk.added_lines) - len(hunk.removed_lines)
            print(f"✅ Replaced {len(hunk.removed_lines)} lines with {len(hunk.added_lines)} lines at position {actual_pos + 1}")
            
            return file_lines, new_offset
        else:
            # No removal, just insertion
            return self._apply_insertion_hunk(file_lines, hunk, expected_pos, offset)
    
    def _resolve_file_path(self, file_path: str) -> Optional[Path]:
        """Resolve a file path relative to the repository root."""
        # Remove common prefixes
        clean_path = file_path
        for prefix in ['a/', 'b/', './']:
            if clean_path.startswith(prefix):
                clean_path = clean_path[len(prefix):]
        
        # Resolve relative to repo root
        target_file = self.repo_root / clean_path
        
        # Validate path is within repo (security check)
        try:
            target_file.resolve().relative_to(self.repo_root.resolve())
            return target_file
        except ValueError:
            print(f"⚠️ File path outside repository: {file_path}")
            return None
    
    def _create_file_backup(self, file_path: Path) -> Path:
        """Create a backup of a file and return backup path."""
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create unique backup filename
        timestamp = int(time.time())
        backup_name = f"{file_path.name}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        # Copy file to backup
        shutil.copy2(file_path, backup_path)
        
        print(f"📸 Created backup: {backup_name}")
        return backup_path
    
    def _commit_transaction(self, transaction_id: str):
        """Commit a transaction by removing temporary backups."""
        if transaction_id != self.transaction_id:
            print(f"⚠️ Transaction ID mismatch during commit")
            return
        
        # Remove transaction backups (they're no longer needed)
        txn_backup_dir = self.backup_dir / transaction_id
        if txn_backup_dir.exists():
            shutil.rmtree(txn_backup_dir)
        
        # Clear transaction state
        self.transaction_id = None
        self.transaction_backups = {}
        
        print(f"✅ Transaction {transaction_id} committed successfully")
    
    def _abort_transaction(self, transaction_id: str):
        """Abort a transaction by restoring all files from backups."""
        if transaction_id != self.transaction_id:
            print(f"⚠️ Transaction ID mismatch during abort")
            return
        
        print(f"🔄 Aborting transaction {transaction_id} - restoring all files...")
        
        # Restore all files from their backups
        for file_path, backup_path in self.transaction_backups.items():
            try:
                target_file = self._resolve_file_path(file_path)
                if target_file and backup_path.exists():
                    shutil.copy2(backup_path, target_file)
                    print(f"↩️ Restored {file_path}")
            except Exception as e:
                print(f"❌ Failed to restore {file_path}: {e}")
        
        # Clean up transaction backup directory
        txn_backup_dir = self.backup_dir / transaction_id
        if txn_backup_dir.exists():
            shutil.rmtree(txn_backup_dir)
        
        # Clear transaction state
        self.transaction_id = None
        self.transaction_backups = {}
        
        print(f"🔄 Transaction {transaction_id} aborted - all changes rolled back")


class DiffSanitizer:
    """Legacy diff sanitizer for compatibility."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
    
    def sanitize_patch(self, patch_text: str) -> str:
        """Minimal sanitization for backward compatibility."""
        return patch_text
    
    def validate_diff_structure(self, patch_text: str) -> Tuple[bool, List[str]]:
        """Basic validation."""
        issues = []
        
        if '*** Begin Patch' not in patch_text:
            issues.append("Missing '*** Begin Patch' sentinel")
        
        if '*** End Patch' not in patch_text:
            issues.append("Missing '*** End Patch' sentinel")
        
        return len(issues) == 0, issues


class AtomicDiffBuilder:
    """
    Main class for applying AI-generated diffs with corruption protection.
    
    ENHANCED: Now uses corruption-aware diff application.
    """
    
    def __init__(self, app_dir: str, patch_file: str):
        self.patch_file = patch_file
        self.app_dir = Path(app_dir)
        self.applier = CorruptionAwareDiffApplier(self.app_dir)
        self.sanitizer = DiffSanitizer(self.app_dir)
        self.max_retries = 2
    
    def apply_patch_atomically(self) -> List[PatchResult]:
        """
        Apply patch with corruption protection and automatic recovery.
        
        ENHANCED: Includes corruption detection and recovery.
        """
        if not Path(self.patch_file).exists():
            return [PatchResult(False, "", f"Patch file not found: {self.patch_file}")]
        
        try:
            with open(self.patch_file, 'r') as f:
                patch_content = f.read()
        except Exception as e:
            return [PatchResult(False, "", f"Failed to read patch file: {e}")]
        
        print(f"🔧 Applying patch with corruption protection: {self.patch_file}")
        
        # Apply patch with corruption detection
        results = self.applier.apply_patch(patch_content)
        
        return results


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