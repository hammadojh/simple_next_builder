"""
File Corruption Detection System

Detects when files are fundamentally corrupted and need restoration
rather than incremental patching.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CorruptionLevel(Enum):
    """Severity levels of file corruption."""
    CLEAN = "clean"
    MINOR = "minor"          # Small syntax issues, fixable with patches
    MODERATE = "moderate"    # Structural issues, requires careful handling  
    SEVERE = "severe"        # Fundamental corruption, needs restoration
    CRITICAL = "critical"    # File is unrecoverable


@dataclass
class CorruptionIssue:
    """Represents a specific corruption issue found in a file."""
    issue_type: str
    line_number: Optional[int]
    description: str
    severity: CorruptionLevel
    suggested_fix: str


@dataclass 
class CorruptionReport:
    """Complete corruption analysis report for a file."""
    file_path: str
    overall_level: CorruptionLevel
    issues: List[CorruptionIssue]
    is_recoverable: bool
    recommended_action: str
    confidence: float  # 0.0 to 1.0


class FileCorruptionDetector:
    """
    Detects various types of file corruption that prevent successful patching.
    
    IMPROVED: Better scope awareness and reduced false positives.
    
    Corruption Types Detected:
    1. Syntax corruption (malformed code)
    2. Structural corruption (missing/duplicate declarations)
    3. Encoding corruption (character issues)
    4. Format corruption (mixed syntax, broken JSX)
    5. Semantic corruption (impossible code patterns)
    """
    
    def __init__(self):
        self.typescript_patterns = self._init_typescript_patterns()
        self.react_patterns = self._init_react_patterns()
        self.common_patterns = self._init_common_patterns()
    
    def analyze_file(self, file_path: str) -> CorruptionReport:
        """
        Analyze a file for corruption and return detailed report.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            CorruptionReport with detailed findings
        """
        path = Path(file_path)
        
        if not path.exists():
            return CorruptionReport(
                file_path=file_path,
                overall_level=CorruptionLevel.CRITICAL,
                issues=[CorruptionIssue("missing_file", None, "File does not exist", CorruptionLevel.CRITICAL, "Restore from backup")],
                is_recoverable=False,
                recommended_action="restore_from_backup",
                confidence=1.0
            )
        
        try:
            content = path.read_text()
        except Exception as e:
            return CorruptionReport(
                file_path=file_path,
                overall_level=CorruptionLevel.CRITICAL,
                issues=[CorruptionIssue("read_error", None, f"Cannot read file: {e}", CorruptionLevel.CRITICAL, "Check file permissions")],
                is_recoverable=False,
                recommended_action="restore_from_backup", 
                confidence=1.0
            )
        
        # Analyze different corruption types
        issues = []
        issues.extend(self._detect_syntax_corruption(content, file_path))
        issues.extend(self._detect_structural_corruption(content, file_path))
        issues.extend(self._detect_format_corruption(content, file_path))
        issues.extend(self._detect_semantic_corruption(content, file_path))
        
        # Determine overall corruption level
        overall_level = self._calculate_overall_level(issues)
        is_recoverable = overall_level not in [CorruptionLevel.CRITICAL]
        recommended_action = self._get_recommended_action(overall_level, issues)
        confidence = self._calculate_confidence(issues)
        
        return CorruptionReport(
            file_path=file_path,
            overall_level=overall_level,
            issues=issues,
            is_recoverable=is_recoverable,
            recommended_action=recommended_action,
            confidence=confidence
        )
    
    def _detect_syntax_corruption(self, content: str, file_path: str) -> List[CorruptionIssue]:
        """Detect syntax-level corruption."""
        issues = []
        lines = content.split('\n')
        
        # Check for common syntax corruption patterns
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Malformed import statements - be more specific
            if 'import' in line and 'from' in line:
                # Only flag truly malformed imports, not just missing semicolons
                if not re.match(r'^import\s+.*from\s+[\'"][^\'"]+[\'"];?\s*$', line_stripped) and len(line_stripped) > 50:
                    # Only flag if line is unusually long (likely concatenated)
                    issues.append(CorruptionIssue(
                        "malformed_import", i, f"Severely malformed import statement: {line_stripped[:50]}...",
                        CorruptionLevel.MODERATE, "Fix import syntax"
                    ))
            
            # Missing semicolons in critical places - be more lenient
            if re.search(r'(import.*from\s+[\'"][^\'"]+[\'"])\s*$', line_stripped):
                if not line_stripped.endswith(';'):
                    issues.append(CorruptionIssue(
                        "missing_semicolon", i, f"Missing semicolon: {line_stripped}",
                        CorruptionLevel.MINOR, "Add semicolon"
                    ))
            
            # Unclosed brackets/braces - only flag severe imbalances
            open_brackets = line.count('(') - line.count(')')
            open_braces = line.count('{') - line.count('}')
            open_squares = line.count('[') - line.count(']')
            
            if abs(open_brackets) > 3 or abs(open_braces) > 3 or abs(open_squares) > 3:
                issues.append(CorruptionIssue(
                    "unbalanced_brackets", i, f"Severely unbalanced brackets: {line_stripped}",
                    CorruptionLevel.SEVERE, "Fix bracket matching"
                ))
        
        # Check overall bracket balance
        total_open_parens = content.count('(') - content.count(')')
        total_open_braces = content.count('{') - content.count('}')
        total_open_squares = content.count('[') - content.count(']')
        
        if abs(total_open_parens) > 2:
            issues.append(CorruptionIssue(
                "unbalanced_parens", None, f"Unbalanced parentheses: {total_open_parens}",
                CorruptionLevel.SEVERE, "Balance parentheses"
            ))
        
        if abs(total_open_braces) > 2:
            issues.append(CorruptionIssue(
                "unbalanced_braces", None, f"Unbalanced braces: {total_open_braces}",
                CorruptionLevel.SEVERE, "Balance braces"
            ))
        
        return issues
    
    def _detect_structural_corruption(self, content: str, file_path: str) -> List[CorruptionIssue]:
        """Detect structural corruption with better scope awareness."""
        issues = []
        lines = content.split('\n')
        
        # Track declarations with scope awareness
        global_functions = []
        interface_names = []
        exports = []
        
        # Parse function scope boundaries
        current_scope_level = 0
        function_scopes = {}  # scope_level -> list of function names
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Track scope level changes
            current_scope_level += line.count('{') - line.count('}')
            
            # Function declarations - only check global scope for duplicates
            func_match = re.search(r'function\s+(\w+)', line_stripped)
            if func_match and current_scope_level <= 1:  # Global or top-level only
                func_name = func_match.group(1)
                if func_name in global_functions:
                    issues.append(CorruptionIssue(
                        "duplicate_function", i, f"Duplicate function declaration: {func_name}",
                        CorruptionLevel.SEVERE, "Remove duplicate declaration"
                    ))
                global_functions.append(func_name)
            
            # Arrow function declarations - only check global/export level
            arrow_match = re.search(r'const\s+(\w+)\s*=.*=>', line_stripped)
            if arrow_match and current_scope_level <= 1:  # Global or top-level only
                func_name = arrow_match.group(1)
                if func_name in global_functions:
                    issues.append(CorruptionIssue(
                        "duplicate_function", i, f"Duplicate arrow function: {func_name}",
                        CorruptionLevel.SEVERE, "Remove duplicate declaration"
                    ))
                global_functions.append(func_name)
            
            # Interface declarations
            interface_match = re.search(r'interface\s+(\w+)', line_stripped)
            if interface_match:
                interface_name = interface_match.group(1)
                if interface_name in interface_names:
                    issues.append(CorruptionIssue(
                        "duplicate_interface", i, f"Duplicate interface: {interface_name}",
                        CorruptionLevel.SEVERE, "Remove duplicate interface"
                    ))
                interface_names.append(interface_name)
            
            # Export statements
            if 'export' in line_stripped:
                exports.append(i)
        
        # Check for missing default export in React components
        if file_path.endswith(('.tsx', '.jsx')):
            has_default_export = any('export default' in line for line in lines)
            has_component = any(re.search(r'(function|const)\s+[A-Z]\w*', line) for line in lines)
            
            if has_component and not has_default_export:
                issues.append(CorruptionIssue(
                    "missing_default_export", None, "React component missing default export",
                    CorruptionLevel.MODERATE, "Add export default"
                ))
        
        return issues
    
    def _detect_format_corruption(self, content: str, file_path: str) -> List[CorruptionIssue]:
        """Detect format-level corruption with better JSX handling."""
        issues = []
        lines = content.split('\n')
        
        if file_path.endswith(('.tsx', '.jsx')):
            # Check for JSX corruption - be more lenient
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Only flag obviously broken JSX patterns, not normal multi-line JSX
                if len(line) > 300:  # Very long lines that might be concatenated
                    jsx_tag_count = len(re.findall(r'<\w+', line))
                    if jsx_tag_count > 3:  # Multiple JSX elements on one line
                        issues.append(CorruptionIssue(
                            "jsx_concatenation", i, f"Multiple JSX elements concatenated: {jsx_tag_count} elements",
                            CorruptionLevel.MODERATE, "Split JSX elements into separate lines"
                        ))
                
                # Check for malformed JSX attributes - be much more specific
                if '<' in line and '>' in line and len(line) > 200:
                    # Look for attributes split across lines incorrectly
                    if re.search(r'className=\{.*\n.*\}', line) or 'className={`' in line and line.count('`') == 1:
                        issues.append(CorruptionIssue(
                            "split_jsx_attr", i, f"JSX attribute split incorrectly: {line_stripped[:80]}...",
                            CorruptionLevel.MODERATE, "Fix JSX attribute formatting"
                        ))
        
        # Check for line concatenation issues (missing line breaks) - be more specific
        for i, line in enumerate(lines, 1):
            if len(line) > 300:  # Very long lines might indicate concatenation
                # Check for multiple statements on one line
                statement_indicators = [';', 'const ', 'let ', 'var ', 'function ', 'if (', 'for (']
                statement_count = sum(line.count(indicator) for indicator in statement_indicators)
                
                if statement_count > 4:
                    issues.append(CorruptionIssue(
                        "line_concatenation", i, f"Multiple statements concatenated: {len(line)} chars",
                        CorruptionLevel.MODERATE, "Split into multiple lines"
                    ))
        
        return issues
    
    def _detect_semantic_corruption(self, content: str, file_path: str) -> List[CorruptionIssue]:
        """Detect semantic corruption - only truly impossible patterns."""
        issues = []
        
        # Check for impossible patterns - be more specific
        impossible_patterns = [
            (r'export\s+default.*export\s+default', "duplicate_default_export", "Multiple export default statements"),
            (r'import.*import.*from.*from', "concatenated_imports", "Multiple import statements concatenated"),
        ]
        
        for pattern, issue_type, description in impossible_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                issues.append(CorruptionIssue(
                    issue_type, None, description,
                    CorruptionLevel.SEVERE, "Fix duplicate statements"
                ))
        
        return issues
    
    def _calculate_overall_level(self, issues: List[CorruptionIssue]) -> CorruptionLevel:
        """Calculate overall corruption level based on individual issues."""
        if not issues:
            return CorruptionLevel.CLEAN
        
        severity_counts = {}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        # Determine overall level based on worst issues
        if severity_counts.get(CorruptionLevel.CRITICAL, 0) > 0:
            return CorruptionLevel.CRITICAL
        elif severity_counts.get(CorruptionLevel.SEVERE, 0) >= 3:  # Multiple severe issues
            return CorruptionLevel.CRITICAL
        elif severity_counts.get(CorruptionLevel.SEVERE, 0) > 0:
            return CorruptionLevel.SEVERE
        elif severity_counts.get(CorruptionLevel.MODERATE, 0) >= 5:  # Many moderate issues
            return CorruptionLevel.SEVERE
        elif severity_counts.get(CorruptionLevel.MODERATE, 0) > 0:
            return CorruptionLevel.MODERATE
        else:
            return CorruptionLevel.MINOR
    
    def _get_recommended_action(self, level: CorruptionLevel, issues: List[CorruptionIssue]) -> str:
        """Get recommended action based on corruption level."""
        if level == CorruptionLevel.CLEAN:
            return "continue_with_patches"
        elif level == CorruptionLevel.MINOR:
            return "apply_targeted_fixes"
        elif level == CorruptionLevel.MODERATE:
            return "apply_targeted_fixes"  # Changed from restore_and_reapply
        elif level == CorruptionLevel.SEVERE:
            return "restore_from_backup"
        else:  # CRITICAL
            return "restore_from_backup_or_recreate"
    
    def _calculate_confidence(self, issues: List[CorruptionIssue]) -> float:
        """Calculate confidence level in the corruption assessment."""
        if not issues:
            return 0.95  # High confidence in clean files
        
        # Higher confidence with more specific issues detected
        specific_issues = len([i for i in issues if i.line_number is not None])
        total_issues = len(issues)
        
        if total_issues == 0:
            return 0.95
        
        base_confidence = 0.7
        specificity_bonus = (specific_issues / total_issues) * 0.2
        issue_count_bonus = min(total_issues * 0.05, 0.1)
        
        return min(base_confidence + specificity_bonus + issue_count_bonus, 0.95)
    
    def _init_typescript_patterns(self) -> Dict:
        """Initialize TypeScript-specific corruption patterns."""
        return {
            'malformed_interface': r'interface\s+\w+\s*[^{]',
            'malformed_type': r'type\s+\w+\s*[^=]',
            'malformed_generic': r'<[^>]*[^>]$',
        }
    
    def _init_react_patterns(self) -> Dict:
        """Initialize React-specific corruption patterns.""" 
        return {
            'malformed_hook': r'use\w+\s*\([^)]*[^)]$',
            'malformed_jsx': r'<\w+[^>]*[^>]$',
            'malformed_component': r'function\s+[A-Z]\w*\s*[^(]',
        }
    
    def _init_common_patterns(self) -> Dict:
        """Initialize common corruption patterns."""
        return {
            'concatenated_lines': r'.{300,}',  # Increased threshold
            'missing_quotes': r'import.*from\s+[^\'"][^\s]+',
            'unbalanced_brackets': r'[\(\)\{\}\[\]]',
        } 