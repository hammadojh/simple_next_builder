"""
File Corruption Detection System

This module provides comprehensive detection of file corruption patterns
that commonly occur during AI-assisted code editing, particularly:
1. Syntax errors (malformed JSX, TypeScript issues)
2. Structural corruption (duplicate elements, misplaced code)
3. Import/export issues
4. Hook placement problems
5. Missing closing tags or braces
"""

import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class CorruptionSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CorruptionIssue:
    """Represents a detected corruption issue."""
    file_path: str
    line_number: int
    issue_type: str
    severity: CorruptionSeverity
    description: str
    suggested_fix: str
    confidence: float  # 0.0 to 1.0


class FileCorruptionDetector:
    """
    Detects various types of file corruption in TypeScript/TSX files.
    
    This detector identifies common corruption patterns that occur during
    AI-assisted editing, particularly from intent-based and diff-based systems.
    """
    
    def __init__(self):
        self.corruption_patterns = self._initialize_patterns()
        self.syntax_validators = self._initialize_validators()
    
    def detect_corruption(self, file_path: str, content: str) -> List[CorruptionIssue]:
        """
        Detect all types of corruption in a file.
        
        Args:
            file_path: Path to the file being analyzed
            content: File content as string
            
        Returns:
            List of corruption issues found
        """
        issues = []
        
        # 1. Detect syntax corruption
        syntax_issues = self._detect_syntax_corruption(file_path, content)
        issues.extend(syntax_issues)
        
        # 2. Detect structural corruption
        structural_issues = self._detect_structural_corruption(file_path, content)
        issues.extend(structural_issues)
        
        # 3. Detect import/export corruption
        import_issues = self._detect_import_corruption(file_path, content)
        issues.extend(import_issues)
        
        # 4. Detect JSX corruption
        if file_path.endswith(('.tsx', '.jsx')):
            jsx_issues = self._detect_jsx_corruption(file_path, content)
            issues.extend(jsx_issues)
        
        # 5. Detect hook corruption
        if file_path.endswith(('.tsx', '.jsx', '.ts', '.js')):
            hook_issues = self._detect_hook_corruption(file_path, content)
            issues.extend(hook_issues)
        
        return issues
    
    def is_file_corrupted(self, file_path: str, content: str) -> Tuple[bool, CorruptionSeverity]:
        """
        Quick check if file is corrupted and severity level.
        
        Returns:
            (is_corrupted, max_severity)
        """
        issues = self.detect_corruption(file_path, content)
        
        if not issues:
            return False, CorruptionSeverity.LOW
        
        max_severity = max(issue.severity for issue in issues)
        is_corrupted = any(issue.severity in [CorruptionSeverity.HIGH, CorruptionSeverity.CRITICAL] 
                          for issue in issues)
        
        return is_corrupted, max_severity
    
    def _detect_syntax_corruption(self, file_path: str, content: str) -> List[CorruptionIssue]:
        """Detect basic syntax corruption patterns."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Detect misplaced imports
            if 'import ' in line and not line.strip().startswith('import'):
                # Check if it's inside a function
                if self._is_inside_function(lines, i):
                    issues.append(CorruptionIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type="misplaced_import",
                        severity=CorruptionSeverity.HIGH,
                        description=f"Import statement inside function body at line {line_num}",
                        suggested_fix="Move import to top of file",
                        confidence=0.9
                    ))
            
            # Detect incomplete statements
            if line.strip().endswith('{') and not self._has_matching_brace(lines, i):
                issues.append(CorruptionIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type="incomplete_statement",
                    severity=CorruptionSeverity.MEDIUM,
                    description=f"Incomplete statement or missing closing brace at line {line_num}",
                    suggested_fix="Add missing closing brace",
                    confidence=0.7
                ))
            
            # Detect malformed return statements
            if 'return (' in line and not line.strip().startswith('return'):
                issues.append(CorruptionIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type="malformed_return",
                    severity=CorruptionSeverity.HIGH,
                    description=f"Malformed return statement at line {line_num}",
                    suggested_fix="Fix return statement structure",
                    confidence=0.8
                ))
        
        return issues
    
    def _detect_structural_corruption(self, file_path: str, content: str) -> List[CorruptionIssue]:
        """Detect structural corruption like duplicate elements."""
        issues = []
        lines = content.split('\n')
        
        # Detect duplicate interface declarations
        interfaces = {}
        for i, line in enumerate(lines):
            match = re.match(r'export interface (\w+)', line.strip())
            if match:
                interface_name = match.group(1)
                if interface_name in interfaces:
                    issues.append(CorruptionIssue(
                        file_path=file_path,
                        line_number=i + 1,
                        issue_type="duplicate_interface",
                        severity=CorruptionSeverity.HIGH,
                        description=f"Duplicate interface declaration '{interface_name}' at line {i + 1}",
                        suggested_fix=f"Remove duplicate interface declaration",
                        confidence=0.95
                    ))
                else:
                    interfaces[interface_name] = i + 1
        
        # Detect duplicate JSX elements
        jsx_tags = {}
        for i, line in enumerate(lines):
            # Look for opening JSX tags
            match = re.search(r'<(\w+)(?:\s|>)', line)
            if match:
                tag_name = match.group(1)
                # Check for specific problematic duplicates
                if tag_name in ['CardFooter', 'CardHeader', 'Card'] and tag_name in jsx_tags:
                    # Verify it's not just nested usage
                    if not self._is_nested_jsx_usage(lines, jsx_tags[tag_name], i):
                        issues.append(CorruptionIssue(
                            file_path=file_path,
                            line_number=i + 1,
                            issue_type="duplicate_jsx_element",
                            severity=CorruptionSeverity.HIGH,
                            description=f"Potential duplicate JSX element '{tag_name}' at line {i + 1}",
                            suggested_fix=f"Remove duplicate JSX element",
                            confidence=0.8
                        ))
                jsx_tags[tag_name] = i
        
        return issues
    
    def _detect_import_corruption(self, file_path: str, content: str) -> List[CorruptionIssue]:
        """Detect import/export related corruption."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Detect incomplete export statements
            if line.strip().startswith('export interface') and not line.strip().endswith('{'):
                if not self._has_interface_body(lines, i):
                    issues.append(CorruptionIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type="incomplete_export",
                        severity=CorruptionSeverity.HIGH,
                        description=f"Incomplete export interface at line {line_num}",
                        suggested_fix="Complete the interface definition",
                        confidence=0.9
                    ))
        
        return issues
    
    def _detect_jsx_corruption(self, file_path: str, content: str) -> List[CorruptionIssue]:
        """Detect JSX-specific corruption patterns."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            # Detect JSX elements on their own lines (potential corruption)
            jsx_element_pattern = r'^<(\w+)(?:\s[^>]*)?>$'
            match = re.match(jsx_element_pattern, stripped)
            if match:
                element_name = match.group(1)
                # Check if this looks like a duplicate element
                if self._is_suspicious_jsx_element(lines, i, element_name):
                    issues.append(CorruptionIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type="suspicious_jsx_element",
                        severity=CorruptionSeverity.MEDIUM,
                        description=f"Suspicious standalone JSX element '{element_name}' at line {line_num}",
                        suggested_fix="Verify JSX element structure and nesting",
                        confidence=0.7
                    ))
        
        return issues
    
    def _detect_hook_corruption(self, file_path: str, content: str) -> List[CorruptionIssue]:
        """Detect React hook corruption patterns."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Detect hooks outside component scope
            hook_pattern = r'const\s+\{[^}]+\}\s*=\s*use\w+\('
            if re.search(hook_pattern, line):
                if not self._is_inside_component_function(lines, i):
                    issues.append(CorruptionIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type="hook_outside_component",
                        severity=CorruptionSeverity.CRITICAL,
                        description=f"React hook outside component function at line {line_num}",
                        suggested_fix="Move hook inside component function",
                        confidence=0.95
                    ))
        
        return issues
    
    def _initialize_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for corruption detection."""
        return {
            'malformed_jsx': r'<\w+[^>]*>\s*<\w+[^>]*>',
            'incomplete_interface': r'export interface \w+ \{$',
            'misplaced_import': r'^\s+import\s+',
            'duplicate_element': r'(<(\w+)[^>]*>).*\1',
        }
    
    def _initialize_validators(self) -> Dict[str, callable]:
        """Initialize syntax validators."""
        return {
            'typescript': self._validate_typescript_syntax,
            'jsx': self._validate_jsx_syntax,
        }
    
    def _is_inside_function(self, lines: List[str], line_index: int) -> bool:
        """Check if a line is inside a function definition."""
        # Look backwards for function declaration
        for i in range(line_index - 1, -1, -1):
            line = lines[i].strip()
            if re.match(r'function\s+\w+|export\s+default\s+function|\w+\s*=\s*\([^)]*\)\s*=>', line):
                return True
            if line in ['', '}'] and i > 0:
                continue
            if line.startswith('export') or line.startswith('import'):
                return False
        return False
    
    def _is_inside_component_function(self, lines: List[str], line_index: int) -> bool:
        """Check if a line is inside a React component function."""
        # Look backwards for React component function
        for i in range(line_index - 1, -1, -1):
            line = lines[i].strip()
            # React component patterns
            if re.match(r'export\s+default\s+function\s+\w+|function\s+\w+.*\{|const\s+\w+.*=.*\{', line):
                return True
            if line.startswith('export') or line.startswith('import'):
                return False
        return False
    
    def _has_matching_brace(self, lines: List[str], line_index: int) -> bool:
        """Check if an opening brace has a matching closing brace."""
        brace_count = 0
        for i in range(line_index, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and i > line_index:
                return True
        return False
    
    def _has_interface_body(self, lines: List[str], line_index: int) -> bool:
        """Check if an interface declaration has a proper body."""
        # Look for opening brace on same line or next few lines
        current_line = lines[line_index]
        if '{' in current_line:
            return True
        
        # Check next few lines for opening brace or completion
        for i in range(line_index + 1, min(line_index + 5, len(lines))):
            line = lines[i].strip()
            if '{' in line:
                return True
            # If we see another declaration or end of interface, check if it's complete
            if line.endswith('}') or line.endswith('{}'):
                return True
            # Multi-line interface with extends - check if it ends properly
            if ('extends' in current_line or 'extends' in line) and (line.endswith('{}') or line.endswith('{ }')):
                return True
        
        return False
    
    def _is_nested_jsx_usage(self, lines: List[str], first_occurrence: int, second_occurrence: int) -> bool:
        """Check if JSX elements are properly nested rather than duplicated."""
        # Simple heuristic: if there's significant indentation difference, likely nested
        first_indent = len(lines[first_occurrence]) - len(lines[first_occurrence].lstrip())
        second_indent = len(lines[second_occurrence]) - len(lines[second_occurrence].lstrip())
        
        return abs(first_indent - second_indent) > 2
    
    def _is_suspicious_jsx_element(self, lines: List[str], line_index: int, element_name: str) -> bool:
        """Check if a JSX element appears suspicious (likely corruption)."""
        current_line = lines[line_index].strip()
        
        # Don't flag common single elements that are normal in JSX
        if element_name in ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'Link', 'Button', 'Badge']:
            return False
        
        # Don't flag if this looks like normal JSX structure
        if current_line.startswith('<') and (current_line.endswith('>') or current_line.endswith('/>')):
            # Check if it's within a return statement or JSX context
            for i in range(max(0, line_index - 10), line_index):
                prev_line = lines[i].strip()
                if 'return (' in prev_line or prev_line.startswith('return'):
                    return False  # This is in a return statement, probably fine
                if '<' in prev_line and '>' in prev_line:
                    return False  # This is in JSX context, probably fine
        
        # Only flag if we see the exact same element very close by (likely duplication)
        for i in range(max(0, line_index - 3), min(len(lines), line_index + 3)):
            if i != line_index and lines[i].strip() == current_line:
                return True  # Exact duplicate nearby - this is suspicious
        
        return False
    
    def _validate_typescript_syntax(self, content: str) -> List[str]:
        """Validate TypeScript syntax (basic checks)."""
        errors = []
        
        # Check for common TypeScript syntax errors
        if 'export interface' in content and content.count('{') != content.count('}'):
            errors.append("Mismatched braces in interface definitions")
        
        return errors
    
    def _validate_jsx_syntax(self, content: str) -> List[str]:
        """Validate JSX syntax (basic checks)."""
        errors = []
        
        # Check for common JSX syntax errors
        jsx_tags = re.findall(r'<(\w+)', content)
        closing_tags = re.findall(r'</(\w+)>', content)
        
        # Simple check for unmatched tags (not foolproof but catches obvious issues)
        if len(jsx_tags) != len(closing_tags):
            errors.append("Potential unmatched JSX tags")
        
        return errors


def detect_app_corruption(app_path: str) -> Dict[str, List[CorruptionIssue]]:
    """
    Scan an entire app directory for corruption issues.
    
    Args:
        app_path: Path to the app directory
        
    Returns:
        Dictionary mapping file paths to their corruption issues
    """
    detector = FileCorruptionDetector()
    app_root = Path(app_path)
    corruption_report = {}
    
    # Scan TypeScript/TSX/JavaScript files
    for file_pattern in ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx']:
        for file_path in app_root.glob(file_pattern):
            # Skip node_modules and build directories
            if 'node_modules' in str(file_path) or '.next' in str(file_path):
                continue
            
            try:
                content = file_path.read_text()
                issues = detector.detect_corruption(str(file_path), content)
                
                if issues:
                    corruption_report[str(file_path)] = issues
                    
            except Exception as e:
                # If we can't read the file, that's also a corruption issue
                corruption_report[str(file_path)] = [CorruptionIssue(
                    file_path=str(file_path),
                    line_number=1,
                    issue_type="file_read_error",
                    severity=CorruptionSeverity.CRITICAL,
                    description=f"Cannot read file: {str(e)}",
                    suggested_fix="Check file permissions and integrity",
                    confidence=1.0
                )]
    
    return corruption_report


def print_corruption_report(corruption_report: Dict[str, List[CorruptionIssue]]):
    """Print a formatted corruption report."""
    if not corruption_report:
        print("✅ No corruption detected in the app")
        return
    
    print("🚨 CORRUPTION DETECTED:")
    print("=" * 60)
    
    for file_path, issues in corruption_report.items():
        print(f"\n📁 {file_path}")
        print("-" * 40)
        
        for issue in issues:
            severity_icon = {
                CorruptionSeverity.LOW: "ℹ️",
                CorruptionSeverity.MEDIUM: "⚠️",
                CorruptionSeverity.HIGH: "❌",
                CorruptionSeverity.CRITICAL: "🚨"
            }[issue.severity]
            
            print(f"  {severity_icon} Line {issue.line_number}: {issue.description}")
            print(f"     Type: {issue.issue_type}")
            print(f"     Fix: {issue.suggested_fix}")
            print(f"     Confidence: {issue.confidence:.0%}")
            print() 