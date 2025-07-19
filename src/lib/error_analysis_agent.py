"""
Error Analysis Agent

Analyzes edit failures and provides intelligent recovery strategies.
Breaks the infinite loop pattern by:
1. Understanding WHY edits fail
2. Detecting file corruption patterns  
3. Providing adaptive solutions
4. Learning from failure patterns
5. Implementing progressive simplification
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class FailureType(Enum):
    """Types of edit failures we can detect and handle."""
    FUZZY_MATCH_FAILED = "fuzzy_match_failed"
    CONTEXT_NOT_FOUND = "context_not_found"
    FILE_CORRUPTED = "file_corrupted"
    SYNTAX_ERROR = "syntax_error"
    STRUCTURAL_MISMATCH = "structural_mismatch"
    LINE_COUNT_MISMATCH = "line_count_mismatch"
    DUPLICATE_CONTENT = "duplicate_content"


@dataclass
class FailureAnalysis:
    """Analysis of why an edit failed."""
    failure_type: FailureType
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    expected_context: List[str]
    actual_context: List[str]
    suggested_fix: str
    confidence: float  # 0.0 to 1.0
    file_path: str
    line_number: Optional[int] = None


@dataclass
class FileCorruption:
    """Details about detected file corruption."""
    corruption_type: str
    line_numbers: List[int]
    description: str
    severity: str
    can_auto_fix: bool
    suggested_action: str


class ErrorAnalysisAgent:
    """
    Intelligent error analysis and recovery agent.
    
    This agent:
    1. Analyzes why edits fail (instead of just retrying)
    2. Detects file corruption patterns
    3. Provides adaptive solutions based on actual file content
    4. Learns from failure patterns to avoid repeating mistakes
    5. Implements progressive simplification strategies
    """
    
    def __init__(self):
        self.failure_history: List[FailureAnalysis] = []
        self.corruption_patterns: Dict[str, int] = {}
        self.successful_adaptations: Dict[str, str] = {}
        
    def analyze_fuzzy_match_failure(self, expected_context: List[str], file_content: str, 
                                  file_path: str, expected_line: int) -> FailureAnalysis:
        """
        Analyze why fuzzy matching failed and suggest adaptive solutions.
        
        This is the core method that breaks the infinite retry loop.
        """
        print(f"🔬 Analyzing fuzzy match failure at line {expected_line} in {file_path}")
        
        file_lines = file_content.split('\n')
        
        # 1. Check if file is corrupted
        corruption = self._detect_file_corruption(file_content, file_path)
        if corruption.severity in ['high', 'critical']:
            return FailureAnalysis(
                failure_type=FailureType.FILE_CORRUPTED,
                severity=corruption.severity,
                description=f"File corruption detected: {corruption.description}",
                expected_context=expected_context,
                actual_context=self._get_actual_context(file_lines, expected_line),
                suggested_fix=corruption.suggested_action,
                confidence=0.9,
                file_path=file_path,
                line_number=expected_line
            )
        
        # 2. Analyze context mismatch
        actual_context = self._get_actual_context(file_lines, expected_line)
        similarity = self._calculate_context_similarity(expected_context, actual_context)
        
        # 3. Find the best alternative context
        alternative_matches = self._find_alternative_contexts(expected_context, file_lines)
        
        # 4. Determine failure type and suggest fix
        if similarity < 0.3:
            failure_type = FailureType.STRUCTURAL_MISMATCH
            suggested_fix = f"File structure has changed. Found {len(alternative_matches)} potential alternatives."
        elif similarity < 0.7:
            failure_type = FailureType.CONTEXT_NOT_FOUND
            suggested_fix = "Context partially matches but needs adaptation."
        else:
            failure_type = FailureType.FUZZY_MATCH_FAILED
            suggested_fix = "Context is similar - likely minor formatting differences."
        
        return FailureAnalysis(
            failure_type=failure_type,
            severity='medium' if alternative_matches else 'high',
            description=f"Expected context not found. Similarity: {similarity:.2f}",
            expected_context=expected_context,
            actual_context=actual_context,
            suggested_fix=suggested_fix,
            confidence=similarity,
            file_path=file_path,
            line_number=expected_line
        )
    
    def _detect_file_corruption(self, content: str, file_path: str) -> FileCorruption:
        """Detect various patterns of file corruption."""
        lines = content.split('\n')
        issues = []
        line_numbers = []
        
        # Check for duplicate elements (common corruption pattern)
        seen_lines = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and len(stripped) > 10:  # Skip empty/short lines
                if stripped in seen_lines:
                    issues.append(f"Duplicate line: '{stripped[:50]}...'")
                    line_numbers.extend([seen_lines[stripped], i])
                else:
                    seen_lines[stripped] = i
        
        # Check for orphaned/malformed JSX elements
        jsx_stack = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Simple JSX tag detection
            if '<' in stripped and '>' in stripped:
                # Check for orphaned opening tags
                if stripped.endswith('>') and not stripped.endswith('/>') and '=' not in stripped:
                    # This might be an orphaned opening tag
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if not next_line or next_line.startswith('<'):
                        issues.append(f"Potential orphaned JSX element: '{stripped}'")
                        line_numbers.append(i)
        
        # Check for malformed function/component structure
        function_starts = []
        for i, line in enumerate(lines):
            if 'export default function' in line or 'function' in line and '{' in line:
                function_starts.append(i)
        
        # Determine severity
        if len(issues) >= 3:
            severity = 'critical'
            action = "File needs complete reconstruction"
        elif len(issues) >= 1:
            severity = 'high'
            action = "File needs cleanup before edits"
        else:
            severity = 'low'
            action = "File appears clean"
        
        return FileCorruption(
            corruption_type='structural_duplication' if 'Duplicate' in str(issues) else 'jsx_malformation',
            line_numbers=line_numbers,
            description='; '.join(issues) if issues else 'No corruption detected',
            severity=severity,
            can_auto_fix=severity != 'critical',
            suggested_action=action
        )
    
    def _get_actual_context(self, file_lines: List[str], expected_line: int, context_size: int = 3) -> List[str]:
        """Get the actual context around a line number."""
        start = max(0, expected_line - context_size)
        end = min(len(file_lines), expected_line + context_size)
        return file_lines[start:end]
    
    def _calculate_context_similarity(self, expected: List[str], actual: List[str]) -> float:
        """Calculate similarity between expected and actual context."""
        if not expected or not actual:
            return 0.0
        
        # Normalize both contexts (remove extra whitespace, etc.)
        expected_norm = [self._normalize_line(line) for line in expected]
        actual_norm = [self._normalize_line(line) for line in actual]
        
        # Calculate line-by-line similarity
        matches = 0
        total_comparisons = 0
        
        for exp_line in expected_norm:
            if not exp_line:  # Skip empty lines
                continue
            best_match = 0
            for act_line in actual_norm:
                if not act_line:
                    continue
                similarity = self._line_similarity(exp_line, act_line)
                best_match = max(best_match, similarity)
            matches += best_match
            total_comparisons += 1
        
        return matches / total_comparisons if total_comparisons > 0 else 0.0
    
    def _normalize_line(self, line: str) -> str:
        """Normalize a line for comparison."""
        # Remove extra whitespace, normalize quotes, etc.
        normalized = re.sub(r'\s+', ' ', line.strip())
        normalized = normalized.replace('"', "'")  # Normalize quotes
        return normalized.lower()
    
    def _line_similarity(self, line1: str, line2: str) -> float:
        """Calculate similarity between two normalized lines."""
        if line1 == line2:
            return 1.0
        
        # Use set intersection for word-level similarity
        words1 = set(line1.split())
        words2 = set(line2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union
    
    def _find_alternative_contexts(self, expected_context: List[str], file_lines: List[str]) -> List[Tuple[int, float]]:
        """Find alternative locations where the context might match."""
        alternatives = []
        
        if not expected_context:
            return alternatives
        
        # Look for partial matches throughout the file
        for i in range(len(file_lines) - len(expected_context) + 1):
            actual_context = file_lines[i:i + len(expected_context)]
            similarity = self._calculate_context_similarity(expected_context, actual_context)
            
            if similarity > 0.5:  # Potential match
                alternatives.append((i, similarity))
        
        # Sort by similarity
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def suggest_adaptive_strategy(self, analysis: FailureAnalysis) -> Dict[str, any]:
        """
        Suggest an adaptive strategy based on failure analysis.
        
        This method provides specific, actionable strategies to break the loop.
        """
        strategy = {
            'approach': None,
            'confidence': analysis.confidence,
            'actions': [],
            'simplified_change': None
        }
        
        if analysis.failure_type == FailureType.FILE_CORRUPTED:
            strategy['approach'] = 'file_reconstruction'
            strategy['actions'] = [
                'Create clean version of file',
                'Apply changes to clean version',
                'Replace corrupted file'
            ]
            strategy['simplified_change'] = self._extract_core_change_intent(analysis)
        
        elif analysis.failure_type == FailureType.STRUCTURAL_MISMATCH:
            strategy['approach'] = 'progressive_application'
            strategy['actions'] = [
                'Break change into smaller parts',
                'Apply one small change at a time',
                'Verify each step before proceeding'
            ]
            strategy['simplified_change'] = self._simplify_change(analysis)
        
        elif analysis.failure_type == FailureType.CONTEXT_NOT_FOUND:
            strategy['approach'] = 'context_adaptation'
            strategy['actions'] = [
                'Use actual file context instead of expected',
                'Generate new diff based on real content',
                'Apply with relaxed matching'
            ]
            strategy['simplified_change'] = self._adapt_to_actual_context(analysis)
        
        else:
            strategy['approach'] = 'fallback_simplification'
            strategy['actions'] = [
                'Use line-based replacement',
                'Target specific functions only',
                'Avoid complex structural changes'
            ]
        
        return strategy
    
    def _extract_core_change_intent(self, analysis: FailureAnalysis) -> str:
        """Extract the core intent of what the change was trying to achieve."""
        # This would analyze the expected changes and extract the high-level intent
        # For now, return a simplified description
        return "Apply minimal changes to achieve the core functionality"
    
    def _simplify_change(self, analysis: FailureAnalysis) -> str:
        """Break down a complex change into simpler steps."""
        return "Break change into individual line modifications"
    
    def _adapt_to_actual_context(self, analysis: FailureAnalysis) -> str:
        """Adapt the change to work with actual file content."""
        return "Generate new diff based on actual file structure"
    
    def record_failure(self, analysis: FailureAnalysis):
        """Record a failure for learning and pattern detection."""
        self.failure_history.append(analysis)
        
        # Track patterns
        pattern_key = f"{analysis.failure_type}:{analysis.file_path}"
        self.corruption_patterns[pattern_key] = self.corruption_patterns.get(pattern_key, 0) + 1
        
        print(f"📊 Recorded failure: {analysis.failure_type} in {analysis.file_path}")
        if self.corruption_patterns[pattern_key] > 2:
            print(f"⚠️ Pattern detected: {pattern_key} has failed {self.corruption_patterns[pattern_key]} times")
    
    def should_emergency_reset(self, file_path: str) -> bool:
        """Determine if a file needs emergency reset due to repeated failures."""
        pattern_key = f"FILE_CORRUPTED:{file_path}"
        return self.corruption_patterns.get(pattern_key, 0) >= 2
    
    def get_failure_summary(self) -> Dict[str, any]:
        """Get a summary of failure patterns for debugging."""
        return {
            'total_failures': len(self.failure_history),
            'failure_types': {ft.value: sum(1 for f in self.failure_history if f.failure_type == ft) 
                            for ft in FailureType},
            'corruption_patterns': self.corruption_patterns,
            'most_problematic_files': sorted(
                [(k, v) for k, v in self.corruption_patterns.items()], 
                key=lambda x: x[1], reverse=True
            )[:5]
        }
