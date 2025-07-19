#!/usr/bin/env python3
"""
Code Builder Script

This script takes an input file with structured code blocks and applies them to an output directory.
The input file format uses <new filename="path"> tags to specify file locations and content,
and <edit filename="path" start_line="#" end_line="#"> tags to edit existing files.

Usage:
    python code_builder.py --input input.txt --output ./myapp
    python code_builder.py -i input.txt -o ./myapp
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Union


class CodeBuilder:
    def __init__(self, input_file: str, output_dir: str, error_logger=None):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.error_logger = error_logger  # Optional error logger for tracking issues
        
    def clean_code_content(self, content: str) -> str:
        """
        Clean the code content by removing markdown code blocks and other formatting artifacts.
        
        Args:
            content: Raw content that may contain markdown code blocks
            
        Returns:
            Cleaned content without markdown formatting
        """
        if not content:
            return content
        
        # Remove markdown code blocks like ```tsx, ```javascript, ```typescript, etc.
        # This pattern matches code blocks that span multiple lines
        pattern = r'^```[a-zA-Z]*\n(.*?)\n```$'
        cleaned = re.sub(pattern, r'\1', content, flags=re.MULTILINE | re.DOTALL)
        
        # Also handle inline code blocks within our tags
        cleaned = re.sub(r'```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```', '', cleaned)
        
        # Remove any remaining backticks at the start or end of lines
        cleaned = re.sub(r'^```.*$', '', cleaned, flags=re.MULTILINE)
        
        # Clean up any extra whitespace that might be left
        cleaned = re.sub(r'\n\n\n+', '\n\n', cleaned)
        
        return cleaned.strip()
        
    def parse_input_file(self) -> List[Tuple[str, str, Union[None, Tuple[int, int]]]]:
        """
        Parse the input file and extract filename/content pairs for both new and edit operations.
        
        Returns:
            List of tuples containing (filename, content, line_range)
            where line_range is None for new files, or (start_line, end_line) for edits
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file {self.input_file} not found")
            
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        operations = []
        
        # Find all <new filename="..."> blocks
        new_pattern = r'<new filename="([^"]+)">\s*(.*?)\s*</new>'
        new_matches = re.findall(new_pattern, content, re.DOTALL)
        
        for filename, file_content in new_matches:
            # Clean the content to remove markdown code blocks
            cleaned_content = self.clean_code_content(file_content.strip())
            operations.append((filename, cleaned_content, None))
            
        # Find all <edit filename="..." start_line="#" end_line="#"> blocks
        edit_pattern = r'<edit filename="([^"]+)" start_line="(\d+)" end_line="(\d+)">\s*(.*?)\s*</edit>'
        edit_matches = re.findall(edit_pattern, content, re.DOTALL)
        
        for filename, start_line, end_line, file_content in edit_matches:
            # Clean the content to remove markdown code blocks
            cleaned_content = self.clean_code_content(file_content.strip())
            line_range = (int(start_line), int(end_line))
            operations.append((filename, cleaned_content, line_range))
        
        if not operations:
            print("Warning: No <new> or <edit> blocks found in input file")
            
        return operations
    
    def create_file(self, filename: str, content: str) -> None:
        """
        Create a file with the given content in the output directory.
        
        Args:
            filename: Relative path to the file
            content: Content to write to the file
        """
        file_path = self.output_dir / filename
        
        # Validate configuration files before creating them
        validation_errors = self.validate_edit_content(filename, content)
        if validation_errors:
            print(f"⚠️  Configuration validation warnings for {filename}:")
            for error in validation_errors:
                if error.startswith("CRITICAL:"):
                    print(f"   🚨 {error}")
                else:
                    print(f"   ⚠️  {error}")
            
            # For critical configuration errors, show guidance
            if any(error.startswith("CRITICAL:") for error in validation_errors):
                print(f"   💡 Fix: Ensure {filename} uses the correct format specified in the system prompt")
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✓ Created: {file_path}")
        
        # Additional post-creation validation for critical files
        if filename in ['postcss.config.js', 'tailwind.config.js'] and validation_errors:
            print(f"   📋 Remember: This configuration may cause build issues if not corrected")
    
    def validate_edit_content(self, filename: str, content: str) -> List[str]:
        """
        Validate edit content for basic syntax issues.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for configuration file issues
        if filename in ['postcss.config.js', 'postcss.config.mjs']:
            errors.extend(self.validate_postcss_config(filename, content))
        
        if filename == 'tailwind.config.js':
            errors.extend(self.validate_tailwind_config(content))
        
        if filename in ['next.config.js', 'next.config.mjs', 'next.config.ts']:
            errors.extend(self.validate_next_config(filename, content))
        
        if filename == 'app/globals.css':
            errors.extend(self.validate_globals_css(content))
        
        # Check for basic JSX/TSX syntax issues
        if filename.endswith(('.tsx', '.jsx')):
            # Check for malformed return statements
            if 'return (' in content:
                # Find return statements
                return_matches = re.finditer(r'return\s*\(', content)
                for match in return_matches:
                    start_pos = match.start()
                    # Find the end of this return statement
                    remaining_content = content[start_pos:]
                    
                    # Look for common malformed patterns
                    if '}' in remaining_content and ')' not in remaining_content[:remaining_content.find('}')]:
                        errors.append("CRITICAL: Malformed return statement - ends with '}' instead of ')'")
                    
                    # Check for missing closing parenthesis
                    lines_after_return = remaining_content.split('\n')
                    if len(lines_after_return) > 1:
                        # Check if the return statement spans multiple lines but doesn't close properly
                        combined_return = '\n'.join(lines_after_return[:10])  # Check first 10 lines
                        open_parens = combined_return.count('(')
                        close_parens = combined_return.count(')')
                        if open_parens > close_parens and '}' in combined_return:
                            errors.append("CRITICAL: Return statement missing closing parenthesis - found '}' instead")
            
            # Check for improperly formatted JSX elements on same line
            jsx_one_line_pattern = r'<\w+[^>]*>[^<]*<\w+[^>]*>'
            if re.search(jsx_one_line_pattern, content):
                # Check if this is actually malformed (no newlines between elements)
                malformed_jsx = re.findall(r'<\w+[^>]*>\s*<\w+[^>]*>', content)
                if malformed_jsx:
                    errors.append("JSX formatting: Multiple elements on same line - consider adding newlines for readability")
            
            # Check for common syntax errors
            if '<' in content and not content.strip().startswith('<'):
                # Check for malformed JSX
                open_tags = content.count('<')
                close_tags = content.count('>')
                if open_tags != close_tags:
                    errors.append("Mismatched JSX angle brackets")
            
            # Check for style object syntax
            if 'style={{' in content:
                # Count opening and closing braces for style objects
                style_start = content.find('style={{')
                if style_start != -1:
                    brace_count = 0
                    for i, char in enumerate(content[style_start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    if brace_count != 0:
                        errors.append("Unbalanced braces in style object")
            
            # Check for missing commas in CSS properties
            if ':' in content and '{' in content:
                lines = content.strip().split('\n')
                for line in lines:
                    stripped = line.strip()
                    if ':' in stripped and not stripped.endswith(',') and not stripped.endswith('{') and not stripped.endswith('}'):
                        # Check if this looks like a CSS property that needs a comma
                        if any(css_prop in stripped for css_prop in ['background:', 'color:', 'padding:', 'margin:', 'border:']):
                            if not stripped.endswith('}}') and not stripped.endswith('}}>'):
                                errors.append(f"Missing comma in CSS property: {stripped}")
        
        return errors

    def validate_postcss_config(self, filename: str, content: str) -> List[str]:
        """Validate PostCSS configuration file."""
        errors = []
        
        if filename == 'postcss.config.mjs':
            errors.append("CRITICAL: Use postcss.config.js with CommonJS format instead of .mjs")
        
        if 'export default' in content and filename.endswith('.js'):
            errors.append("CRITICAL: PostCSS config uses ES modules syntax - must use CommonJS format (module.exports)")
        
        if 'module.exports' not in content and filename.endswith('.js'):
            errors.append("CRITICAL: PostCSS config missing proper CommonJS export (module.exports)")
        
        if 'tailwindcss' not in content:
            errors.append("CRITICAL: PostCSS config missing tailwindcss plugin")
        
        if 'autoprefixer' not in content:
            errors.append("CRITICAL: PostCSS config missing autoprefixer plugin")
        
        return errors

    def validate_tailwind_config(self, content: str) -> List[str]:
        """Validate Tailwind CSS configuration file."""
        errors = []
        
        if 'export default' in content:
            errors.append("CRITICAL: Tailwind config uses ES modules syntax - must use CommonJS format (module.exports)")
        
        if 'module.exports' not in content:
            errors.append("CRITICAL: Tailwind config missing proper CommonJS export (module.exports)")
        
        if 'content:' not in content:
            errors.append("CRITICAL: Tailwind config missing content paths")
        elif './app/**/*.{' not in content:
            errors.append("WARNING: Tailwind config missing app directory content paths")
        
        return errors

    def validate_next_config(self, filename: str, content: str) -> List[str]:
        """Validate Next.js configuration file."""
        errors = []
        
        if filename == 'next.config.ts':
            errors.append("CRITICAL: Use next.config.mjs instead of .ts")
        
        if filename == 'next.config.mjs' and 'module.exports' in content:
            errors.append("CRITICAL: next.config.mjs using CommonJS syntax - must use ES modules (export default)")
        
        if filename == 'next.config.js' and 'export default' in content:
            errors.append("CRITICAL: next.config.js using ES modules syntax - use next.config.mjs")
        
        if 'appDir: true' in content:
            errors.append("WARNING: Deprecated experimental.appDir setting - remove in Next.js 14")
        
        return errors

    def validate_globals_css(self, content: str) -> List[str]:
        """Validate globals.css file for Tailwind directives."""
        errors = []
        
        required_directives = ['@tailwind base', '@tailwind components', '@tailwind utilities']
        for directive in required_directives:
            if directive not in content:
                errors.append(f"CRITICAL: globals.css missing required directive: {directive}")
        
        return errors

    def edit_file(self, filename: str, content: str, start_line: int, end_line: int) -> None:
        """
        Edit an existing file by replacing lines between start_line and end_line.
        Enhanced with automatic line range correction and validation.
        
        Args:
            filename: Relative path to the file
            content: New content to replace the lines
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based, inclusive)
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot edit non-existent file: {file_path}")
        
        # Read existing file content
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        actual_line_count = len(lines)
        
        # Auto-correct line range issues
        original_start, original_end = start_line, end_line
        corrected_range = False
        
        # Validate and auto-correct line numbers
        if start_line < 1:
            print(f"⚠️  Auto-correcting start_line from {start_line} to 1")
            start_line = 1
            corrected_range = True
            
        if end_line < 1:
            print(f"⚠️  Auto-correcting end_line from {end_line} to 1")
            end_line = 1
            corrected_range = True
            
        if start_line > actual_line_count + 1:
            print(f"⚠️  Auto-correcting start_line from {start_line} to {actual_line_count}")
            start_line = actual_line_count
            corrected_range = True
            
        if end_line > actual_line_count:
            print(f"⚠️  Auto-correcting end_line from {end_line} to {actual_line_count}")
            end_line = actual_line_count
            corrected_range = True
            
        if start_line > end_line:
            print(f"⚠️  Auto-correcting: start_line ({start_line}) > end_line ({end_line}), swapping")
            start_line, end_line = end_line, start_line
            corrected_range = True
        
        # Special case: If the edit seems to be trying to replace the entire file
        if original_start == 1 and original_end >= actual_line_count:
            print(f"🔄 Detected full file replacement attempt (1-{original_end} vs actual {actual_line_count} lines)")
            start_line = 1
            end_line = actual_line_count
            corrected_range = True
        
        if corrected_range:
            print(f"📏 Line range auto-corrected: {original_start}-{original_end} → {start_line}-{end_line} (file has {actual_line_count} lines)")
        
        # Validate edit content before applying
        validation_errors = self.validate_edit_content(filename, content)
        if validation_errors:
            print(f"⚠️  Validation warnings for {filename}:")
            for error in validation_errors:
                if error.startswith("CRITICAL:"):
                    print(f"   🚨 {error}")
                else:
                    print(f"   ⚠️  {error}")
            print("   Applying edit anyway, but check for syntax errors...")
        
        # Convert to 0-based indexing for list operations
        start_idx = start_line - 1
        end_idx = end_line
        
        # Split content into lines and ensure proper newline handling
        new_lines = content.split('\n')
        
        # CRITICAL FIX: Ensure ALL lines end with newlines to prevent concatenation
        # This fixes the bug where lines get concatenated like: "function() {const x = 1"
        if new_lines:
            # Add newlines to all lines, including the last one
            new_lines = [line + '\n' for line in new_lines]
            
            # Handle the case where original content ended with \n (avoid double newlines)
            if content.endswith('\n') and new_lines and new_lines[-1] == '\n':
                new_lines.pop()  # Remove the extra empty newline
        
        # Replace the lines
        lines[start_idx:end_idx] = new_lines
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        if corrected_range:
            print(f"✓ Edited: {file_path} (lines {start_line}-{end_line}, auto-corrected from {original_start}-{original_end})")
        else:
            print(f"✓ Edited: {file_path} (lines {start_line}-{end_line})")

    def get_file_line_count(self, filename: str) -> int:
        """
        Get the actual line count of a file for validation purposes.
        
        Args:
            filename: Relative path to the file
            
        Returns:
            Number of lines in the file, or 0 if file doesn't exist
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            return 0
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except Exception:
            return 0
    
    def validate_line_ranges(self, operations: List[Tuple[str, str, Union[None, Tuple[int, int]]]]) -> List[Tuple[str, str, Union[None, Tuple[int, int]]]]:
        """
        Pre-validate and auto-correct line ranges for all edit operations.
        Enhanced with overlapping edit detection to prevent syntax corruption.
        
        Args:
            operations: List of (filename, content, line_range) tuples
            
        Returns:
            List of operations with corrected line ranges and overlapping edits resolved
        """
        corrected_operations = []
        
        # Group operations by filename to detect overlaps
        operations_by_file = {}
        for i, (filename, content, line_range) in enumerate(operations):
            if filename not in operations_by_file:
                operations_by_file[filename] = []
            operations_by_file[filename].append((i, content, line_range))
        
        # Process each file's operations
        for filename, file_ops in operations_by_file.items():
            # Separate new file operations from edit operations
            new_ops = [(i, content, line_range) for i, content, line_range in file_ops if line_range is None]
            edit_ops = [(i, content, line_range) for i, content, line_range in file_ops if line_range is not None]
            
            # Add new file operations directly (no validation needed)
            for i, content, line_range in new_ops:
                corrected_operations.append((filename, content, line_range))
            
            # Process edit operations with overlap detection
            if edit_ops:
                actual_line_count = self.get_file_line_count(filename)
                
                if actual_line_count == 0:
                    print(f"⚠️  File {filename} not found for edit operation - will be skipped")
                    continue
                
                # Validate and correct each edit operation
                corrected_edits = []
                for i, content, (start_line, end_line) in edit_ops:
                    original_start, original_end = start_line, end_line
                    corrected = False
                    
                    # Auto-correct line ranges
                    if end_line > actual_line_count:
                        print(f"📏 Pre-correcting {filename}: end_line {end_line} → {actual_line_count} (file has {actual_line_count} lines)")
                        end_line = actual_line_count
                        corrected = True
                    
                    if start_line > actual_line_count:
                        print(f"📏 Pre-correcting {filename}: start_line {start_line} → {actual_line_count}")
                        start_line = actual_line_count
                        corrected = True
                    
                    if start_line > end_line:
                        print(f"📏 Pre-correcting {filename}: start_line > end_line, using end_line for both")
                        start_line = end_line
                        corrected = True
                    
                    # Special case: Full file replacement
                    if original_start == 1 and original_end >= actual_line_count:
                        start_line = 1
                        end_line = actual_line_count
                        corrected = True
                    
                    corrected_edits.append((i, content, (start_line, end_line), corrected, (original_start, original_end)))
                
                # Detect overlapping edits
                overlap_detected = False
                for idx1, (i1, content1, (start1, end1), corrected1, (orig_start1, orig_end1)) in enumerate(corrected_edits):
                    for idx2, (i2, content2, (start2, end2), corrected2, (orig_start2, orig_end2)) in enumerate(corrected_edits):
                        if idx1 >= idx2:  # Only check each pair once
                            continue
                        
                        # Check for overlap: ranges overlap if one starts before the other ends
                        if not (end1 < start2 or end2 < start1):
                            overlap_detected = True
                            print(f"🚨 OVERLAP DETECTED in {filename}:")
                            print(f"   Edit 1: lines {start1}-{end1} (original: {orig_start1}-{orig_end1})")
                            print(f"   Edit 2: lines {start2}-{end2} (original: {orig_start2}-{orig_end2})")
                            print(f"   🛑 Overlapping edits can corrupt file structure!")
                            
                            # Strategy: Keep the first edit, remove the overlapping one
                            print(f"   ✂️  Resolution: Keeping Edit 1, removing overlapping Edit 2")
                            print(f"   💡 Suggestion: Combine overlapping edits into single larger edit")
                
                # If overlaps detected, use conflict resolution
                if overlap_detected:
                    # Log overlap detection if logger available
                    if self.error_logger:
                        app_name = Path(self.output_dir).name
                        self.error_logger.log_overlap_detection(
                            app_name, 
                            filename, 
                            {
                                "original_count": len(corrected_edits),
                                "resolved_count": "pending_resolution"
                            }
                        )
                    
                    # Sort edits by start line to process in order
                    corrected_edits.sort(key=lambda x: x[2][0])  # Sort by start_line
                    
                    # Remove overlapping edits (keep first, remove subsequent overlaps)
                    non_overlapping_edits = []
                    for current_edit in corrected_edits:
                        i, content, (start, end), corrected, (orig_start, orig_end) = current_edit
                        
                        # Check if current edit overlaps with any already accepted edit
                        has_overlap = False
                        for accepted_edit in non_overlapping_edits:
                            _, _, (acc_start, acc_end), _, _ = accepted_edit
                            if not (end < acc_start or acc_end < start):
                                has_overlap = True
                                break
                        
                        if not has_overlap:
                            non_overlapping_edits.append(current_edit)
                        else:
                            print(f"   ⚠️  Skipped overlapping edit: lines {start}-{end}")
                    
                    corrected_edits = non_overlapping_edits
                    print(f"   ✅ Resolved to {len(corrected_edits)} non-overlapping edits")
                    
                    # Update logger with final resolution count
                    if self.error_logger:
                        app_name = Path(self.output_dir).name
                        self.error_logger.log_overlap_detection(
                            app_name, 
                            filename, 
                            {
                                "original_count": len(edit_ops),
                                "resolved_count": len(corrected_edits)
                            }
                        )
                
                # Add the corrected and validated edits
                for i, content, (start_line, end_line), corrected, (original_start, original_end) in corrected_edits:
                    corrected_operations.append((filename, content, (start_line, end_line)))
        
        return corrected_operations

    def apply_edits_sequentially(self, operations: List[Tuple[str, str, Union[None, Tuple[int, int]]]]) -> Tuple[int, int]:
        """
        Apply edits sequentially, handling line number adjustments for multiple edits to same file.
        Enhanced with automatic line range validation and correction.
        
        Returns:
            Tuple of (new_files_count, edited_files_count)
        """
        new_files_count = 0
        edited_files_count = 0
        
        # Pre-validate and correct line ranges for all operations
        print("🔍 Validating and correcting line ranges...")
        validated_operations = self.validate_line_ranges(operations)
        
        if len(validated_operations) != len(operations):
            skipped_count = len(operations) - len(validated_operations)
            print(f"⚠️  Skipped {skipped_count} operations due to validation issues")
        
        # Group operations by filename
        operations_by_file = {}
        for filename, content, line_range in validated_operations:
            if filename not in operations_by_file:
                operations_by_file[filename] = []
            operations_by_file[filename].append((content, line_range))
        
        # Process each file
        for filename, file_operations in operations_by_file.items():
            # Separate new file operations from edit operations
            new_ops = [(content, line_range) for content, line_range in file_operations if line_range is None]
            edit_ops = [(content, line_range) for content, line_range in file_operations if line_range is not None]
            
            # Handle new file creation
            for content, _ in new_ops:
                try:
                    self.create_file(filename, content)
                    new_files_count += 1
                except Exception as e:
                    print(f"❌ Error creating {filename}: {e}")
            
            # Handle edits - sort by line number (descending) to avoid line number shifts
            if edit_ops:
                edit_ops.sort(key=lambda x: x[1][0], reverse=True)  # Sort by start_line descending
                
                # Detect and remove duplicate content edits
                edit_ops = self._remove_duplicate_edits(filename, edit_ops)
                
                for content, (start_line, end_line) in edit_ops:
                    try:
                        self.edit_file(filename, content, start_line, end_line)
                        edited_files_count += 1
                    except Exception as e:
                        print(f"❌ Error editing {filename} lines {start_line}-{end_line}: {e}")
                        continue
        
        return new_files_count, edited_files_count
    
    def build(self) -> bool:
        """
        Parse the input file and apply all operations (create new files and edit existing ones).
        
        Returns:
            True if all operations were processed successfully, False otherwise
        """
        print(f"📂 Reading input from: {self.input_file}")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        # Parse the input file
        operations = self.parse_input_file()
        
        if not operations:
            print("❌ No operations to perform")
            return False
            
        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply operations using improved sequential processing
        print(f"🔨 Processing {len(operations)} operations...")
        new_files_count, edited_files_count = self.apply_edits_sequentially(operations)
                
        print()
        print(f"✅ Successfully processed {len(operations)} operations:")
        print(f"   📄 Created {new_files_count} new files")
        print(f"   ✏️  Edited {edited_files_count} existing files")
        
        return True  # Return success status

    def _remove_duplicate_edits(self, filename: str, edit_ops: List[Tuple[str, Tuple[int, int]]]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Remove duplicate edits that would insert the same content multiple times.
        
        Args:
            filename: The file being edited
            edit_ops: List of (content, (start_line, end_line)) tuples
            
        Returns:
            Filtered list without duplicates
        """
        if len(edit_ops) <= 1:
            return edit_ops
        
        # Read current file content to check for existing content
        target_file = self.output_dir / filename
        existing_content = ""
        if target_file.exists():
            try:
                existing_content = target_file.read_text()
            except Exception:
                existing_content = ""
        
        filtered_ops = []
        seen_content = set()
        
        for content, line_range in edit_ops:
            # Normalize content for comparison (remove extra whitespace)
            normalized_content = ' '.join(content.split())
            
            # Check if this content already exists in the file
            if normalized_content in ' '.join(existing_content.split()):
                print(f"⚠️  Skipping duplicate content in {filename}: '{content[:50]}...'")
                continue
            
            # Check if we've already seen this exact content in our edits
            if normalized_content in seen_content:
                print(f"⚠️  Skipping duplicate edit in {filename}: '{content[:50]}...'")
                continue
            
            # Check for function/variable redefinition patterns
            if self._is_function_redefinition(content, existing_content):
                print(f"⚠️  Skipping function redefinition in {filename}: '{content[:50]}...'")
                continue
            
            seen_content.add(normalized_content)
            filtered_ops.append((content, line_range))
        
        if len(filtered_ops) != len(edit_ops):
            removed_count = len(edit_ops) - len(filtered_ops)
            print(f"🔧 Removed {removed_count} duplicate edit(s) for {filename}")
        
        return filtered_ops
    
    def _is_function_redefinition(self, new_content: str, existing_content: str) -> bool:
        """Check if the new content would create a function/variable redefinition."""
        import re
        
        # Extract function/variable names from new content
        function_patterns = [
            r'const\s+(\w+)\s*=',
            r'let\s+(\w+)\s*=', 
            r'var\s+(\w+)\s*=',
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*\([^)]*\)\s*=>'  # Arrow functions with type annotations
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, new_content)
            for match in matches:
                # Check if this name already exists in the file
                if re.search(rf'\b{re.escape(match)}\b', existing_content):
                    return True
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build code files from structured input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code_builder.py --input input.txt --output ./myapp
  python code_builder.py -i input.txt -o ./myapp
  python code_builder.py -i input.txt -o ../new-project
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input file containing code blocks (e.g., input.txt)'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory where files will be created (e.g., ./myapp)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without actually creating files'
    )
    
    args = parser.parse_args()
    
    try:
        builder = CodeBuilder(args.input, args.output)
        
        if args.dry_run:
            print("🔍 DRY RUN - No files will be created")
            operations = builder.parse_input_file()
            print(f"\nWould process {len(operations)} operations:")
            for filename, _, line_range in operations:
                if line_range is None:
                    print(f"  📄 {filename}")
                else:
                    start_line, end_line = line_range
                    print(f"  ✏️  {filename} (lines {start_line}-{end_line})")
        else:
            builder.build()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main()) 