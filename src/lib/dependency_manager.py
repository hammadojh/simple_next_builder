#!/usr/bin/env python3
"""
Dependency Manager

This module automatically detects required dependencies from generated code
and adds them to package.json, then installs them.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class DependencyInfo:
    """Information about a dependency that needs to be installed."""
    name: str
    version: str
    is_dev_dependency: bool = False
    reason: str = ""  # Why this dependency is needed


class DependencyManager:
    """
    Automatically detects and manages dependencies for NextJS apps.
    
    Analyzes generated code to find import statements and automatically
    adds required packages to package.json and installs them.
    """
    
    def __init__(self, app_directory: str):
        self.app_directory = Path(app_directory)
        self.package_json_path = self.app_directory / "package.json"
        
        # Common frontend-only dependency mappings
        self.dependency_map = {
            # Styling and UI
            'styled-components': DependencyInfo('styled-components', '^6.1.0', False, 'Styled components for CSS-in-JS'),
            '@emotion/react': DependencyInfo('@emotion/react', '^11.11.0', False, 'Emotion CSS-in-JS library'),
            '@emotion/styled': DependencyInfo('@emotion/styled', '^11.11.0', False, 'Emotion styled components'),
            'framer-motion': DependencyInfo('framer-motion', '^10.16.0', False, 'Animation library'),
            'react-spring': DependencyInfo('react-spring', '^9.7.0', False, 'Spring animations'),
            
            # State Management
            'zustand': DependencyInfo('zustand', '^4.4.0', False, 'Lightweight state management'),
            'jotai': DependencyInfo('jotai', '^2.6.0', False, 'Atomic state management'),
            'react-query': DependencyInfo('@tanstack/react-query', '^5.0.0', False, 'Data fetching library'),
            '@tanstack/react-query': DependencyInfo('@tanstack/react-query', '^5.0.0', False, 'Data fetching library'),
            
            # Utilities
            'lodash': DependencyInfo('lodash', '^4.17.0', False, 'Utility library'),
            'date-fns': DependencyInfo('date-fns', '^2.30.0', False, 'Date utility library'),
            'clsx': DependencyInfo('clsx', '^2.0.0', False, 'Conditional CSS classes'),
            'classnames': DependencyInfo('classnames', '^2.3.0', False, 'Conditional CSS classes'),
            
            # Icons and Assets
            'lucide-react': DependencyInfo('lucide-react', '^0.294.0', False, 'Icon library'),
            'react-icons': DependencyInfo('react-icons', '^4.12.0', False, 'Icon library'),
            'heroicons': DependencyInfo('@heroicons/react', '^2.0.0', False, 'Heroicons icon library'),
            '@heroicons/react': DependencyInfo('@heroicons/react', '^2.0.0', False, 'Heroicons icon library'),
            
            # UI Libraries
            'react-dnd': DependencyInfo('react-dnd', '^16.0.0', False, 'Drag and drop library'),
            'react-beautiful-dnd': DependencyInfo('react-beautiful-dnd', '^13.1.0', False, 'Beautiful drag and drop'),
            'react-sortable-hoc': DependencyInfo('react-sortable-hoc', '^2.0.0', False, 'Sortable higher-order components'),
            'react-modal': DependencyInfo('react-modal', '^3.16.0', False, 'Modal components'),
            'react-select': DependencyInfo('react-select', '^5.8.0', False, 'Select components'),
            'react-datepicker': DependencyInfo('react-datepicker', '^4.25.0', False, 'Date picker component'),
            
            # Charts and Visualization
            'recharts': DependencyInfo('recharts', '^2.8.0', False, 'Chart library'),
            'chart.js': DependencyInfo('chart.js', '^4.4.0', False, 'Chart library'),
            'react-chartjs-2': DependencyInfo('react-chartjs-2', '^5.2.0', False, 'React wrapper for Chart.js'),
            
            # Forms
            'react-hook-form': DependencyInfo('react-hook-form', '^7.48.0', False, 'Form library'),
            'formik': DependencyInfo('formik', '^2.4.0', False, 'Form library'),
            'yup': DependencyInfo('yup', '^1.4.0', False, 'Schema validation'),
            'zod': DependencyInfo('zod', '^3.22.0', False, 'Schema validation'),
            
            # Dev dependencies
            '@types/lodash': DependencyInfo('@types/lodash', '^4.14.0', True, 'TypeScript types for lodash'),
            '@types/react-modal': DependencyInfo('@types/react-modal', '^3.16.0', True, 'TypeScript types for react-modal'),
            '@types/react-datepicker': DependencyInfo('@types/react-datepicker', '^4.25.0', True, 'TypeScript types for react-datepicker'),
            '@types/react-beautiful-dnd': DependencyInfo('@types/react-beautiful-dnd', '^13.1.0', True, 'TypeScript types for react-beautiful-dnd'),
        }
        
        print(f"🔍 Dependency Manager initialized for: {self.app_directory}")
    
    def analyze_code_dependencies(self) -> Set[str]:
        """
        Analyze all code files to detect required dependencies.
        
        Returns:
            Set of dependency names that need to be installed
        """
        required_deps = set()
        
        # File patterns to analyze
        patterns = ['**/*.tsx', '**/*.ts', '**/*.jsx', '**/*.js']
        
        for pattern in patterns:
            for file_path in self.app_directory.glob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        deps = self._extract_imports_from_content(content)
                        required_deps.update(deps)
                except Exception as e:
                    print(f"⚠️ Could not analyze {file_path}: {e}")
        
        print(f"🔍 Found {len(required_deps)} potential dependencies in code")
        return required_deps
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_dirs = {'node_modules', '.next', 'dist', 'build', '.git'}
        return any(part in skip_dirs for part in file_path.parts)
    
    def _extract_imports_from_content(self, content: str) -> Set[str]:
        """Extract import statements from file content."""
        imports = set()
        
        # Pattern for import statements
        import_patterns = [
            r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",  # import ... from 'package'
            r"import\s+['\"]([^'\"]+)['\"]",  # import 'package'
            r"require\(['\"]([^'\"]+)['\"]\)",  # require('package')
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Extract package name (ignore relative imports)
                if not match.startswith('.') and not match.startswith('/'):
                    # Handle scoped packages (@scope/package) and sub-imports (package/submodule)
                    if match.startswith('@'):
                        parts = match.split('/')
                        if len(parts) >= 2:
                            package_name = f"{parts[0]}/{parts[1]}"
                        else:
                            package_name = parts[0]
                    else:
                        package_name = match.split('/')[0]
                    
                    imports.add(package_name)
        
        return imports
    
    def get_current_dependencies(self) -> Dict[str, str]:
        """Get currently installed dependencies from package.json."""
        if not self.package_json_path.exists():
            return {}
        
        try:
            with open(self.package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            deps = {}
            deps.update(package_data.get('dependencies', {}))
            deps.update(package_data.get('devDependencies', {}))
            
            return deps
        except Exception as e:
            print(f"⚠️ Could not read package.json: {e}")
            return {}
    
    def get_missing_dependencies(self, required_deps: Set[str]) -> List[DependencyInfo]:
        """
        Compare required dependencies with installed ones.
        
        Args:
            required_deps: Set of required dependency names
            
        Returns:
            List of DependencyInfo for missing dependencies
        """
        current_deps = self.get_current_dependencies()
        missing_deps = []
        
        for dep_name in required_deps:
            if dep_name in self.dependency_map and dep_name not in current_deps:
                missing_deps.append(self.dependency_map[dep_name])
                
                # Automatically add TypeScript types if available
                # Check for exact @types package names we have defined
                types_package = f"@types/{dep_name}"
                if types_package in self.dependency_map and types_package not in current_deps:
                    missing_deps.append(self.dependency_map[types_package])
        
        return missing_deps
    
    def add_dependencies_to_package_json(self, dependencies: List[DependencyInfo]) -> bool:
        """
        Add missing dependencies to package.json.
        
        Args:
            dependencies: List of DependencyInfo to add
            
        Returns:
            True if successful, False otherwise
        """
        if not dependencies:
            return True
        
        try:
            # Read current package.json
            with open(self.package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            # Add dependencies
            for dep in dependencies:
                if dep.is_dev_dependency:
                    if 'devDependencies' not in package_data:
                        package_data['devDependencies'] = {}
                    package_data['devDependencies'][dep.name] = dep.version
                else:
                    if 'dependencies' not in package_data:
                        package_data['dependencies'] = {}
                    package_data['dependencies'][dep.name] = dep.version
                
                print(f"📦 Adding {'dev ' if dep.is_dev_dependency else ''}dependency: {dep.name}@{dep.version}")
            
            # Write updated package.json
            with open(self.package_json_path, 'w', encoding='utf-8') as f:
                json.dump(package_data, f, indent=2)
            
            print(f"✅ Added {len(dependencies)} dependencies to package.json")
            return True
            
        except Exception as e:
            print(f"❌ Error updating package.json: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """
        Run npm install to install dependencies.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("📦 Installing dependencies...")
            result = subprocess.run(
                ['npm', 'install'],
                cwd=self.app_directory,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ npm install failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ npm install timed out")
            return False
        except Exception as e:
            print(f"❌ Error running npm install: {e}")
            return False
    
    def auto_manage_dependencies(self) -> bool:
        """
        Complete workflow: analyze, detect, add, and install dependencies.
        
        Returns:
            True if successful, False otherwise
        """
        print("🔍 Starting automatic dependency management...")
        
        # Step 1: Analyze code for required dependencies
        required_deps = self.analyze_code_dependencies()
        
        if not required_deps:
            print("✅ No external dependencies detected")
            return True
        
        # Step 2: Find missing dependencies
        missing_deps = self.get_missing_dependencies(required_deps)
        
        if not missing_deps:
            print("✅ All required dependencies are already installed")
            return True
        
        print(f"📦 Found {len(missing_deps)} missing dependencies:")
        for dep in missing_deps:
            print(f"  • {dep.name}@{dep.version} - {dep.reason}")
        
        # Step 3: Add to package.json
        if not self.add_dependencies_to_package_json(missing_deps):
            return False
        
        # Step 4: Install dependencies
        return self.install_dependencies() 