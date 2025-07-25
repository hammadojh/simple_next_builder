#!/usr/bin/env python3
"""
NextJS App Builder & Editor
A Python script that creates and edits NextJS applications using AI assistance.
"""

import os
import sys
import time
import json
import subprocess
import argparse
import logging
import re
import shutil
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    # dotenv not installed, continue without it
    pass

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.app_builder import MultiLLMAppBuilder
from lib.code_builder import CodeBuilder


class BuildErrorLogger:
    """Handles logging of build errors and fix attempts to a file."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.log_file = self.project_root / "build_errors.log"
        
        # Setup logging
        self.logger = logging.getLogger('BuildErrorLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler if it doesn't exist
        if not hasattr(self.logger, 'handler_set'):
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.handler_set = True
        
        # Log initialization
        self.log_info("Build Error Logging System Initialized", {
            "log_file": str(self.log_file),
            "project_root": str(self.project_root)
        })
    
    def log_info(self, message: str, data: dict = None):
        """Log an info message with optional structured data."""
        if data:
            formatted_data = " | ".join([f"{k}: {v}" for k, v in data.items()])
            self.logger.info(f"{message} | {formatted_data}")
        else:
            self.logger.info(message)
    
    def log_build_attempt(self, app_name: str, operation: str, success: bool = None):
        """Log a build attempt."""
        status = "SUCCESS" if success else "FAILED" if success is False else "STARTED"
        self.log_info(f"BUILD ATTEMPT {status}", {
            "app": app_name,
            "operation": operation,
            "status": status
        })
    
    def log_build_errors(self, app_name: str, errors: List[str], attempt_number: int = 1):
        """Log build errors with details."""
        self.log_info(f"BUILD ERRORS DETECTED", {
            "app": app_name,
            "attempt": attempt_number,
            "error_count": len(errors)
        })
        
        # Log each error on a separate line for clarity
        for i, error in enumerate(errors, 1):
            # Clean up error for logging (remove newlines, limit length)
            cleaned_error = error.replace('\n', ' | ').replace('\r', '')[:500]
            if len(error) > 500:
                cleaned_error += "..."
            
            self.log_info(f"  ERROR {i}/{len(errors)}", {"detail": cleaned_error})
    
    def log_fix_attempt(self, app_name: str, fix_type: str, files_affected: List[str] = None):
        """Log a fix attempt."""
        data = {
            "app": app_name,
            "fix_type": fix_type
        }
        
        if files_affected:
            data["files"] = ", ".join(files_affected)
        
        self.log_info("FIX ATTEMPT", data)
    
    def log_overlap_detection(self, app_name: str, filename: str, overlap_details: dict):
        """Log when overlapping edits are detected and resolved."""
        self.log_info("OVERLAP DETECTED & RESOLVED", {
            "app": app_name,
            "file": filename,
            "original_edits": overlap_details.get("original_count", "unknown"),
            "resolved_edits": overlap_details.get("resolved_count", "unknown")
        })
    
    def log_session_summary(self, app_name: str, total_attempts: int, success: bool, elapsed_time: float):
        """Log a summary of the entire build/fix session."""
        self.log_info("SESSION SUMMARY", {
            "app": app_name,
            "total_attempts": total_attempts,
            "final_result": "SUCCESS" if success else "FAILED",
            "elapsed_time_seconds": round(elapsed_time, 1)
        })
        
        # Add separator for readability
        self.logger.info("-" * 80)


class MasterBuilder:
    def __init__(self):
        """Initialize the master builder with all components."""
        # Get current working directory as project root
        self.project_root = Path.cwd()
        self.apps_dir = self.project_root / "apps"
        self.inputs_dir = self.project_root / "inputs"
        
        # Initialize error logger
        self.error_logger = BuildErrorLogger(str(self.project_root))
        
        # Initialize the AI app builder
        self.app_builder = MultiLLMAppBuilder()
        
        print(f"🏗️  Master Builder initialized in: {self.project_root}")
        print(f"📁 Apps directory: {self.apps_dir}")
        print(f"📄 Inputs directory: {self.inputs_dir}")
        print(f"📋 Build error log: {self.error_logger.log_file}")
        
        # Create directories if they don't exist
        self.apps_dir.mkdir(exist_ok=True)
        self.inputs_dir.mkdir(exist_ok=True)
        
    def get_next_app_name(self, base_name: str = "myapp") -> str:
        """Get the next available app name (myapp4, myapp5, etc.)."""
        version = 1
        while (self.apps_dir / f"{base_name}{version}").exists():
            version += 1
        return f"{base_name}{version}"
    
    def find_available_port(self, start_port: int = 3000) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + 100):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                return port
            except OSError:
                continue
        raise Exception(f"No available ports found in range {start_port}-{start_port + 99}")
    
    def create_template_nextjs_app(self, app_name: str) -> bool:
        """Copy the NextJS template from nextjs_temp directory."""
        print(f"🏗️  Creating NextJS app from template: {app_name}")
        
        try:
            # Define paths
            template_dir = self.apps_dir / "nextjs_temp"
            app_dir = self.apps_dir / app_name
            
            # Check if template exists
            if not template_dir.exists():
                print(f"❌ Template directory 'nextjs_temp' not found in {self.apps_dir}")
                return False
            
            # Check if target directory already exists
            if app_dir.exists():
                print(f"❌ Directory '{app_name}' already exists")
                return False
            
            print(f"📂 Copying template from {template_dir} to {app_dir}")
            
            # Copy the entire template directory
            shutil.copytree(template_dir, app_dir, 
                          ignore=shutil.ignore_patterns('node_modules', '.git', '.next'))
            
            # Update package.json with new app name
            package_json_path = app_dir / "package.json"
            if package_json_path.exists():
                import json
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                
                # Update the name field
                package_data['name'] = app_name
                
                with open(package_json_path, 'w', encoding='utf-8') as f:
                    json.dump(package_data, f, indent=2)
                
                print(f"📝 Updated package.json name to '{app_name}'")
            
            # Fix next.config.ts issue - convert to next.config.mjs if needed
            next_config_ts = app_dir / "next.config.ts"
            next_config_mjs = app_dir / "next.config.mjs"
            
            if next_config_ts.exists() and not next_config_mjs.exists():
                print("🔧 Converting next.config.ts to next.config.mjs (NextJS requirement)")
                
                # Create the correct .mjs version
                mjs_content = """/** @type {import('next').NextConfig} */
const nextConfig = {
  /* config options here */
};

export default nextConfig;"""
                
                with open(next_config_mjs, 'w', encoding='utf-8') as f:
                    f.write(mjs_content)
                
                # Remove the .ts version
                next_config_ts.unlink()
                print("✅ Converted next.config.ts → next.config.mjs")
            
            print(f"✅ Template app '{app_name}' created successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error creating template app: {str(e)}")
            return False
    
    def generate_ai_changes(self, app_idea: str, app_name: str) -> Optional[str]:
        """Generate AI changes and save to file."""
        print(f"🤖 Generating AI changes for: {app_idea}")
        
        try:
            # Generate the app content
            generated_content = self.app_builder.generate_app(app_idea)
            
            if not generated_content:
                print("❌ Failed to generate AI content")
                return None
            
            # Create app-specific input directory
            app_input_dir = self.inputs_dir / app_name
            app_input_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to file with metadata
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = app_input_dir / f"input_v1_{timestamp}.txt"
            
            header = f"""<!-- 
Generated NextJS Application
App Idea: {app_idea}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Multi-LLM Builder with validation
-->

"""
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(header + generated_content)
            
            print(f"✅ AI changes saved to: {output_filename}")
            return str(output_filename)
            
        except Exception as e:
            print(f"❌ Error generating AI changes: {str(e)}")
            return None
    
    def apply_changes(self, input_file: str, app_directory: str) -> bool:
        """
        Apply code changes from input file to app directory using CodeBuilder.
        
        Args:
            input_file: Path to the input file with code changes
            app_directory: Path to the target app directory
            
        Returns:
            True if changes were applied successfully, False otherwise
        """
        try:
            # Create and run the code builder with error logging
            builder = CodeBuilder(input_file, app_directory, self.error_logger)
            builder.build()
            return True
        except Exception as e:
            print(f"❌ Error applying changes: {str(e)}")
            return False
    
    def install_dependencies(self, app_directory: str) -> bool:
        """Install npm dependencies for the NextJS app."""
        print(f"📦 Installing dependencies in {app_directory}")
        
        try:
            result = subprocess.run(
                ["npm", "install"], 
                cwd=app_directory, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Error installing dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {str(e)}")
            return False
    
    def run_nextjs_app(self, app_directory: str, port: Optional[int] = None) -> None:
        """Start the NextJS development server on an available port."""
        # Find an available port if none specified
        if port is None:
            port = self.find_available_port(3000)
            print(f"🔍 Auto-detected available port: {port}")
        else:
            # Check if specified port is available
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                print(f"✅ Port {port} is available")
            except OSError:
                print(f"⚠️  Port {port} is in use, finding alternative...")
                port = self.find_available_port(port)
                print(f"🔍 Using available port: {port}")
        
        print(f"🚀 Starting NextJS app in {app_directory} on port {port}")
        print(f"🌐 App will be available at: http://localhost:{port}")
        print("📝 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        try:
            # Start the development server
            subprocess.run(
                ["npm", "run", "dev", "--", "--port", str(port)],
                cwd=app_directory
            )
        except KeyboardInterrupt:
            print("\n🛑 Development server stopped")
        except Exception as e:
            print(f"❌ Error running NextJS app: {str(e)}")
    
    def build_and_run(self, app_idea: str, app_name: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Complete workflow to build and run a NextJS app:
        1. Create template
        2. Generate AI changes
        3. Apply changes
        4. Validate build
        5. Run the app
        Enhanced with comprehensive logging.
        """
        if app_name is None:
            app_name = self.get_next_app_name()
        
        app_directory = str(self.apps_dir / app_name)
        
        print("🚀 NextJS App Builder")
        print("=" * 50)
        print(f"App Name: {app_name}")
        print(f"App Idea: {app_idea}")
        print(f"Directory: {app_directory}")
        print("-" * 50)
        
        # Log the start of app creation
        self.error_logger.log_build_attempt(app_name, f"create: {app_idea[:50]}...", None)
        
        try:
            # Step 1: Create NextJS template
            print("\n📦 Creating NextJS template...")
            if not self.create_template_nextjs_app(app_name):
                print("❌ Failed to create template")
                self.error_logger.log_build_attempt(app_name, f"create: {app_idea[:50]}...", False)
                return False
        
            # Step 2: Install dependencies automatically
            print("\n📦 Installing dependencies...")
            if not self.install_dependencies(app_directory):
                print("❌ Failed to install dependencies")
                self.error_logger.log_build_attempt(app_name, f"create: {app_idea[:50]}...", False)
                return False
        
            # Step 3: Generate AI changes
            print("\n🤖 Generating AI changes...")
            input_file = self.generate_ai_changes(app_idea, app_name)
            if not input_file:
                print("❌ Failed to generate AI changes")
                self.error_logger.log_build_attempt(app_name, f"create: {app_idea[:50]}...", False)
                return False
        
            # Step 4: Apply changes
            print("\n🔧 Applying changes...")
            if not self.apply_changes(input_file, app_directory):
                print("❌ Failed to apply changes")
                self.error_logger.log_build_attempt(app_name, f"create: {app_idea[:50]}...", False)
                return False
        
            # Step 5: Validate and auto-fix build
            print("\n🔍 Validating and fixing build...")
            if not self.validate_and_fix_build(app_directory):
                print("⚠️  Build validation failed, but app was created")
        
            print(f"🎉 Successfully created {app_name}!")
            self.error_logger.log_build_attempt(app_name, f"create: {app_idea[:50]}...", True)
        
            # Step 5: Optionally run the app
            if port is not None:
                print(f"\n🚀 Starting app on port {port}...")
                self.run_nextjs_app(app_directory, port)
            else:
                # Ask user if they want to run it
                response = input("\n🚀 Start the development server? [Y/n]: ").strip().lower()
                if response == '' or response == 'y' or response == 'yes':
                    self.run_nextjs_app(app_directory)
        
            return True
            
        except Exception as e:
            print(f"❌ Error in build and run: {str(e)}")
            self.error_logger.log_build_attempt(app_name, f"create: {app_idea[:50]}...", False)
            return False

    def analyze_app_structure(self, app_directory: str) -> str:
        """Analyze an existing NextJS app structure and return formatted content."""
        app_path = Path(app_directory)
        
        if not app_path.exists():
            raise FileNotFoundError(f"App directory does not exist: {app_directory}")
        
        structure_info = []
        structure_info.append(f"APP DIRECTORY: {app_directory}")
        structure_info.append("=" * 50)
        
        # Key files to analyze
        key_files = [
            "package.json",
            "app/layout.tsx", 
            "app/page.tsx",
            "app/globals.css"
        ]
        
        # Add any additional component files
        app_dir = app_path / "app"
        if app_dir.exists():
            for file_path in app_dir.rglob("*.tsx"):
                relative_path = file_path.relative_to(app_path)
                if str(relative_path) not in key_files:
                    key_files.append(str(relative_path))
        
        for file_path in key_files:
            full_path = app_path / file_path
            if full_path.exists():
                structure_info.append(f"\n📄 FILE: {file_path}")
                structure_info.append("-" * 30)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Add line numbers for easier editing
                    lines = content.split('\n')
                    numbered_content = []
                    for i, line in enumerate(lines, 1):
                        numbered_content.append(f"{i:3d}: {line}")
                    
                    structure_info.append('\n'.join(numbered_content))
                except Exception as e:
                    structure_info.append(f"Error reading file: {e}")
            else:
                structure_info.append(f"\n📄 FILE: {file_path} (NOT FOUND)")
        
        return '\n'.join(structure_info)
    
    def edit_existing_app(self, app_directory: str, edit_idea: str) -> bool:
        """Edit an existing NextJS app using the robust intent-based approach (with diff fallback)."""
        app_name = Path(app_directory).name
        print("✏️  NextJS App Editor (Intent-Based + Robust Fallbacks)")
        print("=" * 50)
        print(f"App Directory: {app_directory}")
        print(f"Edit Request: {edit_idea}")
        print("-" * 50)
        
        # Log the start of edit operation
        self.error_logger.log_build_attempt(app_name, f"edit: {edit_idea[:50]}...", None)
        
        try:
            # Create app builder instance for this specific app
            print("🔧 Initializing robust intent-based editor...")
            app_builder = MultiLLMAppBuilder()
            app_builder.app_name = app_name  # Set the app name
            app_builder.apps_dir = Path(app_directory).parent  # Set the apps directory
            
            # Use the robust intent-based edit_app method (with diff fallback)
            print("🚀 Applying intent-based edits (more robust)...")
            success = app_builder.edit_app(edit_idea, use_intent_based=True)
            
            if success:
                print("✅ Edits applied successfully using robust intent-based approach!")
                self.error_logger.log_build_attempt(app_name, f"edit: {edit_idea[:50]}...", True)
                return True
            else:
                print("❌ Intent-based edit failed (all fallback strategies exhausted)")
                self.error_logger.log_build_attempt(app_name, f"edit: {edit_idea[:50]}...", False)
                return False
            
        except Exception as e:
            print(f"❌ Error editing app: {str(e)}")
            self.error_logger.log_build_attempt(app_name, f"edit: {edit_idea[:50]}...", False)
            return False
    
    def list_existing_apps(self) -> list:
        """List all existing NextJS apps in the apps directory."""
        if not self.apps_dir.exists():
            return []
        
        apps = []
        for item in self.apps_dir.iterdir():
            if item.is_dir() and (item / "package.json").exists():
                apps.append(item.name)
        
        return sorted(apps)

    def validate_and_fix_build(self, app_directory: str, max_time_minutes: int = 10) -> bool:
        """
        Build the NextJS app and automatically fix errors until it compiles successfully.
        Keeps trying until build succeeds or time limit is reached.
        Enhanced with comprehensive error logging.
        
        Args:
            app_directory: Path to the NextJS app directory
            max_time_minutes: Maximum time to spend fixing (safety limit)
            
        Returns:
            True if build is successful, False if time limit exceeded
        """
        app_name = Path(app_directory).name
        print("🔨 Validating NextJS build...")
        
        # Log the start of validation
        self.error_logger.log_build_attempt(app_name, "validation", None)
        
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        attempt = 0
        last_errors = []
        file_attempt_count = {}
        
        while True:
            attempt += 1
            elapsed_time = time.time() - start_time
            
            # Safety check - don't run forever
            if elapsed_time > max_time_seconds:
                print(f"\n⏰ Time limit reached ({max_time_minutes} minutes)")
                print("❌ Unable to fix all build errors within time limit")
                
                # Log session summary
                self.error_logger.log_session_summary(
                    app_name, attempt, False, elapsed_time
                )
                return False
            
            if attempt > 1:
                print(f"\n🔄 Auto-fix attempt #{attempt} (elapsed: {elapsed_time:.1f}s)")
            
            # Try to build the app
            build_result = self.check_nextjs_build(app_directory)
            
            if build_result["success"]:
                print(f"✅ Build successful after {attempt} attempt(s)!")
                
                # Log successful completion
                self.error_logger.log_build_attempt(app_name, "validation", True)
                self.error_logger.log_session_summary(
                    app_name, attempt, True, elapsed_time
                )
                return True
            
            current_errors = build_result["errors"]
            print(f"🐛 Found {len(current_errors)} build error(s)")
            
            # Log the build errors
            self.error_logger.log_build_errors(app_name, current_errors, attempt)
            
            # Extract problematic files from errors
            problematic_files = self.extract_files_from_errors(current_errors)
            
            # Update attempt count for each problematic file
            for file_path in problematic_files:
                file_attempt_count[file_path] = file_attempt_count.get(file_path, 0) + 1
            
            # Check if any file needs complete rewrite (after 3 attempts)
            files_to_rewrite = [f for f, count in file_attempt_count.items() if count >= 3]
            
            if files_to_rewrite:
                print(f"🔥 Rewriting {len(files_to_rewrite)} file(s) after 3 failed attempts:")
                for file_path in files_to_rewrite:
                    print(f"   📄 {file_path}")
                
                # Log the rewrite attempt
                self.error_logger.log_fix_attempt(
                    app_name, "complete_rewrite", files_to_rewrite
                )
                
                # Attempt complete file rewrite
                if not self.rewrite_problematic_files(app_directory, files_to_rewrite, current_errors):
                    print("❌ File rewrite failed")
                    continue
                
                # Reset attempt count for rewritten files
                for file_path in files_to_rewrite:
                    file_attempt_count[file_path] = 0
                    
            else:
                # Regular incremental fix attempt
                # Check if we're making progress (errors changing)
                if attempt > 1 and current_errors == last_errors:
                    print("⚠️  Same errors as last attempt - trying different fix strategy...")
                    # Use a more aggressive fix approach
                    fix_idea = self.generate_aggressive_fix_idea(current_errors, attempt)
                    fix_type = "aggressive_fix"
                else:
                    fix_idea = self.generate_error_fix_idea(current_errors)
                    fix_type = "incremental_fix"
                
                # Log the fix attempt
                self.error_logger.log_fix_attempt(
                    app_name, fix_type, problematic_files
                )
                
                # Try to auto-fix the errors
                print("🔧 Attempting incremental auto-fix...")
                if not self.auto_fix_build_errors(app_directory, current_errors, fix_idea):
                    print("❌ Unable to generate fixes, trying manual rebuild...")
                    # Sometimes a clean rebuild helps
                    if attempt % 5 == 0:  # Every 5th attempt
                        self.clean_and_rebuild(app_directory)
                    continue
            
            last_errors = current_errors
            time.sleep(1)  # Brief pause between attempts
        
        return False
        
    def extract_files_from_errors(self, errors: list) -> list:
        """Extract file paths from build error messages."""
        files = set()
        for error in errors:
            # Look for common file path patterns in errors
            if '.tsx' in error or '.jsx' in error or '.ts' in error or '.js' in error:
                # Extract file paths like "./app/page.tsx" or "app/page.tsx"
                import re
                file_matches = re.findall(r'[./]*(?:app|src|components)/[^:\s]+\.(?:tsx?|jsx?)', error)
                for match in file_matches:
                    # Normalize path
                    normalized = match.lstrip('./')
                    files.add(normalized)
        return list(files)
    
    def rewrite_problematic_files(self, app_directory: str, file_paths: list, current_errors: list) -> bool:
        """Completely rewrite problematic files based on errors and original structure."""
        try:
            # Get semantic context for file rewriting based on errors
            error_query = f"rewrite files with errors: {'; '.join(str(e) for e in current_errors[:3])}"
            semantic_context = self.app_builder.get_semantic_context_for_request(
                user_request=error_query,
                app_directory=app_directory
            )
            
            # Generate complete file rewrites
            rewrite_instructions = self.app_builder.generate_file_rewrite_response(
                file_paths, current_errors, semantic_context
            )
            
            if not rewrite_instructions:
                return False
            
            # Save and apply the rewrite
            rewrite_filename = f"rewrite_{int(time.time())}.txt"
            rewrite_file_path = self.inputs_dir / Path(app_directory).name / rewrite_filename
            rewrite_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(rewrite_file_path, 'w', encoding='utf-8') as f:
                f.write(rewrite_instructions)
            
            print(f"💾 File rewrite instructions saved to: {rewrite_file_path}")
            
            # Apply the rewrite
            return self.apply_changes(str(rewrite_file_path), app_directory)
            
        except Exception as e:
            print(f"❌ File rewrite failed: {str(e)}")
            return False
    
    def check_nextjs_build(self, app_directory: str) -> dict:
        """
        Check if NextJS app builds successfully.
        
        Returns:
            Dict with 'success' (bool) and 'errors' (list) keys
        """
        try:
            # Run next build to check for compilation errors
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=app_directory,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode == 0:
                return {"success": True, "errors": []}
            
            # Parse build errors
            errors = self.parse_build_errors(result.stderr + result.stdout)
            return {"success": False, "errors": errors}
            
        except subprocess.TimeoutExpired:
            return {"success": False, "errors": ["Build timeout after 60 seconds"]}
        except Exception as e:
            return {"success": False, "errors": [f"Build check failed: {str(e)}"]}
    
    def parse_build_errors(self, build_output: str) -> list:
        """Parse NextJS build output to extract specific errors with enhanced detail and context."""
        errors = []
        lines = build_output.split('\n')
        
        # First, check for common infrastructure errors
        infrastructure_errors = self.detect_infrastructure_errors(build_output)
        if infrastructure_errors:
            return infrastructure_errors
        
        current_error = []
        in_error_block = False
        current_file = None
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Skip empty lines unless we're in an error block
            if not line and not in_error_block:
                continue
            
            # Detect start of compilation error blocks
            if line.startswith("Failed to compile"):
                if current_error:
                    errors.append('\n'.join(current_error))
                current_error = [line]
                in_error_block = True
                continue
            
            # Detect file paths more accurately
            if any(ext in line for ext in ['.tsx', '.jsx', '.ts', '.js']):
                # Look for file paths that start with ./ or contain app/, src/, components/
                if line.startswith('./') or any(path in line for path in ['app/', 'src/', 'components/']):
                    # Clean up the file path
                    file_path = line.split()[0] if ' ' in line else line
                    if current_error and not any(file_path in err for err in current_error):
                        current_error.append(f"FILE: {file_path}")
                    current_file = file_path
                    in_error_block = True
                    continue
            
            # Capture detailed error messages
            error_keywords = [
                'error:', 'syntax error', 'type error', 'unexpected token', 'unexpected eof',
                'expected', 'missing', 'cannot find module', 'property does not exist',
                'parse error', 'compilation error', 'module not found', 'import error',
                'x unexpected', 'x expected', '✗', '×'
            ]
            
            # Check if this line contains detailed error information
            if any(keyword in line.lower() for keyword in error_keywords):
                if in_error_block:
                    current_error.append(f"DETAIL: {line}")
                else:
                    current_error = [f"ERROR: {line}"]
                    in_error_block = True
                continue
            
            # Capture error context and line information
            if in_error_block:
                # Look for line numbers, position indicators, and error details
                context_indicators = [
                    '>', '|', '→', 'at line', 'line ', ':', 'pos:', 'column:',
                    'caused by', 'reason:', 'hint:', 'suggestion:', 'maybe you meant',
                    'note:', 'help:', 'info:'
                ]
                
                # Check if this line provides useful context
                if (any(indicator in line for indicator in context_indicators) or 
                    line.isdigit() or  # Line numbers
                    re.match(r'^\s*\d+\s*\|', line) or  # Line number with content
                    re.match(r'^\s*\^+\s*$', line) or  # Error pointer
                    '─' in line or '═' in line):  # Visual separators
                    
                    current_error.append(f"CONTEXT: {line}")
                    continue
                
                # End of error block detection
                end_indicators = [
                    'import trace for', '✓ compiled', '○ compiling', 'ready in',
                    'local:', 'fast refresh', 'waiting for file changes',
                    '- ready in', '- compiled', 'build completed'
                ]
                
                if any(indicator in line.lower() for indicator in end_indicators):
                    # End current error and start fresh
                    if current_error:
                        errors.append('\n'.join(current_error))
                    current_error = []
                    in_error_block = False
                    current_file = None
                    continue
                
                # Collect additional error details if line has content
                if line and len(current_error) < 15:  # Reasonable limit for error detail
                    # Skip common noise lines
                    noise_patterns = [
                        'at eval', 'at Object.eval', 'at Module.eval',
                        'webpack:', 'chunk ', 'asset ', 'entrypoint ',
                        'warning in', 'info '
                    ]
                    
                    if not any(noise in line for noise in noise_patterns):
                        current_error.append(f"INFO: {line}")
                
            # Handle empty lines in error blocks
            elif line == '' and in_error_block and len(current_error) > 1:
                # Check if next lines contain more error content
                has_more_content = False
                look_ahead = 2
                for j in range(i + 1, min(i + 1 + look_ahead, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and any(keyword in next_line.lower() for keyword in error_keywords + ['>', '|', 'at ']):
                        has_more_content = True
                        break
                
                if not has_more_content:
                    errors.append('\n'.join(current_error))
                    current_error = []
                    in_error_block = False
                    current_file = None
        
        # Add any remaining error
        if current_error:
            errors.append('\n'.join(current_error))
        
        # Clean and format errors
        cleaned_errors = []
        for error in errors:
            error_lines = error.strip().split('\n')
            
            # Skip very minimal errors
            if len(error_lines) < 2 and all(len(line.strip()) < 15 for line in error_lines):
                continue
                
            # Remove duplicate lines
            seen = set()
            unique_lines = []
            for line in error_lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
            
            if unique_lines:
                cleaned_error = '\n'.join(unique_lines)
                cleaned_errors.append(cleaned_error)
        
        # Enhanced fallback: try to extract specific error patterns if we didn't get good errors
        if not cleaned_errors or all(len(err.split('\n')) < 3 for err in cleaned_errors):
            print("⚠️  Limited error detail captured, attempting enhanced extraction...")
            
            # Look for specific NextJS/TypeScript error patterns
            enhanced_errors = []
            full_text = build_output.lower()
            
            # Common specific error patterns
            specific_patterns = [
                (r'unexpected eof', 'Syntax Error: Unexpected end of file - likely missing closing brace or bracket'),
                (r"expected '[^']+', got '[^']+'", 'Syntax Error: Unexpected token - check for missing punctuation'),
                (r'missing semicolon', 'Syntax Error: Missing semicolon'),
                (r'unmatched', 'Syntax Error: Unmatched brackets or braces'),
                (r'cannot find module [\'"][^\'"]+[\'"]', 'Import Error: Module not found'),
                (r'property [\'"][^\'"]+[\'"] does not exist', 'Type Error: Property does not exist')
            ]
            
            for pattern, description in specific_patterns:
                matches = re.findall(pattern, build_output, re.IGNORECASE)
                if matches:
                    enhanced_errors.append(f"{description}\nPattern found: {matches[0] if isinstance(matches[0], str) else matches[0]}")
            
            if enhanced_errors:
                cleaned_errors.extend(enhanced_errors)
        
        # Debug output for error quality
        if cleaned_errors:
            total_lines = sum(len(err.split('\n')) for err in cleaned_errors)
            print(f"🔍 Captured {len(cleaned_errors)} error(s) with {total_lines} total lines of detail")
        else:
            print("⚠️  No detailed errors captured - using generic build failure")
            cleaned_errors = ["Build failed with compilation errors - check syntax and imports"]
            
        return cleaned_errors[:5]  # Limit to first 5 errors for focus
    
    def generate_aggressive_fix_idea(self, errors: list, attempt: int) -> str:
        """Generate more aggressive fix strategies when normal fixes aren't working."""
        error_text = ' '.join(errors).lower()
        
        # Progressive aggressive strategies based on attempt number
        if attempt <= 3:
            return "Completely rewrite the problematic component with proper syntax - start fresh with basic structure"
        elif attempt <= 6:
            return "Remove all complex styling and use simple inline styles - eliminate potential CSS syntax issues"
        elif attempt <= 9:
            return "Simplify the component to basic HTML elements only - remove all advanced React features temporarily"
        else:
            return "Create a minimal working component and gradually add features back - nuclear option"
    
    def clean_and_rebuild(self, app_directory: str) -> None:
        """Clean Next.js cache and force rebuild."""
        try:
            print("🧹 Cleaning Next.js cache...")
            
            # Remove .next directory
            next_dir = Path(app_directory) / ".next"
            if next_dir.exists():
                shutil.rmtree(next_dir)
                print("✓ Removed .next directory")
            
            # Clear npm cache
            subprocess.run(
                ["npm", "cache", "clean", "--force"],
                cwd=app_directory,
                capture_output=True,
                text=True
            )
            print("✓ Cleared npm cache")
            
        except Exception as e:
            print(f"⚠️  Clean failed: {str(e)}")

    def auto_fix_build_errors(self, app_directory: str, errors: list, fix_idea: str = None) -> bool:
        """
        Attempt to automatically fix specific build errors.
        
        Args:
            app_directory: Path to the app directory
            errors: List of specific error messages from build
            fix_idea: Custom fix strategy (optional, fallback for generic fixes)
        
        Returns:
            True if fixes were attempted, False if unable to fix
        """
        if not errors:
            return False
        
        # Try to apply the fix using our edit system
        try:
            # Check for infrastructure errors first and handle them
            infrastructure_errors = self.detect_infrastructure_errors('\n'.join(errors))
            if infrastructure_errors:
                print("🔧 Infrastructure errors detected - attempting automatic fixes...")
                
                # Log infrastructure fix attempt
                self.error_logger.log_fix_attempt(
                    Path(app_directory).name, "infrastructure_fix", ["dependencies"]
                )
                
                # Try to fix missing dependencies
                if any("npm install" in error for error in infrastructure_errors):
                    print("📦 Attempting to install missing dependencies...")
                    if self.install_dependencies(app_directory):
                        print("✅ Dependencies installed successfully")
                        return True
                    else:
                        print("❌ Failed to install dependencies")
                        return False
                
                # Other infrastructure fixes can be added here
                print("⚠️  Infrastructure errors detected but no automatic fix available")
                for error in infrastructure_errors:
                    print(f"   - {error}")
                return False
            
            # Get semantic context for build error fixing
            error_query = f"fix build errors: {'; '.join(errors[:3])}"
            app_structure = self.app_builder.get_semantic_context_for_request(
                user_request=error_query,
                app_directory=app_directory
            )
            
            # Use build-specific error fixing if we have specific errors
            has_specific_errors = any(
                any(keyword in error.lower() for keyword in [
                    'line', 'unexpected token', 'syntax error', '.tsx', '.jsx'
                ])
                for error in errors
            )
            
            if has_specific_errors:
                print(f"🎯 Using targeted build error fixing for {len(errors)} specific error(s)")
                print("📋 Sending these exact errors to LLM:")
                for i, error in enumerate(errors, 1):
                    # Show a more detailed preview of each error
                    error_lines = error.split('\n')
                    
                    # Extract key information from the error
                    key_info = []
                    for line in error_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Focus on the most important error details
                        if any(indicator in line.lower() for indicator in [
                            'error:', 'detail:', 'file:', 'unexpected', 'expected', 
                            'syntax error', 'type error', 'missing', 'cannot find'
                        ]):
                            # Clean up prefixes from our parsing
                            cleaned_line = line
                            for prefix in ['ERROR:', 'DETAIL:', 'FILE:', 'CONTEXT:', 'INFO:']:
                                if cleaned_line.startswith(prefix):
                                    cleaned_line = cleaned_line[len(prefix):].strip()
                            
                            if cleaned_line:
                                key_info.append(cleaned_line)
                    
                    # Display the key information
                    if key_info:
                        print(f"   {i}. {key_info[0]}")  # Most important line first
                        if len(key_info) > 1:
                            for additional in key_info[1:3]:  # Show up to 2 additional lines
                                print(f"      └─ {additional[:100]}{'...' if len(additional) > 100 else ''}")
                    else:
                        # Fallback to showing first non-empty line
                        first_line = next((line.strip() for line in error_lines if line.strip()), "Unknown error")
                        print(f"   {i}. {first_line[:150]}{'...' if len(first_line) > 150 else ''}")
                    
                # Generate fix instructions based on specific build errors
                fix_instructions = self.app_builder.generate_build_fix_response(errors, app_structure)
            else:
                # Fallback to generic fix approach
                if not fix_idea:
                    fix_idea = self.generate_error_fix_idea(errors)
                if not fix_idea:
                    return False
                print(f"🤖 Using generic auto-fix strategy: {fix_idea}")
                fix_instructions = self.app_builder.generate_edit_response(fix_idea, app_structure)
            
            if not fix_instructions:
                return False
            
            # Save and apply the fix
            fix_filename = f"autofix_{int(time.time())}.txt"
            fix_file_path = self.inputs_dir / Path(app_directory).name / fix_filename
            fix_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(fix_file_path, 'w', encoding='utf-8') as f:
                f.write(fix_instructions)
            
            # Apply the fix
            return self.apply_changes(str(fix_file_path), app_directory)
            
        except Exception as e:
            print(f"❌ Auto-fix failed: {str(e)}")
            return False
    
    def generate_error_fix_idea(self, errors: list) -> str:
        """Generate a fix idea based on common error patterns."""
        error_text = ' '.join(errors).lower()
        
        # Common error patterns and their fixes
        if 'syntax error' in error_text or 'unexpected token' in error_text:
            return "Fix syntax errors in JSX/TypeScript code - check for missing commas, unbalanced braces, and proper JSX structure"
        
        if 'cannot find module' in error_text or 'module not found' in error_text:
            return "Fix missing module imports - add proper import statements"
        
        if 'type error' in error_text or 'property does not exist' in error_text:
            return "Fix TypeScript type errors - add proper type annotations and fix property access"
        
        if 'expected' in error_text and 'jsx' in error_text:
            return "Fix JSX structure errors - ensure proper component structure and closing tags"
        
        if 'style' in error_text or 'css' in error_text:
            return "Fix CSS styling errors - ensure proper style object syntax with commas and quotes"
        
        if 'use client' in error_text:
            return "Add 'use client' directive to components that use client-side features"
        
        # Generic fix for other errors
        return "Fix compilation errors - review and correct syntax, imports, and type issues"

    def detect_infrastructure_errors(self, build_output: str) -> list:
        """Detect common infrastructure errors that prevent builds from running."""
        errors = []
        
        if "command not found" in build_output.lower():
            if "next:" in build_output.lower():
                errors.append("INFRASTRUCTURE ERROR: Next.js not installed - dependencies missing. Run 'npm install' first.")
            elif "npm:" in build_output.lower():
                errors.append("INFRASTRUCTURE ERROR: npm not found - Node.js may not be installed properly.")
            else:
                errors.append("INFRASTRUCTURE ERROR: Command not found - check if required tools are installed.")
        
        if "enoent" in build_output.lower() and "node_modules" in build_output.lower():
            errors.append("INFRASTRUCTURE ERROR: node_modules missing - run 'npm install' to install dependencies.")
        
        if "cannot find module" in build_output.lower() and ("next" in build_output.lower() or "react" in build_output.lower()):
            errors.append("INFRASTRUCTURE ERROR: Core dependencies missing - run 'npm install' to install required packages.")
        
        return errors


def main():
    """Main function to run the master builder."""
    parser = argparse.ArgumentParser(
        description="Master NextJS App Builder - Complete automation from idea to running app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --idea "A todo list app with dark mode"
  python src/main.py --idea "Calculator app" --name "my-calculator" --port 3001
  python src/main.py --edit myapp2 --idea "Add a reset button"
  python src/main.py --interactive
        """
    )
    
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--openrouter-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--idea", help="The NextJS app idea to build or edit instruction")
    parser.add_argument("--name", help="Custom name for the app (default: auto-generated)")
    parser.add_argument("--port", type=int, default=None, help="Port for the development server (default: auto-detect starting from 3000)")
    parser.add_argument("--edit", help="Edit existing app by name (e.g., myapp2)")
    parser.add_argument("--list", action="store_true", help="List existing NextJS apps")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize the master builder
        master = MasterBuilder()
        
        # Handle list mode
        if args.list:
            existing_apps = master.list_existing_apps()
            if existing_apps:
                print("📱 Existing NextJS Apps:")
                print("=" * 30)
                for app in existing_apps:
                    print(f"  • {app}")
                print(f"\nTotal: {len(existing_apps)} apps")
            else:
                print("📱 No existing NextJS apps found")
            sys.exit(0)
        
        # Handle edit mode
        if args.edit:
            if not args.idea:
                print("❌ Error: --idea is required when using --edit")
                sys.exit(1)
            
            app_directory = master.apps_dir / args.edit
            if not app_directory.exists():
                print(f"❌ Error: App '{args.edit}' not found")
                existing_apps = master.list_existing_apps()
                if existing_apps:
                    print("\nAvailable apps:")
                    for app in existing_apps:
                        print(f"  • {app}")
                sys.exit(1)
            
            success = master.edit_existing_app(str(app_directory), args.idea)
            if success:
                print(f"\n🎉 Successfully edited {args.edit}!")
                # Ask if user wants to run the app
                try:
                    response = input("🚀 Start the development server? [Y/n]: ").strip().lower()
                    if response in ['', 'y', 'yes']:
                        master.run_nextjs_app(str(app_directory), args.port)
                except KeyboardInterrupt:
                    print("\n👋 Edit complete!")
                sys.exit(0)
            else:
                sys.exit(1)
        
        if args.interactive or not args.idea:
            # Interactive mode
            print("🎯 NextJS Master Builder - Interactive Mode")
            print("=" * 50)
            
            # Show existing apps
            existing_apps = master.list_existing_apps()
            if existing_apps:
                print("📱 Existing apps:", ", ".join(existing_apps))
                print()
            
            print("Commands:")
            print("  • Enter app idea to create new app")
            print("  • Type 'edit <app_name>' to edit existing app")
            print("  • Type 'list' to show existing apps")
            print("  • Type 'quit' to exit")
            print()
            
            while True:
                try:
                    user_input = input("💡 Command or app idea: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if user_input.lower() == 'list':
                        existing_apps = master.list_existing_apps()
                        if existing_apps:
                            print("📱 Existing apps:")
                            for app in existing_apps:
                                print(f"  • {app}")
                        else:
                            print("📱 No existing apps found")
                        continue
                    
                    if user_input.lower().startswith('edit '):
                        app_name = user_input[5:].strip()
                        if not app_name:
                            print("⚠️  Please specify app name: edit <app_name>")
                            continue
                        
                        app_directory = master.apps_dir / app_name
                        if not app_directory.exists():
                            print(f"❌ App '{app_name}' not found")
                            continue
                        
                        edit_idea = input(f"✏️  What changes for {app_name}? ").strip()
                        if not edit_idea:
                            print("⚠️  Please enter edit instruction")
                            continue
                        
                        print()
                        success = master.edit_existing_app(str(app_directory), edit_idea)
                        
                        if success:
                            print(f"\n🎉 Successfully edited {app_name}!")
                            # Ask if user wants to run the app
                            try:
                                response = input("🚀 Start the development server? [Y/n]: ").strip().lower()
                                if response in ['', 'y', 'yes']:
                                    master.run_nextjs_app(str(app_directory), None)
                            except KeyboardInterrupt:
                                print("\n👋 Edit complete!")
                        else:
                            print(f"\n❌ Failed to edit {app_name}")
                        
                        print("\n" + "=" * 50)
                        continue
                    
                    if not user_input:
                        print("⚠️  Please enter a command or app idea")
                        continue
                    
                    # Create new app
                    # Ask for custom name
                    custom_name = input("📱 App name (press Enter for auto): ").strip()
                    if not custom_name:
                        custom_name = None
                    
                    # Ask for port
                    port_input = input(f"🌐 Port [auto-detect]: ").strip()
                    port = int(port_input) if port_input else None
                    
                    print()
                    
                    # Build and run
                    success = master.build_and_run(user_input, custom_name, port)
                    
                    if success:
                        print(f"\n🎉 Success! App created and ready to use.")
                    else:
                        print("\n❌ Failed to create app")
                    
                    print("\n" + "=" * 50)
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except ValueError:
                    print("⚠️  Invalid input")
                    continue
        else:
            # Single run mode
            success = master.build_and_run(args.idea, args.name, args.port)
            if success:
                sys.exit(0)
            else:
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 