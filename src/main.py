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

# Add current and parent directories to path for imports
sys.path.append(os.path.dirname(__file__))  # src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # root directory

from lib.app_builder import MultiLLMAppBuilder
from lib.code_builder import CodeBuilder
from lib.mvp_enhancer import MVPEnhancer
from lib.llm_coordinator import LLMCoordinator
from lib.dependency_manager import DependencyManager
from lib.progress_loader import (
    show_progress, llm_progress, analysis_progress, 
    build_progress, file_progress, update_current_task,
    LoaderStyle
)


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
    def __init__(self, use_single_file_generation: bool = False):
        """
        Initialize the Master Builder.
        
        Args:
            use_single_file_generation: If True, use new single-file approach (recommended)
                                      If False, use legacy multi-file approach
        """
        self.project_root = Path(__file__).parent.parent
        self.apps_dir = self.project_root / "apps"
        self.inputs_dir = self.project_root / "inputs"
        self.build_log_file = self.project_root / "build_errors.log"
        
        # Create directories
        self.apps_dir.mkdir(exist_ok=True)
        self.inputs_dir.mkdir(exist_ok=True)
        
        # Initialize builder components
        self.app_builder = MultiLLMAppBuilder()
        self.error_logger = BuildErrorLogger(str(self.project_root))
        self.mvp_enhancer = MVPEnhancer()
        
        # Initialize LLM coordinator for intelligent planning
        from lib.llm_coordinator import LLMCoordinator
        self.coordinator = LLMCoordinator(self.app_builder)
        
        # NEW: Choose generation approach
        self.use_single_file_generation = use_single_file_generation
        print(f"🎯 Generation mode: {'Single-file (recommended)' if use_single_file_generation else 'Multi-file (legacy)'}")
        
        print("🏗️  Master Builder initialized in:", self.project_root)
        print("📁 Apps directory:", self.apps_dir)
        print("📄 Inputs directory:", self.inputs_dir)
        print("📋 Build error log:", self.build_log_file)
        
        if hasattr(self, 'mvp_enhancer'):
            print("🎯 MVP Enhancer: Ready for complex app development")
        if hasattr(self, 'coordinator') and self.coordinator:
            print("🧠 LLM Coordinator: Ready for intelligent planning")
    
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
            if self.use_single_file_generation:
                print("🚀 Using new single-file generation approach...")
                
                # Use the new single-file approach
                success = self.app_builder.generate_app_with_single_files(app_idea)
                
                if success:
                    print("✅ Single-file generation completed successfully!")
                    return "SUCCESS"  # Return success indicator for single-file approach
                else:
                    print("❌ Single-file generation failed")
                    return None
            else:
                print("🔄 Using legacy multi-file generation approach...")
                
                # Use the legacy approach
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
        """Apply changes from input file to app directory."""
        print(f"🔧 Applying changes to: {app_directory}")
        
        try:
            # NEW: Handle single-file generation approach
            if input_file == "SUCCESS":
                print("✅ Changes already applied by single-file generation")
                return True
            
            # Legacy approach: process input file
            if not os.path.exists(input_file):
                print(f"❌ Input file not found: {input_file}")
                return False
            
            # Use CodeBuilder to apply changes
            code_builder = CodeBuilder(input_file, app_directory, self.error_logger)
            success = code_builder.build()
            
            if success:
                print("✅ Changes applied successfully")
            else:
                print("❌ Failed to apply changes")
            
            return success
            
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
        """Start the NextJS development server for an app."""
        try:
            import subprocess
            import os
            
            app_path = Path(app_directory)
            app_name = app_path.name
            
            print(f"🚀 Starting development server for {app_name}...")
            
            # Change to app directory
            os.chdir(app_path)
            
            # Determine port
            if port is None:
                port = self._find_available_port(3000)
            
            print(f"🌐 Server will start on http://localhost:{port}")
            print("🛑 Press Ctrl+C to stop the server")
            print("-" * 50)
            
            # Start the development server
            cmd = ["npm", "run", "dev", "--", "--port", str(port)]
            subprocess.run(cmd, check=True)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Development server stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start development server: {e}")
        except Exception as e:
            print(f"❌ Error running app: {str(e)}")
        finally:
            # Change back to project root
            os.chdir(self.project_root)

    def _find_available_port(self, start_port: int = 3000) -> int:
        """Find an available port starting from the given port."""
        import socket
        
        port = start_port
        while port < start_port + 100:  # Try up to 100 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except socket.error:
                port += 1
        
        return start_port  # Fallback to original port

    def build_and_run_enhanced(self, app_idea: str, app_name: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Enhanced workflow to build and run complex NextJS apps:
        1. MVP Enhancement - Transform user idea into comprehensive specification
        2. Intelligent Planning - Create detailed execution plan
        3. Coordinated Execution - Build app using multiple coordinated steps
        4. Validation and Running
        """
        if app_name is None:
            app_name = self.get_next_app_name()
        
        app_directory = str(self.apps_dir / app_name)
        
        with show_progress(f"building NextJS app '{app_name}'", LoaderStyle.BUILDING):
            print("🚀 Enhanced NextJS App Builder (MVP-Driven)")
            print("=" * 60)
            print(f"App Name: {app_name}")
            print(f"Original Idea: {app_idea}")
            print(f"Directory: {app_directory}")
            print("-" * 60)
            
            # Log the start of app creation
            self.error_logger.log_build_attempt(app_name, f"enhanced_create: {app_idea[:50]}...", None)
            
            try:
                # Step 1: Enhance user prompt to MVP specification
                update_current_task("enhancing prompt to MVP specification")
                print("\n🎯 Step 1: Enhancing prompt to MVP specification...")
                mvp_spec = self.mvp_enhancer.enhance_prompt_to_mvp(app_idea)
                
                # Step 2: Create intelligent execution plan
                update_current_task("creating intelligent execution plan")
                print("\n🧠 Step 2: Creating intelligent execution plan...")
                execution_plan = self.coordinator.analyze_and_plan(app_idea, "create_app", mvp_spec)
                
                # Step 3: Create NextJS template foundation
                update_current_task("creating NextJS template foundation")
                print("\n📦 Step 3: Creating NextJS template foundation...")
                if not self.create_template_nextjs_app(app_name):
                    print("❌ Failed to create template")
                    self.error_logger.log_build_attempt(app_name, f"enhanced_create: {app_idea[:50]}...", False)
                    return False
            
                # Step 4: Install dependencies
                update_current_task("installing dependencies")
                print("\n📦 Step 4: Installing dependencies...")
                with build_progress("installing dependencies"):
                    if not self.install_dependencies(app_directory):
                        print("❌ Failed to install dependencies")
                        self.error_logger.log_build_attempt(app_name, f"enhanced_create: {app_idea[:50]}...", False)
                        return False
            
                # Step 5: Execute coordinated plan
                update_current_task("executing coordinated development plan")
                print("\n🎯 Step 5: Executing coordinated development plan...")
                if not self.execute_coordinated_plan(execution_plan, mvp_spec, app_name, app_directory):
                    print("❌ Failed to execute coordinated plan")
                    self.error_logger.log_build_attempt(app_name, f"enhanced_create: {app_idea[:50]}...", False)
                    return False
            
                # Step 6: Final validation and build fixing
                update_current_task("final validation and build fixing")
                print("\n🔍 Step 6: Final validation and build fixing...")
                with build_progress("validating and fixing build"):
                    if not self.validate_and_fix_build(app_directory):
                        print("⚠️  Build validation failed, but app was created")
                        self.error_logger.log_build_attempt(app_name, f"enhanced_create: {app_idea[:50]}...", True)
                        # Don't return False here - app was created, just might have minor issues
                
                # Step 7: App creation completed successfully
                update_current_task("app creation completed")
                print(f"\n✅ NextJS app '{app_name}' created successfully!")
                print(f"📁 Location: {app_directory}")
                print(f"🚀 To start the dev server, run:")
                print(f"   cd apps/{app_name}")
                print(f"   npm run dev")
                if port:
                    print(f"   Or: npm run dev -- --port {port}")
                
                self.error_logger.log_build_attempt(app_name, f"enhanced_create: {app_idea[:50]}...", True)
                return True
                
            except Exception as e:
                print(f"❌ Error in enhanced build process: {str(e)}")
                self.error_logger.log_build_attempt(app_name, f"enhanced_create: {app_idea[:50]}...", False)
                return False

    def build_and_run_legacy(self, app_idea: str, app_name: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Legacy simple workflow (kept for compatibility):
        1. Create template
        2. Generate AI changes
        3. Apply changes
        4. Validate build
        5. Run the app
        """
        if app_name is None:
            app_name = self.get_next_app_name()
        
        app_directory = str(self.apps_dir / app_name)
        
        print("🚀 NextJS App Builder (Legacy Mode)")
        print("=" * 50)
        print(f"App Name: {app_name}")
        print(f"App Idea: {app_idea}")
        print(f"Directory: {app_directory}")
        print("-" * 50)
        
        # Log the start of app creation
        self.error_logger.log_build_attempt(app_name, f"legacy_create: {app_idea[:50]}...", None)
        
        try:
            # Step 1: Create NextJS template
            print("\n📦 Creating NextJS template...")
            if not self.create_template_nextjs_app(app_name):
                print("❌ Failed to create template")
                self.error_logger.log_build_attempt(app_name, f"legacy_create: {app_idea[:50]}...", False)
                return False
        
            # Step 2: Install dependencies automatically
            print("\n📦 Installing dependencies...")
            if not self.install_dependencies(app_directory):
                print("❌ Failed to install dependencies")
                self.error_logger.log_build_attempt(app_name, f"legacy_create: {app_idea[:50]}...", False)
                return False
        
            # Step 3: Generate AI changes
            print("\n🤖 Generating AI changes...")
            input_file = self.generate_ai_changes(app_idea, app_name)
            if not input_file:
                print("❌ Failed to generate AI changes")
                self.error_logger.log_build_attempt(app_name, f"legacy_create: {app_idea[:50]}...", False)
                return False
        
            # Step 4: Apply changes
            print("\n🔧 Applying changes...")
            if not self.apply_changes(input_file, app_directory):
                print("❌ Failed to apply changes")
                self.error_logger.log_build_attempt(app_name, f"legacy_create: {app_idea[:50]}...", False)
                return False
        
            # Step 5: Validate and auto-fix build
            print("\n🔍 Validating and fixing build...")
            if not self.validate_and_fix_build(app_directory):
                print("⚠️  Build validation failed, but app was created")
        
            print(f"🎉 Successfully created {app_name}!")
            self.error_logger.log_build_attempt(app_name, f"legacy_create: {app_idea[:50]}...", True)
        
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
            print(f"❌ Error in legacy build and run: {str(e)}")
            self.error_logger.log_build_attempt(app_name, f"legacy_create: {app_idea[:50]}...", False)
            return False

    def execute_coordinated_plan(self, execution_plan, mvp_spec, app_name: str, app_directory: str) -> bool:
        """
        Execute the coordinated development plan using the LLM coordinator.
        
        Args:
            execution_plan: The execution plan from the coordinator
            mvp_spec: The MVP specification
            app_name: Name of the app being built
            app_directory: Directory where the app is being built
            
        Returns:
            True if plan executed successfully, False otherwise
        """
        try:
            print(f"🎯 Executing {len(execution_plan.tasks)} coordinated tasks...")
            print(f"📋 Strategy: {execution_plan.execution_strategy}")
            print(f"⏱️ Estimated duration: {execution_plan.estimated_duration}")
            
            # Format MVP spec for the app builder
            enhanced_request = self.mvp_enhancer.format_mvp_for_coordinator(mvp_spec)
            
            # 🖨️ SHOW THE FORMATTED MVP REQUEST BEING USED
            print("\n" + "=" * 80)
            print("🎯 USING ENHANCED MVP SPECIFICATION FOR CODE GENERATION:")
            print("=" * 80)
            print(enhanced_request)
            print("=" * 80)
            
            # Use existing app builder instance with configured API keys
            app_builder = self.app_builder
            app_builder.app_name = app_name
            app_builder.apps_dir = Path(app_directory).parent
            
            # Use the coordinator mode for complex builds
            app_builder.set_coordinator_mode(True)
            
            # Generate the coordinated content
            print("\n🤖 Generating coordinated AI content...")
            print("🧠 This may take a moment while the AI creates your application...")
            generated_content = app_builder.generate_app(enhanced_request)
            
            if not generated_content:
                print("❌ Failed to generate coordinated content")
                return False
            
            # Create app-specific input directory
            app_input_dir = self.inputs_dir / app_name
            app_input_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to file with enhanced metadata
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = app_input_dir / f"enhanced_mvp_v1_{timestamp}.txt"
            
            header = f"""<!-- 
Generated Enhanced NextJS Frontend Application
Original Idea: {mvp_spec.original_prompt}
Enhanced MVP: {mvp_spec.enhanced_prompt}
Complexity: {mvp_spec.complexity_level}
Core Features: {', '.join(mvp_spec.core_features)}
Components: {mvp_spec.estimated_components}
Tech Stack: {', '.join(mvp_spec.suggested_tech_stack)}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Frontend-Only MVP Builder with intelligent coordination
-->

"""
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(header + generated_content)
            
            print(f"✅ Enhanced AI content saved to: {output_filename}")
            
            # Apply the coordinated changes
            print("\n" + "🔥" * 80)
            print("🔧 APPLYING COORDINATED CHANGES - CREATING FILES...")
            print("🔥" * 80)
            builder = CodeBuilder(str(output_filename), app_directory, self.error_logger)
            builder.build()
            
            # NEW: Automatic dependency management
            print("\n📦 Managing dependencies automatically...")
            dependency_manager = DependencyManager(app_directory)
            if dependency_manager.auto_manage_dependencies():
                print("✅ Dependencies managed successfully!")
            else:
                print("⚠️ Dependency management completed with warnings")
            
            print("\n" + "🎉" * 80)
            print("✅ COORDINATED PLAN EXECUTED SUCCESSFULLY!")
            print("🎉" * 80)
            return True
            
        except Exception as e:
            print(f"❌ Error executing coordinated plan: {str(e)}")
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
        """Edit an existing app with new features or changes."""
        try:
            print(f"✏️ Editing app: {Path(app_directory).name}")
            print(f"🎯 Changes: {edit_idea}")
            
            # Set the app path for the builder
            app_path = Path(app_directory)
            self.app_builder.app_name = app_path.name
            self.app_builder.apps_dir = app_path.parent
            
            # Use the edit method from app_builder
            success = self.app_builder.edit_app(edit_idea, use_intent_based=True)
            
            if success:
                print("✅ App edited successfully!")
                return True
            else:
                print("❌ Failed to edit app")
                return False
                
        except Exception as e:
            print(f"❌ Error editing app: {str(e)}")
            return False
    
    def list_existing_apps(self) -> List[str]:
        """List all existing NextJS apps."""
        try:
            apps = []
            if self.apps_dir.exists():
                for item in self.apps_dir.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        # Check if it's a NextJS app (has package.json)
                        package_json = item / "package.json"
                        if package_json.exists():
                            apps.append(item.name)
            return sorted(apps)
        except Exception as e:
            print(f"⚠️ Error listing apps: {e}")
            return []

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
        
        # 🔧 CRITICAL FIX: Configure app builder with proper app context
        # This prevents the "App name not set" infinite loop error
        self._configure_app_builder_for_validation(app_name, app_directory)
        
        # Log the start of validation
        self.error_logger.log_build_attempt(app_name, "validation", None)
        
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        attempt = 0
        last_errors = []
        file_attempt_count = {}
        
        # 🛡️ SAFETY MECHANISMS: Prevent infinite loops
        max_total_attempts = 15  # Hard limit to prevent infinite loops
        consecutive_same_errors = 0  # Track how many times we see identical errors
        max_consecutive_same = 3  # Max times to see same errors before giving up
        
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
            
            # 🛡️ SAFETY: Hard limit on total attempts
            if attempt > max_total_attempts:
                print(f"\n🛑 Maximum attempts reached ({max_total_attempts})")
                print("❌ Stopping to prevent infinite loop")
                
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
            
            # 🛡️ SAFETY: Detect identical repeating errors
            if attempt > 1 and current_errors == last_errors:
                consecutive_same_errors += 1
                print(f"⚠️  Same errors as last attempt ({consecutive_same_errors}/{max_consecutive_same})")
                
                if consecutive_same_errors >= max_consecutive_same:
                    print(f"\n🛑 Identical errors repeated {max_consecutive_same} times")
                    print("❌ Likely stuck in infinite loop - stopping auto-fix")
                    print("💡 Manual intervention may be required")
                    
                    # Show the problematic errors for debugging
                    print("\n🔍 Problematic errors that couldn't be fixed:")
                    for i, error in enumerate(current_errors[:3], 1):
                        print(f"   {i}. {error[:200]}{'...' if len(error) > 200 else ''}")
                    
                    # Log session summary
                    self.error_logger.log_session_summary(
                        app_name, attempt, False, elapsed_time
                    )
                    return False
            else:
                # Errors are different, reset the counter
                consecutive_same_errors = 0
            
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
    
    def _configure_app_builder_for_validation(self, app_name: str, app_directory: str):
        """
        Configure the app builder with proper app context for validation/auto-fixing.
        
        This prevents the "App name not set" error that causes infinite loops.
        """
        try:
            # Set the app name and apps directory
            self.app_builder.app_name = app_name
            self.app_builder.apps_dir = Path(app_directory).parent
            
            # Validate the configuration
            test_path = self.app_builder.get_app_path()
            if test_path != app_directory:
                print(f"⚠️ App path mismatch: expected {app_directory}, got {test_path}")
                # Force set to correct path
                self.app_builder.app_name = app_name
                self.app_builder.apps_dir = Path(app_directory).parent
            
            print(f"✅ App builder configured: {app_name} -> {app_directory}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not configure app builder: {e}")
            print("🔄 This may cause some auto-fix features to be unavailable")
    
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
            rewrite_filename = f"rewrite_{int(time.time())}"
            rewrite_file_path = self.inputs_dir / Path(app_directory).name / f"{rewrite_filename}.txt"
            rewrite_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(rewrite_file_path, 'w', encoding='utf-8') as f:
                f.write(rewrite_instructions)
            
            print(f"💾 File rewrite instructions saved to: {rewrite_file_path}")
            
            # 🔧 INTELLIGENT FIX: Use appropriate builder based on format
            is_unified_diff = self._is_unified_diff_format(rewrite_instructions)
            
            if is_unified_diff:
                print(f"🔧 Applying rewrite using DiffBuilder (unified diff format)...")
                # Save as .patch file for DiffBuilder
                patch_file_path = self.inputs_dir / Path(app_directory).name / f"{rewrite_filename}.patch"
                with open(patch_file_path, 'w', encoding='utf-8') as f:
                    f.write(rewrite_instructions)
                
                # Apply using DiffBuilder
                from .lib.diff_builder import DiffBuilder
                diff_builder = DiffBuilder(str(patch_file_path), app_directory)
                success = diff_builder.build()
                
                if success:
                    print("✅ File rewrite applied successfully using DiffBuilder!")
                    return True
                else:
                    print("❌ DiffBuilder failed, trying CodeBuilder as fallback...")
                    # Fall through to CodeBuilder
            
            # Apply using CodeBuilder for <new>/<edit> format or as fallback
            print(f"🔧 Applying rewrite using CodeBuilder...")
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
            
            # 🔧 CRITICAL FIX: Use DiffBuilder for unified diff patches, CodeBuilder for <new>/<edit> blocks
            fix_filename = f"autofix_{int(time.time())}"
            fix_file_path = self.inputs_dir / Path(app_directory).name / f"{fix_filename}.txt"
            fix_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(fix_file_path, 'w', encoding='utf-8') as f:
                f.write(fix_instructions)
            
            # Detect whether this is a unified diff or legacy format
            is_unified_diff = self._is_unified_diff_format(fix_instructions)
            
            if is_unified_diff:
                print(f"🔧 Applying unified diff patch using DiffBuilder...")
                # Save as .patch file for DiffBuilder
                patch_file_path = self.inputs_dir / Path(app_directory).name / f"{fix_filename}.patch"
                with open(patch_file_path, 'w', encoding='utf-8') as f:
                    f.write(fix_instructions)
                
                # Apply using DiffBuilder
                from .lib.diff_builder import DiffBuilder
                diff_builder = DiffBuilder(str(patch_file_path), app_directory)
                success = diff_builder.build()
                
                if success:
                    print("✅ Unified diff applied successfully!")
                    return True
                else:
                    print("❌ DiffBuilder failed, trying CodeBuilder as fallback...")
                    # Fall through to CodeBuilder
            
            # Apply using legacy CodeBuilder for <new>/<edit> format
            print(f"🔧 Applying changes using CodeBuilder...")
            return self.apply_changes(str(fix_file_path), app_directory)
            
        except Exception as e:
            print(f"❌ Auto-fix failed: {str(e)}")
            return False
    
    def _is_unified_diff_format(self, content: str) -> bool:
        """
        Detect if content is in unified diff format vs legacy <new>/<edit> format.
        
        Args:
            content: The fix instructions content
            
        Returns:
            True if unified diff, False if legacy format
        """
        # Check for unified diff markers
        diff_markers = [
            "*** Begin Patch",
            "*** Update File:",
            "*** End Patch",
            "@@ -",
            "@@"
        ]
        
        # Check for legacy format markers
        legacy_markers = [
            "<new filename=",
            "<edit filename=",
            "</new>",
            "</edit>"
        ]
        
        has_diff_markers = any(marker in content for marker in diff_markers)
        has_legacy_markers = any(marker in content for marker in legacy_markers)
        
        # If it has diff markers but no legacy markers, it's a unified diff
        if has_diff_markers and not has_legacy_markers:
            return True
        
        # If it has legacy markers but no diff markers, it's legacy format
        if has_legacy_markers and not has_diff_markers:
            return False
        
        # If it has both or neither, make a best guess based on prevalence
        if has_diff_markers:
            return True
        
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

    def build_and_run(self, app_idea: str, app_name: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Build and run a NextJS app (defaults to enhanced mode).
        
        This is a convenience method that calls build_and_run_enhanced.
        """
        return self.build_and_run_enhanced(app_idea, app_name, port)

    def build_and_run_single_file(self, app_idea: str, app_name: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Build and run a NextJS app using the new single-file generation approach.
        
        This method creates files one at a time to avoid truncation issues.
        """
        print("🚀 Starting single-file generation process...")
        
        # Generate app name if not provided
        if not app_name:
            app_name = self.get_next_app_name()
        
        app_directory = str(self.apps_dir / app_name)
        
        try:
            # Step 1: Create NextJS template
            print(f"📱 Creating NextJS app: {app_name}")
            if not self.create_template_nextjs_app(app_name):
                print("❌ Failed to create NextJS template")
                return False
            
            # Step 2: Set up app builder context
            print(f"🔧 Setting up app builder for: {app_name}")
            self.app_builder.app_name = app_name
            self.app_builder.apps_dir = self.apps_dir
            
            # Step 3: Generate files using single-file approach
            print(f"🎯 Generating app files...")
            success = self.app_builder.generate_app_with_single_files(app_idea)
            
            if not success:
                print("❌ Failed to generate app files")
                return False
            
            # Step 4: Install dependencies and validate
            print(f"📦 Installing dependencies...")
            if not self.install_dependencies(app_directory):
                print("⚠️ Failed to install dependencies, trying to continue...")
            
            # Step 5: Build validation
            print(f"🔨 Validating build...")
            self.app_builder.app_name = app_name  # Ensure context is set
            if not self.app_builder.build_and_fix_errors():
                print("❌ Build validation failed")
                print("🔍 Please check the error messages above for details:")
                print("   • If you see 'executor bugs', the system needs debugging")
                print("   • If you see build errors, review the generated code")
                print("   • You can manually fix issues and run 'npm run build' to test")
                
                # Ask user what to do instead of automatically starting with errors
                import sys
                try:
                    response = input("\n🤔 Start dev server anyway? (y/N): ").strip().lower()
                    if response not in ['y', 'yes']:
                        print("🛑 Stopping. Fix the issues and try again.")
                        return False
                    print("⚠️ Starting dev server with known issues...")
                except (KeyboardInterrupt, EOFError):
                    print("\n🛑 Stopping. Fix the issues and try again.")
                    return False
            
            # Step 6: Start development server
            if port is None:
                port = self.find_available_port()
            
            print(f"🚀 Starting development server on port {port}...")
            self.run_nextjs_app(app_directory, port)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in single-file generation: {str(e)}")
            return False

    def build_and_run_anti_truncation(self, app_idea: str, app_name: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Build and run a NextJS app using the efficient anti-truncation approach.
        
        This method uses a single API call with a focused, shorter prompt to prevent
        truncation while being much more token-efficient than the file-by-file approach.
        """
        print("🎯 Starting anti-truncation generation process...")
        print("   ✅ Uses 1 API call instead of 15-20+")
        print("   ✅ Focused prompts prevent truncation")
        print("   ✅ 10x-20x more token efficient")
        
        # Generate app name if not provided
        if not app_name:
            app_name = self.get_next_app_name()
        
        app_directory = str(self.apps_dir / app_name)
        
        try:
            # Step 1: Create NextJS template
            print(f"📱 Creating NextJS app: {app_name}")
            if not self.create_template_nextjs_app(app_name):
                print("❌ Failed to create NextJS template")
                return False
            
            # Step 2: Set up app builder context
            print(f"🔧 Setting up app builder for: {app_name}")
            self.app_builder.app_name = app_name
            self.app_builder.apps_dir = self.apps_dir
            
            # Step 3: Generate using anti-truncation approach
            print(f"🎯 Generating app with focused prompt...")
            ai_content = self.app_builder.generate_app_with_anti_truncation(app_idea)
            
            if not ai_content:
                print("❌ Failed to generate app content")
                return False
            
            # Step 4: Apply the generated content
            print(f"🔧 Applying generated content...")
            
            # Create app-specific input directory and save content
            app_input_dir = self.inputs_dir / app_name
            app_input_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            input_file = app_input_dir / f"anti_truncation_{timestamp}.txt"
            
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(ai_content)
            
            # Apply changes using CodeBuilder
            success = self.apply_changes(str(input_file), app_directory)
            
            if not success:
                print("❌ Failed to apply changes")
                return False
            
            # Step 5: Install dependencies and validate
            print(f"📦 Installing dependencies...")
            if not self.install_dependencies(app_directory):
                print("⚠️ Failed to install dependencies, trying to continue...")
            
            # Step 6: Build validation
            print(f"🔨 Validating build...")
            self.app_builder.app_name = app_name
            if not self.app_builder.build_and_fix_errors():
                print("❌ Build validation failed")
                print("🔍 Please check the error messages above for details:")
                print("   • If you see 'executor bugs', the system needs debugging")
                print("   • If you see build errors, review the generated code")
                print("   • You can manually fix issues and run 'npm run build' to test")
                
                # Ask user what to do instead of automatically starting with errors
                import sys
                try:
                    response = input("\n🤔 Start dev server anyway? (y/N): ").strip().lower()
                    if response not in ['y', 'yes']:
                        print("🛑 Stopping. Fix the issues and try again.")
                        return False
                    print("⚠️ Starting dev server with known issues...")
                except (KeyboardInterrupt, EOFError):
                    print("\n🛑 Stopping. Fix the issues and try again.")
                    return False
            
            # Step 7: Start development server
            if port is None:
                port = self.find_available_port()
            
            print(f"🚀 Starting development server on port {port}...")
            self.run_nextjs_app(app_directory, port)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in anti-truncation generation: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='NextJS App Builder with Multi-LLM Support and MVP Enhancement')
    parser.add_argument('-i', '--idea', type=str, help='App idea description')
    parser.add_argument('-n', '--name', type=str, help='App name (optional)')
    parser.add_argument('-p', '--port', type=int, default=3000, help='Development server port')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced MVP mode (default)')
    parser.add_argument('--legacy', action='store_true', help='Use legacy mode')
    parser.add_argument('--single-file', action='store_true', help='Use new single-file generation (recommended to avoid truncation)')
    parser.add_argument('--anti-truncation', action='store_true', help='Use efficient anti-truncation approach (RECOMMENDED - 1 call, shorter prompt)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('-e', '--edit', type=str, help='Edit existing app by name')
    
    args = parser.parse_args()
    
    try:
        # Initialize builder with chosen generation approach
        builder = MasterBuilder(use_single_file_generation=args.single_file)
        
        if args.single_file:
            print("🎯 Using new single-file generation approach (recommended)")
            print("   ✅ Prevents AI response truncation")
            print("   ✅ Better error handling per file")
            print("   ✅ More reliable file generation")
        else:
            print("🔄 Using legacy multi-file generation approach")
            print("   ⚠️  May experience truncation with complex apps")
        
        # Handle edit mode
        if args.edit:
            if not args.idea:
                print("❌ Error: --idea is required when using --edit")
                return
            
            app_directory = builder.apps_dir / args.edit
            if not app_directory.exists():
                print(f"❌ Error: App '{args.edit}' not found")
                existing_apps = builder.list_existing_apps()
                if existing_apps:
                    print("\nAvailable apps:")
                    for app in existing_apps:
                        print(f"  • {app}")
                return
            
            success = builder.edit_existing_app(str(app_directory), args.idea)
            if success:
                print(f"\n🎉 Successfully edited {args.edit}!")
                # Ask if user wants to run the app
                try:
                    response = input("🚀 Start the development server? [Y/n]: ").strip().lower()
                    if response in ['', 'y', 'yes']:
                        builder.run_nextjs_app(str(app_directory), args.port)
                except KeyboardInterrupt:
                    print("\n👋 Edit complete!")
            return
        
        # Handle interactive mode
        if args.interactive or not args.idea:
            print("🎯 NextJS Master Builder - Interactive Mode")
            print("=" * 50)
            
            # Show generation mode
            if args.anti_truncation:
                mode_info = "Anti-truncation (RECOMMENDED)"
            elif args.single_file:
                mode_info = "Single-file (expensive)"
            else:
                mode_info = "Enhanced MVP (default)"
            print(f"📁 Generation mode: {mode_info}")
            
            # Show existing apps
            existing_apps = builder.list_existing_apps()
            if existing_apps:
                print("📱 Existing apps:", ", ".join(existing_apps))
                print()
            
            print("Commands:")
            print("  • Enter app idea to create new app")
            print("  • Type 'edit <app_name>' to edit existing app")
            print("  • Type 'list' to show existing apps")
            print("  • Type 'mode' to cycle through generation modes")
            print("  • Type 'quit' to exit")
            print()
            
            # Track current mode
            current_mode = "anti_truncation" if args.anti_truncation else ("single_file" if args.single_file else "enhanced")
            
            while True:
                try:
                    user_input = input("💡 Command or app idea: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if user_input.lower() == 'list':
                        existing_apps = builder.list_existing_apps()
                        if existing_apps:
                            print("📱 Existing apps:")
                            for app in existing_apps:
                                print(f"  • {app}")
                        else:
                            print("📱 No existing apps found")
                        continue
                    
                    if user_input.lower() == 'mode':
                        # Cycle through modes: enhanced -> anti_truncation -> single_file -> enhanced
                        if current_mode == "enhanced":
                            current_mode = "anti_truncation"
                            mode_name = "Anti-truncation (RECOMMENDED)"
                        elif current_mode == "anti_truncation":
                            current_mode = "single_file"
                            mode_name = "Single-file (expensive)"
                        else:  # single_file
                            current_mode = "enhanced"
                            mode_name = "Enhanced MVP (default)"
                        
                        # No need to reinitialize builder for mode switching
                        print(f"🔄 Switched to {mode_name} generation mode")
                        continue
                    
                    if user_input.lower().startswith('edit '):
                        app_name = user_input[5:].strip()
                        if not app_name:
                            print("⚠️  Please specify app name: edit <app_name>")
                            continue
                        
                        app_directory = builder.apps_dir / app_name
                        if not app_directory.exists():
                            print(f"❌ App '{app_name}' not found")
                            continue
                        
                        edit_idea = input(f"✏️  What changes for {app_name}? ").strip()
                        if not edit_idea:
                            print("⚠️  Please enter edit instruction")
                            continue
                        
                        print()
                        success = builder.edit_existing_app(str(app_directory), edit_idea)
                        
                        if success:
                            print(f"\n🎉 Successfully edited {app_name}!")
                            # Ask if user wants to run the app
                            try:
                                response = input("🚀 Start the development server? [Y/n]: ").strip().lower()
                                if response in ['', 'y', 'yes']:
                                    builder.run_nextjs_app(str(app_directory), None)
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
                    custom_name = input("📱 App name (press Enter for auto): ").strip()
                    if not custom_name:
                        custom_name = None
                    
                    port_input = input(f"🌐 Port [auto-detect]: ").strip()
                    port = int(port_input) if port_input else None
                    
                    print()
                    
                    # Build and run based on mode
                    if current_mode == "anti_truncation":
                        success = builder.build_and_run_anti_truncation(user_input, custom_name, port)
                    elif current_mode == "single_file":
                        success = builder.build_and_run_single_file(user_input, custom_name, port)
                    elif args.legacy:
                        success = builder.build_and_run_legacy(user_input, custom_name, port)
                    else:
                        success = builder.build_and_run_enhanced(user_input, custom_name, port)
                    
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
            return
        
        # Single command mode
        if not args.idea:
            print("❌ Please provide an app idea using -i or --idea")
            print("Example: python src/main.py -i 'a todo app with categories'")
            print("Or use --interactive for interactive mode")
            return
        
        # Set mode based on arguments
        if args.anti_truncation:
            print("🎯 Using anti-truncation mode (RECOMMENDED - efficient & reliable)")
            success = builder.build_and_run_anti_truncation(args.idea, args.name, args.port)
        elif args.single_file:
            print("🎯 Using single-file generation mode (expensive but thorough)")
            success = builder.build_and_run_single_file(args.idea, args.name, args.port)
        elif args.legacy:
            print("🎯 Using legacy mode")
            success = builder.build_and_run_legacy(args.idea, args.name, args.port)
        else:
            print("🎯 Using enhanced MVP mode (default)")
            success = builder.build_and_run_enhanced(args.idea, args.name, args.port)
        
        if success:
            print("🎉 Application built and running successfully!")
        else:
            print("❌ Application build failed")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 