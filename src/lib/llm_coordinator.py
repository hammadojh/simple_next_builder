#!/usr/bin/env python3
"""
LLM Coordinator System

This module implements an intelligent LLM coordinator that:
1. Understands user prompts and creates detailed plans
2. Breaks down complex requests into manageable tasks
3. Decides what context and information is needed for each task
4. Coordinates multiple LLM calls strategically
5. Tests and validates results at each step
6. Enables much more complex app creation and editing workflows

The coordinator acts as a "senior developer" that plans and manages the work,
while the individual LLMs act as "junior developers" executing specific tasks.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Task:
    """Represents a single task in the execution plan."""
    id: str
    type: str  # 'create_file', 'edit_file', 'analyze_context', 'validate_build', etc.
    description: str
    dependencies: List[str]  # Task IDs this task depends on
    context_needed: List[str]  # What context/files need to be analyzed
    priority: int  # 1 (high) to 5 (low)
    estimated_complexity: str  # 'simple', 'moderate', 'complex'
    llm_instructions: str  # Specific instructions for the LLM
    validation_criteria: List[str]  # How to validate task completion
    status: str = "pending"  # 'pending', 'in_progress', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionPlan:
    """Represents the complete execution plan for a user request."""
    request_id: str
    user_request: str
    request_type: str  # 'create_app', 'edit_app', 'add_feature', etc.
    complexity_assessment: str
    tasks: List[Task]
    execution_strategy: str
    estimated_duration: str
    success_criteria: List[str]
    created_at: str
    status: str = "planned"  # 'planned', 'executing', 'completed', 'failed'


class LLMCoordinator:
    """
    Intelligent LLM coordinator that plans and orchestrates complex app development tasks.
    
    This coordinator acts as a "senior developer" that:
    - Analyzes user requests and creates detailed execution plans
    - Breaks down complex tasks into smaller, manageable subtasks
    - Determines optimal sequencing and dependencies
    - Coordinates multiple specialized LLM calls
    - Validates results at each step
    - Handles failures and retries strategically
    """
    
    def __init__(self, app_builder=None):
        """Initialize the coordinator with access to the app builder."""
        self.app_builder = app_builder
        self.execution_history = []
        self.current_plan: Optional[ExecutionPlan] = None
        
        # Initialize OpenAI client for coordination
        self.openai_client = None
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
            except ImportError:
                print("⚠️ OpenAI package not installed")
        
        print("🧠 LLM Coordinator initialized")
    
    def analyze_and_plan(self, user_request: str, request_type: str = "auto") -> ExecutionPlan:
        """
        Analyze the user request and create a detailed execution plan.
        
        Args:
            user_request: The user's request/idea
            request_type: Type of request ('create_app', 'edit_app', 'add_feature', 'auto')
            
        Returns:
            ExecutionPlan: Detailed plan for executing the request
        """
        print("🧠 Analyzing request and creating execution plan...")
        print(f"📝 Request: {user_request}")
        
        # Auto-detect request type if not specified
        if request_type == "auto":
            request_type = self._classify_request_type(user_request)
        
        print(f"🔍 Detected request type: {request_type}")
        
        # Use coordinator LLM to create detailed plan
        planning_prompt = self._get_planning_prompt(user_request, request_type)
        
        plan_response = self._call_coordinator_llm(planning_prompt, "planning")
        
        if not plan_response:
            # Fallback to simple plan
            return self._create_fallback_plan(user_request, request_type)
        
        # Parse the plan from LLM response
        execution_plan = self._parse_execution_plan(plan_response, user_request, request_type)
        
        self.current_plan = execution_plan
        
        print(f"📋 Created execution plan with {len(execution_plan.tasks)} tasks")
        print(f"⏱️ Estimated complexity: {execution_plan.complexity_assessment}")
        print(f"🎯 Strategy: {execution_plan.execution_strategy}")
        
        return execution_plan
    
    def execute_plan(self, plan: ExecutionPlan) -> bool:
        """
        Execute the planned tasks in the optimal order.
        
        Args:
            plan: The execution plan to follow
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        print("🚀 Starting coordinated execution...")
        print(f"📋 Executing plan: {plan.request_id}")
        
        plan.status = "executing"
        
        # Sort tasks by dependencies and priority
        execution_order = self._determine_execution_order(plan.tasks)
        
        completed_tasks = set()
        failed_tasks = set()
        
        for task in execution_order:
            print(f"\n🔄 Executing task: {task.description}")
            
            # Check if dependencies are met
            if not self._dependencies_satisfied(task, completed_tasks):
                print(f"⏳ Waiting for dependencies: {task.dependencies}")
                continue
            
            # Execute the task
            task.status = "in_progress"
            success = self._execute_task(task)
            
            if success:
                task.status = "completed"
                completed_tasks.add(task.id)
                print(f"✅ Task completed: {task.description}")
            else:
                task.status = "failed"
                failed_tasks.add(task.id)
                print(f"❌ Task failed: {task.description}")
                
                # Attempt recovery or adjust plan
                if not self._handle_task_failure(task, plan):
                    print("🛑 Critical failure - stopping execution")
                    plan.status = "failed"
                    return False
        
        # Validate overall success
        if self._validate_plan_completion(plan):
            plan.status = "completed"
            print("🎉 Plan execution completed successfully!")
            return True
        else:
            plan.status = "failed"
            print("❌ Plan execution failed validation")
            return False
    
    def coordinate_app_creation(self, app_idea: str) -> Tuple[bool, Optional[str]]:
        """
        Coordinate the creation of a new app using the planning system.
        
        Args:
            app_idea: Description of the app to create
            
        Returns:
            Tuple of (success, error_message)
        """
        print("🎯 Coordinating app creation with intelligent planning...")
        
        try:
            # Step 1: Create execution plan
            plan = self.analyze_and_plan(app_idea, "create_app")
            
            # Step 2: Execute the plan
            success = self.execute_plan(plan)
            
            if success:
                return True, None
            else:
                return False, f"Plan execution failed for: {app_idea}"
                
        except Exception as e:
            print(f"❌ Coordination error: {str(e)}")
            return False, str(e)
    
    def coordinate_app_editing(self, edit_request: str, app_directory: str) -> Tuple[bool, Optional[str]]:
        """
        Coordinate the editing of an existing app using the planning system.
        
        Args:
            edit_request: Description of the changes to make
            app_directory: Path to the app to edit
            
        Returns:
            Tuple of (success, error_message)
        """
        print("🎯 Coordinating app editing with intelligent planning...")
        
        try:
            # Set app context for coordinator
            self._set_app_context(app_directory)
            
            # Step 1: Create execution plan for editing
            plan = self.analyze_and_plan(edit_request, "edit_app")
            
            # Step 2: Execute the plan
            success = self.execute_plan(plan)
            
            if success:
                return True, None
            else:
                return False, f"Edit plan execution failed for: {edit_request}"
                
        except Exception as e:
            print(f"❌ Coordination error: {str(e)}")
            return False, str(e)
    
    def _classify_request_type(self, user_request: str) -> str:
        """Classify the type of user request."""
        request_lower = user_request.lower()
        
        if any(word in request_lower for word in ['create', 'build', 'make', 'new app']):
            return "create_app"
        elif any(word in request_lower for word in ['edit', 'modify', 'change', 'update', 'add', 'remove']):
            return "edit_app"
        elif any(word in request_lower for word in ['feature', 'functionality', 'component']):
            return "add_feature"
        else:
            return "create_app"  # Default
    
    def _get_planning_prompt(self, user_request: str, request_type: str) -> str:
        """Create the prompt for the coordinator LLM to plan the execution."""
        return f"""You are a Senior Software Architect AI coordinating a complex NextJS development task.
Your role is to create a detailed execution plan by breaking down the user request into smaller, manageable tasks.

USER REQUEST: {user_request}
REQUEST TYPE: {request_type}

You must respond with a JSON plan in this EXACT format:

{{
  "complexity_assessment": "simple|moderate|complex|very_complex",
  "execution_strategy": "Description of overall approach",
  "estimated_duration": "estimated time",
  "tasks": [
    {{
      "id": "task_1",
      "type": "create_file|edit_file|analyze_context|validate_build|install_deps",
      "description": "What this task accomplishes",
      "dependencies": ["task_id_that_must_complete_first"],
      "context_needed": ["files or info needed"],
      "priority": 1,
      "estimated_complexity": "simple|moderate|complex",
      "llm_instructions": "Specific instructions for the LLM executing this task",
      "validation_criteria": ["How to check if task succeeded"]
    }}
  ],
  "success_criteria": ["How to know if overall request is successful"]
}}

KEY PRINCIPLES:
1. Break complex requests into 3-8 smaller tasks
2. Each task should be focused and testable
3. Consider dependencies (some tasks need others to complete first)
4. Specify what context/files each task needs to examine
5. Provide clear validation criteria for each task
6. Make LLM instructions specific and actionable

TASK TYPES:
- "analyze_context": Understand existing codebase/requirements
- "create_file": Generate a new file
- "edit_file": Modify existing file
- "validate_build": Test that app builds/runs correctly
- "install_deps": Add new dependencies
- "create_component": Create reusable component
- "setup_routing": Configure app routing
- "add_styling": Add CSS/styling
- "integrate_api": Connect to external APIs

COMPLEXITY GUIDELINES:
- simple: Single file creation, basic edits
- moderate: Multiple files, some logic
- complex: Architecture changes, complex features
- very_complex: Major refactoring, multiple complex features

Create a comprehensive plan that maximizes success and minimizes risk."""

    def _call_coordinator_llm(self, prompt: str, context: str) -> Optional[str]:
        """Call the coordinator LLM with the given prompt."""
        if not self.openai_client:
            print("⚠️ No OpenAI client available for coordination")
            return None
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Senior Software Architect AI that creates detailed execution plans for NextJS development tasks. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Coordinator LLM call failed: {str(e)}")
            return None
    
    def _parse_execution_plan(self, llm_response: str, user_request: str, request_type: str) -> ExecutionPlan:
        """Parse the LLM response into an ExecutionPlan object."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = llm_response[json_start:json_end]
            plan_data = json.loads(json_str)
            
            # Convert tasks to Task objects
            tasks = []
            for i, task_data in enumerate(plan_data.get('tasks', [])):
                task = Task(
                    id=task_data.get('id', f'task_{i+1}'),
                    type=task_data.get('type', 'create_file'),
                    description=task_data.get('description', ''),
                    dependencies=task_data.get('dependencies', []),
                    context_needed=task_data.get('context_needed', []),
                    priority=task_data.get('priority', 3),
                    estimated_complexity=task_data.get('estimated_complexity', 'moderate'),
                    llm_instructions=task_data.get('llm_instructions', ''),
                    validation_criteria=task_data.get('validation_criteria', [])
                )
                tasks.append(task)
            
            # Create ExecutionPlan
            plan = ExecutionPlan(
                request_id=f"req_{int(time.time())}",
                user_request=user_request,
                request_type=request_type,
                complexity_assessment=plan_data.get('complexity_assessment', 'moderate'),
                tasks=tasks,
                execution_strategy=plan_data.get('execution_strategy', 'Sequential execution'),
                estimated_duration=plan_data.get('estimated_duration', 'Unknown'),
                success_criteria=plan_data.get('success_criteria', []),
                created_at=datetime.now().isoformat()
            )
            
            return plan
            
        except Exception as e:
            print(f"❌ Failed to parse execution plan: {str(e)}")
            return self._create_fallback_plan(user_request, request_type)
    
    def _create_fallback_plan(self, user_request: str, request_type: str) -> ExecutionPlan:
        """Create a simple fallback plan when AI planning fails."""
        print("🔄 Creating fallback execution plan...")
        
        tasks = []
        
        if request_type == "create_app":
            tasks = [
                Task(
                    id="analyze_requirements",
                    type="analyze_context",
                    description="Analyze user requirements",
                    dependencies=[],
                    context_needed=["user_request"],
                    priority=1,
                    estimated_complexity="simple",
                    llm_instructions=f"Analyze this request and determine what type of app to build: {user_request}",
                    validation_criteria=["Requirements are clear"]
                ),
                Task(
                    id="create_main_page",
                    type="create_file",
                    description="Create main application page",
                    dependencies=["analyze_requirements"],
                    context_needed=["requirements"],
                    priority=1,
                    estimated_complexity="moderate",
                    llm_instructions="Create the main app/page.tsx file with core functionality",
                    validation_criteria=["File created", "TypeScript compiles"]
                ),
                Task(
                    id="validate_build",
                    type="validate_build",
                    description="Validate app builds correctly",
                    dependencies=["create_main_page"],
                    context_needed=["all_files"],
                    priority=2,
                    estimated_complexity="simple",
                    llm_instructions="Test that the app builds without errors",
                    validation_criteria=["npm run build succeeds"]
                )
            ]
        
        return ExecutionPlan(
            request_id=f"fallback_{int(time.time())}",
            user_request=user_request,
            request_type=request_type,
            complexity_assessment="moderate",
            tasks=tasks,
            execution_strategy="Simple sequential execution with fallback",
            estimated_duration="5-10 minutes",
            success_criteria=["App builds successfully", "Core functionality works"],
            created_at=datetime.now().isoformat()
        )
    
    def _determine_execution_order(self, tasks: List[Task]) -> List[Task]:
        """Determine optimal execution order based on dependencies and priority."""
        # Simple topological sort considering dependencies and priority
        ordered_tasks = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unfulfilled dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep_id in [t.id for t in ordered_tasks] for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Break cycles by picking highest priority task
                ready_tasks = [min(remaining_tasks, key=lambda t: t.priority)]
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda t: t.priority)
            
            # Add the highest priority ready task
            next_task = ready_tasks[0]
            ordered_tasks.append(next_task)
            remaining_tasks.remove(next_task)
        
        return ordered_tasks
    
    def _dependencies_satisfied(self, task: Task, completed_tasks: set) -> bool:
        """Check if all dependencies for a task are satisfied."""
        return all(dep_id in completed_tasks for dep_id in task.dependencies)
    
    def _execute_task(self, task: Task) -> bool:
        """Execute a single task using the appropriate method."""
        print(f"  🔧 Executing: {task.type} - {task.description}")
        
        try:
            if task.type == "analyze_context":
                return self._execute_analyze_context_task(task)
            elif task.type == "create_file":
                return self._execute_create_file_task(task)
            elif task.type == "edit_file":
                return self._execute_edit_file_task(task)
            elif task.type == "validate_build":
                return self._execute_validate_build_task(task)
            elif task.type == "install_deps":
                return self._execute_install_deps_task(task)
            else:
                print(f"⚠️ Unknown task type: {task.type}")
                return False
                
        except Exception as e:
            print(f"❌ Task execution error: {str(e)}")
            return False
    
    def _execute_analyze_context_task(self, task: Task) -> bool:
        """Execute a context analysis task."""
        # This would analyze the current codebase or requirements
        # For now, we'll simulate this
        print("  📊 Analyzing context...")
        task.result = {"analysis": "Context analyzed", "insights": []}
        return True
    
    def _execute_create_file_task(self, task: Task) -> bool:
        """Execute a file creation task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  📄 Creating file based on: {task.llm_instructions}")
        
        try:
            # Build prompt for file creation
            context_info = ""
            if task.context_needed:
                context_info = self._gather_context_for_task(task)
            
            create_prompt = f"""You are creating a specific file for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}

Generate the file content using the <new filename="path/to/file"> syntax.
Focus ONLY on this specific file and task.
Make sure the file is complete and follows NextJS/TypeScript best practices.
"""
            
            # Call LLM to generate file content
            is_valid, response = self.app_builder.make_openai_request(create_prompt, "create")
            
            if not is_valid:
                print("❌ Failed to generate file content")
                return False
            
            # Apply the generated content using CodeBuilder
            from .code_builder import CodeBuilder
            import tempfile
            import os
            
            # Save to temporary file
            temp_file = f"temp_task_{task.id}_{int(time.time())}.txt"
            with open(temp_file, 'w') as f:
                f.write(response)
            
            # Apply using CodeBuilder
            if hasattr(self.app_builder, 'get_app_path'):
                app_path = self.app_builder.get_app_path()
            else:
                # Fallback for new app creation
                app_path = "."
            
            code_builder = CodeBuilder(temp_file, app_path)
            success = code_builder.build()
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            if success:
                task.result = {"file_created": True, "method": "llm_generated"}
                print(f"  ✅ File created successfully")
                return True
            else:
                print(f"  ❌ Failed to apply file creation")
                return False
                
        except Exception as e:
            print(f"  ❌ Error creating file: {str(e)}")
            return False
    
    def _execute_edit_file_task(self, task: Task) -> bool:
        """Execute a file editing task."""
        print("  ✏️ Editing file...")
        
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        try:
            # Gather context for editing
            context_info = self._gather_context_for_task(task)
            
            # Create focused edit prompt
            edit_prompt = f"""You are editing a specific file in a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CURRENT CONTEXT:
{context_info}

Generate the changes using the intent-based editing approach.
Be precise and make only the necessary changes to accomplish the task.
"""
            
            # Use intent-based editing for more reliability
            success = self.app_builder._edit_app_with_intents(edit_prompt)
            
            if success:
                task.result = {"file_edited": True, "method": "intent_based"}
                print(f"  ✅ File edited successfully")
                return True
            else:
                print(f"  ❌ Failed to edit file")
                return False
                
        except Exception as e:
            print(f"  ❌ Error editing file: {str(e)}")
            return False
    
    def _execute_validate_build_task(self, task: Task) -> bool:
        """Execute a build validation task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print("  🔨 Validating build...")
        return self.app_builder.build_and_run(auto_install_deps=False)
    
    def _execute_install_deps_task(self, task: Task) -> bool:
        """Execute a dependency installation task."""
        print("  📦 Installing dependencies...")
        # This would run npm install or add specific packages
        return True
    
    def _handle_task_failure(self, task: Task, plan: ExecutionPlan) -> bool:
        """Handle task failure and attempt recovery."""
        print(f"🔄 Handling failure for task: {task.id}")
        
        # Simple retry logic
        if task.status == "failed":
            print("  🔁 Attempting retry...")
            success = self._execute_task(task)
            if success:
                task.status = "completed"
                return True
        
        # If retry fails, check if this is a critical task
        critical_types = ["create_file", "validate_build"]
        if task.type in critical_types:
            print("  🚨 Critical task failed - stopping execution")
            return False
        
        # For non-critical tasks, continue
        print("  ⚠️ Non-critical task failed - continuing")
        return True
    
    def _validate_plan_completion(self, plan: ExecutionPlan) -> bool:
        """Validate that the plan was completed successfully."""
        # Check that all critical tasks completed
        critical_tasks = [t for t in plan.tasks if t.type in ["create_file", "validate_build"]]
        critical_completed = all(t.status == "completed" for t in critical_tasks)
        
        # Check success criteria
        # For now, we'll just check critical tasks
        return critical_completed
    
    def _set_app_context(self, app_directory: str):
        """Set the app context for editing operations."""
        if self.app_builder:
            app_name = Path(app_directory).name
            self.app_builder.app_name = app_name
            self.app_builder.apps_dir = Path(app_directory).parent
    
    def _gather_context_for_task(self, task: Task) -> str:
        """Gather relevant context for a specific task."""
        if not self.app_builder or not hasattr(self.app_builder, 'app_name'):
            return "No app context available"
        
        try:
            app_path = os.path.join(self.app_builder.apps_dir, self.app_builder.app_name)
            context_info = f"App: {self.app_builder.app_name}\nLocation: {app_path}\n\n"
            
            # Get file list
            if os.path.exists(app_path):
                context_info += "Available files:\n"
                for root, dirs, files in os.walk(app_path):
                    for file in files:
                        if file.endswith(('.tsx', '.ts', '.js', '.jsx')):
                            rel_path = os.path.relpath(os.path.join(root, file), app_path)
                            context_info += f"  - {rel_path}\n"
            
            # Add specific context based on task type
            if task.task_type == "create_file":
                context_info += f"\nTask: Create {task.description}\n"
                context_info += "Important: Use proper NextJS App Router syntax with 'use client' directive for client components\n"
                context_info += "Important: Use useRouter from 'next/navigation' not 'next/router'\n"
                context_info += "Important: Ensure all JSX return statements are properly closed with parentheses\n"
            elif task.task_type == "edit_file":
                context_info += f"\nTask: Edit {task.description}\n"
                context_info += "Important: When generating diffs, account for 'use client' directives at the top of files\n"
                context_info += "Important: Provide exact context matching for diff application\n"
            
            return context_info
            
        except Exception as e:
            return f"Error gathering context: {str(e)}"
    
    def _get_basic_file_structure(self, app_path: str) -> str:
        """Get basic file structure for context."""
        try:
            from pathlib import Path
            app_dir = Path(app_path)
            
            if not app_dir.exists():
                return "App directory does not exist"
            
            structure = []
            for item in app_dir.rglob("*"):
                if item.is_file() and item.suffix in ['.tsx', '.ts', '.js', '.jsx']:
                    relative_path = item.relative_to(app_dir)
                    structure.append(str(relative_path))
            
            return "\n".join(sorted(structure)[:10])  # Limit to first 10 files
            
        except Exception as e:
            return f"Error getting file structure: {str(e)}"
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution state."""
        if not self.current_plan:
            return {"status": "no_plan"}
        
        completed_tasks = sum(1 for t in self.current_plan.tasks if t.status == "completed")
        total_tasks = len(self.current_plan.tasks)
        
        return {
            "request_id": self.current_plan.request_id,
            "status": self.current_plan.status,
            "progress": f"{completed_tasks}/{total_tasks}",
            "completion_percentage": int((completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0,
            "complexity": self.current_plan.complexity_assessment,
            "strategy": self.current_plan.execution_strategy
        } 