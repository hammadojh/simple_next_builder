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

# Import progress loader system
from .progress_loader import (
    show_progress, llm_progress, analysis_progress, 
    build_progress, file_progress, update_current_task,
    LoaderStyle
)


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


class TaskMemory:
    """
    Shared memory system for tasks to share context and results.
    Enables tasks to build incrementally on previous work.
    """
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self.files_created: List[str] = []
        self.files_modified: List[str] = []
        self.build_status: Optional[bool] = None
        
    def store_result(self, task_id: str, result: Any):
        """Store the result of a completed task."""
        self.results[task_id] = result
        print(f"  💾 Stored result for task {task_id}: {type(result).__name__}")
    
    def get_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a specific task."""
        return self.results.get(task_id)
    
    def get_dependencies_results(self, task: Task) -> Dict[str, Any]:
        """Get results from all dependency tasks."""
        dep_results = {}
        for dep_id in task.dependencies:
            if dep_id in self.results:
                dep_results[dep_id] = self.results[dep_id]
        return dep_results
    
    def store_context(self, key: str, value: Any):
        """Store shared context information."""
        self.context[key] = value
    
    def get_context(self, key: str, default=None):
        """Get shared context information."""
        return self.context.get(key, default)
    
    def add_file_created(self, filename: str):
        """Track a newly created file."""
        if filename not in self.files_created:
            self.files_created.append(filename)
    
    def add_file_modified(self, filename: str):
        """Track a modified file."""
        if filename not in self.files_modified:
            self.files_modified.append(filename)
    
    def get_all_affected_files(self) -> List[str]:
        """Get all files that have been created or modified."""
        return list(set(self.files_created + self.files_modified))
    
    def clear(self):
        """Clear all stored data."""
        self.results.clear()
        self.context.clear()
        self.files_created.clear()
        self.files_modified.clear()
        self.build_status = None


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
        self.task_memory = TaskMemory()
        
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
    
    def analyze_and_plan(self, user_request: str, request_type: str = "auto", mvp_spec=None) -> ExecutionPlan:
        """
        Analyze the user request and create a detailed execution plan.
        
        Args:
            user_request: The user's request/idea (can be original or enhanced MVP)
            request_type: Type of request ('create_app', 'edit_app', 'add_feature', 'auto')
            mvp_spec: Enhanced MVP specification if available
            
        Returns:
            ExecutionPlan: Detailed plan for executing the request
        """
        with analysis_progress("user request and creating execution plan"):
            # Use MVP spec if available, otherwise use original request
            planning_request = user_request
            if mvp_spec:
                update_current_task("using enhanced MVP specification")
                print(f"📋 Using enhanced MVP specification")
                print(f"🎯 Complexity level: {mvp_spec.complexity_level}")
                print(f"📊 Estimated components: {mvp_spec.estimated_components}")
                planning_request = mvp_spec.enhanced_prompt
            else:
                update_current_task("using original request")
                print(f"📝 Using original request: {user_request}")
            
            # Auto-detect request type if not specified
            if request_type == "auto":
                update_current_task("classifying request type")
                request_type = self._classify_request_type(planning_request)
            
            print(f"🔍 Detected request type: {request_type}")
            
            # Use coordinator LLM to create detailed plan
            print("🎯 Creating detailed execution plan...")
            planning_prompt = self._get_planning_prompt(planning_request, request_type, mvp_spec)
            
            # Stop the current progress loader before streaming
            from .progress_loader import progress_manager
            progress_manager.cleanup_all()
            
            plan_response = self._call_coordinator_llm(planning_prompt, "planning")
            
            if not plan_response:
                update_current_task("creating fallback plan")
                # Fallback to simple plan
                return self._create_fallback_plan(planning_request, request_type, mvp_spec)
            
            # Parse the plan from LLM response
            update_current_task("parsing execution plan")
            execution_plan = self._parse_execution_plan(plan_response, planning_request, request_type)
            
            self.current_plan = execution_plan
            
        print(f"📋 Created execution plan with {len(execution_plan.tasks)} tasks")
        print(f"⏱️ Estimated complexity: {execution_plan.complexity_assessment}")
        print(f"🎯 Strategy: {execution_plan.execution_strategy}")
        
        # Validate the plan before returning
        validation_result = self._validate_execution_plan(execution_plan)
        if not validation_result["valid"]:
            print(f"⚠️ Plan validation issues found: {', '.join(validation_result['issues'])}")
            # Fix common issues in the plan
            execution_plan = self._fix_plan_issues(execution_plan, validation_result["issues"])
            print(f"🔧 Fixed plan issues automatically")
        
        return execution_plan
    
    def execute_plan(self, plan: ExecutionPlan) -> bool:
        """
        Execute the planned tasks in the optimal order.
        
        Args:
            plan: The execution plan to follow
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        # Disable animated spinner to avoid conflicts with LLM output
        print(f"🚀 Executing {len(plan.tasks)} coordinated tasks...")
        print("🚀 Starting coordinated execution...")
        print(f"📋 Executing plan: {plan.request_id}")
        
        plan.status = "executing"
        
        # Clear task memory for new execution
        self.task_memory.clear()
        
        # Sort tasks by dependencies and priority
        update_current_task("determining execution order")
        execution_order = self._determine_execution_order(plan.tasks)
        
        completed_tasks = set()
        failed_tasks = set()
        
        for i, task in enumerate(execution_order, 1):
            update_current_task(f"[{i}/{len(execution_order)}] {task.description}")
            print(f"\n🔄 Executing task: {task.description}")
            
            # Check if dependencies are met
            if not self._dependencies_satisfied(task, completed_tasks):
                print(f"⏳ Waiting for dependencies: {task.dependencies}")
                # Skip this task for now, will be retried in next iteration
                continue
            
            # Execute the task with access to dependency results
            task.status = "in_progress"
            success = self._execute_task_with_context(task)
            
            if success:
                task.status = "completed"
                completed_tasks.add(task.id)
                print(f"✅ Task completed: {task.description}")
                
                # Store task result in memory if available
                if task.result:
                    self.task_memory.store_result(task.id, task.result)
                    
            else:
                task.status = "failed"
                failed_tasks.add(task.id)
                print(f"❌ Task failed: {task.description}")
                
                # Attempt recovery or adjust plan
                update_current_task(f"handling failure for task {task.id}")
                if not self._handle_task_failure(task, plan):
                    print("🛑 Critical failure - stopping execution")
                    plan.status = "failed"
                    return False
        
        # Check if all tasks completed
        incomplete_tasks = [t for t in plan.tasks if t.status != "completed"]
        if incomplete_tasks:
            print(f"⚠️ {len(incomplete_tasks)} tasks still incomplete - retrying...")
            # Try one more pass for incomplete tasks
            max_retries = 2
            for retry in range(max_retries):
                remaining = [t for t in incomplete_tasks if t.status != "completed"]
                if not remaining:
                    break
                
                print(f"🔄 Retry {retry + 1}/{max_retries} for {len(remaining)} tasks")
                for task in remaining:
                    if self._dependencies_satisfied(task, completed_tasks):
                        task.status = "in_progress"
                        success = self._execute_task_with_context(task)
                        if success:
                            task.status = "completed"
                            completed_tasks.add(task.id)
                            if task.result:
                                self.task_memory.store_result(task.id, task.result)
        
        # Validate overall success
        update_current_task("validating plan completion")
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
        Coordinate the editing of an existing app using context-first intelligent planning.
        
        ENHANCED: Now analyzes codebase FIRST before creating plans
        
        Args:
            edit_request: Description of the changes to make
            app_directory: Path to the app to edit
            
        Returns:
            Tuple of (success, error_message)
        """
        print("🎯 Coordinating app editing with context-first intelligent planning...")
        
        try:
            # Set app context for coordinator
            self._set_app_context(app_directory)
            
            # Step 1: FIRST analyze the existing codebase
            print("🔍 Phase 1: Analyzing existing codebase...")
            codebase_analysis = self._analyze_existing_codebase(app_directory, edit_request)
            
            if not codebase_analysis["success"]:
                print(f"❌ Codebase analysis failed: {codebase_analysis['error']}")
                return False, f"Could not analyze codebase: {codebase_analysis['error']}"
            
            # Step 2: Create context-aware execution plan
            print("📋 Phase 2: Creating context-aware execution plan...")
            plan = self.analyze_and_plan_with_context(edit_request, "edit_app", codebase_analysis)
            
            # Step 3: Execute the plan
            print("🚀 Phase 3: Executing focused plan...")
            success = self.execute_plan(plan)
            
            if success:
                # Step 4: MANDATORY build validation after any edit
                print("🔨 Phase 4: Final build validation...")
                build_success = self._validate_final_build()
                
                if build_success:
                    print("✅ All edits completed and build validated successfully!")
                    return True, None
                else:
                    print("⚠️ Edits completed but build has issues - attempting auto-fixes...")
                    # Try to auto-fix build issues
                    fix_success = self._auto_fix_build_issues()
                    if fix_success:
                        print("✅ Build issues resolved!")
                        return True, None
                    else:
                        return False, "Edits completed but build validation failed"
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
    
    def _get_planning_prompt(self, user_request: str, request_type: str, mvp_spec=None) -> str:
        """Create the prompt for the coordinator LLM to plan the execution."""
        
        # Build MVP context if available
        mvp_context = ""
        if mvp_spec:
            mvp_context = f"""
MVP SPECIFICATION PROVIDED:
- Core Features: {', '.join(mvp_spec.core_features)}
- Technical Requirements: {', '.join(mvp_spec.technical_requirements)}
- UI Components: {', '.join(mvp_spec.ui_components)}
- Routing Structure: {', '.join(mvp_spec.routing_structure)}
- Styling Approach: {mvp_spec.styling_approach}
- Complexity Level: {mvp_spec.complexity_level}
- Estimated Components: {mvp_spec.estimated_components}
- Tech Stack: {', '.join(mvp_spec.suggested_tech_stack)}

This is a comprehensive MVP that should result in a production-ready, shareable application.
Plan accordingly for higher complexity and more thorough implementation."""
        
        return f"""You are a Senior Software Architect AI coordinating a complex FRONTEND-ONLY NextJS development task.
Your role is to create a detailed execution plan by breaking down the user request into smaller, manageable frontend tasks.

USER REQUEST: {user_request}
REQUEST TYPE: {request_type}
{mvp_context}

CRITICAL: This is a FRONTEND-ONLY application. NO authentication, NO database, NO backend APIs.
Focus on rich UI/UX, component architecture, and client-side functionality.

You must respond with a JSON plan in this EXACT format:

{{
  "complexity_assessment": "simple|moderate|complex|very_complex",
  "execution_strategy": "Description of frontend-focused approach",
  "estimated_duration": "estimated time",
  "tasks": [
    {{
      "id": "task_1",
      "type": "create_file|edit_file|analyze_context|validate_build|create_component|setup_routing|setup_styling",
      "description": "What this frontend task accomplishes",
      "dependencies": ["task_id_that_must_complete_first"],
      "context_needed": ["files or info needed"],
      "priority": 1,
      "estimated_complexity": "simple|moderate|complex",
      "llm_instructions": "Specific frontend-focused instructions for the LLM executing this task",
      "validation_criteria": ["How to check if frontend task succeeded"]
    }}
  ],
  "success_criteria": ["How to know if frontend application is successful"]
}}

KEY PRINCIPLES FOR FRONTEND-ONLY PLANNING:
1. For MVP specs, break into 5-15 tasks for comprehensive frontend implementation
        2. Include production-ready patterns (error handling, loading states, responsive design, modern UI components)
3. Plan for proper component architecture and reusability
4. Consider client-side data flow and state management needs
5. Include proper routing setup and navigation (client-side only)
6. Plan for styling consistency and theme implementation
7. Focus on localStorage/sessionStorage for data persistence
8. Each task should be focused and testable
9. Consider dependencies (some tasks need others to complete first)
10. Specify what context/files each task needs to examine
11. Provide clear validation criteria for each task
12. NO authentication, NO database, NO backend APIs

ENHANCED TASK TYPES FOR FRONTEND:
- "analyze_context": Understand existing codebase/requirements
- "create_file": Generate a new frontend file
- "edit_file": Modify existing frontend file
- "validate_build": Test that app builds/runs correctly (frontend only)
- "create_component": Create reusable React component with state management
- "setup_routing": Configure NextJS client-side routing and navigation
        - "setup_styling": Configure styling system (shadcn/ui with Tailwind, etc.)
- "create_layout": Create app layout and navigation structure
- "setup_state": Configure client-side state management (useState, localStorage)
- "add_features": Implement specific frontend feature functionality
- "setup_interactions": Add interactivity (forms, modals, animations)
- "optimize_ux": Enhance user experience and accessibility

Focus on creating rich, interactive, production-ready frontend applications."""

    def _call_coordinator_llm(self, prompt: str, context: str) -> Optional[str]:
        """Call the coordinator LLM with real-time streaming."""
        
        # Prefer Anthropic Claude 4 Sonnet if available, fallback to OpenAI GPT-4
        if self.app_builder and self.app_builder.anthropic_client:
            try:
                print("\n🧠 Claude 4 Sonnet Planning:")
                
                messages = [{"role": "user", "content": prompt}]
                
                # Use streaming API for real-time response
                full_response = ""
                with self.app_builder.anthropic_client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=3000,
                    temperature=0.3,
                    system="You are a Senior Software Architect AI that creates detailed execution plans for NextJS development tasks. Always respond with valid JSON.",
                    messages=messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        # Stream each token as it arrives
                        print(text, end="", flush=True)
                
                print("\n")
                return full_response.strip()
                
            except Exception as e:
                print(f"❌ Claude coordination failed, falling back to GPT-4: {str(e)}")
        
        # Fallback to OpenAI GPT-4
        if not self.openai_client:
            print("⚠️ No LLM client available for coordination")
            return None
        
        try:
            print("\n🧠 GPT-4 Planning:")
            
            # Use streaming API for real-time response
            response_stream = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Senior Software Architect AI that creates detailed execution plans for NextJS development tasks. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000,
                stream=True  # Enable streaming
            )
            
            full_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Stream each token as it arrives
                    print(content, end="", flush=True)
            
            print("\n")
            return full_response.strip()
            
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
    
    def _parse_plan_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse the LLM response for context-aware planning into a dictionary."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = llm_response[json_start:json_end]
            plan_data = json.loads(json_str)
            
            # Ensure all required fields exist with defaults
            plan_data.setdefault('complexity_assessment', 'moderate')
            plan_data.setdefault('execution_strategy', 'Context-aware focused editing')
            plan_data.setdefault('estimated_duration', '30-60 minutes')
            plan_data.setdefault('tasks', [])
            plan_data.setdefault('success_criteria', ['Edit completed successfully'])
            
            # Ensure each task has validation_criteria
            for task_data in plan_data.get('tasks', []):
                task_data.setdefault('validation_criteria', ['Task completed successfully'])
                task_data.setdefault('id', f"task_{len(plan_data['tasks'])}")
                task_data.setdefault('type', 'edit_file')
                task_data.setdefault('description', 'Complete the assigned task')
                task_data.setdefault('dependencies', [])
                task_data.setdefault('context_needed', [])
                task_data.setdefault('priority', 1)
                task_data.setdefault('estimated_complexity', 'moderate')
                task_data.setdefault('llm_instructions', 'Complete the task as described')
            
            return plan_data
            
        except Exception as e:
            print(f"❌ Failed to parse plan response: {str(e)}")
            # Return minimal fallback dictionary
            return {
                'complexity_assessment': 'moderate',
                'execution_strategy': 'Fallback minimal edit',
                'estimated_duration': '30 minutes',
                'tasks': [{
                    'id': 'fallback_task',
                    'type': 'edit_file',
                    'description': 'Apply requested changes',
                    'dependencies': [],
                    'context_needed': [],
                    'priority': 1,
                    'estimated_complexity': 'moderate',
                    'llm_instructions': 'Apply the requested changes',
                    'validation_criteria': ['Changes applied successfully']
                }],
                'success_criteria': ['Edit completed successfully']
            }
    
    def _create_fallback_plan(self, user_request: str, request_type: str, mvp_spec=None) -> ExecutionPlan:
        """Create a simple fallback plan when AI planning fails."""
        print("🔄 Creating fallback execution plan...")
        
        tasks = []
        complexity = "moderate"
        
        if request_type == "create_app":
            if mvp_spec:
                # Enhanced fallback for MVP specs
                print(f"📋 Creating enhanced fallback for MVP ({mvp_spec.complexity_level})")
                complexity = mvp_spec.complexity_level
                
                tasks = [
                    Task(
                        id="setup_foundation",
                        type="create_file",
                        description="Setup app foundation and layout",
                        dependencies=[],
                        context_needed=["mvp_spec"],
                        priority=1,
                        estimated_complexity="moderate",
                        llm_instructions=f"Create the foundational layout and structure for: {mvp_spec.enhanced_prompt}. Include proper TypeScript types, responsive design, and frontend-only architecture. NO auth, NO database, NO backend APIs.",
                        validation_criteria=["Layout created", "TypeScript compiles", "Responsive design", "Frontend-only architecture"]
                    ),
                    Task(
                        id="setup_routing",
                        type="setup_routing",
                        description="Configure NextJS routing structure",
                        dependencies=["setup_foundation"],
                        context_needed=["layout", "mvp_spec"],
                        priority=1,
                        estimated_complexity="simple",
                        llm_instructions=f"Setup client-side routing for: {', '.join(mvp_spec.routing_structure)}. Focus on frontend navigation only.",
                        validation_criteria=["Routes configured", "Navigation works", "Client-side routing only"]
                    ),
                    Task(
                        id="create_components",
                        type="create_component",
                        description="Create core UI components",
                        dependencies=["setup_routing"],
                        context_needed=["layout", "routing"],
                        priority=1,
                        estimated_complexity="complex",
                        llm_instructions=f"Create these frontend components: {', '.join(mvp_spec.ui_components)}. Include proper TypeScript types, error handling, loading states, and rich interactivity. Use React state and localStorage for data management.",
                        validation_criteria=["Components created", "TypeScript compiles", "Components render correctly", "Interactive features work"]
                    ),
                    Task(
                        id="implement_features",
                        type="add_features",
                        description="Implement core frontend features",
                        dependencies=["create_components"],
                        context_needed=["components", "routing"],
                        priority=1,
                        estimated_complexity="complex",
                        llm_instructions=f"Implement these core frontend features: {', '.join(mvp_spec.core_features)}. Make it production-ready with proper error handling, local state management, localStorage persistence. NO backend dependencies.",
                        validation_criteria=["Features implemented", "Error handling added", "Loading states included", "Local persistence works"]
                    ),
                    Task(
                        id="setup_styling",
                        type="setup_styling",
                        description="Configure styling system",
                        dependencies=["implement_features"],
                        context_needed=["components", "layout"],
                        priority=2,
                        estimated_complexity="moderate",
                        llm_instructions=f"Setup {mvp_spec.styling_approach} styling system with consistent theme, responsive design, and modern UI patterns. Use shadcn/ui components like Button, Input, Card, etc. for professional UI. Focus on excellent user experience with modern component library.",
                        validation_criteria=["Styling configured", "Theme consistent", "Responsive design", "Modern UI patterns"]
                    ),
                    Task(
                        id="validate_frontend",
                        type="validate_build",
                        description="Validate complete frontend application",
                        dependencies=["setup_styling"],
                        context_needed=["all_files"],
                        priority=3,
                        estimated_complexity="simple",
                        llm_instructions="Test that the complete frontend application builds and runs without errors. Verify all features work with frontend-only architecture.",
                        validation_criteria=["npm run build succeeds", "App starts correctly", "All features work", "No backend dependencies"]
                    )
                ]
            else:
                # Simple fallback for basic requests
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
            complexity_assessment=complexity,
            tasks=tasks,
            execution_strategy="Enhanced MVP-aware sequential execution" if mvp_spec else "Simple sequential execution with fallback",
            estimated_duration="15-30 minutes" if mvp_spec else "5-10 minutes",
            success_criteria=["Production-ready MVP completed", "All features working", "App is shareable"] if mvp_spec else ["App builds successfully", "Core functionality works"],
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
            elif task.type == "create_component":
                return self._execute_create_component_task(task)
            elif task.type == "setup_styling":
                return self._execute_setup_styling_task(task)
            elif task.type == "setup_routing":
                return self._execute_setup_routing_task(task)
            elif task.type == "create_layout":
                return self._execute_create_layout_task(task)
            elif task.type == "setup_state":
                return self._execute_setup_state_task(task)
            elif task.type == "add_features":
                return self._execute_add_features_task(task)
            elif task.type == "setup_interactions":
                return self._execute_setup_interactions_task(task)
            elif task.type == "optimize_ux":
                return self._execute_optimize_ux_task(task)
            elif task.type == "install_dependency":
                return self._execute_install_dependency_task(task)
            else:
                print(f"⚠️ Unknown task type: {task.type}")
                return False
                
        except Exception as e:
            print(f"❌ Task execution error: {str(e)}")
            return False
    
    def _execute_task_with_context(self, task: Task) -> bool:
        """Execute a task with access to shared memory and dependency results."""
        print(f"  🔧 Executing: {task.type} - {task.description}")
        
        # Get results from dependency tasks
        dep_results = self.task_memory.get_dependencies_results(task)
        if dep_results:
            print(f"  📊 Using results from {len(dep_results)} dependency tasks")
        
        try:
            if task.type == "analyze_context":
                return self._execute_analyze_context_task(task)
            elif task.type == "create_file":
                return self._execute_create_file_task(task, dep_results)
            elif task.type == "edit_file":
                return self._execute_edit_file_task(task, dep_results)
            elif task.type == "validate_build":
                return self._execute_validate_build_task(task)
            elif task.type == "install_deps":
                return self._execute_install_deps_task(task)
            elif task.type == "create_component":
                return self._execute_create_component_task(task, dep_results)
            elif task.type == "setup_styling":
                return self._execute_setup_styling_task(task, dep_results)
            elif task.type == "setup_routing":
                return self._execute_setup_routing_task(task, dep_results)
            elif task.type == "create_layout":
                return self._execute_create_layout_task(task, dep_results)
            elif task.type == "setup_state":
                return self._execute_setup_state_task(task, dep_results)
            elif task.type == "add_features":
                return self._execute_add_features_task(task, dep_results)
            elif task.type == "setup_interactions":
                return self._execute_setup_interactions_task(task, dep_results)
            elif task.type == "optimize_ux":
                return self._execute_optimize_ux_task(task, dep_results)
            elif task.type == "install_dependency":
                return self._execute_install_dependency_task(task)
            else:
                print(f"⚠️ Unknown task type: {task.type}")
                return False
                
        except Exception as e:
            print(f"❌ Task execution error: {str(e)}")
            return False
    
    def _execute_analyze_context_task(self, task: Task) -> bool:
        """Execute a context analysis task."""
        with analysis_progress("context and requirements"):
            # This would analyze the current codebase or requirements
            # For now, we'll simulate this
            print("  📊 Analyzing context...")
            task.result = {"analysis": "Context analyzed", "insights": []}
            return True
    
    def _execute_create_file_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a file creation task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        with file_progress(f"creating file for {task.description}"):
            print(f"  📄 Creating file based on: {task.llm_instructions}")
            
            try:
                # Build prompt for file creation
                update_current_task("gathering context information")
                context_info = ""
                if task.context_needed:
                    context_info = self._gather_context_for_task(task)
                
                # Add dependency results to context
                dep_context = ""
                if dep_results:
                    dep_context = "\n\nDEPENDENCY RESULTS:\n"
                    for dep_id, result in dep_results.items():
                        dep_context += f"- {dep_id}: {result}\n"
                
                create_prompt = f"""You are creating a specific file for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Generate the file content using the <new filename="path/to/file"> syntax.
Focus ONLY on this specific file and task.
Make sure the file is complete and follows NextJS/TypeScript best practices.
"""
                
                # Call LLM to generate file content
                update_current_task("generating file content with AI")
                with llm_progress(f"Code Generation AI"):
                    is_valid, response = self.app_builder.make_openai_request(create_prompt, "create")
                
                if not is_valid:
                    print("❌ Failed to generate file content")
                    return False
                
                # Apply the generated content using CodeBuilder
                update_current_task("applying generated content")
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
                
                with build_progress("applying file changes"):
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
    
    def _execute_edit_file_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a file editing task."""
        print("  ✏️ Editing file...")
        
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        try:
            # Gather context for editing
            context_info = self._gather_context_for_task(task)
            
            # Add dependency results to context
            dep_context = ""
            if dep_results:
                dep_context = "\n\nDEPENDENCY RESULTS:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            # Enhance the task description for better specificity
            try:
                from .request_enhancer import RequestEnhancer
                enhancer = RequestEnhancer()
                
                if hasattr(self.app_builder, 'get_app_path'):
                    app_path = self.app_builder.get_app_path()
                    enhanced_description = enhancer.enhance_edit_request(
                        user_request=task.description,
                        app_path=app_path,
                        context_summary=(context_info + dep_context)[:500]  # Brief context summary
                    )
                else:
                    enhanced_description = task.description
            except Exception as e:
                print(f"⚠️ Request enhancement failed: {e}")
                enhanced_description = task.description
            
            # Create focused edit prompt with proper diff format
            edit_prompt = f"""You are editing a specific file in a NextJS application.

TASK: {enhanced_description}
ORIGINAL TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CURRENT CONTEXT:
{context_info}

STEP-BY-STEP APPROACH:
Think through this edit systematically:

1. **ANALYZE THE TASK**: What specific changes are needed?
2. **IDENTIFY TARGET FILES**: Which files need modification?
3. **PLAN IMPLEMENTATION**: What code changes will accomplish this?
4. **CONSIDER IMPACTS**: What other files might be affected?
5. **EXECUTE PRECISELY**: Generate the exact diff needed.

🚨 CRITICAL JSON GENERATION RULES 🚨

You MUST respond with JSON in this EXACT format:

```json
{{
    "intents": [
        {{
            "file_path": "components/ThemeContext.tsx",
            "action": "replace",
            "target": "const [isDarkMode, setIsDarkMode] = useState(false);",
            "replacement": "const [isDarkMode, setIsDarkMode] = useState(false);\n  const [isLoading, setIsLoading] = useState(true);",
            "context": "add loading state for theme initialization"
        }}
    ]
}}
```

AVAILABLE ACTIONS:
- **replace**: Replace specific text/code with new text
- **modify**: Add content after a target (for insertions)
- **insert**: Insert new content at specific location
- **delete**: Remove specific content

CRITICAL REQUIREMENTS:
1. Use ONLY valid JSON format with "intents" array
2. Each intent must have: file_path, action, target, replacement, context
3. Use EXACT quotes and escaping for JSON
4. Target must be SPECIFIC enough to uniquely identify location
5. Include proper context description for each change
6. Use forward slashes for file paths (not backslashes)

EXAMPLE FOR NEXTJS DARK MODE:
```json
{{
    "intents": [
        {{
            "file_path": "components/ThemeContext.tsx",
            "action": "replace", 
            "target": "export const ThemeProvider = ({{ children }}: {{ children: ReactNode }}) => {{",
            "replacement": "export const ThemeProvider = ({{ children }}: {{ children: ReactNode }}) => {{\n  const [isDarkMode, setIsDarkMode] = useState(false);\n  const toggleTheme = () => setIsDarkMode(!isDarkMode);",
            "context": "add dark mode state and toggle function"
        }}
    ]
}}
```

Generate structured editing intents in JSON format that implement the requested changes."""
            
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
        
        if not self.app_builder:
            print("  ❌ No app builder available")
            return False
        
        try:
            app_path = self.app_builder.get_app_path()
            from .dependency_manager import DependencyManager
            
            dependency_manager = DependencyManager(app_path)
            success = dependency_manager.auto_manage_dependencies()
            
            if success:
                print("  ✅ Dependencies analyzed and installed successfully!")
                return True
            else:
                print("  ⚠️ Dependency management completed with warnings")
                return True  # Still consider it successful
                
        except Exception as e:
            print(f"  ❌ Error installing dependencies: {str(e)}")
            return False
    
    def _execute_install_dependency_task(self, task: Task) -> bool:
        """Execute a specific dependency installation task."""
        print(f"  📦 Installing specific dependency...")
        
        if not self.app_builder:
            print("  ❌ No app builder available")
            return False
        
        try:
            app_path = self.app_builder.get_app_path()
            from .dependency_manager import DependencyManager
            
            dependency_manager = DependencyManager(app_path)
            
            # Extract package name from task description if possible
            description = task.description.lower()
            package_name = None
            
            # Look for package name in description
            if "framer-motion" in description:
                package_name = "framer-motion"
            elif "react-hook-form" in description:
                package_name = "react-hook-form"
            # Add more package detection as needed
            
            if package_name:
                print(f"  🎯 Targeting package: {package_name}")
                
            # Run full dependency analysis (this will catch any missing packages)
            success = dependency_manager.auto_manage_dependencies()
            
            if success:
                print(f"  ✅ Dependencies analyzed and installed successfully!")
                return True
            else:
                print(f"  ⚠️ Dependency management completed with warnings")
                return True  # Still consider it successful
                
        except Exception as e:
            print(f"  ❌ Error installing dependency: {str(e)}")
            return False
    
    def _validate_final_build(self) -> bool:
        """Validate the final build after all edits are complete."""
        if not self.app_builder:
            print("❌ No app builder available for validation")
            return False
        
        try:
            print("🔨 Running final build validation...")
            success = self.app_builder.build_and_run(auto_install_deps=False)
            
            if success:
                print("✅ Final build validation successful!")
                return True
            else:
                print("❌ Final build validation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during final build validation: {str(e)}")
            return False
    
    def _auto_fix_build_issues(self) -> bool:
        """Attempt to automatically fix build issues using smart analysis."""
        if not self.app_builder:
            print("❌ No app builder available for auto-fixing")
            return False
        
        try:
            print("🧠 Smart auto-fix: Analyzing build issues...")
            
            # First, try smart build error analysis
            app_path = self.app_builder.get_app_path()
            
            import subprocess
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=app_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✅ Build is already successful!")
                return True
            
            # Use smart build error analyzer
            from .build_error_analyzer import SmartBuildErrorAnalyzer
            from .smart_task_executor import SmartTaskExecutor
            
            analyzer = SmartBuildErrorAnalyzer(app_path)
            analysis = analyzer.analyze_build_error(result.stderr or result.stdout)
            
            print(f"   📋 Analysis: {analysis.error_summary}")
            print(f"   🎯 Root cause: {analysis.root_cause}")
            print(f"   📊 Confidence: {analysis.confidence:.1%}")
            
            if analysis.confidence > 0.4 and analysis.tasks:
                executor = SmartTaskExecutor(app_path)
                success, task_results = executor.execute_analysis(analysis)
                
                if success:
                    print("✅ Smart auto-fix completed successfully!")
                    # Verify the fix worked
                    verify_result = subprocess.run(
                        ["npm", "run", "build"],
                        cwd=app_path,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if verify_result.returncode == 0:
                        print("✅ Build validation after smart fix: SUCCESS")
                        return True
                    else:
                        print("⚠️ Build still has issues after smart fix")
                        return False
                else:
                    print("⚠️ Smart auto-fix partially succeeded")
                    print(f"   {executor.get_execution_summary()}")
            else:
                print(f"❌ Low confidence analysis ({analysis.confidence:.1%}), trying fallback...")
            
            # Fallback to traditional comprehensive build fixing
            print("🔧 Fallback: Using traditional build fixing...")
            success = self.app_builder.build_and_fix_errors(auto_install_deps=True, max_retries=3)
            
            if success:
                print("✅ Traditional build issues automatically resolved!")
                return True
            else:
                print("❌ Could not automatically resolve all build issues")
                return False
                
        except Exception as e:
            print(f"❌ Error during auto-fix: {str(e)}")
            return False
    
    def _execute_create_component_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a component creation task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  🧩 Creating component: {task.description}")
        
        try:
            # Build context from dependencies and task requirements
            context_info = self._gather_context_for_task(task)
            
            # Add dependency results to context
            dep_context = ""
            if dep_results:
                dep_context = "\n\nDEPENDENCY RESULTS:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            create_prompt = f"""You are creating a React component for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Generate the component using the <new filename="components/ComponentName.tsx"> syntax.
Ensure the component:
- Uses TypeScript interfaces
- Follows React best practices
- Includes proper prop types
- Uses shadcn/ui components where appropriate
- Has responsive design with Tailwind CSS
- Includes error handling and loading states if needed
"""
            
            # Generate component code
            with llm_progress("Creating React component"):
                is_valid, response = self.app_builder.make_openai_request(create_prompt, "create")
            
            if not is_valid:
                print("❌ Failed to generate component")
                return False
            
            # Apply the component using CodeBuilder
            from .code_builder import CodeBuilder
            import tempfile
            import os
            
            temp_file = f"temp_component_{task.id}_{int(time.time())}.txt"
            with open(temp_file, 'w') as f:
                f.write(response)
            
            app_path = self.app_builder.get_app_path() if hasattr(self.app_builder, 'get_app_path') else "."
            code_builder = CodeBuilder(temp_file, app_path)
            success = code_builder.build()
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
            
            if success:
                task.result = {"component_created": True, "component_type": "react"}
                self.task_memory.add_file_created(f"components/{task.description}")
                print(f"  ✅ Component created successfully")
                return True
            else:
                print(f"  ❌ Failed to create component")
                return False
                
        except Exception as e:
            print(f"  ❌ Error creating component: {str(e)}")
            return False
    
    def _execute_setup_styling_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a styling setup task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  🎨 Setting up styling: {task.description}")
        
        try:
            context_info = self._gather_context_for_task(task)
            
            # Add dependency results
            dep_context = ""
            if dep_results:
                dep_context = "\n\nPREVIOUS WORK:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            styling_prompt = f"""You are configuring the styling system for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

🚨 CRITICAL JSON GENERATION RULES 🚨

You MUST respond with JSON in this EXACT format:

```json
{{
    "intents": [
        {{
            "file_path": "tailwind.config.js",
            "action": "replace",
            "target": "module.exports = {{",
            "replacement": "module.exports = {{\n  darkMode: ['class'],",
            "context": "enable dark mode support in Tailwind"
        }},
        {{
            "file_path": "app/globals.css",
            "action": "modify",
            "target": "@tailwind utilities;",
            "replacement": "@tailwind utilities;\n\n@layer base {{\n  .dark {{\n    --background: hsl(222.2 84% 4.9%);\n  }}\n}}",
            "context": "add dark mode CSS variables"
        }}
    ]
}}
```

AVAILABLE ACTIONS:
- **replace**: Replace specific text/code with new text
- **modify**: Add content after a target (for insertions)
- **insert**: Insert new content at specific location
- **delete**: Remove specific content

Focus on:
- Tailwind CSS dark mode configuration
- CSS variables for theme switching
- Component styling updates for dark mode
- Consistent theme implementation

Generate structured editing intents in JSON format that configure the styling system.
"""
            
            with llm_progress("Configuring styling system"):
                is_valid, response = self.app_builder.make_openai_request(styling_prompt, "intent")
            
            if not is_valid:
                print("❌ Failed to generate styling configuration")
                return False
            
            # Apply styling changes
            success = self.app_builder._edit_app_with_intents(response)
            
            if success:
                task.result = {"styling_configured": True, "system": "tailwind_shadcn"}
                self.task_memory.store_context("styling_system", "tailwind_shadcn")
                print(f"  ✅ Styling configured successfully")
                return True
            else:
                print(f"  ❌ Failed to configure styling")
                return False
                
        except Exception as e:
            print(f"  ❌ Error configuring styling: {str(e)}")
            return False
    
    def _execute_setup_routing_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a routing setup task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  🛣️ Setting up routing: {task.description}")
        
        try:
            context_info = self._gather_context_for_task(task)
            
            # Add dependency context
            dep_context = ""
            if dep_results:
                dep_context = "\n\nBUILDING ON:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            routing_prompt = f"""You are setting up routing for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Set up NextJS app router structure by creating the necessary page files.
Use <new filename="app/route/page.tsx"> syntax for each route.
Ensure:
- Proper NextJS 13+ app router structure
- TypeScript interfaces
- Client-side navigation with Link components
- Loading and error boundaries where needed
"""
            
            with llm_progress("Setting up routing structure"):
                is_valid, response = self.app_builder.make_openai_request(routing_prompt, "create")
            
            if not is_valid:
                print("❌ Failed to generate routing structure")
                return False
            
            # Apply routing structure
            from .code_builder import CodeBuilder
            import tempfile
            import os
            
            temp_file = f"temp_routing_{task.id}_{int(time.time())}.txt"
            with open(temp_file, 'w') as f:
                f.write(response)
            
            app_path = self.app_builder.get_app_path() if hasattr(self.app_builder, 'get_app_path') else "."
            code_builder = CodeBuilder(temp_file, app_path)
            success = code_builder.build()
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
            
            if success:
                task.result = {"routing_configured": True, "type": "app_router"}
                self.task_memory.store_context("routing_system", "app_router")
                print(f"  ✅ Routing configured successfully")
                return True
            else:
                print(f"  ❌ Failed to configure routing")
                return False
                
        except Exception as e:
            print(f"  ❌ Error configuring routing: {str(e)}")
            return False
    
    def _execute_create_layout_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a layout creation task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  📐 Creating layout: {task.description}")
        
        try:
            context_info = self._gather_context_for_task(task)
            
            # Include previous work context
            dep_context = ""
            if dep_results:
                dep_context = "\n\nPREVIOUS COMPONENTS:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            layout_prompt = f"""You are creating the layout structure for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Create layout components and update the main layout file.
Use <new> or unified diff format as appropriate.
Include:
- Navigation components
- Header/footer structure
- Responsive layout patterns
- Accessibility features
"""
            
            with llm_progress("Creating layout structure"):
                is_valid, response = self.app_builder.make_openai_request(layout_prompt, "edit")
            
            if not is_valid:
                print("❌ Failed to generate layout")
                return False
            
            # Apply layout changes
            success = self.app_builder._edit_app_with_intents(response)
            
            if success:
                task.result = {"layout_created": True, "responsive": True}
                self.task_memory.store_context("layout_system", "responsive")
                print(f"  ✅ Layout created successfully")
                return True
            else:
                print(f"  ❌ Failed to create layout")
                return False
                
        except Exception as e:
            print(f"  ❌ Error creating layout: {str(e)}")
            return False
    
    def _execute_setup_state_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a state management setup task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  🔄 Setting up state management: {task.description}")
        
        try:
            context_info = self._gather_context_for_task(task)
            
            # Build on previous work
            dep_context = ""
            if dep_results:
                dep_context = "\n\nEXISTING COMPONENTS:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            state_prompt = f"""You are setting up state management for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Implement state management by:
- Adding useState hooks where needed
- Setting up localStorage persistence
- Creating state interfaces
- Implementing state update functions
Use unified diff format for modifications.
"""
            
            with llm_progress("Setting up state management"):
                is_valid, response = self.app_builder.make_openai_request(state_prompt, "edit")
            
            if not is_valid:
                print("❌ Failed to generate state management")
                return False
            
            # Apply state management
            success = self.app_builder._edit_app_with_intents(response)
            
            if success:
                task.result = {"state_configured": True, "persistence": "localStorage"}
                self.task_memory.store_context("state_system", "react_hooks_localStorage")
                print(f"  ✅ State management configured successfully")
                return True
            else:
                print(f"  ❌ Failed to configure state management")
                return False
                
        except Exception as e:
            print(f"  ❌ Error configuring state: {str(e)}")
            return False
    
    def _execute_add_features_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a feature addition task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  ⚡ Adding features: {task.description}")
        
        try:
            context_info = self._gather_context_for_task(task)
            
            # Comprehensive dependency context
            dep_context = ""
            if dep_results:
                dep_context = "\n\nAVAILABLE CONTEXT:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
                
                # Add files created/modified info
                files_info = "\n\nFILES AVAILABLE:\n"
                for file in self.task_memory.get_all_affected_files():
                    files_info += f"- {file}\n"
                dep_context += files_info
            
            features_prompt = f"""You are implementing features for a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Implement the requested features by modifying existing files or creating new ones.
Use unified diff format for edits and <new> syntax for new files.
Ensure:
- Feature completeness
- Error handling
- Loading states
- User feedback
- Responsive design
"""
            
            with llm_progress("Implementing features"):
                is_valid, response = self.app_builder.make_openai_request(features_prompt, "edit")
            
            if not is_valid:
                print("❌ Failed to generate feature implementation")
                return False
            
            # Apply feature implementation
            success = self.app_builder._edit_app_with_intents(response)
            
            if success:
                task.result = {"features_added": True, "count": "multiple"}
                print(f"  ✅ Features implemented successfully")
                return True
            else:
                print(f"  ❌ Failed to implement features")
                return False
                
        except Exception as e:
            print(f"  ❌ Error implementing features: {str(e)}")
            return False
    
    def _execute_setup_interactions_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute an interactions setup task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  🎭 Setting up interactions: {task.description}")
        
        try:
            context_info = self._gather_context_for_task(task)
            
            # Use all available context
            dep_context = ""
            if dep_results:
                dep_context = "\n\nCOMPONENTS & FEATURES:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            interactions_prompt = f"""You are adding interactivity to a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Add interactive elements by:
- Implementing click handlers
- Adding form interactions
- Creating modals/dialogs
- Adding animations and transitions
- Implementing keyboard navigation
Use unified diff format for modifications.
"""
            
            with llm_progress("Adding interactivity"):
                is_valid, response = self.app_builder.make_openai_request(interactions_prompt, "edit")
            
            if not is_valid:
                print("❌ Failed to generate interactions")
                return False
            
            # Apply interactions
            success = self.app_builder._edit_app_with_intents(response)
            
            if success:
                task.result = {"interactions_added": True, "type": "comprehensive"}
                print(f"  ✅ Interactions configured successfully")
                return True
            else:
                print(f"  ❌ Failed to configure interactions")
                return False
                
        except Exception as e:
            print(f"  ❌ Error configuring interactions: {str(e)}")
            return False
    
    def _execute_optimize_ux_task(self, task: Task, dep_results: Dict[str, Any] = None) -> bool:
        """Execute a UX optimization task."""
        if not self.app_builder:
            print("❌ No app builder available")
            return False
        
        print(f"  ✨ Optimizing UX: {task.description}")
        
        try:
            context_info = self._gather_context_for_task(task)
            
            # Include all previous work
            dep_context = ""
            if dep_results:
                dep_context = "\n\nCOMPLETE APPLICATION CONTEXT:\n"
                for dep_id, result in dep_results.items():
                    dep_context += f"- {dep_id}: {result}\n"
            
            ux_prompt = f"""You are optimizing the user experience of a NextJS application.

TASK: {task.description}
INSTRUCTIONS: {task.llm_instructions}

CONTEXT:
{context_info}
{dep_context}

Optimize UX by:
- Improving accessibility (ARIA labels, keyboard nav)
- Adding loading states and skeleton screens
- Enhancing error handling and user feedback
- Optimizing responsive design
- Adding micro-interactions and polish
Use unified diff format for all changes.
"""
            
            with llm_progress("Optimizing user experience"):
                is_valid, response = self.app_builder.make_openai_request(ux_prompt, "edit")
            
            if not is_valid:
                print("❌ Failed to generate UX optimizations")
                return False
            
            # Apply UX optimizations
            success = self.app_builder._edit_app_with_intents(response)
            
            if success:
                task.result = {"ux_optimized": True, "accessibility": True}
                print(f"  ✅ UX optimized successfully")
                return True
            else:
                print(f"  ❌ Failed to optimize UX")
                return False
                
        except Exception as e:
            print(f"  ❌ Error optimizing UX: {str(e)}")
            return False
    
    def _handle_task_failure(self, task: Task, plan: ExecutionPlan) -> bool:
        """
        Handle task failure with intelligent recovery and plan adaptation.
        
        Enhanced to:
        - Analyze failure reasons
        - Try alternative approaches
        - Adapt the plan dynamically
        - Learn from failures
        """
        print(f"🔄 Handling failure for task: {task.id}")
        
        # Analyze the failure
        failure_analysis = self._analyze_task_failure(task)
        print(f"📊 Failure analysis: {failure_analysis['reason']}")
        
        # Strategy 1: Intelligent retry with modifications
        if failure_analysis["can_retry"]:
            print("  🔁 Attempting intelligent retry...")
            success = self._retry_task_intelligently(task, failure_analysis)
            if success:
                task.status = "completed"
                print(f"✅ Task recovered successfully: {task.id}")
                return True
        
        # Strategy 2: Try alternative implementation
        if failure_analysis["has_alternatives"]:
            print("  🔄 Trying alternative implementation...")
            success = self._try_alternative_implementation(task, plan)
            if success:
                task.status = "completed"
                print(f"✅ Alternative approach succeeded: {task.id}")
                return True
        
        # Strategy 3: Check if task is still necessary
        if self._can_skip_task(task, plan):
            print(f"⚠️ Task not critical, skipping: {task.type}")
            task.status = "skipped"
            return True
        
        # Critical task failure - assess plan impact
        critical_types = ["create_file", "validate_build", "setup_routing"]
        if task.type in critical_types:
            print(f"💥 Critical task failed: {task.type}")
            
            # Try to adapt the plan
            if self._can_adapt_plan_for_failure(task, plan):
                print("🔧 Adapting plan to work around failure...")
                self._adapt_plan_for_failure(task, plan)
                task.status = "adapted"
                return True
            
            print("  🚨 Critical task failed - stopping execution")
            return False
        
        # For non-critical tasks, continue
        print("  ⚠️ Non-critical task failed - continuing")
        return True
    
    def _analyze_task_failure(self, task: Task) -> Dict[str, Any]:
        """Analyze why a task failed and determine recovery options."""
        analysis = {
            "reason": "unknown",
            "can_retry": True,
            "has_alternatives": False,
            "can_decompose": False,
            "suggested_fixes": []
        }
        
        # Analyze based on task type
        if task.type == "create_component":
            analysis["reason"] = "component creation failed"
            analysis["has_alternatives"] = True  # Can try simpler component
            analysis["can_decompose"] = True  # Can break into smaller parts
            analysis["suggested_fixes"] = [
                "Simplify component structure",
                "Use basic HTML instead of complex components",
                "Split into multiple smaller components"
            ]
        
        elif task.type == "setup_styling":
            analysis["reason"] = "styling setup failed"
            analysis["has_alternatives"] = True  # Can use basic CSS
            analysis["suggested_fixes"] = [
                "Use inline styles instead of complex setup",
                "Apply basic Tailwind classes",
                "Skip advanced theming"
            ]
        
        elif task.type == "create_file":
            analysis["reason"] = "file creation failed"
            analysis["can_retry"] = True  # Always worth retrying file creation
            analysis["suggested_fixes"] = [
                "Simplify file content",
                "Use basic template",
                "Check file path validity"
            ]
        
        elif task.type == "add_features":
            analysis["reason"] = "feature implementation failed"
            analysis["can_decompose"] = True  # Features can be broken down
            analysis["has_alternatives"] = True  # Can implement simpler version
            analysis["suggested_fixes"] = [
                "Implement basic version first",
                "Split features into separate tasks",
                "Use simpler implementations"
            ]
        
        return analysis
    
    def _retry_task_intelligently(self, task: Task, analysis: Dict[str, Any]) -> bool:
        """Retry a task with intelligent modifications based on failure analysis."""
        # Modify task instructions based on suggested fixes
        original_instructions = task.llm_instructions
        
        # Apply suggested simplifications
        if "Simplify" in str(analysis["suggested_fixes"]):
            task.llm_instructions += "\n\nIMPORTANT: Use the simplest possible implementation. Focus on basic functionality over advanced features."
        
        if "basic" in str(analysis["suggested_fixes"]).lower():
            task.llm_instructions += "\n\nFALLBACK: If complex implementation fails, use basic HTML/CSS and minimal JavaScript."
        
        # Try the modified task
        try:
            success = self._execute_task_with_context(task)
            if not success:
                # Restore original instructions
                task.llm_instructions = original_instructions
            return success
        except Exception as e:
            print(f"⚠️ Intelligent retry failed: {e}")
            task.llm_instructions = original_instructions
            return False
    
    def _try_alternative_implementation(self, task: Task, plan: ExecutionPlan) -> bool:
        """Try an alternative implementation approach for a failed task."""
        alternative_approaches = {
            "create_component": "create_file",  # Fallback to simple file creation
            "setup_styling": "edit_file",       # Fallback to manual style edits
            "add_features": "edit_file",        # Fallback to direct file editing
            "setup_interactions": "edit_file"   # Fallback to simple edits
        }
        
        if task.type in alternative_approaches:
            # Create alternative task
            alt_task = Task(
                id=f"{task.id}_alt",
                type=alternative_approaches[task.type],
                description=f"Alternative approach: {task.description}",
                dependencies=task.dependencies,
                context_needed=task.context_needed,
                priority=task.priority,
                estimated_complexity="simple",
                llm_instructions=f"ALTERNATIVE APPROACH: {task.llm_instructions}\n\nUse the simplest possible implementation.",
                validation_criteria=task.validation_criteria
            )
            
            # Execute alternative
            return self._execute_task_with_context(alt_task)
        
        return False
    
    def _can_skip_task(self, task: Task, plan: ExecutionPlan) -> bool:
        """Determine if a task can be safely skipped."""
        optional_task_types = ["optimize_ux", "setup_interactions", "setup_styling"]
        return task.type in optional_task_types
    
    def _can_adapt_plan_for_failure(self, task: Task, plan: ExecutionPlan) -> bool:
        """Check if the plan can be adapted to work around a critical task failure."""
        # For now, always try to adapt
        return True
    
    def _adapt_plan_for_failure(self, task: Task, plan: ExecutionPlan):
        """Adapt the execution plan to work around a critical task failure."""
        # Remove tasks that depend on the failed task
        dependent_tasks = [t for t in plan.tasks if task.id in t.dependencies]
        
        for dep_task in dependent_tasks:
            dep_task.dependencies.remove(task.id)
            print(f"  🔧 Removed dependency on failed task from: {dep_task.id}")
        
        # Add alternative tasks if needed
        if task.type == "create_file":
            # Add a simpler file creation task
            simple_task = Task(
                id=f"{task.id}_simple",
                type="create_file",
                description=f"Simplified: {task.description}",
                dependencies=[],
                context_needed=[],
                priority=task.priority,
                estimated_complexity="simple",
                llm_instructions="Create a minimal, basic version of this file with essential functionality only.",
                validation_criteria=["File exists", "Basic functionality works"]
            )
            plan.tasks.append(simple_task)
            print(f"  ➕ Added simplified alternative task: {simple_task.id}")
    
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
            
            # Get file list (exclude node_modules, .next, etc.)
            if os.path.exists(app_path):
                context_info += "Available files:\n"
                exclude_dirs = {'node_modules', '.next', 'dist', 'build', '.git'}
                
                for root, dirs, files in os.walk(app_path):
                    # Remove excluded directories from dirs to prevent walking into them
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        if file.endswith(('.tsx', '.ts', '.js', '.jsx')):
                            rel_path = os.path.relpath(os.path.join(root, file), app_path)
                            # Double-check no excluded paths made it through
                            if not any(excluded in rel_path for excluded in exclude_dirs):
                                context_info += f"  - {rel_path}\n"
            
            # Add specific context based on task type
            if task.type == "create_file":
                context_info += f"\nTask: Create {task.description}\n"
                context_info += "Important: Use proper NextJS App Router syntax with 'use client' directive for client components\n"
                context_info += "Important: Use useRouter from 'next/navigation' not 'next/router'\n"
                context_info += "Important: Ensure all JSX return statements are properly closed with parentheses\n"
            elif task.type == "edit_file":
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
    
    def _validate_execution_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Validate an execution plan to ensure it's executable."""
        issues = []
        
        # Check for valid task types
        valid_task_types = {
            "analyze_context", "create_file", "edit_file", "validate_build", "install_deps",
            "create_component", "setup_styling", "setup_routing", "create_layout", 
            "setup_state", "add_features", "setup_interactions", "optimize_ux"
        }
        
        for task in plan.tasks:
            # Check task type validity
            if task.type not in valid_task_types:
                issues.append(f"Invalid task type: {task.type}")
            
            # Check for circular dependencies
            if task.id in task.dependencies:
                issues.append(f"Task {task.id} has circular dependency on itself")
            
            # Check dependency validity
            task_ids = {t.id for t in plan.tasks}
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    issues.append(f"Task {task.id} depends on non-existent task {dep_id}")
            
            # Check for required fields
            if not task.description:
                issues.append(f"Task {task.id} missing description")
            if not task.llm_instructions:
                issues.append(f"Task {task.id} missing LLM instructions")
        
        # Check for dependency cycles
        cycle_issues = self._check_dependency_cycles(plan.tasks)
        issues.extend(cycle_issues)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def _check_dependency_cycles(self, tasks: List[Task]) -> List[str]:
        """Check for circular dependencies in the task graph."""
        issues = []
        task_map = {t.id: t for t in tasks}
        
        def has_cycle(task_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            if task_id in task_map:
                for dep_id in task_map[task_id].dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id, visited, rec_stack):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        visited = set()
        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id, visited, set()):
                    issues.append(f"Circular dependency detected involving task {task.id}")
        
        return issues
    
    def _fix_plan_issues(self, plan: ExecutionPlan, issues: List[str]) -> ExecutionPlan:
        """Fix common issues in execution plans."""
        fixed_tasks = []
        task_ids = {t.id for t in plan.tasks}
        
        for task in plan.tasks:
            # Fix invalid task types
            if "Invalid task type" in str(issues):
                # Map common invalid types to valid ones
                type_mapping = {
                    "create_components": "create_component",
                    "setup_styles": "setup_styling",
                    "add_feature": "add_features",
                    "create_pages": "create_file",
                    "setup_routes": "setup_routing"
                }
                if task.type in type_mapping:
                    task.type = type_mapping[task.type]
            
            # Fix dependency issues
            valid_deps = [dep for dep in task.dependencies if dep in task_ids]
            task.dependencies = valid_deps
            
            # Ensure required fields
            if not task.description:
                task.description = f"Execute {task.type} task"
            if not task.llm_instructions:
                task.llm_instructions = f"Complete the {task.type} operation as described"
            
            fixed_tasks.append(task)
        
        plan.tasks = fixed_tasks
        return plan
    
    def _analyze_existing_codebase(self, app_directory: str, edit_request: str) -> Dict[str, Any]:
        """
        Analyze the existing codebase to understand its structure and current state.
        This is the FIRST step before creating any edit plans.
        """
        try:
            analysis_result = {
                "success": True,
                "app_structure": {},
                "key_files": [],
                "current_components": [],
                "routing_setup": {},
                "styling_approach": "",
                "dependencies": [],
                "specific_targets": [],
                "edit_complexity": "unknown",
                "focused_scope": []
            }
            
            # Analyze app structure
            analysis_result["app_structure"] = self._analyze_app_structure(app_directory)
            
            # Identify key files
            analysis_result["key_files"] = self._identify_key_files(app_directory)
            
            # Analyze current components
            analysis_result["current_components"] = self._identify_current_components(app_directory)
            
            # Identify specific targets for the edit request
            analysis_result["specific_targets"] = self._identify_specific_targets(edit_request, analysis_result)
            
            # Assess mobile/responsive state
            analysis_result["mobile_state"] = self._assess_mobile_state(app_directory)
            
            # Determine focused scope based on edit request
            analysis_result["focused_scope"] = self._determine_focused_scope(edit_request, analysis_result)
            
            # Assess edit complexity
            analysis_result["edit_complexity"] = self._assess_edit_complexity(edit_request, analysis_result)
            
            print(f"✅ Codebase analysis complete. Found {len(analysis_result['key_files'])} key files")
            return analysis_result
            
        except Exception as e:
            print(f"❌ Codebase analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_app_structure(self, app_directory: str) -> Dict[str, Any]:
        """Analyze the basic structure of the app."""
        structure = {
            "has_src_dir": False,
            "has_components": False,
            "has_pages": False,
            "has_styles": False,
            "framework": "unknown"
        }
        
        try:
            app_path = Path(app_directory)
            
            # Check for src directory
            if (app_path / "src").exists():
                structure["has_src_dir"] = True
                base_path = app_path / "src"
            else:
                base_path = app_path
            
            # Check for components
            if (base_path / "components").exists():
                structure["has_components"] = True
            
            # Check for pages/app directory (Next.js)
            if (base_path / "pages").exists() or (base_path / "app").exists():
                structure["has_pages"] = True
                structure["framework"] = "nextjs"
            
            # Check for styles
            if (base_path / "styles").exists() or (app_path / "styles").exists():
                structure["has_styles"] = True
                
        except Exception as e:
            print(f"Warning: Could not analyze app structure: {e}")
        
        return structure
    
    def _identify_key_files(self, app_directory: str) -> List[str]:
        """Identify key files in the application."""
        key_files = []
        
        try:
            app_path = Path(app_directory)
            
            # Common important files
            important_files = [
                "package.json",
                "next.config.js",
                "tailwind.config.js",
                "src/app/layout.tsx",
                "src/app/page.tsx",
                "src/app/globals.css",
                "app/layout.tsx",
                "app/page.tsx",
                "pages/_app.tsx",
                "pages/index.tsx"
            ]
            
            for file_path in important_files:
                full_path = app_path / file_path
                if full_path.exists():
                    key_files.append(str(full_path))
                    
        except Exception as e:
            print(f"Warning: Could not identify key files: {e}")
        
        return key_files
    
    def _identify_current_components(self, app_directory: str) -> List[str]:
        """Identify existing React components."""
        components = []
        
        try:
            app_path = Path(app_directory)
            
            # Look for components in common locations
            component_dirs = [
                app_path / "src" / "components",
                app_path / "components",
                app_path / "src" / "app" / "components"
            ]
            
            for comp_dir in component_dirs:
                if comp_dir.exists():
                    for file_path in comp_dir.rglob("*.tsx"):
                        components.append(file_path.name.replace(".tsx", ""))
                    for file_path in comp_dir.rglob("*.jsx"):
                        components.append(file_path.name.replace(".jsx", ""))
                        
        except Exception as e:
            print(f"Warning: Could not identify components: {e}")
        
        return components
    
    def _identify_specific_targets(self, edit_request: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify specific files/components that the edit request targets."""
        targets = []
        request_lower = edit_request.lower()
        
        # Look for component names mentioned in the request
        for component in analysis.get("current_components", []):
            if component.lower() in request_lower:
                targets.append(f"component:{component}")
        
        # Look for specific file mentions
        file_keywords = ["layout", "page", "style", "css", "config"]
        for keyword in file_keywords:
            if keyword in request_lower:
                targets.append(f"file_type:{keyword}")
        
        # Look for feature areas
        feature_keywords = ["navigation", "header", "footer", "sidebar", "mobile", "responsive"]
        for feature in feature_keywords:
            if feature in request_lower:
                targets.append(f"feature:{feature}")
        
        return targets
    
    def _assess_mobile_state(self, app_directory: str) -> Dict[str, Any]:
        """Assess the current mobile/responsive state of the app."""
        mobile_state = {
            "has_responsive_design": False,
            "mobile_navigation": False,
            "breakpoint_usage": False
        }
        
        try:
            # Check for responsive classes or mobile-specific code
            # This is a simplified check - could be enhanced
            app_path = Path(app_directory)
            
            # Look for Tailwind responsive classes in key files
            for file_path in app_path.rglob("*.tsx"):
                try:
                    content = file_path.read_text()
                    if any(cls in content for cls in ["sm:", "md:", "lg:", "xl:", "mobile", "responsive"]):
                        mobile_state["has_responsive_design"] = True
                        break
                except:
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not assess mobile state: {e}")
        
        return mobile_state
    
    def _determine_focused_scope(self, edit_request: str, analysis: Dict[str, Any]) -> List[str]:
        """Determine the focused scope for the edit based on analysis."""
        scope = []
        
        targets = analysis.get("specific_targets", [])
        if targets:
            scope.extend(targets)
        else:
            # Fallback to general categorization
            request_lower = edit_request.lower()
            if any(word in request_lower for word in ["mobile", "responsive"]):
                scope.append("responsive_design")
            elif any(word in request_lower for word in ["style", "css", "design"]):
                scope.append("styling")
            elif any(word in request_lower for word in ["component", "ui"]):
                scope.append("components")
            else:
                scope.append("general")
        
        return scope
    
    def _assess_edit_complexity(self, edit_request: str, analysis: Dict[str, Any]) -> str:
        """Assess the complexity of the requested edit."""
        request_lower = edit_request.lower()
        
        # Simple edits
        if any(word in request_lower for word in ["color", "text", "font", "margin", "padding"]):
            return "simple"
        
        # Moderate edits
        if any(word in request_lower for word in ["responsive", "mobile", "layout", "navigation"]):
            return "moderate"
        
        # Complex edits
        if any(word in request_lower for word in ["architecture", "refactor", "complete", "rebuild"]):
            return "complex"
        
        return "moderate"  # Default
    
    def analyze_and_plan_with_context(self, edit_request: str, request_type: str, codebase_analysis: Dict[str, Any]) -> ExecutionPlan:
        """
        Create a context-aware execution plan based on codebase analysis.
        This replaces generic planning with focused, minimal plans.
        """
        print("🎯 Creating context-aware execution plan...")
        
        # Create a focused prompt that includes codebase context
        context_prompt = self._build_context_aware_prompt(edit_request, request_type, codebase_analysis)
        
                    # Generate plan using the LLM with context
        try:
            plan_response = self._call_coordinator_llm(context_prompt, "context_aware_planning")
            raw_plan = self._parse_plan_response(plan_response)
            
            # Create execution plan with context
            plan = ExecutionPlan(
                request_id=f"edit_{int(time.time())}",
                user_request=edit_request,
                request_type=request_type,
                complexity_assessment=raw_plan.get("complexity_assessment", "moderate"),
                execution_strategy=raw_plan.get("execution_strategy", "Context-aware focused editing"),
                estimated_duration=raw_plan.get("estimated_duration", "30-60 minutes"),
                tasks=[Task(**task_data) for task_data in raw_plan.get("tasks", [])],
                success_criteria=raw_plan.get("success_criteria", ["Edit completed successfully"]),
                created_at=datetime.now().isoformat()
            )
            
            # Validate and fix the plan
            validation_result = self._validate_execution_plan(plan)
            if not validation_result["valid"]:
                print(f"⚠️ Fixing plan issues: {validation_result['issues']}")
                plan = self._fix_plan_issues(plan, validation_result["issues"])
            
            print(f"✅ Context-aware plan created with {len(plan.tasks)} focused tasks")
            return plan
            
        except Exception as e:
            print(f"❌ Context-aware planning failed: {str(e)}")
            # Return minimal fallback plan
            return self._create_fallback_plan(edit_request, request_type)
    
    def _build_context_aware_prompt(self, edit_request: str, request_type: str, analysis: Dict[str, Any]) -> str:
        """Build a context-aware prompt that includes codebase analysis."""
        
        context_info = f"""
CODEBASE ANALYSIS CONTEXT:
- App Structure: {analysis.get('app_structure', {})}
- Key Files Found: {len(analysis.get('key_files', []))} files
- Existing Components: {analysis.get('current_components', [])}
- Specific Targets: {analysis.get('specific_targets', [])}
- Mobile State: {analysis.get('mobile_state', {})}
- Focused Scope: {analysis.get('focused_scope', [])}
- Edit Complexity: {analysis.get('edit_complexity', 'moderate')}

IMPORTANT: Create a MINIMAL, FOCUSED plan that targets only the specific areas identified above.
Do NOT create generic, overly complex plans. Focus on the exact targets and scope identified.
"""
        
        base_prompt = f"""You are a Senior Software Architect AI coordinating a FOCUSED NextJS edit task.
Your role is to create a MINIMAL execution plan based on the codebase analysis provided.

USER REQUEST: {edit_request}
REQUEST TYPE: {request_type}
{context_info}

CRITICAL REQUIREMENTS:
1. This is a FRONTEND-ONLY application. NO authentication, NO database, NO backend APIs.
2. Create a MINIMAL plan that addresses ONLY the specific request
3. Focus on the identified targets and scope from the analysis
4. Do NOT add unnecessary complexity or generic tasks
5. Keep the plan small and focused

You must respond with a JSON plan in this EXACT format:

{{
  "complexity_assessment": "{analysis.get('edit_complexity', 'moderate')}",
  "execution_strategy": "Focused edit targeting specific areas identified in analysis",
  "estimated_duration": "15-45 minutes",
  "tasks": [
    {{
      "id": "task_1",
      "type": "edit_file|create_component|setup_styling|etc",
      "description": "Specific focused task description",
      "dependencies": [],
      "context_needed": ["specific files identified"],
      "priority": 1,
      "estimated_complexity": "simple|moderate",
      "llm_instructions": "Clear, specific instructions"
    }}
  ],
  "success_criteria": ["Specific, measurable criteria"]
}}

Remember: MINIMAL and FOCUSED. Only include tasks that directly address the edit request."""
        
        return base_prompt
    
    def _create_fallback_plan(self, edit_request: str, request_type: str) -> ExecutionPlan:
        """Create a minimal fallback plan when context-aware planning fails."""
        fallback_task = Task(
            id="fallback_edit",
            type="edit_file",
            description=f"Apply requested edit: {edit_request}",
            dependencies=[],
            context_needed=["src/app/page.tsx"],
            priority=1,
            estimated_complexity="moderate",
            llm_instructions=f"Apply the following edit to the main app: {edit_request}",
            validation_criteria=["Edit applied successfully", "App still builds"]
        )
        
        return ExecutionPlan(
            request_id=f"fallback_{int(time.time())}",
            user_request=edit_request,
            request_type=request_type,
            complexity_assessment="moderate",
            execution_strategy="Fallback minimal edit",
            estimated_duration="30 minutes",
            tasks=[fallback_task],
            success_criteria=["Edit applied successfully"],
            created_at=datetime.now().isoformat()
        )

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