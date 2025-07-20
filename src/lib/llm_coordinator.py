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
        print("🧠 Analyzing request and creating execution plan...")
        
        # Use MVP spec if available, otherwise use original request
        planning_request = user_request
        if mvp_spec:
            print(f"📋 Using enhanced MVP specification")
            print(f"🎯 Complexity level: {mvp_spec.complexity_level}")
            print(f"📊 Estimated components: {mvp_spec.estimated_components}")
            planning_request = mvp_spec.enhanced_prompt
        else:
            print(f"📝 Using original request: {user_request}")
        
        # Auto-detect request type if not specified
        if request_type == "auto":
            request_type = self._classify_request_type(planning_request)
        
        print(f"🔍 Detected request type: {request_type}")
        
        # Use coordinator LLM to create detailed plan
        planning_prompt = self._get_planning_prompt(planning_request, request_type, mvp_spec)
        
        plan_response = self._call_coordinator_llm(planning_prompt, "planning")
        
        if not plan_response:
            # Fallback to simple plan
            return self._create_fallback_plan(planning_request, request_type, mvp_spec)
        
        # Parse the plan from LLM response
        execution_plan = self._parse_execution_plan(plan_response, planning_request, request_type)
        
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
        """Call the coordinator LLM with the given prompt."""
        # Prefer Anthropic Claude 4 Sonnet if available, fallback to OpenAI GPT-4
        if self.app_builder and self.app_builder.anthropic_client:
            try:
                print("🧠 Using Claude 4 Sonnet for coordination...")
                messages = [{"role": "user", "content": prompt}]
                response = self.app_builder.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=3000,
                    temperature=0.3,
                    system="You are a Senior Software Architect AI that creates detailed execution plans for NextJS development tasks. Always respond with valid JSON.",
                    messages=messages
                )
                return response.content[0].text.strip()
            except Exception as e:
                print(f"❌ Claude coordination failed, falling back to GPT-4: {str(e)}")
        
        # Fallback to OpenAI GPT-4
        if not self.openai_client:
            print("⚠️ No LLM client available for coordination")
            return None
        
        try:
            print("🧠 Using GPT-4 for coordination...")
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
            
            # Enhance the task description for better specificity
            try:
                from .request_enhancer import RequestEnhancer
                enhancer = RequestEnhancer()
                
                if hasattr(self.app_builder, 'get_app_path'):
                    app_path = self.app_builder.get_app_path()
                    enhanced_description = enhancer.enhance_edit_request(
                        user_request=task.description,
                        app_path=app_path,
                        context_summary=context_info[:500]  # Brief context summary
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

🚨 CRITICAL DIFF GENERATION RULES 🚨

You MUST respond with a UNIFIED DIFF in this EXACT format:

*** Begin Patch
*** Update File: path/to/file.tsx
@@ -old_start,old_count +new_start,new_count @@ optional_context
- old line to remove
+ new line to add
 unchanged context line
*** End Patch

KEY REQUIREMENTS:
1. Use ONLY the unified diff format shown above
2. Include 2-3 lines of context before and after changes for accurate matching
3. Use proper +/- prefixes for added/removed lines
4. Use space prefix for unchanged context lines
5. Multiple files = multiple "*** Update File:" sections in ONE patch
6. NO line-based <edit> tags - ONLY unified diffs
7. Context lines must EXACTLY match existing file content

EXAMPLE:
*** Begin Patch
*** Update File: app/layout.tsx
@@ -3,6 +3,7 @@
 import './globals.css';
+import Navigation from '@/components/Navigation';
 
 export default function RootLayout({{ children }}: {{ children: React.ReactNode }}) {{
   return (
     <html lang="en">
       <body className="bg-gray-50">
+        <Navigation />
         {{children}}
       </body>
*** End Patch

Generate a unified diff that implements the requested changes with proper context matching."""
            
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