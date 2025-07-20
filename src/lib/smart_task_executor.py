"""
Smart Task Executor

Executes structured tasks returned by the build error analyzer.
Handles dependencies, progress tracking, and error recovery.
"""

import os
import subprocess
import json
import time
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from .build_error_analyzer import FixTask, TaskType, BuildErrorAnalysis

@dataclass
class TaskResult:
    """Result of executing a task"""
    task_id: str
    success: bool
    message: str
    duration: float = 0.0
    error_details: Optional[str] = None

class SmartTaskExecutor:
    """
    Executes structured fix tasks with dependency handling and progress tracking.
    """
    
    def __init__(self, app_path: str):
        self.app_path = app_path
        self.executed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.task_results: List[TaskResult] = []
        
    def execute_analysis(self, analysis: BuildErrorAnalysis) -> Tuple[bool, List[TaskResult]]:
        """
        Execute all tasks from a build error analysis.
        
        Args:
            analysis: BuildErrorAnalysis with tasks to execute
            
        Returns:
            Tuple of (success, list of task results)
        """
        print(f"🚀 Executing {len(analysis.tasks)} fix tasks...")
        print(f"   📊 Estimated complexity: {analysis.estimated_complexity}")
        print(f"   🎯 Success probability: {analysis.success_probability:.1%}")
        
        # Reset state
        self.executed_tasks.clear()
        self.failed_tasks.clear()
        self.task_results.clear()
        
        # Build dependency graph
        task_by_id = {task.id: task for task in analysis.tasks}
        dependency_graph = self._build_dependency_graph(analysis.tasks)
        
        # Execute tasks in dependency order
        execution_order = self._topological_sort(dependency_graph)
        
        print(f"   📋 Execution order: {' → '.join(execution_order)}")
        
        overall_success = True
        
        for task_id in execution_order:
            if task_id in task_by_id:
                task = task_by_id[task_id]
                
                # Check dependencies
                if not self._dependencies_satisfied(task):
                    result = TaskResult(
                        task_id=task_id,
                        success=False,
                        message=f"Dependencies not satisfied: {task.dependencies}"
                    )
                    self.task_results.append(result)
                    self.failed_tasks.add(task_id)
                    overall_success = False
                    continue
                
                # Execute task
                print(f"   🔧 Executing: {task.description}")
                result = self._execute_task(task)
                self.task_results.append(result)
                
                if result.success:
                    self.executed_tasks.add(task_id)
                    print(f"   ✅ {task.description}")
                else:
                    self.failed_tasks.add(task_id)
                    print(f"   ❌ {task.description}: {result.message}")
                    overall_success = False
                    
                    # Stop on critical failures
                    if self._is_critical_task(task):
                        print(f"   🛑 Critical task failed, stopping execution")
                        break
        
        print(f"🏁 Task execution complete: {len(self.executed_tasks)}/{len(analysis.tasks)} successful")
        
        # Validate that the fixes actually work by running a quick build check
        if overall_success and len(self.executed_tasks) > 0:
            print("🔍 Validating that fixes actually resolved the build errors...")
            if self._validate_fixes_worked():
                print("✅ Build validation: Fixes successfully resolved the errors!")
                return True, self.task_results
            else:
                print("❌ Build validation: Tasks completed but errors persist")
                # Mark as partial success since tasks executed but didn't solve the problem
                return False, self.task_results
        
        return overall_success, self.task_results
    
    def _build_dependency_graph(self, tasks: List[FixTask]) -> Dict[str, List[str]]:
        """Build dependency graph from tasks"""
        graph = {}
        for task in tasks:
            graph[task.id] = task.dependencies.copy()
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Topological sort for dependency resolution"""
        # Kahn's algorithm
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degrees
        for node in graph:
            for dep in graph[node]:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find nodes with no dependencies
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Remove this node from graph
            for neighbor in graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    def _dependencies_satisfied(self, task: FixTask) -> bool:
        """Check if all dependencies for a task are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.executed_tasks:
                return False
        return True
    
    def _execute_task(self, task: FixTask) -> TaskResult:
        """Execute a single task"""
        import time
        start_time = time.time()
        
        try:
            if task.type == TaskType.EDIT_FILE:
                return self._execute_edit_file(task, start_time)
            elif task.type == TaskType.CREATE_FILE:
                return self._execute_create_file(task, start_time)
            elif task.type == TaskType.DELETE_FILE:
                return self._execute_delete_file(task, start_time)
            elif task.type == TaskType.RUN_COMMAND:
                return self._execute_run_command(task, start_time)
            elif task.type == TaskType.INSTALL_DEPENDENCY:
                return self._execute_install_dependency(task, start_time)
            elif task.type == TaskType.ADD_IMPORT:
                return self._execute_add_import(task, start_time)
            elif task.type == TaskType.FIX_SYNTAX:
                return self._execute_fix_syntax(task, start_time)
            else:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    message=f"Unknown task type: {task.type}",
                    duration=time.time() - start_time
                )
                
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Exception during execution: {str(e)}",
                duration=time.time() - start_time,
                error_details=str(e)
            )
    
    def _execute_edit_file(self, task: FixTask, start_time: float) -> TaskResult:
        """Execute edit_file task"""
        if not task.file_path or not task.content:
            return TaskResult(
                task_id=task.id,
                success=False,
                message="Missing file_path or content for edit_file task",
                duration=time.time() - start_time
            )
        
        try:
            file_path = os.path.join(self.app_path, task.file_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(task.content)
            
            return TaskResult(
                task_id=task.id,
                success=True,
                message=f"Successfully edited {task.file_path}",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Failed to edit {task.file_path}: {str(e)}",
                duration=time.time() - start_time,
                error_details=str(e)
            )
    
    def _execute_create_file(self, task: FixTask, start_time: float) -> TaskResult:
        """Execute create_file task"""
        if not task.file_path or not task.content:
            return TaskResult(
                task_id=task.id,
                success=False,
                message="Missing file_path or content for create_file task",
                duration=time.time() - start_time
            )
        
        try:
            file_path = os.path.join(self.app_path, task.file_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(task.content)
            
            return TaskResult(
                task_id=task.id,
                success=True,
                message=f"Successfully created {task.file_path}",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Failed to create {task.file_path}: {str(e)}",
                duration=time.time() - start_time,
                error_details=str(e)
            )
    
    def _execute_delete_file(self, task: FixTask, start_time: float) -> TaskResult:
        """Execute delete_file task"""
        if not task.file_path:
            return TaskResult(
                task_id=task.id,
                success=False,
                message="Missing file_path for delete_file task",
                duration=time.time() - start_time
            )
        
        try:
            file_path = os.path.join(self.app_path, task.file_path)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                message = f"Successfully deleted {task.file_path}"
            else:
                message = f"File {task.file_path} does not exist (already deleted)"
            
            return TaskResult(
                task_id=task.id,
                success=True,
                message=message,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Failed to delete {task.file_path}: {str(e)}",
                duration=time.time() - start_time,
                error_details=str(e)
            )
    
    def _execute_run_command(self, task: FixTask, start_time: float) -> TaskResult:
        """Execute run_command task"""
        if not task.command:
            return TaskResult(
                task_id=task.id,
                success=False,
                message="Missing command for run_command task",
                duration=time.time() - start_time
            )
        
        try:
            # Run command in app directory
            result = subprocess.run(
                task.command.split(),
                cwd=self.app_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            message = f"Command '{task.command}' {'succeeded' if success else 'failed'}"
            
            if not success and result.stderr:
                message += f": {result.stderr.strip()}"
            
            return TaskResult(
                task_id=task.id,
                success=success,
                message=message,
                duration=time.time() - start_time,
                error_details=result.stderr if not success else None
            )
            
        except subprocess.TimeoutExpired:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Command '{task.command}' timed out",
                duration=time.time() - start_time
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Failed to run command '{task.command}': {str(e)}",
                duration=time.time() - start_time,
                error_details=str(e)
            )
    
    def _execute_install_dependency(self, task: FixTask, start_time: float) -> TaskResult:
        """Execute install_dependency task"""
        command = task.command or f"npm install"
        
        # Use dependency manager if available
        try:
            from .dependency_manager import DependencyManager
            dep_manager = DependencyManager(self.app_path)
            
            if dep_manager.auto_manage_dependencies():
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    message="Dependencies managed successfully",
                    duration=time.time() - start_time
                )
        except:
            pass  # Fall back to command execution
        
        # Fall back to direct command execution
        task.command = command
        return self._execute_run_command(task, start_time)
    
    def _execute_add_import(self, task: FixTask, start_time: float) -> TaskResult:
        """Execute add_import task"""
        if not task.file_path or not task.content:
            return TaskResult(
                task_id=task.id,
                success=False,
                message="Missing file_path or import statement for add_import task",
                duration=time.time() - start_time
            )
        
        try:
            file_path = os.path.join(self.app_path, task.file_path)
            
            if not os.path.exists(file_path):
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    message=f"File {task.file_path} does not exist",
                    duration=time.time() - start_time
                )
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find the best place to insert import (after existing imports)
            import_line = task.content.strip() + '\n'
            
            # Check if import already exists
            if any(import_line.strip() in line.strip() for line in lines):
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    message=f"Import already exists in {task.file_path}",
                    duration=time.time() - start_time
                )
            
            # Find insertion point
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('const '):
                    insert_index = i + 1
                elif line.strip() == '' and i < len(lines) - 1:
                    continue
                else:
                    break
            
            # Insert import
            lines.insert(insert_index, import_line)
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return TaskResult(
                task_id=task.id,
                success=True,
                message=f"Successfully added import to {task.file_path}",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Failed to add import to {task.file_path}: {str(e)}",
                duration=time.time() - start_time,
                error_details=str(e)
            )
    
    def _execute_fix_syntax(self, task: FixTask, start_time: float) -> TaskResult:
        """Execute fix_syntax task"""
        if not task.file_path:
            return TaskResult(
                task_id=task.id,
                success=False,
                message="Missing file_path for fix_syntax task",
                duration=time.time() - start_time
            )
        
        try:
            file_path = os.path.join(self.app_path, task.file_path)
            
            if not os.path.exists(file_path):
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    message=f"File {task.file_path} does not exist",
                    duration=time.time() - start_time
                )
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple syntax fixes
            fixed_content = content
            
            # Fix common issues
            if not content.strip().endswith('}') and 'function' in content:
                # Likely missing closing brace
                fixed_content = content.rstrip() + '\n}'
            
            # If we have specific content to replace with
            if task.content:
                fixed_content = task.content
            
            # Write back if changed
            if fixed_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    message=f"Fixed syntax in {task.file_path}",
                    duration=time.time() - start_time
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    message=f"No syntax fixes needed in {task.file_path}",
                    duration=time.time() - start_time
                )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                message=f"Failed to fix syntax in {task.file_path}: {str(e)}",
                duration=time.time() - start_time,
                error_details=str(e)
            )
    
    def _is_critical_task(self, task: FixTask) -> bool:
        """Determine if a task is critical (failure should stop execution)"""
        critical_types = [TaskType.INSTALL_DEPENDENCY]
        return task.type in critical_types
    
    def get_execution_summary(self) -> Dict:
        """Get a summary of the execution results"""
        total = len(self.task_results)
        successful = len(self.executed_tasks)
        failed = len(self.failed_tasks)
        
        failed_tasks = []
        for result in self.task_results:
            if not result.success:
                task_info = {
                    'task_id': result.task_id,
                    'error': result.message,
                    'duration': getattr(result, 'duration', 0)
                }
                if hasattr(result, 'error_details'):
                    task_info['error_details'] = result.error_details
                failed_tasks.append(task_info)
        
        return {
            'total_tasks': total,
            'successful_tasks': successful,
            'failed_tasks': failed_tasks,
            'summary_text': f"📊 Execution Summary: {successful}/{total} tasks successful"
        }
    
    def get_execution_summary_text(self) -> str:
        """Get a text summary of the execution results (for backward compatibility)"""
        summary_dict = self.get_execution_summary()
        text = summary_dict['summary_text']
        
        if summary_dict['failed_tasks']:
            text += f"\n❌ Failed tasks:"
            for task_info in summary_dict['failed_tasks']:
                text += f"\n  • {task_info['task_id']}: {task_info['error']}"
        
        return text
    
    def _validate_fixes_worked(self) -> bool:
        """Validate that the applied fixes actually resolved the build errors"""
        try:
            import subprocess
            
            # Run a quick build check
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.app_path,
                capture_output=True,
                text=True,
                timeout=60  # Quick validation, shorter timeout
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("   ⚠️ Build validation timeout - assuming fixes need more time")
            return False
        except Exception as e:
            print(f"   ⚠️ Build validation error: {e}")
            return False 