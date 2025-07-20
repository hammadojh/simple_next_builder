#!/usr/bin/env python3
"""
Progress Loader System

A centralized system for showing animated progress indicators during long-running tasks.
Ensures the terminal is never idle and always shows what task is being processed.
"""

import sys
import time
import threading
from typing import Optional, List, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


class LoaderStyle(Enum):
    """Different styles of progress loaders."""
    SPINNER = "spinner"
    DOTS = "dots"
    PROGRESS_BAR = "progress_bar"
    PULSE = "pulse"
    THINKING = "thinking"
    BUILDING = "building"
    NETWORK = "network"


@dataclass
class LoaderConfig:
    """Configuration for a progress loader."""
    task_name: str
    style: LoaderStyle = LoaderStyle.SPINNER
    update_interval: float = 0.1
    show_elapsed: bool = True
    prefix: str = ""
    suffix: str = ""


class ProgressLoader:
    """
    Animated progress loader that keeps the terminal active.
    
    Features:
    - Multiple animation styles
    - Task name display
    - Elapsed time tracking
    - Thread-safe operation
    - Context manager support
    - Never leaves terminal idle
    """
    
    def __init__(self, config: LoaderConfig):
        self.config = config
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.current_frame = 0
        self._stop_event = threading.Event()
        
        # Animation frames for different styles
        self.animations = {
            LoaderStyle.SPINNER: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            LoaderStyle.DOTS: [".", "..", "...", ""],
            LoaderStyle.PULSE: ["◐", "◓", "◑", "◒"],
            LoaderStyle.THINKING: ["🤔", "💭", "🧠", "⚡", "🤖", "🎯"],
            LoaderStyle.BUILDING: ["🔨", "🔧", "⚙️", "🛠️", "🏗️", "📦"],
            LoaderStyle.NETWORK: ["📡", "📶", "🌐", "🔗", "📡", "💫"]
        }
        
        # Special progress bar animation
        self.progress_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    
    def start(self):
        """Start the progress loader animation."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self, success_message: Optional[str] = None):
        """Stop the progress loader and optionally show a success message."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # Clear the current line and show final message
        sys.stdout.write("\r" + " " * 100 + "\r")
        if success_message:
            elapsed = time.time() - (self.start_time or 0)
            sys.stdout.write(f"✅ {success_message} ({elapsed:.1f}s)\n")
        sys.stdout.flush()
    
    def update_task(self, new_task_name: str):
        """Update the task name while the loader is running."""
        self.config.task_name = new_task_name
    
    def _animate(self):
        """Animation loop that runs in a separate thread."""
        while not self._stop_event.is_set():
            frame = self._get_current_frame()
            elapsed = time.time() - (self.start_time or 0)
            
            # Build the display line
            if self.config.show_elapsed:
                elapsed_str = f" ({elapsed:.1f}s)"
            else:
                elapsed_str = ""
            
            line = f"\r{self.config.prefix}{frame} {self.config.task_name}{elapsed_str}{self.config.suffix}"
            
            # Ensure line doesn't exceed terminal width (assume 120 chars max)
            if len(line) > 115:
                line = line[:112] + "..."
            
            sys.stdout.write(line)
            sys.stdout.flush()
            
            self.current_frame = (self.current_frame + 1) % len(self._get_animation_frames())
            time.sleep(self.config.update_interval)
    
    def _get_current_frame(self) -> str:
        """Get the current animation frame."""
        if self.config.style == LoaderStyle.PROGRESS_BAR:
            return self._get_progress_bar_frame()
        else:
            frames = self._get_animation_frames()
            return frames[self.current_frame % len(frames)]
    
    def _get_animation_frames(self) -> List[str]:
        """Get animation frames for the current style."""
        return self.animations.get(self.config.style, self.animations[LoaderStyle.SPINNER])
    
    def _get_progress_bar_frame(self) -> str:
        """Generate a progress bar animation frame."""
        # Create an oscillating progress bar
        bar_length = 10
        pos = self.current_frame % (bar_length * 2)
        if pos > bar_length:
            pos = bar_length * 2 - pos
        
        bar = "▁" * pos + "█" + "▁" * (bar_length - pos - 1)
        return f"[{bar}]"


class ProgressManager:
    """
    Central manager for all progress loaders.
    
    Provides:
    - Context managers for easy use
    - Task stacking (nested loaders)
    - Global loader state management
    - Predefined loader configurations
    """
    
    def __init__(self):
        self.active_loaders: List[ProgressLoader] = []
        self.loader_stack: List[ProgressLoader] = []
    
    @contextmanager
    def show_progress(self, task_name: str, style: LoaderStyle = LoaderStyle.SPINNER):
        """Context manager for showing progress during a task."""
        config = LoaderConfig(task_name=task_name, style=style)
        loader = ProgressLoader(config)
        
        try:
            loader.start()
            self.loader_stack.append(loader)
            yield loader
        finally:
            if loader in self.loader_stack:
                self.loader_stack.remove(loader)
            loader.stop()
    
    @contextmanager
    def llm_request_progress(self, provider_name: str = "AI"):
        """Specialized progress loader for LLM API requests."""
        task_name = f"🤖 Calling {provider_name} API"
        with self.show_progress(task_name, LoaderStyle.THINKING) as loader:
            yield loader
    
    @contextmanager
    def file_operation_progress(self, operation: str, file_count: int = 1):
        """Specialized progress loader for file operations."""
        task_name = f"📁 {operation} ({file_count} file{'s' if file_count != 1 else ''})"
        with self.show_progress(task_name, LoaderStyle.SPINNER) as loader:
            yield loader
    
    @contextmanager
    def build_progress(self, operation: str = "Building"):
        """Specialized progress loader for build operations."""
        task_name = f"🔨 {operation}"
        with self.show_progress(task_name, LoaderStyle.BUILDING) as loader:
            yield loader
    
    @contextmanager
    def network_progress(self, operation: str = "Connecting"):
        """Specialized progress loader for network operations."""
        task_name = f"🌐 {operation}"
        with self.show_progress(task_name, LoaderStyle.NETWORK) as loader:
            yield loader
    
    @contextmanager
    def analysis_progress(self, what: str = "code"):
        """Specialized progress loader for analysis operations."""
        task_name = f"🔍 Analyzing {what}"
        with self.show_progress(task_name, LoaderStyle.PULSE) as loader:
            yield loader
    
    def update_current_task(self, new_task_name: str):
        """Update the current active loader's task name."""
        if self.loader_stack:
            self.loader_stack[-1].update_task(new_task_name)


# Global progress manager instance
progress_manager = ProgressManager()


# Convenience functions for common use cases
@contextmanager
def show_progress(task_name: str, style: LoaderStyle = LoaderStyle.SPINNER):
    """Show progress for a task."""
    with progress_manager.show_progress(task_name, style) as loader:
        yield loader


@contextmanager
def llm_progress(provider_name: str = "AI"):
    """Show progress for LLM API calls."""
    with progress_manager.llm_request_progress(provider_name) as loader:
        yield loader


@contextmanager
def file_progress(operation: str, file_count: int = 1):
    """Show progress for file operations."""
    with progress_manager.file_operation_progress(operation, file_count) as loader:
        yield loader


@contextmanager
def build_progress(operation: str = "Building"):
    """Show progress for build operations."""
    with progress_manager.build_progress(operation) as loader:
        yield loader


@contextmanager
def network_progress(operation: str = "Connecting"):
    """Show progress for network operations."""
    with progress_manager.network_progress(operation) as loader:
        yield loader


@contextmanager
def analysis_progress(what: str = "code"):
    """Show progress for analysis operations."""
    with progress_manager.analysis_progress(what) as loader:
        yield loader


def update_current_task(new_task_name: str):
    """Update the current task name."""
    progress_manager.update_current_task(new_task_name)


# Decorator for automatic progress loading
def with_progress(task_name: str, style: LoaderStyle = LoaderStyle.SPINNER):
    """Decorator to automatically show progress for a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            with show_progress(task_name, style):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo of different loader styles
    print("🎯 Progress Loader Demo")
    print("=" * 50)
    
    styles = [
        (LoaderStyle.SPINNER, "Processing data"),
        (LoaderStyle.DOTS, "Loading content"),
        (LoaderStyle.PULSE, "Analyzing patterns"),
        (LoaderStyle.THINKING, "AI processing"),
        (LoaderStyle.BUILDING, "Building application"),
        (LoaderStyle.NETWORK, "Fetching resources"),
        (LoaderStyle.PROGRESS_BAR, "Converting files")
    ]
    
    for style, task in styles:
        with show_progress(task, style):
            time.sleep(3)
        print()
    
    print("✅ Demo completed!") 