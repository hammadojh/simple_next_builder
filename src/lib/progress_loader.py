#!/usr/bin/env python3
"""
Progress Loader System

A centralized system for showing animated progress indicators during long-running tasks.
Ensures the terminal is never idle and always shows what task is being processed.
Now includes streaming text support for AI responses.
"""

import sys
import time
import threading
from typing import Optional, List, Callable, Any, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import textwrap


class LoaderStyle(Enum):
    """Different styles of progress loaders."""
    SPINNER = "spinner"
    DOTS = "dots"
    PROGRESS_BAR = "progress_bar"
    PULSE = "pulse"
    THINKING = "thinking"
    BUILDING = "building"
    NETWORK = "network"
    STREAMING = "streaming"  # New streaming style


@dataclass
class LoaderConfig:
    """Configuration for a progress loader."""
    task_name: str
    style: LoaderStyle = LoaderStyle.SPINNER
    update_interval: float = 0.15  # Increased from 0.1 to reduce flicker
    show_elapsed: bool = True
    prefix: str = ""
    suffix: str = ""


class StreamingLoader:
    """
    Streaming text loader for AI responses.
    
    Features:
    - Real-time text streaming
    - Word wrapping
    - Progress indication
    - Thread-safe operation
    """
    
    def __init__(self, task_name: str = "AI Response"):
        self.task_name = task_name
        self.is_active = False
        self.streamed_text = ""
        self.current_line = ""
        self._lock = threading.Lock()
        self.terminal_width = 100  # Assume 100 chars width
        self.start_time = time.time()
        
    def start(self):
        """Start the streaming loader."""
        with self._lock:
            if self.is_active:
                return
            self.is_active = True
            self.start_time = time.time()
            self.streamed_text = ""
            self.current_line = ""
        
        # Show initial header
        print(f"\n🤖 {self.task_name}:")
        print("─" * 60)
        sys.stdout.flush()
    
    def stream_text(self, text_chunk: str):
        """Stream a chunk of text to the terminal."""
        with self._lock:
            if not self.is_active:
                return
            
            # Add the new text chunk
            self.current_line += text_chunk
            
            # Process complete words (split on spaces)
            words = self.current_line.split(' ')
            
            # Keep the last word in current_line (might be incomplete)
            if len(words) > 1:
                complete_words = words[:-1]
                self.current_line = words[-1]
                
                # Print complete words
                for word in complete_words:
                    if word.strip():  # Only print non-empty words
                        print(word + " ", end="", flush=True)
                        
                        # Add slight delay for natural reading effect
                        time.sleep(0.02)
    
    def stream_chunk(self, chunk: str):
        """Stream a complete chunk of text (like a sentence or paragraph)."""
        with self._lock:
            if not self.is_active:
                return
        
        # Wrap text to terminal width
        wrapped_lines = textwrap.wrap(chunk, width=self.terminal_width - 4)
        
        for line in wrapped_lines:
            print(f"  {line}")
            time.sleep(0.1)  # Small delay between lines
        
        sys.stdout.flush()
    
    def finish(self, final_message: Optional[str] = None):
        """Finish streaming and show completion."""
        with self._lock:
            if not self.is_active:
                return
            self.is_active = False
        
        # Print any remaining text in current_line
        if self.current_line.strip():
            print(self.current_line)
        
        # Show completion
        elapsed = time.time() - self.start_time
        print()
        print("─" * 60)
        if final_message:
            print(f"✅ {final_message} ({elapsed:.1f}s)")
        else:
            print(f"✅ {self.task_name} completed ({elapsed:.1f}s)")
        print()
        sys.stdout.flush()


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
    - Streaming text support
    """
    
    def __init__(self, config: LoaderConfig):
        self.config = config
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.current_frame = 0
        self._stop_event = threading.Event()
        self._lock = threading.Lock()  # Add lock for thread safety
        
        # Streaming text support
        self.streaming_loader: Optional[StreamingLoader] = None
        
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
        with self._lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.start_time = time.time()
            self._stop_event.clear()
            self.current_frame = 0
            
            # For streaming style, create a streaming loader
            if self.config.style == LoaderStyle.STREAMING:
                self.streaming_loader = StreamingLoader(self.config.task_name)
                self.streaming_loader.start()
            else:
                # Regular animation
                self.thread = threading.Thread(target=self._animate, daemon=True)
                self.thread.start()
    
    def stop(self, success_message: Optional[str] = None):
        """Stop the progress loader and optionally show a success message."""
        with self._lock:
            if not self.is_running:
                return
            
            self.is_running = False
            self._stop_event.set()
            
            # Handle streaming loader
            if self.streaming_loader:
                self.streaming_loader.finish(success_message)
                self.streaming_loader = None
                return
        
        # Wait for thread to finish (outside the lock to avoid deadlock)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)  # Reduced timeout for faster cleanup
            
            # Force stop if thread is still alive
            if self.thread.is_alive():
                # Clear the line and move on
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.flush()
        
        # Clear the current line and show final message
        sys.stdout.write("\r" + " " * 120 + "\r")  # Clear wider area
        if success_message:
            elapsed = time.time() - (self.start_time or 0)
            sys.stdout.write(f"✅ {success_message} ({elapsed:.1f}s)\n")
        sys.stdout.flush()
    
    def stream_text(self, text_chunk: str):
        """Stream text if this is a streaming loader."""
        with self._lock:
            if self.streaming_loader and self.is_running:
                self.streaming_loader.stream_text(text_chunk)
    
    def stream_chunk(self, chunk: str):
        """Stream a complete chunk if this is a streaming loader."""
        with self._lock:
            if self.streaming_loader and self.is_running:
                self.streaming_loader.stream_chunk(chunk)
    
    def update_task(self, new_task_name: str):
        """Update the task name while the loader is running."""
        with self._lock:
            if self.is_running:
                self.config.task_name = new_task_name
    
    def _animate(self):
        """Animation loop that runs in a separate thread."""
        try:
            while not self._stop_event.is_set():
                if not self.is_running:  # Double check
                    break
                    
                frame = self._get_current_frame()
                elapsed = time.time() - (self.start_time or 0)
                
                # Build the display line
                with self._lock:
                    task_name = self.config.task_name
                    show_elapsed = self.config.show_elapsed
                    prefix = self.config.prefix
                    suffix = self.config.suffix
                
                if show_elapsed:
                    elapsed_str = f" ({elapsed:.1f}s)"
                else:
                    elapsed_str = ""
                
                line = f"\r{prefix}{frame} {task_name}{elapsed_str}{suffix}"
                
                # Ensure line doesn't exceed terminal width (assume 120 chars max)
                if len(line) > 115:
                    line = line[:112] + "..."
                
                # Only write if we're still supposed to be running
                if self.is_running and not self._stop_event.is_set():
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except:
                        break  # If stdout writing fails, stop
                
                self.current_frame = (self.current_frame + 1) % len(self._get_animation_frames())
                
                # Use the stop event for timing instead of sleep to be more responsive
                if self._stop_event.wait(self.config.update_interval):
                    break  # Stop event was set
                    
        except Exception:
            pass  # Silently handle any animation errors
        finally:
            # Ensure we mark as not running when thread exits
            with self._lock:
                self.is_running = False
    
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
    - Streaming support for AI responses
    """
    
    def __init__(self):
        self.active_loaders: List[ProgressLoader] = []
        self.loader_stack: List[ProgressLoader] = []
        self._manager_lock = threading.Lock()  # Add manager lock
    
    @contextmanager
    def show_progress(self, task_name: str, style: LoaderStyle = LoaderStyle.SPINNER):
        """Context manager for showing progress during a task."""
        config = LoaderConfig(task_name=task_name, style=style)
        loader = ProgressLoader(config)
        
        try:
            # Stop any existing loader in the stack before starting new one
            with self._manager_lock:
                if self.loader_stack:
                    current_loader = self.loader_stack[-1]
                    current_loader.stop()
                
                self.loader_stack.append(loader)
            
            loader.start()
            yield loader
            
        finally:
            # Always cleanup this loader
            loader.stop()
            
            with self._manager_lock:
                if loader in self.loader_stack:
                    self.loader_stack.remove(loader)
                
                # If there are remaining loaders in the stack, restart the previous one
                if self.loader_stack:
                    previous_loader = self.loader_stack[-1]
                    if not previous_loader.is_running:
                        previous_loader.start()
    
    @contextmanager
    def streaming_progress(self, task_name: str = "AI Response"):
        """Context manager for streaming AI response text."""
        with self.show_progress(task_name, LoaderStyle.STREAMING) as loader:
            yield loader
    
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
        with self._manager_lock:
            if self.loader_stack:
                current_loader = self.loader_stack[-1]
                current_loader.update_task(new_task_name)
    
    def stream_to_current(self, text_chunk: str):
        """Stream text to the current active loader if it supports streaming."""
        with self._manager_lock:
            if self.loader_stack:
                current_loader = self.loader_stack[-1]
                current_loader.stream_text(text_chunk)
    
    def stream_chunk_to_current(self, chunk: str):
        """Stream a complete chunk to the current active loader if it supports streaming."""
        with self._manager_lock:
            if self.loader_stack:
                current_loader = self.loader_stack[-1]
                current_loader.stream_chunk(chunk)
    
    def cleanup_all(self):
        """Emergency cleanup of all loaders."""
        with self._manager_lock:
            for loader in self.loader_stack[:]:  # Copy to avoid modification during iteration
                loader.stop()
            self.loader_stack.clear()


# Global progress manager instance
progress_manager = ProgressManager()


# Convenience functions for common use cases
@contextmanager
def show_progress(task_name: str, style: LoaderStyle = LoaderStyle.SPINNER):
    """Show progress for a task."""
    with progress_manager.show_progress(task_name, style) as loader:
        yield loader


@contextmanager
def streaming_progress(task_name: str = "AI Response"):
    """Show streaming text progress for AI responses."""
    with progress_manager.streaming_progress(task_name) as loader:
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


def stream_text(text_chunk: str):
    """Stream text to the current active loader."""
    progress_manager.stream_to_current(text_chunk)


def stream_chunk(chunk: str):
    """Stream a complete chunk to the current active loader."""
    progress_manager.stream_chunk_to_current(chunk)


def cleanup_all_loaders():
    """Emergency cleanup function."""
    progress_manager.cleanup_all()


# Decorator for automatic progress loading
def with_progress(task_name: str, style: LoaderStyle = LoaderStyle.SPINNER):
    """Decorator to automatically show progress for a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            with show_progress(task_name, style):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Cleanup function for graceful shutdown
import atexit
atexit.register(cleanup_all_loaders)


if __name__ == "__main__":
    # Demo of different loader styles including streaming
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
            time.sleep(2)
        print()
    
    # Demo streaming
    print("🎯 Streaming Demo:")
    with streaming_progress("AI Code Generation"):
        # Simulate streaming AI response
        demo_text = """Creating a NextJS application with modern components. This will include shadcn/ui components for a professional look and feel. The application will be responsive and include proper TypeScript types for all components."""
        
        words = demo_text.split()
        for word in words:
            stream_text(word + " ")
            time.sleep(0.1)
    
    print("✅ Demo completed!") 