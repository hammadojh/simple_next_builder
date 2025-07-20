#!/usr/bin/env python3
"""
Test script to demonstrate the progress loader system.
Run this to see the different loader styles in action.
"""

import sys
import time
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lib.progress_loader import (
    show_progress, llm_progress, analysis_progress, 
    build_progress, file_progress, update_current_task,
    LoaderStyle
)

def simulate_task(duration=2):
    """Simulate a task that takes some time."""
    time.sleep(duration)

def main():
    print("🎯 Progress Loader System Demo")
    print("=" * 50)
    print("This demonstrates the new progress loaders that keep the terminal active!")
    print()
    
    # Test different loader styles
    test_cases = [
        ("Building NextJS application", LoaderStyle.BUILDING, 3),
        ("Analyzing code structure", LoaderStyle.PULSE, 2),
        ("Calling OpenAI API", LoaderStyle.THINKING, 4),
        ("Processing files", LoaderStyle.SPINNER, 2),
        ("Downloading resources", LoaderStyle.NETWORK, 3),
        ("Converting files", LoaderStyle.PROGRESS_BAR, 2),
        ("Loading content", LoaderStyle.DOTS, 2),
    ]
    
    for task_name, style, duration in test_cases:
        print(f"\n🔄 Testing: {style.value}")
        with show_progress(task_name, style):
            # Simulate task updates
            if style == LoaderStyle.THINKING:
                time.sleep(1)
                update_current_task("processing your request")
                time.sleep(1)
                update_current_task("generating response")
                time.sleep(1)
                update_current_task("validating output")
                time.sleep(1)
            else:
                simulate_task(duration)
        
        print(f"✅ {task_name} completed!")
    
    # Test nested loaders
    print(f"\n🔄 Testing nested loaders:")
    with show_progress("Creating NextJS app", LoaderStyle.BUILDING):
        time.sleep(1)
        update_current_task("setting up template")
        time.sleep(1)
        
        with llm_progress("Claude AI"):
            update_current_task("generating app code")
            time.sleep(2)
        
        update_current_task("applying generated code")
        with file_progress("processing", 5):
            time.sleep(1)
            update_current_task("writing files to disk")
            time.sleep(1)
        
        update_current_task("finalizing build")
        time.sleep(1)
    
    print("✅ App creation completed!")
    
    # Test specialized context managers
    print(f"\n🔄 Testing specialized context managers:")
    
    with analysis_progress("user requirements"):
        time.sleep(2)
    
    with build_progress("installing dependencies"):
        time.sleep(2)
    
    with file_progress("creating components", 3):
        time.sleep(2)
    
    print("\n🎉 All progress loader tests completed!")
    print("💡 The terminal was never idle during these operations!")
    print("🚀 Now you can integrate this system into your app builder!")

if __name__ == "__main__":
    main() 