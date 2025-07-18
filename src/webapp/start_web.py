#!/usr/bin/env python3
"""
Start the NextJS Master Builder Web Interface
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import websockets
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\n🔧 To install dependencies, run:")
        print("   pip install -r requirements_web.txt")
        print("\n📦 Or install manually:")
        print("   pip install fastapi uvicorn websockets python-dotenv")
        return False

def check_env():
    """Check if OpenAI API key is available."""
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key found")
        return True
    elif os.path.exists('.env'):
        print("✅ .env file found")
        return True
    else:
        print("⚠️  OpenAI API key not found")
        print("\n🔑 Please set your OpenAI API key:")
        print("   1. Create a .env file with: OPENAI_API_KEY=your_key_here")
        print("   2. Or set environment variable: export OPENAI_API_KEY=your_key_here")
        print("   3. Get your API key from: https://platform.openai.com/api-keys")
        return False

def main():
    print("🚀 NextJS Master Builder Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_env():
        print("\n⚠️  Warning: No API key found. The interface will start but app creation will fail.")
        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            sys.exit(1)
    
    print("\n🌐 Starting web interface...")
    print("📍 URL: http://localhost:8001")
    print("💡 Tip: Keep this terminal open while using the web interface")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the FastAPI server
        os.system("python src/webapp/web_interface.py")
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"\n❌ Error starting web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 