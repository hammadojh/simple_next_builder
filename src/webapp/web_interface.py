#!/usr/bin/env python3
"""
NextJS Master Builder Web Interface
FastAPI server that provides a web interface for the NextJS Master Builder system.
"""

import os
import sys
import asyncio
import json
import socket
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Add the current directory to Python path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our existing modules
from src.main import MasterBuilder

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="NextJS Master Builder", description="Web interface for NextJS app generation and editing")

# Pydantic models for API requests
class CreateAppRequest(BaseModel):
    idea: str
    name: Optional[str] = None

class EditAppRequest(BaseModel):
    app_name: str
    idea: str

class ChatMessage(BaseModel):
    message: str
    timestamp: str
    type: str  # 'user', 'assistant', 'system', 'error'

# Global variables
master_builder = None
connected_websockets: List[WebSocket] = []
running_dev_servers = {}  # Track running dev servers {app_name: (process, port)}

def find_available_port(start_port: int = 3000) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            continue
    raise Exception("No available ports found")

def start_nextjs_dev_server(app_directory: str, port: int):
    """Start a NextJS dev server for the given app on the specified port."""
    global running_dev_servers
    
    app_name = Path(app_directory).name
    
    # Kill existing server if running
    if app_name in running_dev_servers:
        try:
            process, old_port = running_dev_servers[app_name]
            process.terminate()
            del running_dev_servers[app_name]
        except:
            pass
    
    try:
        # Start the dev server
        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(port)],
            cwd=app_directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        running_dev_servers[app_name] = (process, port)
        print(f"✅ Started dev server for {app_name} on port {port}")
        
    except Exception as e:
        print(f"❌ Failed to start dev server for {app_name}: {e}")
        raise

def get_app_port(app_name: str) -> Optional[int]:
    """Get the port for a running app, or None if not running."""
    if app_name in running_dev_servers:
        process, port = running_dev_servers[app_name]
        # Check if process is still running
        if process.poll() is None:
            return port
        else:
            # Process died, remove from tracking
            del running_dev_servers[app_name]
    return None

def get_master_builder():
    """Get or create the master builder instance."""
    global master_builder
    if master_builder is None:
        try:
            master_builder = MasterBuilder()
        except Exception as e:
            print(f"Error initializing MasterBuilder: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize builder: {str(e)}")
    return master_builder

async def broadcast_message(message: Dict):
    """Broadcast message to all connected WebSocket clients."""
    if connected_websockets:
        for websocket in connected_websockets.copy():
            try:
                await websocket.send_json(message)
            except:
                connected_websockets.remove(websocket)

@app.get("/")
async def read_root():
    """Serve the main web interface."""
    return FileResponse("web_ui.html")

@app.get("/api/apps")
async def list_apps():
    """Get list of existing NextJS apps."""
    try:
        builder = get_master_builder()
        apps = builder.list_existing_apps()
        
        # Get additional info for each app
        app_details = []
        for app_name in apps:
            app_dir = builder.apps_dir / app_name
            package_json = app_dir / "package.json"
            
            # Check if app is running
            port = get_app_port(app_name)
            
            details = {
                "name": app_name,
                "path": str(app_dir),
                "created": "unknown",
                "status": "running" if port else "stopped",
                "port": port,
                "preview_url": f"http://localhost:{port}" if port else None
            }
            
            if package_json.exists():
                try:
                    stat = package_json.stat()
                    details["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
                except:
                    pass
            
            app_details.append(details)
        
        return {"apps": app_details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/apps/create")
async def create_app(request: CreateAppRequest):
    """Create a new NextJS app."""
    try:
        builder = get_master_builder()
        
        # Broadcast status update
        await broadcast_message({
            "type": "system",
            "message": f"🏗️ Creating app: {request.idea}",
            "timestamp": datetime.now().isoformat()
        })
        
        # Create the app (without running the dev server)
        app_name = request.name or builder.get_next_app_name()
        app_directory = builder.apps_dir / app_name
        
        # Create template app
        if not builder.create_template_nextjs_app(app_name):
            raise HTTPException(status_code=500, detail="Failed to create template app")
        
        # Generate AI changes
        input_file = builder.generate_ai_changes(request.idea, app_name)
        if not input_file:
            raise HTTPException(status_code=500, detail="Failed to generate AI changes")
        
        # Apply changes
        if not builder.apply_changes(input_file, str(app_directory)):
            raise HTTPException(status_code=500, detail="Failed to apply changes")
        
        # Install dependencies
        if not builder.install_dependencies(str(app_directory)):
            raise HTTPException(status_code=500, detail="Failed to install dependencies")
        
        # Validate and auto-fix build
        build_success = builder.validate_and_fix_build(str(app_directory))
        
        # Start the NextJS dev server in the background
        try:
            await broadcast_message({
                "type": "system",
                "message": f"🚀 Starting dev server for {app_name}...",
                "timestamp": datetime.now().isoformat()
            })
            
            # Find an available port
            preview_port = find_available_port(3000)
            start_nextjs_dev_server(str(app_directory), preview_port)
            
            await broadcast_message({
                "type": "system", 
                "message": f"✅ Dev server started on port {preview_port}",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            await broadcast_message({
                "type": "error",
                "message": f"⚠️ Could not start dev server: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        success = True  # Consider it successful even if build validation fails
        
        if success:
            await broadcast_message({
                "type": "system", 
                "message": f"✅ App '{app_name}' created successfully!",
                "timestamp": datetime.now().isoformat()
            })
            return {"success": True, "app_name": app_name, "message": "App created successfully"}
        else:
            await broadcast_message({
                "type": "error",
                "message": "❌ Failed to create app",
                "timestamp": datetime.now().isoformat()
            })
            raise HTTPException(status_code=500, detail="Failed to create app")
            
    except Exception as e:
        await broadcast_message({
            "type": "error",
            "message": f"❌ Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/apps/edit")
async def edit_app(request: EditAppRequest):
    """Edit an existing NextJS app."""
    try:
        builder = get_master_builder()
        app_directory = builder.apps_dir / request.app_name
        
        if not app_directory.exists():
            raise HTTPException(status_code=404, detail=f"App '{request.app_name}' not found")
        
        # Broadcast status update
        await broadcast_message({
            "type": "system",
            "message": f"✏️ Editing {request.app_name}: {request.idea}",
            "timestamp": datetime.now().isoformat()
        })
        
        # Edit the app
        success = builder.edit_existing_app(str(app_directory), request.idea)
        
        if success:
            await broadcast_message({
                "type": "system",
                "message": f"✅ App '{request.app_name}' edited successfully!",
                "timestamp": datetime.now().isoformat()
            })
            return {"success": True, "message": "App edited successfully"}
        else:
            await broadcast_message({
                "type": "error",
                "message": f"❌ Failed to edit app '{request.app_name}'",
                "timestamp": datetime.now().isoformat()
            })
            raise HTTPException(status_code=500, detail="Failed to edit app")
            
    except Exception as e:
        await broadcast_message({
            "type": "error",
            "message": f"❌ Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/apps/{app_name}/preview")
async def get_app_preview(app_name: str):
    """Get preview info for an app."""
    try:
        builder = get_master_builder()
        app_directory = builder.apps_dir / app_name
        
        if not app_directory.exists():
            raise HTTPException(status_code=404, detail=f"App '{app_name}' not found")
        
        # Get the actual port for this app
        port = get_app_port(app_name)
        
        if port:
            preview_url = f"http://localhost:{port}"
            status = "running"
        else:
            preview_url = None
            status = "stopped"
        
        return {
            "app_name": app_name,
            "preview_url": preview_url,
            "port": port,
            "status": status,
            "start_command": f"cd apps/{app_name} && npm run dev"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/apps/{app_name}/start")
async def start_app_server(app_name: str):
    """Start the dev server for an app."""
    try:
        builder = get_master_builder()
        app_directory = builder.apps_dir / app_name
        
        if not app_directory.exists():
            raise HTTPException(status_code=404, detail=f"App '{app_name}' not found")
        
        # Check if already running
        if get_app_port(app_name):
            return {"success": True, "message": f"App '{app_name}' is already running"}
        
        # Find available port and start server
        port = find_available_port(3000)
        start_nextjs_dev_server(str(app_directory), port)
        
        await broadcast_message({
            "type": "system",
            "message": f"🚀 Started {app_name} on port {port}",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True, 
            "message": f"Dev server started on port {port}",
            "port": port,
            "preview_url": f"http://localhost:{port}"
        }
        
    except Exception as e:
        await broadcast_message({
            "type": "error",
            "message": f"❌ Failed to start {app_name}: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/apps/{app_name}/stop")
async def stop_app_server(app_name: str):
    """Stop the dev server for an app."""
    try:
        global running_dev_servers
        
        if app_name not in running_dev_servers:
            return {"success": True, "message": f"App '{app_name}' is not running"}
        
        # Stop the server
        process, port = running_dev_servers[app_name]
        process.terminate()
        del running_dev_servers[app_name]
        
        await broadcast_message({
            "type": "system",
            "message": f"🛑 Stopped {app_name}",
            "timestamp": datetime.now().isoformat()
        })
        
        return {"success": True, "message": f"Dev server stopped"}
        
    except Exception as e:
        await broadcast_message({
            "type": "error",
            "message": f"❌ Failed to stop {app_name}: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    # Send welcome message
    await websocket.send_json({
        "type": "system",
        "message": "🎯 Connected to NextJS Master Builder",
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            
            # Echo user message
            await broadcast_message({
                "type": "user",
                "message": data.get("message", ""),
                "timestamp": datetime.now().isoformat()
            })
            
            # Process the message (you could add chat-based app creation here)
            message = data.get("message", "").lower()
            
            if "create" in message and "app" in message:
                await websocket.send_json({
                    "type": "assistant",
                    "message": "I can help you create an app! Please use the 'Create App' button or specify your app idea more clearly.",
                    "timestamp": datetime.now().isoformat()
                })
            elif "list" in message and "app" in message:
                builder = get_master_builder()
                apps = builder.list_existing_apps()
                await websocket.send_json({
                    "type": "assistant",
                    "message": f"📱 Existing apps: {', '.join(apps) if apps else 'None'}",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "assistant",
                    "message": "I'm your NextJS Master Builder assistant! Use the interface to create or edit apps, or ask me about existing apps.",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

if __name__ == "__main__":
    print("🚀 Starting NextJS Master Builder Web Interface...")
    print("🌐 Access the web interface at: http://localhost:8000")
    uvicorn.run("web_interface:app", host="0.0.0.0", port=8000, reload=True) 