#!/usr/bin/env python3
"""
Smart startup script that finds an available port and starts the server.
"""

import socket
import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def is_port_available(port):
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except OSError:
        return False

def find_available_port(start_port=8000, max_port=8010):
    """Find the first available port in the range."""
    for port in range(start_port, max_port + 1):
        if is_port_available(port):
            return port
    return None

def start_server():
    """Start the server on an available port."""
    print("🔍 Looking for available port...")
    
    port = find_available_port()
    if not port:
        print("❌ No available ports found in range 8000-8010")
        return False
    
    print(f"✅ Found available port: {port}")
    
    try:
        import uvicorn
        from main import app
        
        print(f"🚀 Starting MCP FastAPI server on port {port}...")
        print(f"📡 Server URL: http://127.0.0.1:{port}")
        print(f"📚 API Docs: http://127.0.0.1:{port}/docs")
        print(f"🔄 Alternative Docs: http://127.0.0.1:{port}/redoc")
        print("\n🛑 Press Ctrl+C to stop the server\n")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_server()
