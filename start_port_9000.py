#!/usr/bin/env python3
"""
Alternative startup script using port 9000 range
"""

import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def main():
    try:
        import uvicorn
        from main import app
        
        print("🚀 Starting MCP FastAPI server on port 9000...")
        print("📡 Server URL: http://127.0.0.1:9000")
        print("📚 API Docs: http://127.0.0.1:9000/docs")
        print("🛑 Press Ctrl+C to stop the server\n")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=9000,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
