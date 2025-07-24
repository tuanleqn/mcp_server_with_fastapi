#!/usr/bin/env python3

print("🚀 Starting MCP Server Test...")

try:
    print("📦 Loading environment variables...")
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded")
    
    print("📦 Importing FastAPI...")
    from fastapi import FastAPI
    print("✅ FastAPI imported")
    
    print("📦 Testing database connection...")
    import psycopg2
    import os
    
    USER_DB_URI = os.getenv("USER_DB_URI")
    if USER_DB_URI:
        print(f"✅ USER_DB_URI found: {USER_DB_URI[:20]}...")
    else:
        print("⚠️  USER_DB_URI not found")
    
    print("📦 Importing MCP servers...")
    from mcp_servers import echo, math
    print("✅ MCP servers imported")
    
    print("🎯 All imports successful! Starting main server...")
    
    # Import and run main
    import main
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
