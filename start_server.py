#!/usr/bin/env python3
"""
Startup script for the MCP FastAPI server project.
This script tests imports and starts the server.
"""

import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def test_imports():
    """Test all required imports before starting the server."""
    print("Testing imports...")
    try:
        # Test main application imports
        from dotenv import load_dotenv
        from fastapi import FastAPI
        import uvicorn
        import contextlib
        
        # Test data science imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        
        # Test other dependencies
        import requests
        import jwt
        import psycopg2
        
        # Test MCP server imports
        from mcp_servers import (
            echo, math, user_db, finance_db_company, finance_db_stock_price, 
            finance_data_ingestion, finance_calculations, finance_portfolio,
            finance_plotting, finance_news_and_insights, finance_analysis_and_predictions
        )
        
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_server():
    """Start the FastAPI server."""
    try:
        from main import app
        import uvicorn
        
        print("Starting MCP FastAPI server...")
        print("Server will be available at: http://localhost:8000")
        print("API documentation at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if test_imports():
        start_server()
    else:
        print("Cannot start server due to import errors.")
        sys.exit(1)
