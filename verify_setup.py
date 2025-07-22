#!/usr/bin/env python3
"""
Final verification script to ensure all components are working before starting the server.
"""

import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def test_environment():
    """Test the environment setup."""
    print("🔍 Testing Python environment...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Project directory: {project_dir}")
    return True

def test_environment_variables():
    """Test that required environment variables are present."""
    print("\n🔧 Testing environment variables...")
    from dotenv import load_dotenv
    load_dotenv()
    
    db_uri = os.getenv("FINANCE_DB_URI")
    user_db_uri = os.getenv("USER_DB_URI")
    
    if not db_uri:
        print("⚠️  Warning: FINANCE_DB_URI not found in environment")
        return False
    if not user_db_uri:
        print("⚠️  Warning: USER_DB_URI not found in environment")
        return False
        
    print("✓ Environment variables found")
    return True

def test_all_imports():
    """Test all imports systematically."""
    print("\n📦 Testing all imports...")
    
    # Standard library
    try:
        from datetime import datetime, timedelta
        from decimal import Decimal
        from io import BytesIO
        import base64
        import contextlib
        print("✓ Standard library imports")
    except Exception as e:
        print(f"❌ Standard library error: {e}")
        return False
    
    # Third-party packages
    try:
        from dotenv import load_dotenv
        from fastapi import FastAPI, Request
        import uvicorn
        import psycopg2
        from psycopg2 import Error
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        import requests
        import jwt
        from mcp.server.fastmcp import FastMCP
        print("✓ Third-party packages")
    except Exception as e:
        print(f"❌ Third-party package error: {e}")
        return False
    
    # MCP server modules
    try:
        from mcp_servers import (
            echo, math, user_db, finance_db_company, finance_db_stock_price, 
            finance_data_ingestion, finance_calculations, finance_portfolio,
            finance_plotting, finance_news_and_insights, finance_analysis_and_predictions
        )
        print("✓ All MCP server modules")
    except Exception as e:
        print(f"❌ MCP server module error: {e}")
        return False
    
    return True

def test_main_application():
    """Test that the main application can be imported."""
    print("\n🚀 Testing main application...")
    try:
        from main import app
        print("✓ Main application imported successfully")
        print(f"✓ FastAPI app title: {app.title}")
        return True
    except Exception as e:
        print(f"❌ Main application error: {e}")
        return False

def run_all_tests():
    """Run all verification tests."""
    print("🧪 Running comprehensive verification tests...\n")
    
    tests = [
        ("Environment", test_environment),
        ("Environment Variables", test_environment_variables),
        ("Imports", test_all_imports),
        ("Main Application", test_main_application)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 VERIFICATION RESULTS:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} | {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Server is ready to start.")
        print("\nTo start the server, run:")
        print("  python start_server.py")
        print("  or")
        print("  python main.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before starting the server.")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
