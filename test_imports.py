#!/usr/bin/env python3

import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing all required imports for the MCP server project...\n")

try:
    # Test standard library imports
    print("Testing standard library imports...")
    from datetime import datetime, timedelta
    from decimal import Decimal
    from io import BytesIO
    import base64
    import contextlib
    print("✓ Standard library imports successful")
    
    # Test third-party package imports
    print("\nTesting third-party package imports...")
    from dotenv import load_dotenv
    print("✓ python-dotenv imported successfully")
    
    from fastapi import FastAPI, Request
    print("✓ FastAPI imported successfully")
    
    import uvicorn
    print("✓ uvicorn imported successfully")
    
    import psycopg2
    from psycopg2 import Error
    print("✓ psycopg2 imported successfully")
    
    import pandas as pd
    print("✓ pandas imported successfully")
    
    import numpy as np
    print("✓ numpy imported successfully")
    
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
    
    from sklearn.ensemble import RandomForestRegressor
    print("✓ scikit-learn imported successfully")
    
    import joblib
    print("✓ joblib imported successfully")
    
    import requests
    print("✓ requests imported successfully")
    
    import jwt
    print("✓ PyJWT imported successfully")
    
    from mcp.server.fastmcp import FastMCP
    print("✓ FastMCP imported successfully")
    
    # Test MCP server imports
    print("\nTesting MCP server module imports...")
    from mcp_servers import echo
    print("✓ echo module imported successfully")
    
    from mcp_servers import math
    print("✓ math module imported successfully")
    
    from mcp_servers import user_db
    print("✓ user_db module imported successfully")
    
    from mcp_servers import finance_db_company
    print("✓ finance_db_company module imported successfully")
    
    from mcp_servers import finance_db_stock_price
    print("✓ finance_db_stock_price module imported successfully")
    
    from mcp_servers import finance_data_ingestion
    print("✓ finance_data_ingestion module imported successfully")
    
    from mcp_servers import finance_calculations
    print("✓ finance_calculations module imported successfully")
    
    from mcp_servers import finance_portfolio
    print("✓ finance_portfolio module imported successfully")
    
    from mcp_servers import finance_plotting
    print("✓ finance_plotting module imported successfully")
    
    from mcp_servers import finance_news_and_insights
    print("✓ finance_news_and_insights module imported successfully")
    
    from mcp_servers import finance_analysis_and_predictions
    print("✓ finance_analysis_and_predictions module imported successfully")
    
    print("\n🎉 All imports successful! The project is ready to run.")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
