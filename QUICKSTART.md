# MCP FastAPI Server - Quick Start Guide

## ✅ Setup Complete!

All required libraries have been installed successfully:
- ✅ FastAPI and Uvicorn (web framework)
- ✅ FastMCP (MCP server framework)
- ✅ PostgreSQL driver (psycopg2-binary)
- ✅ Data science libraries (pandas, numpy, scikit-learn)
- ✅ Plotting library (matplotlib)
- ✅ Authentication (PyJWT)
- ✅ HTTP requests (requests)
- ✅ Environment variables (python-dotenv)

## 🚀 How to Run the Server

### Option 1: Using PowerShell Script (Recommended)
```powershell
.\start_server.ps1
```

### Option 2: Using Python directly
```bash
.venv\Scripts\python.exe main.py
```

### Option 3: Using the startup script
```bash
.venv\Scripts\python.exe start_server.py
```

## 🌐 Access Points

Once the server is running, you can access:

- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

### Available MCP Endpoints:
- `/echo/` - Echo server for testing
- `/math/` - Mathematical operations
- `/user_db/` - User database operations  
- `/finance_db_company/` - Company data management
- `/finance_db_stock_price/` - Stock price data
- `/finance_data_ingestion/` - ML-based stock analysis and predictions
- `/finance_calculations/` - Financial calculations (returns, volatility, comparisons)
- `/finance_portfolio/` - Portfolio management and analysis
- `/finance_plotting/` - Financial data visualization and charts
- `/finance_news_and_insights/` - Financial news and market insights
- `/finance_analysis_and_predictions/` - Advanced financial analysis and predictions

## 🧪 Test Imports

To test if all dependencies are correctly installed:
```bash
.venv\Scripts\python.exe test_imports.py
```

## 📊 Available Finance Tools

The finance data ingestion module provides:
- Stock price prediction using Random Forest
- Historical price analysis
- Technical indicators (SMA, moving averages)
- Machine learning model training and prediction

## 🔧 Environment Configuration

Make sure your `.env` file contains the proper database connection strings for:
- `USER_DB_URI` - User database connection
- `FINANCE_DB_URI` - Finance database connection

## 🎯 Next Steps

1. Start the server using one of the methods above
2. Visit http://localhost:8000/docs to explore the API
3. Test the finance prediction tools
4. Check the various MCP endpoints

Happy coding! 🚀
