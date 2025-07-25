#!/usr/bin/env python3
"""
Finance MCP Server - FastAPI Application
A comprehensive financial analysis server with multiple MCP tools for market data,
analysis, predictions, portfolio management, and more.
"""

import os
import contextlib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Load environment variables
load_dotenv()

# Import MCP servers
from mcp_servers import (
    echo, 
    math, 
    user_db, 
    finance_db_company, 
    finance_db_stock_price, 
    finance_data_ingestion,
    finance_calculations,
    finance_portfolio,
    finance_plotting,
    finance_news_and_insights,
    finance_analysis_and_predictions,
    finance_market_data
)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown of MCP servers
    """
    print("🚀 Starting Finance MCP Server...")
    
    async with contextlib.AsyncExitStack() as stack:
        # Core utility servers
        print("📡 Starting core servers...")
        await stack.enter_async_context(echo.mcp.session_manager.run())
        await stack.enter_async_context(math.mcp.session_manager.run())
        await stack.enter_async_context(user_db.mcp.session_manager.run())
        
        # Database servers
        print("💾 Starting database servers...")
        await stack.enter_async_context(finance_db_company.mcp.session_manager.run())
        await stack.enter_async_context(finance_db_stock_price.mcp.session_manager.run())
        
        # Finance analysis servers
        print("📊 Starting finance analysis servers...")
        await stack.enter_async_context(finance_data_ingestion.mcp.session_manager.run())
        await stack.enter_async_context(finance_calculations.mcp.session_manager.run())
        await stack.enter_async_context(finance_portfolio.mcp.session_manager.run())
        await stack.enter_async_context(finance_plotting.mcp.session_manager.run())
        await stack.enter_async_context(finance_news_and_insights.mcp.session_manager.run())
        await stack.enter_async_context(finance_analysis_and_predictions.mcp.session_manager.run())
        await stack.enter_async_context(finance_market_data.mcp.session_manager.run())
        
        print("✅ All MCP servers started successfully!")
        yield
        print("🛑 Shutting down Finance MCP Server...")


# Create FastAPI application
app = FastAPI(
    title="Finance MCP Server",
    description="""
    ## 🏦 Comprehensive Financial Analysis Platform
    
    A powerful FastAPI server with multiple Model Context Protocol (MCP) tools for:
    
    ### 📈 Market Data & Analysis
    - Real-time market data for stocks, indices, crypto, forex
    - Technical analysis and price predictions
    - Chart data for frontend integration
    
    ### 📊 Portfolio Management
    - Portfolio tracking and optimization
    - Risk analysis and performance metrics
    - Asset allocation recommendations
    
    ### 📰 News & Insights
    - Financial news aggregation
    - Market sentiment analysis
    - Real-time insights and alerts
    
    ### 💾 Data Management
    - Company and stock price databases
    - Data ingestion from multiple sources
    - Historical data storage and retrieval
    
    ### 🔧 Utility Tools
    - Mathematical calculations
    - User management
    - Echo service for testing
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files for dashboard
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("📁 Static files mounted at /static")

# Mount MCP server endpoints
print("🔗 Mounting MCP endpoints...")

# Core utility endpoints
app.mount("/echo/", echo.mcp.streamable_http_app(), name="echo")
app.mount("/math/", math.mcp.streamable_http_app(), name="math")
app.mount("/user_db/", user_db.mcp.streamable_http_app(), name="user_db")

# Database endpoints
app.mount(
    "/finance_db_company/",
    finance_db_company.mcp.streamable_http_app(),
    name="finance_db_company",
)
app.mount(
    "/finance_db_stock_price/",
    finance_db_stock_price.mcp.streamable_http_app(),
    name="finance_db_stock_price",
)

# Finance analysis endpoints
app.mount(
    "/finance_data_ingestion/",
    finance_data_ingestion.mcp.streamable_http_app(),
    name="finance_data_ingestion",
)
app.mount(
    "/finance_calculations/",
    finance_calculations.mcp.streamable_http_app(),
    name="finance_calculations",
)
app.mount(
    "/finance_portfolio/",
    finance_portfolio.mcp.streamable_http_app(),
    name="finance_portfolio",
)
app.mount(
    "/finance_plotting/",
    finance_plotting.mcp.streamable_http_app(),
    name="finance_plotting",
)
app.mount(
    "/finance_news_and_insights/",
    finance_news_and_insights.mcp.streamable_http_app(),
    name="finance_news_and_insights",
)
app.mount(
    "/finance_analysis_and_predictions/",
    finance_analysis_and_predictions.mcp.streamable_http_app(),
    name="finance_analysis_and_predictions",
)
app.mount(
    "/finance_market_data/",
    finance_market_data.mcp.streamable_http_app(),
    name="finance_market_data",
)


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def read_root():
    """
    Root endpoint - returns dashboard or API information
    """
    # If static files exist, redirect to dashboard
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    
    # Otherwise return JSON API info
    return {
        "message": "🏦 Finance MCP Server",
        "status": "operational",
        "version": "1.0.0",
        "description": "Comprehensive financial analysis platform with MCP tools",
        "endpoints": {
            "documentation": "/docs",
            "alternative_docs": "/redoc",
            "health_check": "/health",
            "server_status": "/status",
            "available_tools": "/tools"
        },
        "mcp_servers": [
            "echo", "math", "user_db", 
            "finance_db_company", "finance_db_stock_price", 
            "finance_data_ingestion", "finance_calculations",
            "finance_portfolio", "finance_plotting", 
            "finance_news_and_insights", "finance_analysis_and_predictions",
            "finance_market_data"
        ]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint - returns server health status
    """
    return {
        "status": "healthy",
        "service": "finance_mcp_server",
        "version": "1.0.0",
        "uptime": "operational",
        "mcp_servers": {
            "core": ["echo", "math", "user_db"],
            "database": ["finance_db_company", "finance_db_stock_price"],
            "analysis": [
                "finance_data_ingestion", "finance_calculations",
                "finance_portfolio", "finance_plotting", 
                "finance_news_and_insights", "finance_analysis_and_predictions",
                "finance_market_data"
            ]
        },
        "endpoints_count": 12,
        "environment": {
            "has_api_keys": bool(os.getenv("EXTERNAL_FINANCE_API_KEY")),
            "static_files": os.path.exists("static")
        }
    }


@app.get("/status", tags=["Status"])
async def server_status():
    """
    Detailed server status with configuration info
    """
    return {
        "server": {
            "name": "Finance MCP Server",
            "version": "1.0.0",
            "status": "running",
            "host": "127.0.0.1",
            "port": 8000
        },
        "configuration": {
            "environment_loaded": True,
            "api_keys_configured": {
                "external_finance_api": bool(os.getenv("EXTERNAL_FINANCE_API_KEY")),
                "finnhub": bool(os.getenv("FINNHUB_API_KEY")),
                "news_api": bool(os.getenv("NEWS_API_KEY"))
            },
            "static_files_available": os.path.exists("static"),
            "database_path": os.path.exists("data") if os.path.exists("data") else "Not configured"
        },
        "mcp_endpoints": {
            "/echo/": "Echo service for testing",
            "/math/": "Mathematical calculations",
            "/user_db/": "User management",
            "/finance_db_company/": "Company database operations",
            "/finance_db_stock_price/": "Stock price database",
            "/finance_data_ingestion/": "Data ingestion tools",
            "/finance_calculations/": "Financial calculations",
            "/finance_portfolio/": "Portfolio management",
            "/finance_plotting/": "Chart and plot generation",
            "/finance_news_and_insights/": "News and market insights",
            "/finance_analysis_and_predictions/": "Analysis and predictions",
            "/finance_market_data/": "Real-time market data"
        }
    }


@app.get("/tools", tags=["Tools"])
async def available_tools():
    """
    List all available MCP tools and their capabilities
    """
    return {
        "available_tools": {
            "core_utilities": {
                "echo": "Simple echo service for testing connectivity",
                "math": "Mathematical operations and calculations",
                "user_db": "User management and authentication"
            },
            "database_operations": {
                "finance_db_company": "Company information storage and retrieval",
                "finance_db_stock_price": "Historical stock price data management"
            },
            "data_analysis": {
                "finance_data_ingestion": "Import data from multiple financial sources",
                "finance_calculations": "Advanced financial calculations and metrics",
                "finance_market_data": "Real-time market data for stocks, crypto, forex"
            },
            "portfolio_management": {
                "finance_portfolio": "Portfolio tracking, optimization, and analysis"
            },
            "visualization": {
                "finance_plotting": "Generate charts and visualizations"
            },
            "intelligence": {
                "finance_news_and_insights": "News aggregation and sentiment analysis",
                "finance_analysis_and_predictions": "ML-based predictions and analysis"
            }
        },
        "usage": "Access tools via their respective endpoints (e.g., /finance_market_data/)",
        "documentation": "Visit /docs for interactive API documentation"
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/", "/docs", "/health", "/status", "/tools"
            ],
            "mcp_endpoints": [
                "/echo/", "/math/", "/user_db/",
                "/finance_db_company/", "/finance_db_stock_price/",
                "/finance_data_ingestion/", "/finance_calculations/",
                "/finance_portfolio/", "/finance_plotting/",
                "/finance_news_and_insights/", "/finance_analysis_and_predictions/",
                "/finance_market_data/"
            ]
        }
    )


if __name__ == "__main__":
    print("🏦 Finance MCP Server")
    print("=" * 50)
    print("🚀 Starting server...")
    print("📡 Server will be available at: http://127.0.0.1:8000")
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("🏥 Health Check: http://127.0.0.1:8000/health")
    print("📊 Server Status: http://127.0.0.1:8000/status")
    print("🛠️ Available Tools: http://127.0.0.1:8000/tools")
    print()
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True,
        reload_dirs=["mcp_servers", "utils"],
        reload_includes=["*.py"]
    )
