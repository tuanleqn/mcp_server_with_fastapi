#!/usr/bin/env python3
"""
Finance MCP Server - Simple FastAPI Application
A streamlined financial analysis server with 20 essential MCP tools.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables
load_dotenv()

# Import MCP servers (8 streamlined servers)
from mcp_servers import (
    finance_db_company, 
    finance_db_stock_price, 
    finance_data_ingestion,
    finance_calculations,
    finance_portfolio,
    finance_news_and_insights,
    finance_analysis_and_predictions,
    finance_market_data
)

# Import API routes
from api.direct_finance_api import router as finance_api_router

# Create FastAPI application
app = FastAPI(
    title="Finance MCP Server",
    description="Streamlined financial analysis platform with 20 essential MCP tools",
    version="2.0.0"
)

# Mount Direct Finance API
app.include_router(finance_api_router, tags=["Finance API"])

# Mount MCP server endpoints (8 servers with 20 total tools)
app.mount("/finance_db_company/", finance_db_company.mcp.streamable_http_app(), name="finance_db_company")
app.mount("/finance_db_stock_price/", finance_db_stock_price.mcp.streamable_http_app(), name="finance_db_stock_price")
app.mount("/finance_data_ingestion/", finance_data_ingestion.mcp.streamable_http_app(), name="finance_data_ingestion")
app.mount("/finance_calculations/", finance_calculations.mcp.streamable_http_app(), name="finance_calculations")
app.mount("/finance_portfolio/", finance_portfolio.mcp.streamable_http_app(), name="finance_portfolio")
app.mount("/finance_news_and_insights/", finance_news_and_insights.mcp.streamable_http_app(), name="finance_news_and_insights")
app.mount("/finance_analysis_and_predictions/", finance_analysis_and_predictions.mcp.streamable_http_app(), name="finance_analysis_and_predictions")
app.mount("/finance_market_data/", finance_market_data.mcp.streamable_http_app(), name="finance_market_data")

@app.get("/", tags=["Root"])
async def root():
    """Main endpoint with server information"""
    return JSONResponse(content={
        "message": "🏦 Finance MCP Server",
        "status": "operational",
        "version": "2.0.0",
        "description": "Streamlined financial analysis platform",
        "total_tools": 20,
        "servers": 8,
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health"
        },
        "mcp_servers": [
            "finance_db_company (3 tools)", 
            "finance_db_stock_price (2 tools)", 
            "finance_data_ingestion (2 tools)", 
            "finance_calculations (3 tools)",
            "finance_portfolio (2 tools)", 
            "finance_news_and_insights (2 tools)", 
            "finance_analysis_and_predictions (3 tools)",
            "finance_market_data (3 tools)"
        ]
    })

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "service": "finance_mcp_server",
        "version": "2.0.0",
        "total_tools": 20,
        "servers_count": 8
    }

@app.get("/tools", tags=["Tools"])
async def list_tools():
    """List all available tools by server"""
    return {
        "total_tools": 20,
        "servers": {
            "finance_db_company": {
                "tools": 3,
                "description": "Company database operations and symbol search"
            },
            "finance_db_stock_price": {
                "tools": 2,
                "description": "Stock price database operations"
            },
            "finance_data_ingestion": {
                "tools": 2,
                "description": "ML model training and predictions"
            },
            "finance_calculations": {
                "tools": 3,
                "description": "Financial calculations and returns"
            },
            "finance_portfolio": {
                "tools": 2,
                "description": "Portfolio management operations"
            },
            "finance_news_and_insights": {
                "tools": 2,
                "description": "News and sentiment analysis"
            },
            "finance_analysis_and_predictions": {
                "tools": 3,
                "description": "Technical analysis and ML predictions"
            },
            "finance_market_data": {
                "tools": 3,
                "description": "Real-time market data and status"
            }
        }
    }

if __name__ == "__main__":
    print("🏦 Finance MCP Server v2.0.0")
    print("=" * 40)
    print("📊 20 Essential Tools | 8 Servers")
    print("🚀 Starting server...")
    print("📡 Server: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print()
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True
    )
