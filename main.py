
#!/usr/bin/env python3
"""
Finance MCP Server - FastAPI with MCP Inspector Compatibility
A streamlined financial analysis server with async MCP server initialization.
"""


import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import contextlib


# Load environment variables
load_dotenv()

# Import MCP servers (6 streamlined servers)
# Note: data_ingestion tools converted to helper functions in finance_helpers.py
# Note: market_data functionality integrated into other servers to eliminate redundancy
from mcp_servers import (
    finance_db_company, 
    finance_db_stock_price, 
    finance_calculations,
    finance_portfolio,
    finance_news_and_insights,
    finance_analysis_and_predictions
)

# Async lifespan context for MCP server initialization
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(finance_db_company.mcp.session_manager.run())
        await stack.enter_async_context(finance_db_stock_price.mcp.session_manager.run())
        await stack.enter_async_context(finance_calculations.mcp.session_manager.run())
        await stack.enter_async_context(finance_portfolio.mcp.session_manager.run())
        await stack.enter_async_context(finance_news_and_insights.mcp.session_manager.run())
        await stack.enter_async_context(finance_analysis_and_predictions.mcp.session_manager.run())
        yield


# Import API routes
from api.direct_finance_api import router as finance_api_router

# Create FastAPI application with lifespan
app = FastAPI(
    title="Finance MCP Server",
    description="Streamlined financial analysis platform with 13 essential MCP tools",
    version="2.0.0",
    lifespan=lifespan
)


# Mount Direct Finance API
app.include_router(finance_api_router, tags=["Finance API"])


# Mount MCP server endpoints
app.mount("/finance_db_company/", finance_db_company.mcp.streamable_http_app(), name="finance_db_company")
app.mount("/finance_db_stock_price/", finance_db_stock_price.mcp.streamable_http_app(), name="finance_db_stock_price")
app.mount("/finance_calculations/", finance_calculations.mcp.streamable_http_app(), name="finance_calculations")
app.mount("/finance_portfolio/", finance_portfolio.mcp.streamable_http_app(), name="finance_portfolio")
app.mount("/finance_news_and_insights/", finance_news_and_insights.mcp.streamable_http_app(), name="finance_news_and_insights")
app.mount("/finance_analysis_and_predictions/", finance_analysis_and_predictions.mcp.streamable_http_app(), name="finance_analysis_and_predictions")

@app.get("/", tags=["Root"])
async def root():
    """Main endpoint with server information"""
    return JSONResponse(content={
        "message": "🏦 Finance MCP Server",
        "status": "operational",
        "version": "2.0.0",
        "description": "Streamlined financial analysis platform",
        "total_tools": 12,
        "servers": 6,
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health"
        },
        "mcp_servers": [
            "finance_db_company (1 tool)", 
            "finance_db_stock_price (2 tools)", 
            "finance_calculations (3 tools)",
            "finance_portfolio (2 tools)", 
            "finance_news_and_insights (2 tools)", 
            "finance_analysis_and_predictions (2 tools)"
        ]
    })

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "service": "finance_mcp_server",
        "version": "2.0.0",
        "total_tools": 12,
        "servers_count": 6
    }


@app.get("/tools", tags=["Tools"])
async def list_tools():
    """List all available MCP tools by server (helper functions excluded)"""
    return {
        "total_tools": 12,
        "servers": {
            "finance_db_company": {
                "tools": 1,
                "tool_names": [
                    "search_companies"
                ],
                "description": "Company search with symbol discovery"
            },
            "finance_db_stock_price": {
                "tools": 2,
                "tool_names": [
                    "get_historical_stock_prices", 
                    "update_stock_prices"
                ],
                "description": "Historical price data and external updates"
            },
            "finance_calculations": {
                "tools": 3,
                "tool_names": [
                    "calculate_advanced_technical_analysis",
                    "calculate_portfolio_risk_metrics", 
                    "calculate_financial_ratios"
                ],
                "description": "Advanced technical analysis and risk metrics"
            },
            "finance_portfolio": {
                "tools": 2,
                "tool_names": [
                    "analyze_portfolio",
                    "optimize_equal_risk_portfolio"
                ],
                "description": "Portfolio analysis and optimization"
            },
            "finance_news_and_insights": {
                "tools": 2,
                "tool_names": [
                    "get_financial_news",
                    "get_market_sentiment"
                ],
                "description": "Financial news and sentiment analysis"
            },
            "finance_analysis_and_predictions": {
                "tools": 2,
                "tool_names": [
                    "predict_stock_price",
                    "analyze_stock_trends"
                ],
                "description": "ML predictions and market analysis"
            }
        }
    }

if __name__ == "__main__":
    print("🏦 Finance MCP Server v2.0.0")
    print("=" * 40)
    print("📊 12 Essential Tools | 6 Servers")
    print("🚀 Starting server...")
    print("📡 Server: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print()
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
