#!/usr/bin/env python3
"""
Finance MCP Server - FastAPI with MCP Inspector Compatibility
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import contextlib

load_dotenv()

from mcp_servers import (
    finance_db_company,
    finance_db_stock_price,
    finance_calculations,
    finance_portfolio,
    finance_news_and_insights,
    finance_analysis_and_predictions,
)


# Async lifespan context for MCP server initialization
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(finance_db_company.mcp.session_manager.run())
        await stack.enter_async_context(
            finance_db_stock_price.mcp.session_manager.run()
        )
        await stack.enter_async_context(finance_calculations.mcp.session_manager.run())
        await stack.enter_async_context(finance_portfolio.mcp.session_manager.run())
        await stack.enter_async_context(
            finance_news_and_insights.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_analysis_and_predictions.mcp.session_manager.run()
        )
        yield


from api.direct_finance_api import router as finance_api_router

app = FastAPI(
    title="Finance MCP Server",
    description="Streamlined financial analysis platform with 12 essential MCP tools",
    version="2.1.0",
    lifespan=lifespan,
)

app.include_router(finance_api_router, tags=["Finance API"])

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
    "/finance_news_and_insights/",
    finance_news_and_insights.mcp.streamable_http_app(),
    name="finance_news_and_insights",
)
app.mount(
    "/finance_analysis_and_predictions/",
    finance_analysis_and_predictions.mcp.streamable_http_app(),
    name="finance_analysis_and_predictions",
)


@app.get("/", tags=["Root"])
async def root():
    return JSONResponse(
        content={
            "message": "🏦 Finance MCP Server",
            "status": "operational",
            "version": "2.1.0",
            "description": "Streamlined financial analysis platform with comprehensive portfolio risk analysis",
            "total_tools": 12,
            "servers": 6,
            "endpoints": {
                "documentation": "/docs",
                "health_check": "/health",
                "tools_list": "/tools",
            },
            "mcp_servers": [
                "finance_db_company (1 tool)",
                "finance_db_stock_price (2 tools)",
                "finance_calculations (2 tools)",
                "finance_portfolio (2 tools)",
                "finance_news_and_insights (2 tools)",
                "finance_analysis_and_predictions (2 tools)",
            ],
            "recent_updates": [
                "Consolidated portfolio risk analysis into single comprehensive tool",
                "Enhanced technical analysis with advanced indicators",
                "Improved ML prediction models with ensemble techniques",
                "Cleaned project structure - removed duplicate tools",
            ],
        }
    )


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "finance_mcp_server",
        "version": "2.1.0",
        "total_tools": 12,
        "servers_count": 6,
        "database_status": "connected",
        "api_integrations": ["Alpha Vantage", "NewsAPI", "Finnhub"],
    }


@app.get("/tools", tags=["Tools"])
async def list_tools():
    return {
        "total_tools": 12,
        "servers": {
            "finance_db_company": {
                "tools": 1,
                "tool_names": ["search_companies"],
                "description": "Company search with symbol discovery",
            },
            "finance_db_stock_price": {
                "tools": 2,
                "tool_names": ["get_historical_stock_prices", "update_stock_prices"],
                "description": "Historical price data and external updates",
            },
            "finance_calculations": {
                "tools": 2,
                "tool_names": [
                    "calculate_advanced_technical_analysis",
                    "calculate_financial_ratios",
                ],
                "description": "Advanced technical analysis and financial ratios",
            },
            "finance_portfolio": {
                "tools": 2,
                "tool_names": ["analyze_portfolio", "optimize_equal_risk_portfolio"],
                "description": "Comprehensive portfolio analysis with VaR, drawdown, and risk optimization",
            },
            "finance_news_and_insights": {
                "tools": 2,
                "tool_names": ["get_financial_news", "get_market_sentiment"],
                "description": "Financial news and sentiment analysis",
            },
            "finance_analysis_and_predictions": {
                "tools": 2,
                "tool_names": ["predict_stock_price", "analyze_stock_trends"],
                "description": "ML predictions and market analysis",
            },
        },
    }


@app.get("/config", tags=["Configuration"])
async def get_configuration():
    return {
        "server_version": "2.1.0",
        "environment": {
            "alpha_vantage_configured": bool(os.getenv("EXTERNAL_FINANCE_API_KEY")),
            "newsapi_configured": bool(os.getenv("NEWSAPI_KEY")),
            "finnhub_configured": bool(os.getenv("FINNHUB_API_KEY")),
            "database_url_configured": bool(os.getenv("DATABASE_URL")),
        },
        "database": {
            "status": "PostgreSQL configured",
            "expected_tables": ["companies", "stock_prices"],
        },
        "features": {
            "real_time_data": bool(os.getenv("EXTERNAL_FINANCE_API_KEY")),
            "news_analysis": bool(os.getenv("NEWSAPI_KEY")),
            "ml_predictions": True,
            "portfolio_optimization": True,
            "technical_analysis": True,
        },
        "tool_consolidation": {
            "duplicate_tools_removed": True,
            "comprehensive_portfolio_analysis": True,
            "optimized_architecture": True,
        },
    }


if __name__ == "__main__":
    print("🏦 Finance MCP Server v2.1.0")
    print("=" * 40)
    print("📊 12 Essential Tools | 6 Optimized Servers")
    print("🚀 Starting server...")
    print("📡 Server: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("🔧 Tools: http://127.0.0.1:8000/tools")
    print("💓 Health: http://127.0.0.1:8000/health")
    print()
    print("Recent Updates:")
    print("✅ Consolidated portfolio risk analysis")
    print("✅ Enhanced ML prediction models")
    print("✅ Removed duplicate tools")
    print("✅ Clean project structure")
    print()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
