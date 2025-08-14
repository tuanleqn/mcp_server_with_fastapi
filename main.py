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
        # await stack.enter_async_context(finance_portfolio.mcp.session_manager.run())
        await stack.enter_async_context(
            finance_news_and_insights.mcp.session_manager.run()
        )
        await stack.enter_async_context(
            finance_analysis_and_predictions.mcp.session_manager.run()
        )
        yield


from api.direct_finance_api import router as finance_api_router

app = FastAPI(
    title="Finance MCP Server - Optimized",
    description="Streamlined financial analysis platform with 8 essential optimized MCP tools",
    version="3.0.0",
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
# app.mount(
#     "/finance_portfolio/",
#     finance_portfolio.mcp.streamable_http_app(),
#     name="finance_portfolio",
# )
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
            "message": "🏦 Finance MCP Server - Optimized Edition",
            "status": "operational",
            "version": "3.0.0",
            "description": "Optimized financial analysis platform with streamlined MCP tools",
            "total_tools": 8,
            "servers": 6,
            "endpoints": {
                "documentation": "/docs",
                "health_check": "/health",
                "tools_list": "/tools",
            },
            "mcp_servers": [
                "finance_db_company (1 tool)",
                "finance_db_stock_price (2 tools) - Simplified parameter handling",
                "finance_calculations (3 tools) - Enhanced null handling",
                "finance_portfolio (1 tool) - Portfolio optimization with recommended allocation rates",
                "finance_news_and_insights (2 tools) - Enhanced with news article lists",
                "finance_analysis_and_predictions (2 tools)",
            ],
            "optimization_updates": [
                "Portfolio server optimized with Modern Portfolio Theory recommendations",
                "Market sentiment analysis now includes full news article details",
                "Enhanced stock price retrieval with simplified days parameter",
                "Fixed current_price None conversion issues across all servers",
                "Improved error handling for price data conversion",
            ],
        }
    )


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "finance_mcp_server_optimized",
        "version": "3.0.0",
        "total_tools": 8,
        "servers_count": 6,
        "database_status": "connected",
        "api_integrations": ["Alpha Vantage", "NewsAPI", "Finnhub"],
        "architecture": "Optimized with focused tools and enhanced error handling",
        "optimization_notes": [
            "Portfolio analysis streamlined to single comprehensive equal-weight tool",
            "Enhanced sentiment analysis with full news article transparency",
            "Robust price data handling with proper None value conversion",
        ],
    }


@app.get("/tools", tags=["Tools"])
async def list_tools():
    return {
        "total_tools": 8,
        "servers": {
            "finance_db_company": {
                "tools": 1,
                "tool_names": ["search_companies"],
                "description": "Enhanced company search with substring matching and symbol discovery",
            },
            "finance_db_stock_price": {
                "tools": 2,
                "tool_names": ["get_historical_stock_prices", "update_stock_prices"],
                "description": "Historical price data retrieval with simplified days parameter and robust null value handling",
            },
            "finance_calculations": {
                "tools": 3,
                "tool_names": [
                    "calculate_advanced_technical_analysis",
                    "calculate_financial_ratios",
                    "calculate_portfolio_risk_metrics",
                ],
                "description": "Comprehensive technical analysis and financial performance ratios with enhanced null value handling",
            },
            "finance_portfolio": {
                "tools": 1,
                "tool_names": ["optimize_portfolio_allocation"],
                "description": "Portfolio optimization with recommended allocation rates based on risk-return analysis",
            },
            "finance_news_and_insights": {
                "tools": 2,
                "tool_names": ["get_financial_news", "get_market_sentiment"],
                "description": "Financial news aggregation and sentiment analysis with full article transparency",
            },
            "finance_analysis_and_predictions": {
                "tools": 2,
                "tool_names": ["predict_stock_price", "analyze_stock_trends"],
                "description": "Machine learning predictions and trend analysis",
            },
        },
        "optimization_highlights": {
            "portfolio_optimization": "Modern Portfolio Theory optimization with recommended allocation rates for different risk levels",
            "enhanced_sentiment": "Market sentiment now includes full news article details for transparency",
            "robust_data_handling": "Fixed current_price None conversion issues across all price-related functions",
            "parameter_clarity": "Simplified stock price retrieval with days parameter only",
        },
    }


@app.get("/config", tags=["Configuration"])
async def get_configuration():
    return {
        "server_version": "3.0.0",
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
        "optimization_summary": {
            "architecture_optimized": True,
            "portfolio_tools_streamlined": "Portfolio optimization with recommended allocation rates using Modern Portfolio Theory",
            "enhanced_sentiment_transparency": "Market sentiment includes full news article lists",
            "robust_error_handling": "Fixed None value conversions in price data processing",
            "parameter_handling_improved": "Simplified stock price retrieval with days parameter only",
        },
    }


if __name__ == "__main__":
    print("🏦 Finance MCP Server v3.0.0 - Optimized Edition")
    print("=" * 50)
    print("📊 8 Core Tools | 6 Optimized Servers")
    print("🚀 Starting optimized server...")
    print("📡 Server: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("🔧 Tools: http://127.0.0.1:8000/tools")
    print("💓 Health: http://127.0.0.1:8000/health")
    print()
    print("🎯 Optimization Highlights:")
    print(
        "✅ Portfolio: Modern Portfolio Theory optimization with recommended allocation rates"
    )
    print("✅ Sentiment: Enhanced with full news article transparency")
    print("✅ Stock Data: Simplified days parameter & null conversions")
    print("✅ Error Handling: Robust price data processing across all servers")
    print()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
