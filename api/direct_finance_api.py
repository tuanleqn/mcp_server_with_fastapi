"""
Direct Finance API for Frontend Integration
"""

from fastapi import APIRouter
from datetime import datetime

# Import modular routers
from .core_endpoints import router as core_router
from .advanced_endpoints import router as advanced_router

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(core_router, tags=["Core Market Data"])
router.include_router(advanced_router, tags=["Advanced Analytics"])

# Health check for the modular API
@router.get("/api/health/modular")
async def modular_api_health():
    """Health check for modular finance API structure"""
    return {
        "status": "healthy",
        "service": "modular_finance_api",
        "version": "4.0.0",
        "architecture": "modular",
        "modules": [
            {
                "name": "core_endpoints",
                "description": "Basic market data, stocks, charts, news",
                "endpoints": 12
            },
            {
                "name": "advanced_endpoints", 
                "description": "Portfolio, technical analysis, predictions, calculations",
                "endpoints": 25
            },
            {
                "name": "finance_models",
                "description": "Pydantic models and data structures",
                "models": 15
            }
        ],
        "data_source": "local_database_with_mcp_fallback",
        "symbol_focus": "21_reliable_symbols_across_asset_classes",
        "total_endpoints": 37,
        "features": [
            "modular_architecture",
            "focused_symbol_set",
            "crypto_etf_tracking", 
            "commodities_monitoring",
            "market_indices_coverage",
            "comprehensive_market_overview",
            "local_database_caching",
            "background_data_refresh",
            "mathematical_calculations",
            "user_management",
            "advanced_analytics",
            "portfolio_optimization",
            "risk_analysis",
            "technical_indicators",
            "sentiment_analysis",
            "predictive_modeling"
        ],
        "asset_classes": {
            "big_tech_stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX", "CRM", "ORCL"],
            "market_indices": ["SPY", "QQQ", "DIA", "VTI", "IWM"],
            "commodities": ["GLD", "SLV", "USO", "UNG"], 
            "crypto_etfs": ["BITO", "ETHE"]
        },
        "symbols_tracked": 21,
        "mcp_servers_integrated": 13,
        "maintainability": "high - separated into logical modules",
        "readability": "improved - smaller focused files",
        "last_updated": datetime.now().isoformat()
    }
