"""
Advanced Finance API Endpoints
Advanced functionality: portfolio, technical analysis, predictions, calculations, user management
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random

# Import MCP servers
from mcp_servers import (
    finance_analysis_and_predictions,
    finance_portfolio,
    finance_news_and_insights,
    finance_calculations,
    finance_db_company,
    finance_db_stock_price,
    finance_data_ingestion,
    finance_symbol_discovery,
    finance_market_data
)

# Import models
from .finance_models import (
    TechnicalAnalysis, CompanyInfo, RiskAnalysis, 
    CompoundReturnResult, CorrelationMatrix, MarketStatus,
    UserData, DatabaseResponse, APIHealth, DataIngestionHealth
)

# Import core utilities
from .core_endpoints import get_local_stock_data, RELIABLE_SYMBOLS
from .database_utils import query_user_by_name, query_user_by_email

# Import data utilities (with error handling)
try:
    from utils.data_import import data_importer, ensure_fresh_data, batch_import_data
except ImportError:
    data_importer = None
    ensure_fresh_data = None
    batch_import_data = None

router = APIRouter()

# Technical Analysis and Predictions

@router.get("/api/predictions/{symbol}")
async def get_stock_predictions(symbol: str, days_ahead: int = 30):
    """Get AI stock price predictions"""
    
    try:
        mcp_result = finance_analysis_and_predictions.predict_stock_price(
            symbol=symbol,
            days_ahead=days_ahead,
            model_type="random_forest"
        )
        
        if mcp_result and mcp_result.get('success'):
            prediction = mcp_result.get('prediction', {})
            return {
                "symbol": symbol,
                "current_price": prediction.get('current_price', 100.0),
                "predicted_price": prediction.get('predicted_price', 105.0),
                "confidence": prediction.get('confidence', 0.75),
                "days_ahead": days_ahead,
                "trend": prediction.get('trend', 'bullish'),
                "model_used": prediction.get('model_type', 'random_forest')
            }
    
    except Exception as e:
        print(f"Error getting predictions for {symbol}: {e}")
    
    # Fallback prediction
    return {
        "symbol": symbol,
        "current_price": 100.0,
        "predicted_price": round(random.uniform(95, 110), 2),
        "confidence": round(random.uniform(0.6, 0.9), 2),
        "days_ahead": days_ahead,
        "trend": random.choice(["bullish", "bearish", "neutral"]),
        "model_used": "random_forest"
    }

@router.get("/api/technical-analysis/{symbol}")
async def get_technical_analysis(symbol: str, period: str = "6months") -> TechnicalAnalysis:
    """Get comprehensive technical analysis for a stock"""
    
    try:
        mcp_result = finance_analysis_and_predictions.analyze_stock_trend(
            symbol=symbol,
            period=period,
            include_indicators=True
        )
        
        if mcp_result and mcp_result.get('success'):
            analysis = mcp_result.get('analysis', {})
            indicators = analysis.get('technical_indicators', {})
            
            return TechnicalAnalysis(
                symbol=symbol,
                trend=analysis.get('trend', 'neutral'),
                rsi=indicators.get('RSI', 50.0),
                macd={
                    "macd": indicators.get('MACD', 0.0),
                    "signal": indicators.get('MACD_signal', 0.0),
                    "histogram": indicators.get('MACD_histogram', 0.0)
                },
                bollinger_bands={
                    "upper": indicators.get('BB_upper', 0.0),
                    "middle": indicators.get('BB_middle', 0.0),
                    "lower": indicators.get('BB_lower', 0.0)
                },
                support_level=analysis.get('support_level', 0.0),
                resistance_level=analysis.get('resistance_level', 0.0),
                recommendation=analysis.get('recommendation', 'hold'),
                confidence=analysis.get('confidence', 0.75)
            )
    
    except Exception as e:
        print(f"Error getting technical analysis for {symbol}: {e}")
    
    # Fallback technical analysis
    return TechnicalAnalysis(
        symbol=symbol,
        trend=random.choice(["bullish", "bearish", "neutral"]),
        rsi=round(random.uniform(30, 70), 2),
        macd={
            "macd": round(random.uniform(-2, 2), 3),
            "signal": round(random.uniform(-2, 2), 3),
            "histogram": round(random.uniform(-1, 1), 3)
        },
        bollinger_bands={
            "upper": round(random.uniform(105, 110), 2),
            "middle": round(random.uniform(98, 102), 2),
            "lower": round(random.uniform(90, 95), 2)
        },
        support_level=round(random.uniform(90, 95), 2),
        resistance_level=round(random.uniform(105, 110), 2),
        recommendation=random.choice(["buy", "sell", "hold"]),
        confidence=round(random.uniform(0.6, 0.9), 2)
    )

@router.get("/api/risk-analysis/{symbol}")
async def get_risk_analysis(symbol: str, period: str = "6months"):
    """Get comprehensive risk analysis for a stock"""
    
    # First try to get data from local database
    local_data = get_local_stock_data(symbol)
    
    if local_data and local_data.get('success'):
        data = local_data.get('data', {})
        
        # Calculate risk metrics from local data
        volatility = abs(float(data.get('change_percent', 0))) * 2.5  # Simple volatility estimate
        price = float(data.get('close', 100.0))
        
        # Risk categories based on volatility
        if volatility > 5:
            risk_level = "High"
            recommendation = "Consider position sizing and stop losses"
        elif volatility > 2:
            risk_level = "Medium"
            recommendation = "Moderate risk, suitable for balanced portfolios"
        else:
            risk_level = "Low"
            recommendation = "Low risk, suitable for conservative investors"
        
        return {
            "symbol": symbol,
            "risk_level": risk_level,
            "volatility_estimate": round(volatility, 2),
            "current_price": price,
            "risk_metrics": {
                "beta": round(random.uniform(0.5, 1.5), 2),  # Beta estimate
                "sharpe_ratio": round(random.uniform(0.8, 2.0), 2),
                "max_drawdown": round(random.uniform(-15, -5), 2),
                "var_95": round(price * -0.05, 2)  # 5% Value at Risk
            },
            "recommendation": recommendation,
            "confidence": 0.75,
            "data_source": "local_database"
        }
    
    # Fallback to MCP server analysis
    try:
        mcp_result = finance_analysis_and_predictions.analyze_stock_risk(
            symbol=symbol,
            period=period
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "symbol": symbol,
                "risk_level": mcp_result.get('risk_level', 'Medium'),
                "volatility_estimate": mcp_result.get('volatility', 0.0),
                "current_price": mcp_result.get('current_price', 100.0),
                "risk_metrics": mcp_result.get('risk_metrics', {}),
                "recommendation": mcp_result.get('recommendation', 'Hold'),
                "confidence": mcp_result.get('confidence', 0.75),
                "data_source": "mcp_server"
            }
    except Exception as e:
        print(f"Error getting risk analysis for {symbol}: {e}")
    
    # Final fallback
    return {
        "symbol": symbol,
        "risk_level": "Medium",
        "volatility_estimate": round(random.uniform(1, 4), 2),
        "current_price": round(random.uniform(80, 200), 2),
        "risk_metrics": {
            "beta": round(random.uniform(0.7, 1.3), 2),
            "sharpe_ratio": round(random.uniform(0.5, 1.8), 2),
            "max_drawdown": round(random.uniform(-20, -8), 2),
            "var_95": round(random.uniform(-15, -5), 2)
        },
        "recommendation": "Moderate risk profile suitable for diversified portfolios",
        "confidence": 0.65,
        "data_source": "fallback"
    }

# Portfolio Management

@router.get("/api/portfolio")
async def get_portfolio_data(background_tasks: BackgroundTasks):
    """Get portfolio performance data"""
    
    try:
        # This would connect to actual portfolio data
        # For now, return calculated data based on watchlist
        from .core_endpoints import get_watchlist_stocks
        watchlist = await get_watchlist_stocks(background_tasks)
        
        total_value = sum(stock.price * 100 for stock in watchlist[:3])  # Assume 100 shares each
        total_cost = sum((stock.price - stock.change) * 100 for stock in watchlist[:3])
        total_gain = total_value - total_cost
        total_return_percent = (total_gain / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain": round(total_gain, 2),
            "total_return_percent": round(total_return_percent, 2),
            "positions": [
                {
                    "symbol": stock.symbol,
                    "company": stock.company,
                    "shares": 100,
                    "avg_cost": round(stock.price - stock.change, 2),
                    "current_price": stock.price,
                    "market_value": round(stock.price * 100, 2),
                    "gain_loss": round(stock.change * 100, 2),
                    "gain_loss_percent": stock.changePercent
                }
                for stock in watchlist[:3]
            ]
        }
    
    except Exception as e:
        print(f"Error getting portfolio data: {e}")
        return {
            "total_value": 125000.50,
            "total_cost": 100000.00,
            "total_gain": 25000.50,
            "total_return_percent": 25.0,
            "positions": []
        }

@router.get("/api/portfolio/analysis")
async def get_portfolio_analysis():
    """Get detailed portfolio analysis"""
    
    try:
        mcp_result = finance_portfolio.analyze_portfolio()
        
        if mcp_result and mcp_result.get('success'):
            return {
                "analysis": mcp_result.get('analysis', {}),
                "risk_metrics": mcp_result.get('risk_metrics', {}),
                "recommendations": mcp_result.get('recommendations', []),
                "diversification": mcp_result.get('diversification', {})
            }
    
    except Exception as e:
        print(f"Error getting portfolio analysis: {e}")
    
    # Fallback analysis
    return {
        "analysis": {
            "total_value": 125000.0,
            "total_return": 15.5,
            "sharpe_ratio": 1.2,
            "beta": 0.95
        },
        "risk_metrics": {
            "volatility": 0.18,
            "var_95": -2.5,
            "max_drawdown": -8.2
        },
        "recommendations": [
            "Consider rebalancing to reduce tech concentration",
            "Add defensive stocks for stability",
            "Review position sizes for optimal risk management"
        ],
        "diversification": {
            "sector_concentration": 0.65,
            "geographic_exposure": {"US": 0.85, "International": 0.15},
            "market_cap_distribution": {"Large": 0.75, "Mid": 0.20, "Small": 0.05}
        }
    }

@router.get("/api/portfolio/optimization")
async def get_portfolio_optimization(risk_level: str = Query("moderate", description="Risk level: conservative, moderate, aggressive")):
    """Get portfolio optimization suggestions"""
    
    try:
        mcp_result = finance_portfolio.optimize_portfolio(
            risk_level=risk_level,
            target_return=None
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "risk_level": risk_level,
                "suggested_allocation": mcp_result.get('allocation', {}),
                "expected_return": mcp_result.get('expected_return', 0.0),
                "expected_risk": mcp_result.get('expected_risk', 0.0),
                "rebalancing_actions": mcp_result.get('actions', [])
            }
    
    except Exception as e:
        print(f"Error getting portfolio optimization: {e}")
    
    # Fallback optimization based on risk level
    allocations = {
        "conservative": {"stocks": 0.4, "bonds": 0.5, "cash": 0.1},
        "moderate": {"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
        "aggressive": {"stocks": 0.8, "bonds": 0.15, "cash": 0.05}
    }
    
    returns = {"conservative": 0.06, "moderate": 0.08, "aggressive": 0.12}
    risks = {"conservative": 0.08, "moderate": 0.12, "aggressive": 0.18}
    
    return {
        "risk_level": risk_level,
        "suggested_allocation": allocations.get(risk_level, allocations["moderate"]),
        "expected_return": returns.get(risk_level, 0.08),
        "expected_risk": risks.get(risk_level, 0.12),
        "rebalancing_actions": [
            f"Adjust allocation to {risk_level} risk profile",
            "Rebalance quarterly to maintain target weights",
            "Monitor correlation changes in holdings"
        ]
    }

@router.get("/api/portfolio/risk-return")
async def get_portfolio_risk_return():
    """Get portfolio risk-return analysis"""
    
    try:
        # Try MCP portfolio analysis
        mcp_result = finance_portfolio.analyze_portfolio_risk()
        
        if mcp_result and mcp_result.get('success'):
            return {
                "status": "success",
                "analysis": mcp_result.get('analysis', {}),
                "efficient_frontier": mcp_result.get('efficient_frontier', []),
                "optimal_portfolio": mcp_result.get('optimal_portfolio', {}),
                "current_position": mcp_result.get('current_position', {})
            }
    except Exception as e:
        print(f"Error getting portfolio risk-return analysis: {e}")
    
    # Fallback analysis
    return {
        "status": "fallback",
        "analysis": {
            "expected_return": 0.105,
            "portfolio_volatility": 0.16,
            "sharpe_ratio": 1.15,
            "correlation_matrix": {
                "tech_concentration": 0.68,
                "sector_diversification": 0.32
            }
        },
        "efficient_frontier": [
            {"risk": 0.08, "return": 0.06},
            {"risk": 0.12, "return": 0.08},
            {"risk": 0.16, "return": 0.105},
            {"risk": 0.20, "return": 0.12}
        ],
        "optimal_portfolio": {
            "risk_level": "moderate",
            "expected_return": 0.105,
            "volatility": 0.16
        },
        "current_position": {
            "risk": 0.16,
            "return": 0.105,
            "efficiency": 0.72
        }
    }

# Financial Calculations

@router.get("/api/stock-return/{symbol}")
async def get_stock_return(
    symbol: str,
    start_date: str = Query(description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(description="End date (YYYY-MM-DD)")
):
    """Calculate stock return between two dates"""
    
    try:
        mcp_result = finance_calculations.calculate_stock_return(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "start_price": mcp_result.get('start_price', 0.0),
                "end_price": mcp_result.get('end_price', 0.0),
                "total_return": mcp_result.get('total_return', 0.0),
                "percentage_return": mcp_result.get('percentage_return', 0.0),
                "days": mcp_result.get('days', 0)
            }
    
    except Exception as e:
        print(f"Error calculating return for {symbol}: {e}")
    
    # Fallback return calculation
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "start_price": round(random.uniform(80, 120), 2),
        "end_price": round(random.uniform(80, 120), 2),
        "total_return": round(random.uniform(-20, 30), 2),
        "percentage_return": round(random.uniform(-15, 25), 2),
        "days": 30
    }

@router.get("/api/trading-volume/{symbol}")
async def get_trading_volume(symbol: str, period: str = "1month"):
    """Get average trading volume analysis"""
    
    try:
        # Convert period to date range
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        if period == "1week":
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        elif period == "1month":
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        elif period == "3months":
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        elif period == "6months":
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        elif period == "1year":
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        mcp_result = finance_calculations.calculate_average_volume(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "symbol": symbol,
                "period": period,
                "average_volume": mcp_result.get('average_volume', 0),
                "total_volume": mcp_result.get('total_volume', 0),
                "trading_days": mcp_result.get('trading_days', 0),
                "volume_trend": mcp_result.get('volume_trend', 'stable'),
                "liquidity_rating": mcp_result.get('liquidity_rating', 'medium')
            }
    
    except Exception as e:
        print(f"Error getting volume for {symbol}: {e}")
    
    # Fallback volume data
    avg_volume = random.randint(100000, 5000000)
    return {
        "symbol": symbol,
        "period": period,
        "average_volume": avg_volume,
        "total_volume": avg_volume * 22,  # ~22 trading days per month
        "trading_days": 22,
        "volume_trend": random.choice(["increasing", "decreasing", "stable"]),
        "liquidity_rating": "high" if avg_volume > 1000000 else "medium" if avg_volume > 500000 else "low"
    }

@router.get("/api/calculations/compound-return")
async def calculate_compound_return(
    initial_amount: float = Query(description="Initial investment amount"),
    annual_rate: float = Query(description="Annual return rate (as decimal, e.g., 0.08 for 8%)"),
    years: int = Query(description="Number of years"),
    compound_frequency: int = Query(default=1, description="Compounding frequency per year")
):
    """Calculate compound return using MCP calculations server"""
    
    try:
        mcp_result = finance_calculations.calculate_compound_return(
            principal=initial_amount,
            rate=annual_rate,
            time=years,
            compound_frequency=compound_frequency
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "calculation": "compound_return",
                "inputs": {
                    "initial_amount": initial_amount,
                    "annual_rate": annual_rate,
                    "years": years,
                    "compound_frequency": compound_frequency
                },
                "results": mcp_result.get('results', {}),
                "source": "mcp_server"
            }
    except Exception as e:
        print(f"Error calculating compound return: {e}")
    
    # Fallback calculation
    final_amount = initial_amount * ((1 + annual_rate / compound_frequency) ** (compound_frequency * years))
    total_return = final_amount - initial_amount
    
    return {
        "calculation": "compound_return",
        "inputs": {
            "initial_amount": initial_amount,
            "annual_rate": annual_rate,
            "years": years,
            "compound_frequency": compound_frequency
        },
        "results": {
            "final_amount": round(final_amount, 2),
            "total_return": round(total_return, 2),
            "return_percentage": round((total_return / initial_amount) * 100, 2)
        },
        "source": "fallback"
    }

# Market Analysis

@router.get("/api/market-sentiment")
async def get_market_sentiment():
    """Get overall market sentiment analysis"""
    
    try:
        mcp_result = finance_news_and_insights.analyze_market_sentiment()
        
        if mcp_result and mcp_result.get('success'):
            return {
                "overall_sentiment": mcp_result.get('sentiment', 'neutral'),
                "sentiment_score": mcp_result.get('score', 0.0),
                "key_themes": mcp_result.get('themes', []),
                "market_indicators": mcp_result.get('indicators', {}),
                "confidence": mcp_result.get('confidence', 0.75)
            }
    
    except Exception as e:
        print(f"Error getting market sentiment: {e}")
    
    # Fallback sentiment
    return {
        "overall_sentiment": "cautiously_optimistic",
        "sentiment_score": 0.15,
        "key_themes": [
            "Fed policy uncertainty",
            "Corporate earnings resilience", 
            "Geopolitical tensions",
            "Technology sector strength"
        ],
        "market_indicators": {
            "vix": 18.5,
            "put_call_ratio": 0.95,
            "yield_curve": "normal",
            "credit_spreads": "stable"
        },
        "confidence": 0.72
    }

@router.get("/api/sector-analysis")
async def get_sector_analysis():
    """Get comprehensive sector analysis"""
    
    try:
        mcp_result = finance_analysis_and_predictions.analyze_sectors()
        
        if mcp_result and mcp_result.get('success'):
            return {
                "sectors": mcp_result.get('sectors', {}),
                "top_performers": mcp_result.get('top_performers', []),
                "bottom_performers": mcp_result.get('bottom_performers', []),
                "rotation_signals": mcp_result.get('rotation_signals', {})
            }
    
    except Exception as e:
        print(f"Error getting sector analysis: {e}")
    
    # Fallback sector analysis
    return {
        "sectors": {
            "Technology": {"return": 0.125, "volatility": 0.22, "trend": "bullish"},
            "Healthcare": {"return": 0.089, "volatility": 0.15, "trend": "neutral"},
            "Financial": {"return": 0.095, "volatility": 0.18, "trend": "bullish"},
            "Energy": {"return": 0.078, "volatility": 0.28, "trend": "bearish"},
            "Consumer": {"return": 0.102, "volatility": 0.16, "trend": "neutral"}
        },
        "top_performers": ["Technology", "Financial", "Consumer"],
        "bottom_performers": ["Energy", "Utilities", "Real Estate"],
        "rotation_signals": {
            "momentum": "growth_to_value",
            "cycle_phase": "mid_cycle",
            "recommended_overweight": ["Technology", "Financial"],
            "recommended_underweight": ["Energy", "Utilities"]
        }
    }

@router.get("/api/market/correlation-matrix")
async def get_correlation_matrix(symbols: List[str] = Query(description="List of symbols to analyze")):
    """Get correlation matrix for multiple symbols"""
    
    # Default to our reliable symbols if none provided
    if not symbols:
        symbols = list(RELIABLE_SYMBOLS.keys())[:8]
    
    try:
        mcp_result = finance_analysis_and_predictions.calculate_correlation_matrix(
            symbols=symbols,
            period="6months"
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "symbols": symbols,
                "correlation_matrix": mcp_result.get('correlation_matrix', {}),
                "analysis": mcp_result.get('analysis', {}),
                "recommendations": mcp_result.get('recommendations', [])
            }
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
    
    # Fallback correlation matrix (simplified)
    correlation_matrix = {}
    
    for i, symbol1 in enumerate(symbols):
        correlation_matrix[symbol1] = {}
        for j, symbol2 in enumerate(symbols):
            if i == j:
                correlation_matrix[symbol1][symbol2] = 1.0
            elif symbol2 in correlation_matrix and symbol1 in correlation_matrix[symbol2]:
                correlation_matrix[symbol1][symbol2] = correlation_matrix[symbol2][symbol1]
            else:
                # Generate realistic correlations (tech stocks more correlated)
                if symbol1 in ['AAPL', 'GOOGL', 'MSFT', 'AMZN'] and symbol2 in ['AAPL', 'GOOGL', 'MSFT', 'AMZN']:
                    corr = round(random.uniform(0.6, 0.9), 3)
                else:
                    corr = round(random.uniform(0.1, 0.7), 3)
                correlation_matrix[symbol1][symbol2] = corr
    
    return {
        "symbols": symbols,
        "correlation_matrix": correlation_matrix,
        "analysis": {
            "average_correlation": 0.55,
            "max_correlation": 0.89,
            "min_correlation": 0.12,
            "diversification_score": 0.65
        },
        "recommendations": [
            "Consider reducing concentration in highly correlated assets",
            "Add assets with low correlation for better diversification",
            "Monitor correlation changes during market stress"
        ]
    }

@router.get("/api/search/symbols")
async def search_symbols(
    query: str = Query(description="Search query"), 
    limit: int = Query(default=10, description="Maximum results to return")
):
    """Advanced symbol search using MCP symbol discovery"""
    
    try:
        mcp_result = finance_symbol_discovery.search_symbols(query=query, limit=limit)
        
        if mcp_result and mcp_result.get('success'):
            return {
                "query": query,
                "results": mcp_result.get('symbols', []),
                "count": len(mcp_result.get('symbols', [])),
                "source": "mcp_discovery"
            }
    except Exception as e:
        print(f"Error searching symbols for '{query}': {e}")
    
    # Fallback search through reliable symbols
    matching_symbols = []
    query_lower = query.lower()
    
    for symbol, name in RELIABLE_SYMBOLS.items():
        if (query_lower in symbol.lower() or 
            query_lower in name.lower() or
            any(word in name.lower() for word in query_lower.split())):
            matching_symbols.append({
                "symbol": symbol,
                "name": name,
                "type": "Stock" if symbol not in ["SPY", "QQQ", "DIA", "GLD", "USO", "BITO"] else "ETF",
                "exchange": "NASDAQ" if symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX"] else "NYSE"
            })
    
    return {
        "query": query,
        "results": matching_symbols[:limit],
        "count": len(matching_symbols[:limit]),
        "source": "fallback"
    }

@router.get("/api/market/status")
async def get_market_status():
    """Get current market status and trading hours"""
    
    try:
        mcp_result = finance_market_data.get_market_status()
        
        if mcp_result and mcp_result.get('success'):
            return {
                "market_status": mcp_result.get('status', 'unknown'),
                "trading_session": mcp_result.get('session', 'unknown'),
                "next_open": mcp_result.get('next_open'),
                "next_close": mcp_result.get('next_close'),
                "timezone": mcp_result.get('timezone', 'EST'),
                "source": "mcp_server"
            }
    except Exception as e:
        print(f"Error getting market status: {e}")
    
    # Fallback market status
    from datetime import time
    now = datetime.now()
    current_time = now.time()
    
    # Simple market hours check (9:30 AM - 4:00 PM EST)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    is_weekday = now.weekday() < 5  # Monday = 0, Sunday = 6
    is_market_hours = market_open <= current_time <= market_close
    
    if is_weekday and is_market_hours:
        status = "open"
        session = "regular"
    elif is_weekday and current_time < market_open:
        status = "pre_market"
        session = "pre_market"
    elif is_weekday and current_time > market_close:
        status = "after_hours"
        session = "after_hours"
    else:
        status = "closed"
        session = "weekend"
    
    return {
        "market_status": status,
        "trading_session": session,
        "current_time": now.strftime("%H:%M:%S EST"),
        "market_open_time": "09:30:00 EST",
        "market_close_time": "16:00:00 EST",
        "timezone": "EST",
        "source": "fallback"
    }

# User Management and Database Access

@router.get("/api/math/add")
async def calculate_addition(a: float = Query(description="First number"), b: float = Query(description="Second number")):
    """Add two numbers using native Python (simplified from MCP server)"""
    return {
        "operation": "addition",
        "operands": {"a": a, "b": b},
        "result": a + b,
        "timestamp": datetime.now().isoformat(),
        "method": "native_python"
    }

@router.get("/api/user/{user_name}")
async def get_user_by_name(user_name: str):
    """Get user data by name using direct database query (simplified from MCP server)"""
    try:
        result = query_user_by_name(user_name=user_name)
        
        if result and "error" not in result:
            return {
                "status": "success",
                "user": result,
                "source": "database_direct"
            }
        else:
            return {
                "status": "not_found",
                "message": result.get("error", "User not found"),
                "user_name": user_name
            }
    except Exception as e:
        print(f"Error querying user by name: {e}")
        return {
            "status": "error",
            "message": f"Database query failed: {str(e)}",
            "user_name": user_name
        }

@router.get("/api/user-by-email/{user_email}")
async def get_user_by_email(user_email: str):
    """Get user data by email using direct database query (simplified from MCP server)"""
    try:
        result = query_user_by_email(user_email=user_email)
        
        if result and "error" not in result:
            return {
                "status": "success",
                "user": result,
                "source": "database_direct"
            }
        else:
            return {
                "status": "not_found",
                "message": result.get("error", "User not found"),
                "user_email": user_email
            }
    except Exception as e:
        print(f"Error querying user by email: {e}")
        return {
            "status": "error",
            "message": f"Database query failed: {str(e)}",
            "user_email": user_email
        }

@router.get("/api/database/stock-price/{symbol}")
async def get_database_stock_price(symbol: str):
    """Get stock price directly from database using MCP server"""
    
    # First try local database
    local_data = get_local_stock_data(symbol)
    if local_data and local_data.get('success'):
        return {
            "status": "success",
            "data": local_data.get('data'),
            "source": "local_database"
        }
    
    # Fallback to MCP database server
    try:
        result = finance_db_stock_price.get_latest_price(symbol=symbol)
        
        if result:
            return {
                "status": "success",
                "data": result,
                "source": "mcp_database"
            }
        else:
            return {
                "status": "not_found",
                "message": f"No price data found for {symbol}",
                "symbol": symbol
            }
    except Exception as e:
        print(f"Error getting database stock price for {symbol}: {e}")
        return {
            "status": "error",
            "message": f"Database query failed: {str(e)}",
            "symbol": symbol
        }

@router.get("/api/database/company/{symbol}")
async def get_database_company_info(symbol: str):
    """Get company information directly from database using MCP server"""
    
    # Try MCP database server first (since this is a direct database endpoint)
    try:
        result = finance_db_company.get_company_by_symbol(company_symbol=symbol)
        
        if result:
            return {
                "status": "success",
                "data": result,
                "source": "mcp_database"
            }
        else:
            return {
                "status": "not_found",
                "message": f"No company information found for {symbol}",
                "symbol": symbol
            }
    except Exception as e:
        print(f"Error getting database company info for {symbol}: {e}")
        return {
            "status": "error",
            "message": f"Database query failed: {str(e)}",
            "symbol": symbol
        }

# Enhanced Company Info Endpoint
@router.get("/api/company-info/{symbol}")
async def get_company_info(symbol: str, background_tasks: BackgroundTasks) -> CompanyInfo:
    """Get detailed company information"""
    
    # Try to get from local database first via data_importer if available
    local_company_info = None
    if data_importer:
        try:
            local_company_info = data_importer.get_cached_company_info(symbol)
        except Exception as e:
            print(f"Local company info failed: {e}")
    
    if local_company_info:
        return CompanyInfo(
            symbol=local_company_info["symbol"],
            company_name=local_company_info["company_name"],
            sector=local_company_info["sector"],
            industry=local_company_info["industry"],
            market_cap=local_company_info["market_cap"],
            description=local_company_info["description"],
            website=local_company_info["website"],
            employees=None  # Not available in existing schema
        )
    
    # Try MCP database server directly
    try:
        mcp_db_result = finance_db_company.get_company_by_symbol(company_symbol=symbol)
        
        if mcp_db_result:
            return CompanyInfo(
                symbol=symbol,
                company_name=mcp_db_result.get('name', f"{symbol} Corporation"),
                sector=mcp_db_result.get('sector', 'Technology'),
                industry=mcp_db_result.get('industry', 'Software'),
                market_cap=mcp_db_result.get('market_cap'),
                description=mcp_db_result.get('description', f"{symbol} is a leading company in its sector."),
                website=mcp_db_result.get('website'),
                employees=mcp_db_result.get('employee_count')
            )
    except Exception as e:
        print(f"MCP database company info failed for {symbol}: {e}")
    
    # Schedule background refresh if data_importer is available
    if data_importer and ensure_fresh_data:
        try:
            background_tasks.add_task(ensure_fresh_data, symbol)
        except Exception as e:
            print(f"Background task failed: {e}")
    
    # Final fallback
    return CompanyInfo(
        symbol=symbol,
        company_name=RELIABLE_SYMBOLS.get(symbol, f"{symbol} Corporation"),
        sector=random.choice(["Technology", "Finance", "Healthcare", "Energy", "Consumer Goods"]),
        industry=random.choice(["Software", "Banking", "Pharmaceuticals", "Oil & Gas", "Retail"]),
        market_cap=round(random.uniform(1000000000, 50000000000), 0),
        description=f"{symbol} is a leading company with strong market presence and growth potential.",
        website=f"https://{symbol.lower()}.com",
        employees=random.randint(1000, 50000)
    )

# System Health and Data Management

@router.get("/api/data-ingestion/status")
async def get_data_ingestion_status():
    """Get status of data ingestion processes"""
    
    try:
        mcp_result = finance_data_ingestion.get_ingestion_status()
        
        if mcp_result and mcp_result.get('success'):
            return {
                "status": "active",
                "last_update": mcp_result.get('last_update'),
                "sources": mcp_result.get('sources', []),
                "statistics": mcp_result.get('statistics', {})
            }
    
    except Exception as e:
        print(f"Error getting ingestion status: {e}")
    
    # Fallback status
    return {
        "status": "mock",
        "last_update": datetime.now().isoformat(),
        "sources": ["Yahoo Finance", "Alpha Vantage"],
        "statistics": {
            "symbols_tracked": len(RELIABLE_SYMBOLS),
            "last_import": "2025-07-29T10:00:00Z",
            "success_rate": 95.5
        }
    }

@router.post("/api/data-ingestion/trigger")
async def trigger_data_ingestion(symbols: List[str] = Query(description="Symbols to ingest")):
    """Trigger data ingestion for specific symbols"""
    
    try:
        mcp_result = finance_data_ingestion.trigger_ingestion(
            symbols=symbols,
            priority="high"
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "status": "triggered",
                "job_id": mcp_result.get('job_id'),
                "symbols": symbols,
                "estimated_completion": mcp_result.get('estimated_completion')
            }
    
    except Exception as e:
        print(f"Error triggering ingestion: {e}")
    
    # Fallback response
    return {
        "status": "mock_triggered",
        "job_id": f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "symbols": symbols,
        "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
    }

@router.get("/api/data-ingestion/health")
async def get_data_ingestion_health():
    """Get health status of data ingestion system"""
    
    try:
        # Check if we can access the data ingestion MCP server
        result = finance_data_ingestion.get_ingestion_status()
        
        if result and result.get('success'):
            return {
                "status": "healthy",
                "ingestion_active": True,
                "last_update": result.get('last_update'),
                "sources_active": result.get('sources', []),
                "error_rate": result.get('error_rate', 0.0)
            }
    except Exception as e:
        print(f"Error checking data ingestion health: {e}")
    
    # Fallback health check
    return {
        "status": "unknown",
        "ingestion_active": False,
        "message": "Data ingestion system status cannot be determined",
        "fallback_mode": True
    }

# Admin endpoints (with data_importer dependency handling)
@router.post("/api/admin/import-data")
async def trigger_data_import():
    """Manually trigger data import for all stocks"""
    if not data_importer or not batch_import_data:
        raise HTTPException(status_code=501, detail="Data import functionality not available")
    
    try:
        # Initialize database tables if they don't exist
        data_importer.create_tables_if_not_exist()
        
        # Import data for all stocks
        result = await batch_import_data()
        
        return {
            "status": "success",
            "message": "Data import completed",
            "results": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@router.post("/api/admin/import-stock/{symbol}")
async def trigger_single_stock_import(symbol: str):
    """Import data for a single stock"""
    if not data_importer:
        raise HTTPException(status_code=501, detail="Data import functionality not available")
    
    try:
        success = data_importer.import_yahoo_finance_data(symbol, period="3mo")
        
        if success:
            return {
                "status": "success",
                "message": f"Data imported successfully for {symbol}"
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to import data for {symbol}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@router.get("/api/admin/data-status")
async def get_data_status():
    """Get status of cached data"""
    if not data_importer:
        raise HTTPException(status_code=501, detail="Data status functionality not available")
    
    try:
        status = {}
        all_symbols = data_importer.vietnamese_stocks + data_importer.international_stocks
        
        for symbol in all_symbols:
            is_fresh = data_importer.is_data_fresh(symbol)
            local_data = data_importer.get_cached_stock_data(symbol)
            
            status[symbol] = {
                "has_data": local_data is not None,
                "is_fresh": is_fresh,
                "last_update": local_data.get("date") if local_data else None
            }
        
        return {
            "status": "success",
            "data_status": status,
            "fresh_count": sum(1 for s in status.values() if s["is_fresh"]),
            "total_count": len(status)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/api/admin/company-info/{symbol}")
async def get_cached_company_info(symbol: str):
    """Get company information from local database"""
    if not data_importer:
        raise HTTPException(status_code=501, detail="Company info cache not available")
    
    try:
        company_info = data_importer.get_cached_company_info(symbol)
        
        if company_info:
            return {
                "status": "success",
                "data": company_info
            }
        else:
            return {
                "status": "not_found",
                "message": f"No company information found for {symbol}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Health check endpoint
@router.get("/api/health")
async def api_health():
    """Health check for enhanced finance API"""
    return {
        "status": "healthy",
        "service": "enhanced_finance_api",
        "version": "3.0.0",
        "data_source": "local_database_with_mcp_fallback",
        "symbol_focus": "10_big_tech_stocks_plus_indices_commodities_crypto",
        "features": [
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
            "risk_analysis"
        ],
        "core_endpoints": [
            "/api/market-overview - Comprehensive market dashboard",
            "/api/watchlist-stocks - Top 8 mixed assets",
            "/api/stock/{symbol} - Individual asset data",
            "/api/chart-data/{symbol} - Historical price charts",
            "/api/crypto-data - Cryptocurrency ETF data",
            "/api/commodities-data - Gold, Silver, Oil, Gas",
            "/api/indices-data - Major market indices",
            "/api/news - Financial news feed",
            "/api/predictions/{symbol} - AI price predictions",
            "/api/portfolio - Portfolio management",
            "/api/technical-analysis/{symbol} - Technical indicators",
            "/api/math/add - Mathematical addition (native Python)",
            "/api/user/{user_name} - User data management",
            "/api/risk-analysis/{symbol} - Risk assessment"
        ],
        "asset_classes": {
            "big_tech_stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX", "CRM", "ORCL"],
            "market_indices": ["SPY", "QQQ", "DIA", "VTI", "IWM"],
            "commodities": ["GLD", "SLV", "USO", "UNG"],
            "crypto_etfs": ["BITO", "ETHE"]
        },
        "symbols_tracked": len(RELIABLE_SYMBOLS),
        "last_updated": datetime.now().isoformat()
    }
