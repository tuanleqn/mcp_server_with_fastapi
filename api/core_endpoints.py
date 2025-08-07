from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.finance_helpers import get_historical_prices_helper

router = APIRouter()

@router.get("/core/ping", tags=["Core Market Data"])
async def core_ping():
    """Basic core endpoint for health check."""
    return {"status": "core endpoints operational"}

@router.get("/api/chart-data/{symbol}", tags=["Core Market Data"])
async def get_chart_data(
    symbol: str,
    period: str = Query("1day", description="Time period: 1day, 1week, 1month, 3month, 1year"),
    interval: str = Query("30m", description="Data interval: 5m, 15m, 30m, 1h, 1day")
):
    """Get stock price chart data for frontend visualization"""
    try:
        # Generate reliable mock data based on symbol
        symbol_prices = {
            "AAPL": 175.0, "MSFT": 420.0, "GOOGL": 140.0, "TSLA": 250.0,
            "AMZN": 180.0, "NVDA": 800.0, "META": 520.0
        }
        
        base_price = symbol_prices.get(symbol.upper(), 150.0)
        open_price = base_price * 0.995
        high_price = base_price * 1.025
        low_price = base_price * 0.975
        
        # Generate intraday points
        intraday_times = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
        
        chart_data = []
        price_range = high_price - low_price
        
        for i, time_point in enumerate(intraday_times):
            progress = i / (len(intraday_times) - 1)
            base_prog_price = open_price + (base_price - open_price) * progress
            variation_factor = (i % 3 - 1) * 0.3
            varied_price = base_prog_price + (price_range * 0.1 * variation_factor)
            final_price = max(low_price, min(high_price, varied_price))
            
            chart_data.append({
                "time": time_point,
                "price": round(final_price, 2)
            })
        
        # Calculate metadata
        prices = [point["price"] for point in chart_data]
        min_price = min(prices)
        max_price = max(prices)
        price_change = prices[-1] - prices[0]
        price_change_percent = (price_change / prices[0] * 100) if prices[0] != 0 else 0
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": chart_data,
            "metadata": {
                "minPrice": min_price,
                "maxPrice": max_price,
                "priceChange": round(price_change, 2),
                "priceChangePercent": round(price_change_percent, 2)
            },
            "timestamp": datetime.now().isoformat(),
            "dataPoints": len(chart_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart data: {str(e)}")

@router.get("/api/chart-data-simple/{symbol}", tags=["Core Market Data"])
async def get_simple_chart_data(
    symbol: str,
    points: int = Query(8, description="Number of data points to return (max 20)")
):
    """
    Get simplified mock chart data for frontend testing
    Returns realistic intraday price movements
    """
    try:
        # Limit points to reasonable range
        points = min(max(points, 4), 20)
        
        # Start with reliable mock data
        base_price = 150.0
        open_price = base_price * 0.995
        high_price = base_price * 1.02
        low_price = base_price * 0.98
        
        # Try to enhance with real data if available
        try:
            from mcp_servers.finance_helpers import get_latest_stock_price_helper
            price_data = get_latest_stock_price_helper(symbol)
            
            if (price_data.get("success") and 
                price_data.get("close_price") is not None and 
                str(price_data.get("close_price")).replace('.', '').replace('-', '').isdigit()):
                
                real_base = float(str(price_data["close_price"]))
                if real_base > 0:  # Only use if it's a valid positive number
                    base_price = real_base
                    
                    # Try to get other prices safely
                    if (price_data.get("open_price") is not None and 
                        str(price_data.get("open_price")).replace('.', '').replace('-', '').isdigit()):
                        real_open = float(str(price_data["open_price"]))
                        if real_open > 0:
                            open_price = real_open
                    else:
                        open_price = base_price * 0.995
                    
                    if (price_data.get("high_price") is not None and 
                        str(price_data.get("high_price")).replace('.', '').replace('-', '').isdigit()):
                        real_high = float(str(price_data["high_price"]))
                        if real_high > 0:
                            high_price = real_high
                    else:
                        high_price = base_price * 1.02
                    
                    if (price_data.get("low_price") is not None and 
                        str(price_data.get("low_price")).replace('.', '').replace('-', '').isdigit()):
                        real_low = float(str(price_data["low_price"]))
                        if real_low > 0:
                            low_price = real_low
                    else:
                        low_price = base_price * 0.98
        except:
            # If anything goes wrong, stick with mock data
            pass
        
        # Generate time points
        start_hour = 9
        start_minute = 30
        interval_minutes = max(1, (7 * 60) // points)  # Ensure at least 1 minute intervals
        
        chart_data = []
        price_range = high_price - low_price
        
        for i in range(points):
            # Calculate time
            total_minutes = start_minute + (i * interval_minutes)
            hour = start_hour + total_minutes // 60
            minute = total_minutes % 60
            time_str = f"{hour:02d}:{minute:02d}"
            
            # Calculate realistic price progression
            progress = i / (points - 1) if points > 1 else 0
            
            # Start near open, end near close, vary within high/low
            base_progression = open_price + (base_price - open_price) * progress
            
            # Add realistic variation
            wave_factor = 0.3 * price_range * (0.5 - abs(progress - 0.5))  # Peak in middle
            noise = (i % 3 - 1) * 0.1 * price_range  # Small noise
            
            final_price = base_progression + wave_factor + noise
            final_price = max(low_price, min(high_price, final_price))
            
            chart_data.append({
                "time": time_str,
                "price": round(final_price, 2)
            })
        
        # Calculate metadata
        prices = [point["price"] for point in chart_data]
        min_price = min(prices)
        max_price = max(prices)
        price_change = prices[-1] - prices[0]
        price_change_percent = (price_change / prices[0] * 100) if prices[0] != 0 else 0
        
        return {
            "symbol": symbol.upper(),
            "period": "1day",
            "interval": f"{interval_minutes}m",
            "data": chart_data,
            "metadata": {
                "minPrice": min_price,
                "maxPrice": max_price,
                "priceChange": round(price_change, 2),
                "priceChangePercent": round(price_change_percent, 2)
            },
            "timestamp": datetime.now().isoformat(),
            "dataPoints": len(chart_data),
            "dataSource": "mock_with_real_enhancement" if base_price != 150.0 else "mock"
        }
        
    except Exception as e:
        # Ultimate fallback - return guaranteed working mock data
        return {
            "symbol": symbol.upper(),
            "period": "1day", 
            "interval": "60m",
            "data": [
                {"time": "09:30", "price": 150.0},
                {"time": "10:30", "price": 152.5},
                {"time": "11:30", "price": 148.0},
                {"time": "12:30", "price": 154.0},
                {"time": "13:30", "price": 151.0},
                {"time": "14:30", "price": 153.5},
                {"time": "15:30", "price": 149.5},
                {"time": "16:00", "price": 151.8}
            ],
            "metadata": {
                "minPrice": 148.0,
                "maxPrice": 154.0,
                "priceChange": 1.8,
                "priceChangePercent": 1.2
            },
            "timestamp": datetime.now().isoformat(),
            "dataPoints": 8,
            "dataSource": "fallback_mock",
            "error_handled": str(e)
        }

@router.get("/api/chart-data-mock/{symbol}", tags=["Core Market Data"])
async def get_mock_chart_data(
    symbol: str,
    points: int = Query(8, description="Number of data points to return (max 20)")
):
    """
    Get reliable mock chart data for frontend testing
    Always returns working data without external dependencies
    """
    # Limit points to reasonable range
    points = min(max(points, 4), 20)
    
    # Generate reliable mock data based on symbol
    symbol_prices = {
        "AAPL": 175.0,
        "MSFT": 420.0, 
        "GOOGL": 140.0,
        "TSLA": 250.0,
        "AMZN": 180.0,
        "NVDA": 800.0,
        "META": 520.0
    }
    
    base_price = symbol_prices.get(symbol.upper(), 150.0)
    open_price = base_price * 0.995
    high_price = base_price * 1.025
    low_price = base_price * 0.975
    
    # Generate time points
    start_hour = 9
    start_minute = 30
    interval_minutes = max(1, (7 * 60) // points)
    
    chart_data = []
    price_range = high_price - low_price
    
    for i in range(points):
        # Calculate time
        total_minutes = start_minute + (i * interval_minutes)
        hour = start_hour + total_minutes // 60
        minute = total_minutes % 60
        time_str = f"{hour:02d}:{minute:02d}"
        
        # Calculate realistic price progression
        progress = i / (points - 1) if points > 1 else 0
        
        # Start near open, end near close, vary within high/low
        base_progression = open_price + (base_price - open_price) * progress
        
        # Add realistic variation
        wave_factor = 0.3 * price_range * (0.5 - abs(progress - 0.5))
        noise = (i % 3 - 1) * 0.1 * price_range
        
        final_price = base_progression + wave_factor + noise
        final_price = max(low_price, min(high_price, final_price))
        
        chart_data.append({
            "time": time_str,
            "price": round(final_price, 2)
        })
    
    # Calculate metadata
    prices = [point["price"] for point in chart_data]
    min_price = min(prices)
    max_price = max(prices)
    price_change = prices[-1] - prices[0]
    price_change_percent = (price_change / prices[0] * 100) if prices[0] != 0 else 0
    
    return {
        "symbol": symbol.upper(),
        "period": "1day",
        "interval": f"{interval_minutes}m",
        "data": chart_data,
        "metadata": {
            "minPrice": min_price,
            "maxPrice": max_price,
            "priceChange": round(price_change, 2),
            "priceChangePercent": round(price_change_percent, 2)
        },
        "timestamp": datetime.now().isoformat(),
        "dataPoints": len(chart_data),
        "dataSource": "reliable_mock"
    }

@router.get("/api/symbols", tags=["Core Market Data"])
async def get_available_symbols():
    """
    Get list of available symbols for frontend dropdown
    Returns commonly traded stocks and crypto symbols
    """
    try:
        from mcp_servers.finance_helpers import search_companies_helper
        
        # Get some popular symbols from database
        popular_queries = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"]
        available_symbols = []
        
        for query in popular_queries:
            result = search_companies_helper(query, 1)
            if result.get("success") and result.get("companies"):
                company = result["companies"][0]
                available_symbols.append({
                    "symbol": company["symbol"],
                    "name": company["name"],
                    "type": company.get("asset_type", "stock")
                })
        
        # Add some fallback symbols if database is empty
        if len(available_symbols) == 0:
            available_symbols = [
                {"symbol": "AAPL", "name": "Apple Inc.", "type": "stock"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "stock"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "type": "stock"},
                {"symbol": "TSLA", "name": "Tesla Inc.", "type": "stock"},
                {"symbol": "AMZN", "name": "Amazon.com Inc.", "type": "stock"},
                {"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "stock"},
                {"symbol": "META", "name": "Meta Platforms Inc.", "type": "stock"}
            ]
        
        return {
            "success": True,
            "symbols": available_symbols,
            "count": len(available_symbols),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve symbols: {str(e)}"
        )
