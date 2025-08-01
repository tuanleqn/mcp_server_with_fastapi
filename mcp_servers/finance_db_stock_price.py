import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Stock Price Information (Optimized)")


@mcp.tool(description="Get the latest stock price for a symbol")
def get_latest_price(symbol: str):
    """
    Get the most recent stock price for a symbol (optimized for large datasets)

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with latest stock price information
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cursor:
                # Optimized query using the index on (symbol, date DESC)
                query = """
                SELECT price_id, symbol, date, open_price, high_price, low_price, 
                       close_price, adjusted_close, volume, dividend_amount, split_coefficient
                FROM public.stock_price
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1
                """

                cursor.execute(query, (symbol.upper(),))
                result = cursor.fetchone()

                if result:
                    return {
                        "success": True,
                        "symbol": result[1],
                        "date": result[2].isoformat()
                        if isinstance(result[2], datetime)
                        else str(result[2]),
                        "open_price": float(result[3]),
                        "high_price": float(result[4]),
                        "low_price": float(result[5]),
                        "close_price": float(result[6]),
                        "adjusted_close": float(result[7]),
                        "volume": int(result[8]),
                        "dividend_amount": float(result[9]),
                        "split_coefficient": float(result[10]),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"No price data found for symbol {symbol}",
                        "symbol": symbol.upper()
                    }

    except Error as e:
        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "symbol": symbol.upper()
        }


@mcp.tool(description="Get historical stock prices for a symbol (optimized for large datasets)")
def get_historical_prices(
    symbol: str, start_date: str = None, end_date: str = None, limit: int = 100
):
    """
    Get historical stock prices for a specific period (optimized for large datasets)

    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Maximum number of records to return (max 1000)

    Returns:
        Dictionary with success status and list of historical stock price information
    """
    try:
        # Validate limit to prevent excessive memory usage
        limit = min(limit, 1000)
        
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cursor:
                # Optimized query using indexes
                query = """
                SELECT symbol, date, open_price, high_price, low_price, 
                       close_price, adjusted_close, volume, dividend_amount, split_coefficient
                FROM public.stock_price
                WHERE symbol = %s
                """
                params = [symbol.upper()]

                if start_date:
                    query += " AND date >= %s"
                    params.append(start_date)

                if end_date:
                    query += " AND date <= %s"
                    params.append(end_date)

                query += " ORDER BY date DESC LIMIT %s"
                params.append(limit)

                cursor.execute(query, tuple(params))
                results = cursor.fetchall()

                if results:
                    historical_prices = []
                    for result in results:
                        historical_prices.append({
                            "symbol": result[0],
                            "date": result[1].isoformat()
                            if isinstance(result[1], datetime)
                            else str(result[1]),
                            "open_price": float(result[2]),
                            "high_price": float(result[3]),
                            "low_price": float(result[4]),
                            "close_price": float(result[5]),
                            "adjusted_close": float(result[6]),
                            "volume": int(result[7]),
                            "dividend_amount": float(result[8]),
                            "split_coefficient": float(result[9]),
                        })
                    
                    return {
                        "success": True,
                        "symbol": symbol.upper(),
                        "records_found": len(historical_prices),
                        "date_range": {
                            "start": start_date,
                            "end": end_date
                        },
                        "data": historical_prices
                    }
                else:
                    return {
                        "success": False,
                        "symbol": symbol.upper(),
                        "records_found": 0,
                        "error": f"No price data found for symbol {symbol} in the specified period"
                    }

    except Error as e:
        return {
            "success": False,
            "symbol": symbol.upper(),
            "error": f"Database error: {str(e)}"
        }


@mcp.tool(description="Get price statistics for a symbol over a time period")
def get_price_statistics(symbol: str, days: int = 30):
    """
    Get statistical analysis of price data for a symbol (optimized for large datasets)
    
    Args:
        symbol: Stock symbol
        days: Number of days to analyze (default 30, max 365)
        
    Returns:
        Dictionary with price statistics
    """
    try:
        days = min(days, 365)  # Limit to prevent excessive computation
        
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cursor:
                query = """
                SELECT 
                    COUNT(*) as trading_days,
                    AVG(close_price) as avg_price,
                    MIN(low_price) as min_price,
                    MAX(high_price) as max_price,
                    STDDEV(close_price) as price_volatility,
                    AVG(volume) as avg_volume,
                    MIN(date) as period_start,
                    MAX(date) as period_end
                FROM public.stock_price
                WHERE symbol = %s 
                AND date >= CURRENT_DATE - INTERVAL '%s days'
                """
                
                cursor.execute(query, (symbol.upper(), days))
                result = cursor.fetchone()
                
                if result and result[0] > 0:
                    return {
                        "success": True,
                        "symbol": symbol.upper(),
                        "analysis_period_days": days,
                        "trading_days": int(result[0]),
                        "price_statistics": {
                            "average_price": round(float(result[1]), 2),
                            "min_price": round(float(result[2]), 2),
                            "max_price": round(float(result[3]), 2),
                            "price_volatility": round(float(result[4] or 0), 4),
                            "price_range": round(float(result[3]) - float(result[2]), 2)
                        },
                        "volume_statistics": {
                            "average_volume": int(result[5] or 0)
                        },
                        "period": {
                            "start_date": str(result[6]),
                            "end_date": str(result[7])
                        }
                    }
                else:
                    return {
                        "success": False,
                        "symbol": symbol.upper(),
                        "error": f"No price data found for {symbol} in the last {days} days"
                    }
                    
    except Error as e:
        return {
            "success": False,
            "symbol": symbol.upper(),
            "error": f"Database error: {str(e)}"
        }


@mcp.tool(description="Get multiple symbols' latest prices in a single call")
def get_multiple_latest_prices(symbols: str):
    """
    Get latest prices for multiple symbols efficiently (comma-separated)
    
    Args:
        symbols: Comma-separated list of symbols (e.g., "AAPL,GOOGL,MSFT")
        
    Returns:
        Dictionary with latest prices for all requested symbols
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        if len(symbol_list) > 20:  # Limit to prevent excessive queries
            return {
                "success": False,
                "error": "Too many symbols requested. Maximum 20 symbols allowed."
            }
        
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cursor:
                # Use window function for better performance
                query = """
                WITH latest_prices AS (
                    SELECT symbol, date, close_price, volume,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
                    FROM public.stock_price
                    WHERE symbol = ANY(%s)
                )
                SELECT symbol, date, close_price, volume
                FROM latest_prices
                WHERE rn = 1
                ORDER BY symbol
                """
                
                cursor.execute(query, (symbol_list,))
                results = cursor.fetchall()
                
                prices = {}
                found_symbols = set()
                
                for result in results:
                    symbol = result[0]
                    found_symbols.add(symbol)
                    prices[symbol] = {
                        "symbol": symbol,
                        "date": str(result[1]),
                        "close_price": float(result[2]),
                        "volume": int(result[3])
                    }
                
                # Add missing symbols
                missing_symbols = set(symbol_list) - found_symbols
                for symbol in missing_symbols:
                    prices[symbol] = {
                        "symbol": symbol,
                        "error": "No data found"
                    }
                
                return {
                    "success": True,
                    "requested_symbols": len(symbol_list),
                    "found_symbols": len(found_symbols),
                    "missing_symbols": list(missing_symbols),
                    "data": prices
                }
                
    except Error as e:
        return {
            "success": False,
            "error": f"Database error: {str(e)}"
        }
