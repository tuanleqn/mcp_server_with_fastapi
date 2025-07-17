import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Stock Price Information")


@mcp.tool(description="Get the latest stock price for a symbol")
def get_latest_price(symbol: str):
    """
    Get the most recent stock price for a symbol

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with latest stock price information
    """
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()

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
                "symbol": result[1],
                "date": result[2].isoformat()
                if isinstance(result[2], datetime)
                else result[2],
                "open_price": float(result[3]),
                "high_price": float(result[4]),
                "low_price": float(result[5]),
                "close_price": float(result[6]),
                "adjusted_close": float(result[7]),
                "volume": result[8],
                "dividend_amount": float(result[9]),
                "split_coefficient": float(result[10]),
            }
        else:
            return {"error": f"No price data found for symbol {symbol}"}

    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    finally:
        if conn:
            cursor.close()
            conn.close()


@mcp.tool(description="Get historical stock prices for a symbol")
def get_historical_prices(
    symbol: str, start_date: str = None, end_date: str = None, limit: int = 100
):
    """
    Get historical stock prices for a specific period

    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Maximum number of records to return

    Returns:
        List of dictionaries with historical stock price information
    """
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()

        query = """
        SELECT price_id, symbol, date, open_price, high_price, low_price, 
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
                historical_prices.append(
                    {
                        "symbol": result[1],
                        "date": result[2].isoformat()
                        if isinstance(result[2], datetime)
                        else result[2],
                        "open_price": float(result[3]),
                        "high_price": float(result[4]),
                        "low_price": float(result[5]),
                        "close_price": float(result[6]),
                        "adjusted_close": float(result[7]),
                        "volume": result[8],
                        "dividend_amount": float(result[9]),
                        "split_coefficient": float(result[10]),
                    }
                )
            return historical_prices
        else:
            return {
                "error": f"No price data found for symbol {symbol} in the specified period"
            }

    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    finally:
        if conn:
            cursor.close()
            conn.close()


@mcp.tool(description="Get stock price for a specific date")
def get_price_by_date(symbol: str, date: str):
    """
    Get stock price for a specific date

    Args:
        symbol: Stock symbol
        date: Date (YYYY-MM-DD)

    Returns:
        Dictionary with stock price information for the specified date
    """
    try:
        conn = psycopg2.connect(DB_URI)
        cursor = conn.cursor()

        query = """
        SELECT price_id, symbol, date, open_price, high_price, low_price, 
               close_price, adjusted_close, volume, dividend_amount, split_coefficient
        FROM public.stock_price
        WHERE symbol = %s AND date = %s
        """

        cursor.execute(query, (symbol.upper(), date))
        result = cursor.fetchone()

        if result:
            return {
                "symbol": result[1],
                "date": result[2].isoformat()
                if isinstance(result[2], datetime)
                else result[2],
                "open_price": float(result[3]),
                "high_price": float(result[4]),
                "low_price": float(result[5]),
                "close_price": float(result[6]),
                "adjusted_close": float(result[7]),
                "volume": result[8],
                "dividend_amount": float(result[9]),
                "split_coefficient": float(result[10]),
            }
        else:
            return {"error": f"No price data found for symbol {symbol} on {date}"}

    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    finally:
        if conn:
            cursor.close()
            conn.close()
