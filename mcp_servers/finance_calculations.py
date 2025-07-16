import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal

# Assuming finance_db_stock_price.py is available for direct calls or through MCP
# For simplicity, we'll re-implement the DB connection here or assume it's imported.
# In a real MCP setup, you might call the other MCP tools directly.

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Financial Calculations")

def _get_price_for_calculation(symbol: str, date: str = None) -> Decimal:
    """
    Helper function to get a specific price (close_price) for calculation.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                if date:
                    query = "SELECT close_price FROM public.\"STOCK_PRICES\" WHERE symbol = %s AND date = %s"
                    cur.execute(query, (symbol.upper(), date))
                else: # Get latest price if no date is specified
                    query = "SELECT close_price FROM public.\"STOCK_PRICES\" WHERE symbol = %s ORDER BY date DESC LIMIT 1"
                    cur.execute(query, (symbol.upper(),))
                
                result = cur.fetchone()
                if result:
                    return Decimal(result[0])
                else:
                    return Decimal('0.0') # Or raise an error if preferred
    except Error as e:
        print(f"Database error in _get_price_for_calculation: {e}")
        return Decimal('0.0')


@mcp.tool(description="Calculate the percentage return of a stock between two dates.")
def calculate_percentage_return(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Calculates the percentage return of a stock between a start and an end date.

    Args:
        symbol (str): The stock symbol (e.g., 'AAPL').
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing the symbol, start/end dates, and the calculated
              percentage return. Returns an error if data is not found or calculation fails.
    """
    start_price = _get_price_for_calculation(symbol, start_date)
    end_price = _get_price_for_calculation(symbol, end_date)

    if start_price == Decimal('0.0') or end_price == Decimal('0.0'):
        return {"error": f"Could not retrieve stock prices for {symbol} on the specified dates."}

    if start_price == Decimal('0.0'): # Avoid division by zero
        return {"error": "Start price is zero, cannot calculate percentage return."}

    percentage_return = ((end_price - start_price) / start_price) * Decimal('100.0')
    
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "percentage_return": float(percentage_return)
    }

@mcp.tool(description="Calculate the daily average trading volume for a stock over a period.")
def calculate_average_volume(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Calculates the daily average trading volume for a stock over a specified period.

    Args:
        symbol (str): The stock symbol.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing the symbol, period, and average volume.
              Returns an error if data is not found.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT AVG(volume)
                FROM public."STOCK_PRICES"
                WHERE symbol = %s AND date >= %s AND date <= %s
                """
                cur.execute(query, (symbol.upper(), start_date, end_date))
                avg_volume = cur.fetchone()[0]

                if avg_volume is None:
                    return {"error": f"No volume data found for {symbol} between {start_date} and {end_date}"}
                
                return {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "average_volume": float(avg_volume)
                }
    except Error as e:
        return {"error": f"Database error: {str(e)}"}


@mcp.tool(description="Compares the latest close prices of two stocks.")
def compare_latest_stock_prices(symbol1: str, symbol2: str) -> dict:
    """
    Compares the latest close prices of two given stock symbols.

    Args:
        symbol1 (str): The symbol of the first stock.
        symbol2 (str): The symbol of the second stock.

    Returns:
        dict: A dictionary containing the symbols and their latest close prices,
              and a comparison result. Returns an error if price data is missing.
    """
    price1 = _get_price_for_calculation(symbol1)
    price2 = _get_price_for_calculation(symbol2)

    if price1 == Decimal('0.0') or price2 == Decimal('0.0'):
        return {"error": "Could not retrieve latest prices for one or both symbols."}

    comparison = ""
    if price1 > price2:
        comparison = f"{symbol1} is higher than {symbol2}."
    elif price2 > price1:
        comparison = f"{symbol2} is higher than {symbol1}."
    else:
        comparison = f"{symbol1} and {symbol2} have the same latest price."

    return {
        "symbol1": symbol1,
        "latest_price1": float(price1),
        "symbol2": symbol2,
        "latest_price2": float(price2),
        "comparison": comparison
    }
    
@mcp.tool(description="Calculates the daily price volatility (standard deviation of daily returns) for a stock over a period.")
def calculate_volatility(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Calculates the annualized daily price volatility (standard deviation of daily returns)
    for a stock over a specified period.

    Args:
        symbol (str): The stock symbol.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing the symbol, period, and calculated annualized volatility.
              Returns an error if data is not found or calculation fails.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT close_price
                FROM public."STOCK_PRICES"
                WHERE symbol = %s AND date >= %s AND date <= %s
                ORDER BY date ASC
                """
                cur.execute(query, (symbol.upper(), start_date, end_date))
                rows = cur.fetchall()

                if not rows or len(rows) < 2:
                    return {"error": f"Insufficient data for volatility calculation for {symbol} between {start_date} and {end_date}. Need at least 2 data points."}

                prices = [float(row[0]) for row in rows]
                
                # Calculate daily returns
                returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                
                if not returns:
                    return {"error": f"Could not calculate returns for {symbol}."}

                daily_volatility = np.std(returns)
                
                # Annualize volatility (assuming 252 trading days in a year)
                annualized_volatility = daily_volatility * np.sqrt(252)

                return {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "annualized_volatility": float(annualized_volatility)
                }
    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Volatility calculation failed: {str(e)}"}