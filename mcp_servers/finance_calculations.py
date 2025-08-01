import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal
import numpy as np

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Calculations (Simplified)")

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
                else:
                    query = "SELECT close_price FROM public.\"STOCK_PRICES\" WHERE symbol = %s ORDER BY date DESC LIMIT 1"
                    cur.execute(query, (symbol.upper(),))
                
                result = cur.fetchone()
                if result:
                    return Decimal(result[0])
                else:
                    return Decimal('0.0')
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
        dict: A dictionary containing the percentage return and related information.
    """
    try:
        start_price = _get_price_for_calculation(symbol, start_date)
        end_price = _get_price_for_calculation(symbol, end_date)
        
        if start_price == 0 or end_price == 0:
            return {"error": f"Could not retrieve prices for {symbol} between {start_date} and {end_date}."}
        
        percentage_return = ((end_price - start_price) / start_price) * 100
        
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "start_price": float(start_price),
            "end_price": float(end_price),
            "absolute_change": float(end_price - start_price),
            "percentage_return": round(float(percentage_return), 2),
            "calculation_date": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to calculate percentage return: {str(e)}"}

@mcp.tool(description="Calculate the daily price volatility for a stock over a period.")
def calculate_volatility(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Calculates the daily price volatility (standard deviation of daily returns) for a stock over a period.
    
    Args:
        symbol (str): The stock symbol.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
    
    Returns:
        dict: A dictionary containing volatility metrics.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, close_price 
                FROM public."STOCK_PRICES" 
                WHERE symbol = %s AND date BETWEEN %s AND %s 
                ORDER BY date
                """
                cur.execute(query, (symbol.upper(), start_date, end_date))
                rows = cur.fetchall()
                
                if len(rows) < 2:
                    return {"error": f"Insufficient data for volatility calculation for {symbol}."}
                
                prices = [float(row[1]) for row in rows]
                daily_returns = []
                
                for i in range(1, len(prices)):
                    daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                    daily_returns.append(daily_return)
                
                if not daily_returns:
                    return {"error": f"Could not calculate daily returns for {symbol}."}
                
                volatility = np.std(daily_returns)
                avg_return = np.mean(daily_returns)
                
                return {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_points": len(rows),
                    "daily_volatility": round(volatility, 6),
                    "annualized_volatility": round(volatility * np.sqrt(252), 4),
                    "average_daily_return": round(avg_return, 6),
                    "calculation_date": datetime.now().isoformat()
                }
                
    except Exception as e:
        return {"error": f"Volatility calculation failed: {str(e)}"}

@mcp.tool(description="Calculate compound interest return on an investment.")
def calculate_compound_return(principal: float, rate: float, time: float, compound_frequency: int = 12) -> dict:
    """
    Calculates compound interest return on an investment.
    
    Args:
        principal (float): Initial investment amount.
        rate (float): Annual interest rate (as decimal, e.g., 0.08 for 8%).
        time (float): Time period in years.
        compound_frequency (int): Compounding frequency per year (default 12 for monthly).
    
    Returns:
        dict: Compound interest calculation results.
    """
    try:
        if principal <= 0 or rate < 0 or time <= 0 or compound_frequency <= 0:
            return {"error": "Invalid input parameters. All values must be positive."}
        
        # Compound interest formula: A = P(1 + r/n)^(nt)
        final_amount = principal * (1 + rate / compound_frequency) ** (compound_frequency * time)
        total_return = final_amount - principal
        return_percentage = (total_return / principal) * 100
        
        # Effective annual rate
        effective_rate = (1 + rate / compound_frequency) ** compound_frequency - 1
        
        return {
            "calculation": "compound_return",
            "inputs": {
                "principal": principal,
                "annual_rate": rate,
                "annual_rate_percentage": round(rate * 100, 2),
                "time_years": time,
                "compound_frequency": compound_frequency
            },
            "results": {
                "final_amount": round(final_amount, 2),
                "total_return": round(total_return, 2),
                "return_percentage": round(return_percentage, 2),
                "effective_annual_rate": round(effective_rate, 4)
            },
            "calculation_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Compound return calculation failed: {str(e)}"}

# Simplified calculations server with 3 essential tools
# Removed: calculate_average_volume, compare_stock_prices, and wrapper functions

if __name__ == "__main__":
    mcp.run()
