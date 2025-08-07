"""
Finance Database Stock Price Server
"""

import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from .finance_helpers import (
    get_latest_stock_price_helper,
    get_historical_prices_helper,
    update_stock_price_helper
)
import psycopg2
from psycopg2 import Error

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance Stock Price Database Server")

@mcp.tool(description="Get historical stock prices")
def get_historical_stock_prices(symbol: str, days: int = 30) -> dict:
    days = min(days, 1000)
    
    df = get_historical_prices_helper(symbol, days)
    if df.empty:
        return {
            "success": False,
            "error": f"No historical data found for symbol {symbol}",
            "symbol": symbol.upper()
        }
    
    # Convert DataFrame to list of dictionaries
    historical_data = []
    for index, row in df.iterrows():
        historical_data.append({
            "date": index.isoformat(),
            "open_price": float(row['open_price']),
            "high_price": float(row['high_price']),
            "low_price": float(row['low_price']),
            "close_price": float(row['close_price']),
            "volume": int(row['volume'])
        })
    
    return {
        "success": True,
        "symbol": symbol.upper(),
        "days_requested": days,
        "records_returned": len(historical_data),
        "date_range": {
            "earliest": historical_data[-1]["date"] if historical_data else None,
            "latest": historical_data[0]["date"] if historical_data else None
        },
        "historical_data": historical_data
    }

@mcp.tool(description="Update stock price data from external sources")
def update_stock_prices(symbol: str, force_update: bool = False) -> dict:
    return update_stock_price_helper(symbol, force_update)

if __name__ == "__main__":
    mcp.run()
