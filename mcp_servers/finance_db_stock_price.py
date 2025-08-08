"""
Finance Stock Price Data MCP Server
Provides stock price data with automatic validation and updates.
"""
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Import from local helpers
from .finance_helpers import (
    get_database_connection,
    get_historical_stock_prices_helper,
    update_stock_prices_helper
)

load_dotenv()

mcp = FastMCP(name="Finance Stock Price Server")

@mcp.tool(description="Get historical stock prices for specified number of days")
def get_historical_stock_prices(symbol: str, days: int = 100) -> dict:
    """
    Get historical stock price data for a specified number of days.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        days: Number of days of data to retrieve (default 100)
        
    Returns:
        Historical price data with statistics
        
    Example:
        get_historical_stock_prices("AAPL", days=30)
    """
    try:
        if not symbol or not symbol.strip():
            return {
                "success": False,
                "error": "Symbol cannot be empty"
            }
        
        symbol = symbol.strip().upper()
        
        # Validate days parameter
        if days > 1000:
            days = 1000
        elif days < 1:
            days = 1
        
        # Get historical data using the reliable DataFrame helper
        from .finance_helpers import get_stock_prices_dataframe
        df = get_stock_prices_dataframe(symbol, days=days)
        
        if df.empty:
            return {
                "success": False,
                "error": f"No historical data found for {symbol}"
            }
        
        # Convert DataFrame to records format
        records = []
        for _, row in df.iterrows():
            try:
                # Safely convert prices with None handling
                open_price = 0.0
                high_price = 0.0
                low_price = 0.0
                close_price = 0.0
                volume = 0
                
                if row['open_price'] is not None:
                    open_price = round(float(row['open_price']), 2)
                if row['high_price'] is not None:
                    high_price = round(float(row['high_price']), 2)
                if row['low_price'] is not None:
                    low_price = round(float(row['low_price']), 2)
                if row['close_price'] is not None:
                    close_price = round(float(row['close_price']), 2)
                if row['volume'] is not None:
                    volume = int(row['volume'])
                
                record = {
                    "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume
                }
                records.append(record)
            except (ValueError, TypeError) as e:
                # Skip records with invalid data
                continue
        
        # Get price list for statistics
        prices_list = df['close_price'].tolist()
        
        # Calculate statistics from the price list
        if prices_list:
            # Safely get current price with proper None handling
            current_price_value = None
            if records:
                close_price = records[-1].get('close')
                if close_price is not None:
                    try:
                        current_price_value = float(close_price)
                    except (ValueError, TypeError):
                        current_price_value = 0.0
                else:
                    current_price_value = 0.0
            else:
                current_price_value = 0.0
                
            statistics = {
                "current_price": current_price_value,
                "period_high": round(max(prices_list), 2),
                "period_low": round(min(prices_list), 2),
                "average_price": round(sum(prices_list) / len(prices_list), 2),
                "price_change": round(prices_list[-1] - prices_list[0], 2) if len(prices_list) > 1 else 0,
                "percent_change": round(((prices_list[-1] / prices_list[0]) - 1) * 100, 2) if len(prices_list) > 1 and prices_list[0] != 0 else 0
            }
        else:
            statistics = {
                "current_price": 0.0,  # Use 0.0 instead of None
                "period_high": 0,
                "period_low": 0,
                "average_price": 0,
                "price_change": 0,
                "percent_change": 0
            }
        
        return {
            "success": True,
            "symbol": symbol,
            "records_returned": len(records),
            "date_range": {
                "start": records[0]['date'] if records else None,
                "end": records[-1]['date'] if records else None
            },
            "statistics": statistics,
            "data": records
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting historical prices: {str(e)}"
        }

@mcp.tool(description="Update stock prices from external data sources")
def update_stock_prices(symbols: list, force_update: bool = False) -> dict:
    """
    Update stock price data for specified symbols.
    
    Args:
        symbols: List of stock symbols to update
        force_update: Whether to force update even if data is recent
        
    Returns:
        Update results for each symbol
        
    Example:
        update_stock_prices(["AAPL", "MSFT"])
    """
    try:
        if not symbols:
            return {
                "success": False,
                "error": "Symbols list cannot be empty"
            }
        
        # Convert to uppercase
        symbol_list = [str(s).strip().upper() for s in symbols if str(s).strip()]
        
        if not symbol_list:
            return {
                "success": False,
                "error": "No valid symbols provided"
            }
        
        # Call the helper with the full list of symbols
        update_result = update_stock_prices_helper(symbol_list)
        
        if update_result.get('success'):
            updated_count = update_result.get('updated_count', 0)
            failed_count = update_result.get('failed_count', 0)
            results = update_result.get('results', [])
            
            return {
                "success": True,
                "total_symbols": len(symbol_list),
                "successful_updates": updated_count,
                "failed_updates": failed_count,
                "update_timestamp": datetime.now().isoformat(),
                "results": results
            }
        else:
            return {
                "success": False,
                "error": update_result.get('error', 'Update failed'),
                "total_symbols": len(symbol_list)
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error updating stock prices: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run()
