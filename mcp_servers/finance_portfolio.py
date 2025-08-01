import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)
USER_DB_URI = os.getenv("USER_DB_URI", None)

if not DB_URI or not USER_DB_URI:
    raise ValueError("Database URIs not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Portfolio (Simplified)")

def _get_latest_close_price(symbol: str) -> Decimal:
    """Fetches the most recent close price for a stock symbol."""
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT close_price FROM public.\"STOCK_PRICES\" WHERE symbol = %s ORDER BY date DESC LIMIT 1",
                    (symbol.upper(),)
                )
                result = cur.fetchone()
                if result:
                    return Decimal(str(result[0]))
                else:
                    return Decimal('0.0')
    except Error as e:
        print(f"Error fetching latest price for {symbol}: {e}")
        return Decimal('0.0')


@mcp.tool(description="Adds a stock holding to a user's portfolio.")
def add_stock_holding(user_id: int, symbol: str, quantity: int, purchase_price: float, purchase_date: str) -> dict:
    """
    Adds a specified quantity of a stock to a user's portfolio at a given purchase price and date.

    Args:
        user_id (int): The ID of the user.
        symbol (str): The stock symbol (e.g., 'AAPL').
        quantity (int): The number of shares purchased.
        purchase_price (float): The price per share at the time of purchase.
        purchase_date (str): The date of purchase in YYYY-MM-DD format.

    Returns:
        dict: A dictionary indicating success or failure.
    """
    try:
        # Validate user_id exists (optional but good practice)
        with psycopg2.connect(USER_DB_URI) as conn_user:
            with conn_user.cursor() as cur_user:
                cur_user.execute("SELECT id FROM public.\"USERS\" WHERE id = %s", (user_id,))
                if not cur_user.fetchone():
                    return {"error": f"User with ID {user_id} not found."}

        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # Check if holding already exists for this user and symbol
                cur.execute(
                    "SELECT id, quantity, purchase_price FROM public.\"USER_STOCK_HOLDINGS\" WHERE user_id = %s AND symbol = %s",
                    (user_id, symbol.upper())
                )
                existing_holding = cur.fetchone()

                if existing_holding:
                    # Update existing holding (e.g., average price, sum quantity)
                    existing_id, old_quantity, old_purchase_price_dec = existing_holding
                    old_purchase_price = float(old_purchase_price_dec)

                    new_total_cost = (old_quantity * old_purchase_price) + (quantity * purchase_price)
                    new_total_quantity = old_quantity + quantity
                    new_average_price = new_total_cost / new_total_quantity if new_total_quantity > 0 else 0

                    cur.execute(
                        "UPDATE public.\"USER_STOCK_HOLDINGS\" SET quantity = %s, purchase_price = %s, updatedat = NOW() WHERE id = %s",
                        (new_total_quantity, Decimal(str(new_average_price)), existing_id)
                    )
                    conn.commit()
                    return {"success": True, "message": f"Updated holding for {symbol} for user {user_id}. New quantity: {new_total_quantity}, Average price: {new_average_price:.2f}"}
                else:
                    # Insert new holding
                    cur.execute(
                        "INSERT INTO public.\"USER_STOCK_HOLDINGS\" (user_id, symbol, quantity, purchase_price, purchase_date, createdat, updatedat) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())",
                        (user_id, symbol.upper(), quantity, Decimal(str(purchase_price)), purchase_date)
                    )
                    conn.commit()
                    return {"success": True, "message": f"Added {quantity} shares of {symbol} for user {user_id}."}
    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to add stock holding: {str(e)}"}

@mcp.tool(description="Retrieves all stock holdings for a given user.")
def get_user_portfolio(user_id: int) -> dict:
    """
    Retrieves all stock holdings for a specific user, including current market value.

    Args:
        user_id (int): The ID of the user.

    Returns:
        dict: A dictionary containing the user's portfolio details.
              Each holding includes symbol, quantity, purchase price, current price, and current value.
              Also includes total portfolio value.
    """
    try:
        holdings = []
        total_portfolio_value = Decimal('0.0')

        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT symbol, quantity, purchase_price, purchase_date FROM public.\"USER_STOCK_HOLDINGS\" WHERE user_id = %s",
                    (user_id,)
                )
                rows = cur.fetchall()

                for row in rows:
                    symbol, quantity, purchase_price_dec, purchase_date = row
                    purchase_price = float(purchase_price_dec)
                    latest_price = _get_latest_close_price(symbol)
                    current_value = latest_price * quantity
                    total_portfolio_value += current_value

                    holdings.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "purchase_price": purchase_price,
                        "purchase_date": purchase_date.isoformat() if isinstance(purchase_date, datetime) else str(purchase_date),
                        "latest_price": float(latest_price),
                        "current_value": float(current_value),
                        "gain_loss_percentage": ((latest_price - Decimal(str(purchase_price))) / Decimal(str(purchase_price))) * Decimal('100.0') if purchase_price != 0 else 0
                    })
        
        if not holdings:
            return {"user_id": user_id, "holdings": [], "total_portfolio_value": 0.0, "message": "No holdings found for this user."}

        return {
            "user_id": user_id,
            "holdings": holdings,
            "total_portfolio_value": float(total_portfolio_value)
        }
    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to retrieve portfolio: {str(e)}"}



# Simplified portfolio server with 2 essential tools: add and get
# Removed: remove_stock_holding, analyze_portfolio, optimize_portfolio, analyze_portfolio_risk functions

if __name__ == "__main__":
    mcp.run()
