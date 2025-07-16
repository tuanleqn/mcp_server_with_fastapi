import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Plotting Tools")

@mcp.tool(description="Generates and returns a base64 encoded image of a stock's historical close prices.")
def plot_historical_close_prices(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Generates a line plot of a stock's historical close prices for a specified period
    and returns it as a base64 encoded PNG image.

    Args:
        symbol (str): The stock symbol.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing the symbol and the base64 encoded image string.
              Returns an error if data is not found or plotting fails.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, close_price
                FROM public."STOCK_PRICES"
                WHERE symbol = %s AND date >= %s AND date <= %s
                ORDER BY date ASC
                """
                cur.execute(query, (symbol.upper(), start_date, end_date))
                rows = cur.fetchall()

                if not rows:
                    return {"error": f"No price data found for {symbol} between {start_date} and {end_date} for plotting."}

                df = pd.DataFrame(rows, columns=['date', 'close_price'])
                df['date'] = pd.to_datetime(df['date'])
                df['close_price'] = pd.to_numeric(df['close_price'])
                df = df.set_index('date')

                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['close_price'], marker='o', linestyle='-')
                plt.title(f'{symbol} Historical Close Prices ({start_date} to {end_date})')
                plt.xlabel('Date')
                plt.ylabel('Close Price')
                plt.grid(True)
                plt.tight_layout()

                # Save plot to a bytes buffer
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close() # Close the plot to free up memory
                buffer.seek(0)
                
                # Encode to base64
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                return {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "plot_image_base64": image_base64,
                    "format": "png"
                }

    except Error as e:
        return {"error": f"Database error during plotting: {str(e)}"}
    except Exception as e:
        return {"error": f"Plotting failed: {str(e)}"}


@mcp.tool(description="Generates and returns a base64 encoded image of a stock's historical volume.")
def plot_historical_volume(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Generates a bar plot of a stock's historical trading volume for a specified period
    and returns it as a base64 encoded PNG image.

    Args:
        symbol (str): The stock symbol.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing the symbol and the base64 encoded image string.
              Returns an error if data is not found or plotting fails.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, volume
                FROM public."STOCK_PRICES"
                WHERE symbol = %s AND date >= %s AND date <= %s
                ORDER BY date ASC
                """
                cur.execute(query, (symbol.upper(), start_date, end_date))
                rows = cur.fetchall()

                if not rows:
                    return {"error": f"No volume data found for {symbol} between {start_date} and {end_date} for plotting."}

                df = pd.DataFrame(rows, columns=['date', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df['volume'] = pd.to_numeric(df['volume'])
                df = df.set_index('date')

                plt.figure(figsize=(10, 6))
                plt.bar(df.index, df['volume'])
                plt.title(f'{symbol} Historical Trading Volume ({start_date} to {end_date})')
                plt.xlabel('Date')
                plt.ylabel('Volume')
                plt.grid(True)
                plt.tight_layout()

                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                return {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "plot_image_base64": image_base64,
                    "format": "png"
                }

    except Error as e:
        return {"error": f"Database error during plotting: {str(e)}"}
    except Exception as e:
        return {"error": f"Plotting failed: {str(e)}"}