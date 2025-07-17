import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Market Analysis")


def get_db_connection():
    try:
        conn = psycopg2.connect(DB_URI)
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        raise


@mcp.tool(
    description="Calculate price performance for a stock symbol over a specified period"
)
def calculate_price_performance(symbol: str, period_days: int = 30):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT date, close_price, volume 
            FROM public.stock_price 
            WHERE symbol = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """
        cursor.execute(query, (symbol, start_date_str, end_date_str))

        rows = cursor.fetchall()
        if not rows:
            return {
                "error": f"No data found for symbol {symbol} in the specified period"
            }

        df = pd.DataFrame(rows, columns=["date", "close_price", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df["close_price"] = pd.to_numeric(df["close_price"])

        start_price = df["close_price"].iloc[0]
        end_price = df["close_price"].iloc[-1]
        price_change = end_price - start_price
        percent_change = (price_change / start_price) * 100

        df["daily_return"] = df["close_price"].pct_change()

        volatility = df["daily_return"].std() * np.sqrt(252)

        avg_volume = df["volume"].mean()

        df["cummax"] = df["close_price"].cummax()
        df["drawdown"] = (df["close_price"] - df["cummax"]) / df["cummax"]
        max_drawdown = df["drawdown"].min() * 100

        cursor.close()
        conn.close()

        return {
            "symbol": symbol,
            "period_days": period_days,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "start_price": float(start_price),
            "end_price": float(end_price),
            "price_change": float(price_change),
            "percent_change": float(percent_change),
            "volatility": float(volatility),
            "avg_volume": float(avg_volume),
            "max_drawdown_pct": float(max_drawdown),
        }

    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error calculating performance: {str(e)}"}


@mcp.tool(description="Compare multiple stock symbols over a specified period")
def compare_stocks(symbols: list, start_date: str = None, end_date: str = None):
    try:
        if not symbols:
            return {"error": "No symbols provided"}

        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)
            start_date = start.strftime("%Y-%m-%d")

        conn = get_db_connection()
        cursor = conn.cursor()

        results = {}

        for symbol in symbols:
            query = """
                SELECT date, close_price, volume 
                FROM public.stock_price 
                WHERE symbol = %s AND date BETWEEN %s AND %s
                ORDER BY date
            """
            cursor.execute(query, (symbol, start_date, end_date))

            rows = cursor.fetchall()
            if not rows:
                results[symbol] = {"error": f"No data found for {symbol}"}
                continue

            df = pd.DataFrame(rows, columns=["date", "close_price", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            df["close_price"] = pd.to_numeric(df["close_price"])

            start_price = df["close_price"].iloc[0]
            end_price = df["close_price"].iloc[-1]
            price_change = end_price - start_price
            percent_change = (price_change / start_price) * 100

            df["daily_return"] = df["close_price"].pct_change()
            volatility = df["daily_return"].std() * np.sqrt(252)

            results[symbol] = {
                "start_price": float(start_price),
                "end_price": float(end_price),
                "price_change": float(price_change),
                "percent_change": float(percent_change),
                "volatility": float(volatility),
                "avg_volume": float(df["volume"].mean()),
            }

        cursor.close()
        conn.close()

        if len(symbols) > 1:
            sorted_by_return = sorted(
                symbols,
                key=lambda s: results[s].get("percent_change", float("-inf"))
                if isinstance(results[s], dict)
                else float("-inf"),
                reverse=True,
            )
            for i, s in enumerate(sorted_by_return):
                if isinstance(results[s], dict) and "error" not in results[s]:
                    results[s]["return_rank"] = i + 1

            sorted_by_volatility = sorted(
                symbols,
                key=lambda s: results[s].get("volatility", float("inf"))
                if isinstance(results[s], dict)
                else float("inf"),
            )
            for i, s in enumerate(sorted_by_volatility):
                if isinstance(results[s], dict) and "error" not in results[s]:
                    results[s]["volatility_rank"] = i + 1

        return {
            "comparison_period": {"start_date": start_date, "end_date": end_date},
            "results": results,
        }

    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error comparing stocks: {str(e)}"}


@mcp.tool(description="Get top performers based on a specific metric")
def get_top_performers(metric: str = "daily_change", limit: int = 10, date: str = None):
    try:
        if not date:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        conn = get_db_connection()
        cursor = conn.cursor()

        if metric == "daily_change":
            query = """
                WITH current_day AS (
                    SELECT symbol, close_price, volume, date
                    FROM public.stock_price
                    WHERE date = %s
                ),
                previous_day AS (
                    SELECT symbol, close_price as prev_close
                    FROM public.stock_price
                    WHERE date = (SELECT MAX(date) FROM public.stock_price WHERE date < %s)
                )
                SELECT c.symbol, c.close_price, c.volume, 
                       ((c.close_price - p.prev_close) / p.prev_close * 100) as daily_change
                FROM current_day c
                JOIN previous_day p ON c.symbol = p.symbol
                ORDER BY daily_change DESC
                LIMIT %s
            """
            cursor.execute(query, (date, date, limit))
            columns = ["symbol", "close_price", "volume", "daily_change"]

        elif metric == "volume":
            query = """
                SELECT symbol, close_price, volume, date
                FROM public.stock_price
                WHERE date = %s
                ORDER BY volume DESC
                LIMIT %s
            """
            cursor.execute(query, (date, limit))
            columns = ["symbol", "close_price", "volume", "date"]

        elif metric == "price":
            query = """
                SELECT symbol, close_price, volume, date
                FROM public.stock_price
                WHERE date = %s
                ORDER BY close_price DESC
                LIMIT %s
            """
            cursor.execute(query, (date, limit))
            columns = ["symbol", "close_price", "volume", "date"]

        else:
            return {
                "error": f"Invalid metric: {metric}. Choose from 'daily_change', 'volume', or 'price'"
            }

        rows = cursor.fetchall()
        if not rows:
            return {"error": f"No data found for date {date}"}

        results = []
        for row in rows:
            result = dict(zip(columns, row))
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    result[key] = float(value)
            results.append(result)

        cursor.close()
        conn.close()

        return {"date": date, "metric": metric, "top_performers": results}

    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error getting top performers: {str(e)}"}
