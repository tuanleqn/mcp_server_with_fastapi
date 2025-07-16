import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
# For prediction, you would typically use libraries like tensorflow, sklearn, or pytorch.
# We'll use a placeholder for now as a full ML model is out of scope for a direct code example.
# import numpy as np
# from sklearn.linear_model import LinearRegression # Example for a simple prediction model

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Analysis and Predictions")

@mcp.tool(description="Analyze a stock's historical price range (high, low, average) over a period.")
def analyze_price_range(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Analyzes a stock's historical price range (highest, lowest, and average close price)
    over a specified period.

    Args:
        symbol (str): The stock symbol.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing the symbol, period, highest price, lowest price,
              and average close price. Returns an error if data is not found.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT MAX(high_price), MIN(low_price), AVG(close_price)
                FROM public."STOCK_PRICES"
                WHERE symbol = %s AND date >= %s AND date <= %s
                """
                cur.execute(query, (symbol.upper(), start_date, end_date))
                result = cur.fetchone()

                if result and result[0] is not None:
                    return {
                        "symbol": symbol,
                        "start_date": start_date,
                        "end_date": end_date,
                        "highest_price": float(result[0]),
                        "lowest_price": float(result[1]),
                        "average_close_price": float(result[2])
                    }
                else:
                    return {"error": f"No price data found for {symbol} between {start_date} and {end_date}"}
    except Error as e:
        return {"error": f"Database error: {str(e)}"}


@mcp.tool(description="Predicts a future closing price for a stock (placeholder for ML model).")
def predict_future_price(symbol: str, days_in_future: int) -> dict:
    """
    Predicts a future closing price for a stock.
    NOTE: This is a placeholder and would require integration with a machine learning model.
          For demonstration, it returns a mock prediction.

    Args:
        symbol (str): The stock symbol.
        days_in_future (int): Number of days into the future to predict.

    Returns:
        dict: A dictionary with the symbol, prediction date, and predicted close price.
              Returns an error if prediction logic fails.
    """
    # In a real scenario, you'd load a trained ML model here,
    # fetch historical data using get_historical_prices,
    # preprocess it, and then make a prediction.
    
    # Placeholder: A very simplistic "prediction" based on the latest price + a small arbitrary increase
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = "SELECT close_price, date FROM public.\"STOCK_PRICES\" WHERE symbol = %s ORDER BY date DESC LIMIT 1"
                cur.execute(query, (symbol.upper(),))
                result = cur.fetchone()

                if result:
                    latest_price = float(result[0])
                    latest_date = result[1]
                    
                    # Mock prediction: increase by 0.5% per day for the future
                    predicted_price = latest_price * (1 + 0.005 * days_in_future)
                    
                    # Calculate prediction date
                    prediction_date = latest_date + pd.Timedelta(days=days_in_future)

                    return {
                        "symbol": symbol,
                        "prediction_date": prediction_date.isoformat(),
                        "predicted_close_price": round(predicted_price, 2),
                        "note": "This is a mock prediction and not based on a real ML model."
                    }
                else:
                    return {"error": f"No historical data to base prediction for {symbol}."}

    except Error as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}