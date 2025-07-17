import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # Changed from LinearRegression
import joblib # To save/load models

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Analysis and Predictions")

# Define a path to save/load the trained model
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure the directory exists

def _get_historical_prices_for_ml(symbol: str, days: int) -> pd.DataFrame:
    """
    Fetches historical stock prices suitable for ML model training.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, open_price, high_price, low_price, close_price, adjusted_close, volume, dividend_amount, split_coefficient
                FROM public."STOCK_PRICES"
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
                """
                cur.execute(query, (symbol.upper(), days))
                rows = cur.fetchall()

                if not rows:
                    return pd.DataFrame() # Empty DataFrame

                df = pd.DataFrame(rows, columns=[
                    'date', 'open_price', 'high_price', 'low_price', 'close_price',
                    'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient'
                ])
                df['date'] = pd.to_datetime(df['date'])
                df['close_price'] = pd.to_numeric(df['close_price'])
                df['open_price'] = pd.to_numeric(df['open_price'])
                df['high_price'] = pd.to_numeric(df['high_price'])
                df['low_price'] = pd.to_numeric(df['low_price'])
                df['volume'] = pd.to_numeric(df['volume'])
                df['dividend_amount'] = pd.to_numeric(df['dividend_amount'])
                df['split_coefficient'] = pd.to_numeric(df['split_coefficient'])

                df = df.sort_values('date').set_index('date')
                return df
    except Error as e:
        print(f"Database error in _get_historical_prices_for_ml: {e}")
        return pd.DataFrame()


@mcp.tool(description="Trains a RandomForestRegressor model for stock price prediction based on historical data with basic features.")
def train_stock_prediction_model(symbol: str, lookback_days: int = 90, n_estimators: int = 100, max_depth: int = 10) -> dict:
    """
    Trains a RandomForestRegressor prediction model for a given stock symbol.
    Uses lagged features for close price, volume, and OHLC.

    Args:
        symbol (str): The stock symbol.
        lookback_days (int): Number of past days to use for training.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree.

    Returns:
        dict: Status of the training process.
    """
    # Fetch enough data for lookback and feature creation
    df = _get_historical_prices_for_ml(symbol, lookback_days + 5) # +5 for creating lagged features

    if df.empty or len(df) < (lookback_days + 1):
        return {"error": f"Insufficient data to train model for {symbol} with {lookback_days} lookback days."}

    # Feature Engineering: Lagged prices and volume
    df['next_close'] = df['close_price'].shift(-1) # Target variable: next day's close
    
    # Lagged features (previous day's data)
    df['prev_close'] = df['close_price'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_open'] = df['open_price'].shift(1)
    df['prev_high'] = df['high_price'].shift(1)
    df['prev_low'] = df['low_price'].shift(1)

    # Simple moving averages (example features)
    df['SMA_5'] = df['close_price'].rolling(window=5).mean().shift(1)
    df['SMA_10'] = df['close_price'].rolling(window=10).mean().shift(1)

    df.dropna(inplace=True) # Drop NaN rows created by shifting and rolling

    if df.empty:
        return {"error": f"Not enough data after feature engineering for {symbol}. Try a longer lookback_days or ensure data quality."}

    # Define features (X) and target (y)
    features = [
        'prev_close', 'prev_volume', 'prev_open', 'prev_high', 'prev_low',
        'SMA_5', 'SMA_10'
    ]
    X = df[features]
    y = df['next_close']

    # Ensure X and y have the same number of samples
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]

    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X, y)

    # Save the trained model
    model_file_name = os.path.join(MODEL_DIR, f"{symbol.lower()}_rf_prediction_model.pkl")
    joblib.dump(model, model_file_name)

    return {
        "success": True,
        "symbol": symbol,
        "message": f"RandomForestRegressor model for {symbol} trained and saved.",
        "model_path": model_file_name,
        "features_used": features
    }

@mcp.tool(description="Predicts future closing prices for a stock using a pre-trained RandomForestRegressor ML model.")
def predict_future_price_with_ml(symbol: str, days_in_future: int = 1) -> dict:
    """
    Predicts future closing prices for a stock using a pre-trained RandomForestRegressor model.
    The model needs to be trained first using 'train_stock_prediction_model'.

    Args:
        symbol (str): The stock symbol.
        days_in_future (int): Number of days into the future to predict (currently supports 1-step prediction).

    Returns:
        dict: A dictionary with the symbol, prediction date(s), and predicted close price(s).
              Returns an error if the model is not found or prediction fails.
    """
    model_file_name = os.path.join(MODEL_DIR, f"{symbol.lower()}_rf_prediction_model.pkl")

    if not os.path.exists(model_file_name):
        return {"error": f"RandomForestRegressor model for {symbol} not found. Please train it first using 'train_stock_prediction_model'."}

    try:
        model = joblib.load(model_file_name)
    except Exception as e:
        return {"error": f"Failed to load prediction model for {symbol}: {str(e)}"}

    # Features used by the trained model (must match the features used during training)
    # This should ideally be stored with the model or derived dynamically.
    # For this example, we assume we know the features.
    features = [
        'prev_close', 'prev_volume', 'prev_open', 'prev_high', 'prev_low',
        'SMA_5', 'SMA_10'
    ]

    # Get enough recent historical data to create the input features for prediction
    # Need enough days to calculate SMAs and lagged features (e.g., 10 days for SMA_10)
    history_needed = max(10, days_in_future + 1) # Needs enough history for SMAs and lags
    recent_df = _get_historical_prices_for_ml(symbol, history_needed)

    if recent_df.empty or len(recent_df) < max(2, history_needed): # Ensure at least 2 points for lagged data
        return {"error": f"Insufficient recent data available for {symbol} to make a prediction. Need at least {history_needed} days of data."}

    # Prepare features for prediction using the latest available data
    # Calculate lagged features and SMAs on the recent data
    recent_df['prev_close'] = recent_df['close_price'].shift(1)
    recent_df['prev_volume'] = recent_df['volume'].shift(1)
    recent_df['prev_open'] = recent_df['open_price'].shift(1)
    recent_df['prev_high'] = recent_df['high_price'].shift(1)
    recent_df['prev_low'] = recent_df['low_price'].shift(1)
    recent_df['SMA_5'] = recent_df['close_price'].rolling(window=5).mean().shift(1)
    recent_df['SMA_10'] = recent_df['close_price'].rolling(window=10).mean().shift(1)

    # We need the most recent row that has all features calculated (dropna)
    last_valid_row = recent_df.dropna().iloc[-1]
    latest_features = last_valid_row[features].values.reshape(1, -1) # Reshape for single prediction

    latest_date_in_db = recent_df.index[-1]
    
    predicted_prices = []
    current_features_for_prediction = latest_features.copy()

    # Predict future prices iteratively (multi-step prediction for RandomForest is typically iterative)
    for i in range(days_in_future):
        predicted_price_for_next_day = model.predict(current_features_for_prediction)[0]
        predicted_prices.append(round(float(predicted_price_for_next_day), 2))

        # Update features for the next prediction step using the predicted price
        # This is a very simplistic update; a robust solution would involve forecasting all features.
        # For RandomForest, the model doesn't directly take time series, so we approximate.
        new_prev_close = predicted_price_for_next_day
        # Other features (volume, OHLC, SMAs) would need to be forecasted or approximated.
        # For this example, we'll keep other features static or simply shift them.
        # A more advanced solution would recalculate SMAs based on new predictions.
        current_features_for_prediction[0, features.index('prev_close')] = new_prev_close

        # For SMA_5 and SMA_10, this simplified iteration is not accurate without full history or forecasting
        # You'd need a more robust way to update these if iterating multiple days.
        # For now, we are essentially making a "rolling" 1-day prediction.
        
    prediction_date = latest_date_in_db + timedelta(days=days_in_future)


    return {
        "symbol": symbol,
        "prediction_date": prediction_date.isoformat(),
        "predicted_close_price": predicted_prices[-1], # Return the last predicted price for days_in_future
        "all_predicted_prices": predicted_prices if days_in_future > 1 else None,
        "note": "Prediction based on RandomForestRegressor. Stock prediction is complex and inherently risky. This model is illustrative; consult financial professionals."
    }

# Existing tools from finance_analysis_and_predictions.py
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