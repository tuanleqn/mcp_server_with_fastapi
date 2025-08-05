import os
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

# Define a path to save/load the trained model
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_historical_prices_for_ml(symbol: str, days: int) -> pd.DataFrame:
    """
    Fetches historical stock prices suitable for ML model training.
    Helper function for ML operations.
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
                    return pd.DataFrame()

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
        print(f"Database error in get_historical_prices_for_ml: {e}")
        return pd.DataFrame()


def train_stock_prediction_model_helper(symbol: str, lookback_days: int = 90, n_estimators: int = 100, max_depth: int = 10) -> dict:
    """
    Helper function to train a RandomForestRegressor model for stock price prediction.
    This is called internally by other MCP tools when a trained model is needed.
    """
    # Fetch enough data for lookback and feature creation
    df = get_historical_prices_for_ml(symbol, lookback_days + 5)

    if df.empty or len(df) < (lookback_days + 1):
        return {"error": f"Insufficient data to train model for {symbol} with {lookback_days} lookback days."}

    # Feature Engineering: Lagged prices and volume
    df['next_close'] = df['close_price'].shift(-1)
    
    # Lagged features (previous day's data)
    df['prev_close'] = df['close_price'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_open'] = df['open_price'].shift(1)
    df['prev_high'] = df['high_price'].shift(1)
    df['prev_low'] = df['low_price'].shift(1)

    # Simple moving averages
    df['SMA_5'] = df['close_price'].rolling(window=5).mean().shift(1)
    df['SMA_10'] = df['close_price'].rolling(window=10).mean().shift(1)

    df.dropna(inplace=True)

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


def predict_future_price_with_ml_helper(symbol: str, days_in_future: int = 1) -> dict:
    """
    Helper function to predict future closing prices using a pre-trained RandomForestRegressor model.
    This is called internally by other MCP tools.
    """
    model_file_name = os.path.join(MODEL_DIR, f"{symbol.lower()}_rf_prediction_model.pkl")

    # If model doesn't exist, try to train it automatically
    if not os.path.exists(model_file_name):
        print(f"Model for {symbol} not found, attempting to train...")
        train_result = train_stock_prediction_model_helper(symbol)
        if "error" in train_result:
            return {"error": f"Could not train model for {symbol}: {train_result['error']}"}

    try:
        model = joblib.load(model_file_name)
    except Exception as e:
        return {"error": f"Failed to load prediction model for {symbol}: {str(e)}"}

    # Features used by the trained model
    features = [
        'prev_close', 'prev_volume', 'prev_open', 'prev_high', 'prev_low',
        'SMA_5', 'SMA_10'
    ]

    # Get enough recent historical data to create input features
    history_needed = max(10, days_in_future + 1)
    recent_df = get_historical_prices_for_ml(symbol, history_needed)

    if recent_df.empty or len(recent_df) < max(2, history_needed):
        return {"error": f"Insufficient recent data available for {symbol} to make a prediction. Need at least {history_needed} days of data."}

    # Prepare features for prediction
    recent_df['prev_close'] = recent_df['close_price'].shift(1)
    recent_df['prev_volume'] = recent_df['volume'].shift(1)
    recent_df['prev_open'] = recent_df['open_price'].shift(1)
    recent_df['prev_high'] = recent_df['high_price'].shift(1)
    recent_df['prev_low'] = recent_df['low_price'].shift(1)
    recent_df['SMA_5'] = recent_df['close_price'].rolling(window=5).mean().shift(1)
    recent_df['SMA_10'] = recent_df['close_price'].rolling(window=10).mean().shift(1)

    # Get the most recent row with all features calculated
    last_valid_row = recent_df.dropna().iloc[-1]
    latest_features = last_valid_row[features].values.reshape(1, -1)

    latest_date_in_db = recent_df.index[-1]
    
    predicted_prices = []
    current_features_for_prediction = latest_features.copy()

    # Predict future prices iteratively
    for i in range(days_in_future):
        predicted_price_for_next_day = model.predict(current_features_for_prediction)[0]
        predicted_prices.append(round(float(predicted_price_for_next_day), 2))

        # Update features for the next prediction step
        new_prev_close = predicted_price_for_next_day
        current_features_for_prediction[0, features.index('prev_close')] = new_prev_close
        
    prediction_date = latest_date_in_db + timedelta(days=days_in_future)

    return {
        "symbol": symbol,
        "prediction_date": prediction_date.isoformat(),
        "predicted_close_price": predicted_prices[-1],
        "all_predicted_prices": predicted_prices if days_in_future > 1 else None,
        "note": "Prediction based on RandomForestRegressor. Stock prediction is complex and inherently risky."
    }
