import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Analysis (Simplified)")

# Define a path to save/load the trained model
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def _get_historical_prices_for_ml(symbol: str, days: int) -> pd.DataFrame:
    """
    Fetches historical stock prices suitable for ML model training.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, open_price, high_price, low_price, close_price, volume
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
                    'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'
                ])
                df['date'] = pd.to_datetime(df['date'])
                df['close_price'] = pd.to_numeric(df['close_price'])
                df['volume'] = pd.to_numeric(df['volume'])

                df = df.sort_values('date').set_index('date')
                return df
    except Error as e:
        print(f"Database error in _get_historical_prices_for_ml: {e}")
        return pd.DataFrame()

def _calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic technical indicators for ML model features.
    """
    # Simple moving averages
    df['SMA_20'] = df['close_price'].rolling(window=20).mean()
    df['SMA_50'] = df['close_price'].rolling(window=50).mean()
    
    # Price changes and volatility
    df['price_change'] = df['close_price'].pct_change()
    df['volatility'] = df['price_change'].rolling(window=10).std()
    
    # Volume ratio
    df['volume_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']
    
    return df

@mcp.tool(description="Trains a basic ML model for stock price prediction.")
def train_stock_prediction_model(symbol: str, lookback_days: int = 100) -> dict:
    """
    Trains a simplified ML model for stock price prediction.
    
    Args:
        symbol (str): The stock symbol.
        lookback_days (int): Number of past days to use for training.
    
    Returns:
        dict: Training results with model performance metrics.
    """
    df = _get_historical_prices_for_ml(symbol, lookback_days + 50)
    
    if df.empty or len(df) < (lookback_days + 20):
        return {"error": f"Insufficient data for {symbol}. Need at least {lookback_days + 20} days."}
    
    # Calculate basic indicators
    df = _calculate_basic_indicators(df)
    
    # Create target variable (next day's close price)
    df['target'] = df['close_price'].shift(-1)
    
    # Feature selection - basic feature set
    feature_columns = [
        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
        'SMA_20', 'SMA_50', 'price_change', 'volatility', 'volume_ratio'
    ]
    
    # Clean data
    df = df.dropna()
    
    if len(df) < 50:
        return {"error": f"Insufficient clean data for {symbol} after processing."}
    
    # Prepare features and target
    X = df[feature_columns].values
    y = df['target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return {
        "symbol": symbol,
        "model_type": "random_forest",
        "training_data_points": len(X_train),
        "test_data_points": len(X_test),
        "metrics": {
            "mse": round(mse, 4),
            "r2_score": round(r2, 4),
            "accuracy_pct": round(r2 * 100, 2)
        },
        "model_saved": model_path,
        "scaler_saved": scaler_path,
        "training_date": datetime.now().isoformat()
    }

@mcp.tool(description="Predicts future stock prices using trained ML model.")
def predict_stock_price(symbol: str, days_ahead: int = 1) -> dict:
    """
    Predicts future stock prices using a pre-trained ML model.
    
    Args:
        symbol (str): The stock symbol.
        days_ahead (int): Number of days ahead to predict (max 7).
    
    Returns:
        dict: Prediction results with confidence intervals.
    """
    try:
        # Load model and scaler
        model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            return {"error": f"No trained model found for {symbol}. Please train the model first."}
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Get recent data
        df = _get_historical_prices_for_ml(symbol, 60)
        if df.empty:
            return {"error": f"No recent data available for {symbol}."}
        
        # Calculate indicators
        df = _calculate_basic_indicators(df)
        df = df.dropna()
        
        if len(df) < 1:
            return {"error": f"Insufficient recent data for {symbol}."}
        
        # Get latest features
        feature_columns = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'SMA_20', 'SMA_50', 'price_change', 'volatility', 'volume_ratio'
        ]
        
        latest_features = df[feature_columns].iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        
        # Make predictions
        predictions = []
        current_price = df['close_price'].iloc[-1]
        
        for day in range(1, min(days_ahead + 1, 8)):  # Max 7 days
            pred_price = model.predict(latest_features_scaled)[0]
            confidence = min(90, max(50, 80 - (day - 1) * 5))  # Decreasing confidence
            
            predictions.append({
                "day": day,
                "predicted_price": round(pred_price, 2),
                "confidence": confidence,
                "price_change": round(((pred_price - current_price) / current_price) * 100, 2)
            })
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predictions": predictions,
            "model_type": "random_forest",
            "prediction_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@mcp.tool(description="Analyzes stock trends using basic technical indicators.")
def analyze_stock_trends(symbol: str, period: int = 50) -> dict:
    """
    Analyzes stock price trends using basic technical indicators.
    
    Args:
        symbol (str): The stock symbol.
        period (int): Number of days to analyze.
    
    Returns:
        dict: Trend analysis results.
    """
    try:
        df = _get_historical_prices_for_ml(symbol, period + 20)
        
        if df.empty or len(df) < period:
            return {"error": f"Insufficient data for {symbol}. Need at least {period} days."}
        
        df = _calculate_basic_indicators(df)
        df = df.dropna()
        
        if len(df) < 10:
            return {"error": f"Insufficient clean data for {symbol}."}
        
        # Get latest values
        latest = df.iloc[-1]
        current_price = latest['close_price']
        sma_20 = latest.get('SMA_20', current_price)
        sma_50 = latest.get('SMA_50', current_price)
        
        # Determine trends
        short_trend = "bullish" if current_price > sma_20 else "bearish"
        long_trend = "bullish" if current_price > sma_50 else "bearish"
        overall_trend = "bullish" if (current_price > sma_20 and sma_20 > sma_50) else "bearish" if (current_price < sma_20 and sma_20 < sma_50) else "neutral"
        
        # Calculate trend strength
        recent_changes = df['price_change'].tail(10).mean()
        volatility = df['volatility'].tail(10).mean()
        
        trend_strength = min(100, abs(recent_changes) * 1000)
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "technical_indicators": {
                "SMA_20": round(sma_20, 2),
                "SMA_50": round(sma_50, 2),
                "recent_volatility": round(volatility, 4)
            },
            "trend_analysis": {
                "short_term": short_trend,
                "long_term": long_trend,
                "overall": overall_trend,
                "strength": round(trend_strength, 2)
            },
            "price_levels": {
                "support": round(df['low_price'].tail(20).min(), 2),
                "resistance": round(df['high_price'].tail(20).max(), 2)
            },
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Trend analysis failed: {str(e)}"}

# Simplified analysis server with 3 essential tools
# Removed: sector analysis, correlation matrix, price range analysis, advanced ML features

if __name__ == "__main__":
    mcp.run()
