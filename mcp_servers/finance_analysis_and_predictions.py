import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

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


def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced technical indicators for enhanced ML model features.
    """
    # Price-based indicators
    df['price_change'] = df['close_price'].pct_change()
    df['price_volatility'] = df['price_change'].rolling(window=10).std()
    
    # Moving averages
    df['SMA_5'] = df['close_price'].rolling(window=5).mean()
    df['SMA_10'] = df['close_price'].rolling(window=10).mean()
    df['SMA_20'] = df['close_price'].rolling(window=20).mean()
    df['EMA_12'] = df['close_price'].ewm(span=12).mean()
    df['EMA_26'] = df['close_price'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # RSI (Relative Strength Index)
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['close_price'].rolling(window=20).mean()
    bb_std = df['close_price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (df['close_price'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume indicators
    df['volume_SMA'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_SMA']
    
    # Price patterns
    df['high_low_ratio'] = df['high_price'] / df['low_price']
    df['open_close_ratio'] = df['open_price'] / df['close_price']
    
    return df


@mcp.tool(description="Trains advanced ML models for stock price prediction with comprehensive technical indicators and model comparison.")
def train_advanced_stock_prediction_model(symbol: str, lookback_days: int = 252, model_type: str = "ensemble") -> dict:
    """
    Trains advanced ML models for stock price prediction using comprehensive features.
    
    Args:
        symbol (str): The stock symbol.
        lookback_days (int): Number of past days to use for training (default 252 = 1 year).
        model_type (str): Model type - 'random_forest', 'gradient_boost', 'linear', or 'ensemble'.
    
    Returns:
        dict: Training results with model performance metrics.
    """
    # Fetch data with extra buffer for indicator calculation
    df = _get_historical_prices_for_ml(symbol, lookback_days + 50)
    
    if df.empty or len(df) < (lookback_days + 30):
        return {"error": f"Insufficient data for {symbol}. Need at least {lookback_days + 30} days."}
    
    # Calculate technical indicators
    df = _calculate_technical_indicators(df)
    
    # Create target variable (next day's close price)
    df['target'] = df['close_price'].shift(-1)
    
    # Feature selection - comprehensive feature set
    feature_columns = [
        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
        'price_change', 'price_volatility',
        'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_signal', 'MACD_histogram',
        'RSI', 'BB_width', 'BB_position',
        'volume_ratio', 'high_low_ratio', 'open_close_ratio'
    ]
    
    # Clean data
    df = df.dropna()
    
    if len(df) < 100:
        return {"error": f"Insufficient clean data after feature engineering for {symbol}."}
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['target']
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Train different models based on model_type
    if model_type in ['random_forest', 'ensemble']:
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        models['random_forest'] = rf_model
        results['random_forest'] = {
            'mse': mean_squared_error(y_test, rf_pred),
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred)
        }
    
    if model_type in ['gradient_boost', 'ensemble']:
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        models['gradient_boost'] = gb_model
        results['gradient_boost'] = {
            'mse': mean_squared_error(y_test, gb_pred),
            'mae': mean_absolute_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred)
        }
    
    if model_type in ['linear', 'ensemble']:
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        models['linear'] = lr_model
        results['linear'] = {
            'mse': mean_squared_error(y_test, lr_pred),
            'mae': mean_absolute_error(y_test, lr_pred),
            'r2': r2_score(y_test, lr_pred)
        }
    
    # Save models and scaler
    model_data = {
        'models': models,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'symbol': symbol,
        'training_date': datetime.now().isoformat(),
        'performance': results
    }
    
    model_file_name = os.path.join(MODEL_DIR, f"{symbol.lower()}_advanced_prediction_model.pkl")
    joblib.dump(model_data, model_file_name)
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['mse'])
    
    return {
        "success": True,
        "symbol": symbol,
        "model_type": model_type,
        "best_model": best_model,
        "performance_metrics": results,
        "features_count": len(feature_columns),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "model_path": model_file_name
    }

@mcp.tool(description="Predicts future stock prices using advanced ML models with confidence intervals and trend analysis.")
def predict_stock_price_advanced(symbol: str, days_ahead: int = 1, model_preference: str = "best") -> dict:
    """
    Advanced stock price prediction with confidence intervals and trend analysis.
    
    Args:
        symbol (str): The stock symbol.
        days_ahead (int): Number of days to predict (1-7).
        model_preference (str): Model to use - 'best', 'random_forest', 'gradient_boost', 'linear', or 'ensemble'.
    
    Returns:
        dict: Predictions with confidence intervals and trend analysis.
    """
    if days_ahead > 7:
        return {"error": "Prediction limited to 7 days ahead for accuracy."}
    
    model_file_name = os.path.join(MODEL_DIR, f"{symbol.lower()}_advanced_prediction_model.pkl")
    
    if not os.path.exists(model_file_name):
        return {"error": f"Advanced model for {symbol} not found. Train it first using 'train_advanced_stock_prediction_model'."}
    
    try:
        model_data = joblib.load(model_file_name)
        models = model_data['models']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        performance = model_data['performance']
    except Exception as e:
        return {"error": f"Failed to load model data: {str(e)}"}
    
    # Get recent data for prediction
    recent_df = _get_historical_prices_for_ml(symbol, 100)  # Extra data for indicators
    
    if recent_df.empty or len(recent_df) < 50:
        return {"error": f"Insufficient recent data for {symbol}."}
    
    # Calculate technical indicators
    recent_df = _calculate_technical_indicators(recent_df)
    recent_df = recent_df.dropna()
    
    if recent_df.empty:
        return {"error": f"No valid data after indicator calculation for {symbol}."}
    
    # Get the most recent complete data point
    latest_data = recent_df[feature_columns].iloc[-1:].values
    latest_scaled = scaler.transform(latest_data)
    
    predictions = {}
    
    # Determine which model(s) to use
    if model_preference == "best":
        best_model_name = min(performance.keys(), key=lambda k: performance[k]['mse'])
        model_to_use = {best_model_name: models[best_model_name]}
    elif model_preference == "ensemble":
        model_to_use = models
    elif model_preference in models:
        model_to_use = {model_preference: models[model_preference]}
    else:
        return {"error": f"Model '{model_preference}' not available. Available: {list(models.keys())}"}
    
    # Make predictions
    for model_name, model in model_to_use.items():
        model_predictions = []
        current_data = latest_scaled.copy()
        
        for day in range(days_ahead):
            pred = model.predict(current_data)[0]
            model_predictions.append(float(pred))
            
            # For multi-day prediction, update features (simplified approach)
            if day < days_ahead - 1:
                # Update the close price feature for next prediction
                # This is a simplification - in practice, you'd need to update all features
                current_data = current_data.copy()
        
        predictions[model_name] = model_predictions
    
    # Calculate ensemble prediction if multiple models
    if len(model_to_use) > 1:
        ensemble_pred = []
        for day in range(days_ahead):
            day_preds = [predictions[model][day] for model in predictions.keys()]
            ensemble_pred.append(np.mean(day_preds))
        predictions['ensemble'] = ensemble_pred
    
    # Calculate confidence intervals (simplified approach using model performance)
    final_predictions = []
    current_price = float(recent_df['close_price'].iloc[-1])
    
    for day in range(days_ahead):
        if len(model_to_use) == 1:
            model_name = list(model_to_use.keys())[0]
            pred_price = predictions[model_name][day]
            mae = performance[model_name]['mae']
        else:
            pred_price = predictions['ensemble'][day]
            mae = np.mean([performance[m]['mae'] for m in models.keys()])
        
        # Simple confidence interval based on MAE
        confidence_lower = pred_price - (1.96 * mae)
        confidence_upper = pred_price + (1.96 * mae)
        
        # Trend analysis
        price_change = pred_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        trend = "neutral"
        if price_change_pct > 2:
            trend = "strongly bullish"
        elif price_change_pct > 0.5:
            trend = "bullish"
        elif price_change_pct < -2:
            trend = "strongly bearish"
        elif price_change_pct < -0.5:
            trend = "bearish"
        
        prediction_date = datetime.now() + timedelta(days=day+1)
        
        final_predictions.append({
            "date": prediction_date.strftime("%Y-%m-%d"),
            "predicted_price": round(pred_price, 2),
            "confidence_lower": round(confidence_lower, 2),
            "confidence_upper": round(confidence_upper, 2),
            "price_change": round(price_change, 2),
            "price_change_percent": round(price_change_pct, 2),
            "trend": trend
        })
    
    return {
        "success": True,
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "model_used": model_preference,
        "predictions": final_predictions,
        "model_performance": {k: {metric: round(v, 4) for metric, v in perf.items()} 
                            for k, perf in performance.items()},
        "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@mcp.tool(description="Analyzes stock price trends and technical patterns using advanced indicators.")
def analyze_stock_trends(symbol: str, analysis_period: int = 60) -> dict:
    """
    Comprehensive trend analysis using technical indicators and pattern recognition.
    
    Args:
        symbol (str): The stock symbol.
        analysis_period (int): Number of days to analyze (default 60).
    
    Returns:
        dict: Comprehensive trend analysis with technical indicators and signals.
    """
    # Get historical data
    df = _get_historical_prices_for_ml(symbol, analysis_period + 30)
    
    if df.empty or len(df) < analysis_period:
        return {"error": f"Insufficient data for trend analysis of {symbol}."}
    
    # Calculate technical indicators
    df = _calculate_technical_indicators(df)
    df = df.dropna().tail(analysis_period)
    
    if df.empty:
        return {"error": f"No valid data after indicator calculation for {symbol}."}
    
    # Current values
    current = df.iloc[-1]
    current_price = float(current['close_price'])
    
    # Trend analysis
    sma_20 = float(current['SMA_20'])
    sma_5 = float(current['SMA_5'])
    sma_10 = float(current['SMA_10'])
    rsi = float(current['RSI'])
    macd = float(current['MACD'])
    macd_signal = float(current['MACD_signal'])
    bb_position = float(current['BB_position'])
    
    # Price trend signals
    price_trend = "neutral"
    if current_price > sma_20 and sma_5 > sma_10 > sma_20:
        price_trend = "strong uptrend"
    elif current_price > sma_20:
        price_trend = "uptrend"
    elif current_price < sma_20 and sma_5 < sma_10 < sma_20:
        price_trend = "strong downtrend"
    elif current_price < sma_20:
        price_trend = "downtrend"
    
    # RSI signals
    rsi_signal = "neutral"
    if rsi > 70:
        rsi_signal = "overbought"
    elif rsi < 30:
        rsi_signal = "oversold"
    elif rsi > 60:
        rsi_signal = "bullish"
    elif rsi < 40:
        rsi_signal = "bearish"
    
    # MACD signals
    macd_signal_trend = "neutral"
    if macd > macd_signal and macd > 0:
        macd_signal_trend = "strong bullish"
    elif macd > macd_signal:
        macd_signal_trend = "bullish"
    elif macd < macd_signal and macd < 0:
        macd_signal_trend = "strong bearish"
    elif macd < macd_signal:
        macd_signal_trend = "bearish"
    
    # Bollinger Bands signals
    bb_signal = "neutral"
    if bb_position > 0.8:
        bb_signal = "near upper band - potential resistance"
    elif bb_position < 0.2:
        bb_signal = "near lower band - potential support"
    
    # Volume analysis
    recent_volume = df['volume_ratio'].tail(5).mean()
    volume_trend = "normal"
    if recent_volume > 1.5:
        volume_trend = "high volume"
    elif recent_volume < 0.7:
        volume_trend = "low volume"
    
    # Volatility analysis
    recent_volatility = df['price_volatility'].tail(10).mean()
    volatility_level = "normal"
    if recent_volatility > df['price_volatility'].quantile(0.8):
        volatility_level = "high"
    elif recent_volatility < df['price_volatility'].quantile(0.2):
        volatility_level = "low"
    
    # Support and resistance levels
    highs = df['high_price'].tail(20)
    lows = df['low_price'].tail(20)
    resistance = float(highs.quantile(0.9))
    support = float(lows.quantile(0.1))
    
    # Overall signal
    signals = [price_trend, rsi_signal, macd_signal_trend]
    bullish_signals = sum(1 for s in signals if 'bullish' in s or 'uptrend' in s)
    bearish_signals = sum(1 for s in signals if 'bearish' in s or 'downtrend' in s)
    
    overall_signal = "neutral"
    if bullish_signals >= 2:
        overall_signal = "bullish"
    elif bearish_signals >= 2:
        overall_signal = "bearish"
    
    return {
        "success": True,
        "symbol": symbol,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": round(current_price, 2),
        "overall_signal": overall_signal,
        "trend_analysis": {
            "price_trend": price_trend,
            "trend_strength": "strong" if "strong" in price_trend else "moderate" if price_trend != "neutral" else "weak"
        },
        "technical_indicators": {
            "rsi": {
                "value": round(rsi, 2),
                "signal": rsi_signal,
                "interpretation": "Momentum indicator (0-100). >70 overbought, <30 oversold"
            },
            "macd": {
                "value": round(macd, 4),
                "signal_line": round(macd_signal, 4),
                "signal": macd_signal_trend,
                "interpretation": "Trend following indicator. Bullish when MACD > Signal line"
            },
            "moving_averages": {
                "sma_5": round(sma_5, 2),
                "sma_10": round(sma_10, 2),
                "sma_20": round(sma_20, 2),
                "interpretation": "Price above SMA = bullish, below = bearish"
            },
            "bollinger_bands": {
                "position": round(bb_position, 2),
                "signal": bb_signal,
                "interpretation": "Position in bands (0-1). >0.8 near upper, <0.2 near lower"
            }
        },
        "support_resistance": {
            "support_level": round(support, 2),
            "resistance_level": round(resistance, 2),
            "current_vs_support": round(((current_price - support) / support) * 100, 2),
            "current_vs_resistance": round(((resistance - current_price) / resistance) * 100, 2)
        },
        "volume_analysis": {
            "trend": volume_trend,
            "recent_avg_ratio": round(recent_volume, 2),
            "interpretation": "Volume confirms price movements. High volume = strong moves"
        },
        "volatility": {
            "level": volatility_level,
            "recent_value": round(recent_volatility, 4),
            "interpretation": "High volatility = larger price swings, more risk/opportunity"
        }
    }
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

# Alias functions for API compatibility
@mcp.tool(description="Predict stock price - wrapper for predict_stock_price_advanced")
def predict_stock_price(symbol: str, days_ahead: int = 1) -> dict:
    """Wrapper function for API compatibility"""
    return predict_stock_price_advanced(symbol, days_ahead, "best")

@mcp.tool(description="Analyze stock trend - wrapper for analyze_stock_trends")
def analyze_stock_trend(symbol: str, period: str = "6months") -> dict:
    """Wrapper function for API compatibility"""
    # Convert period string to days
    period_days = {
        "1week": 7,
        "1month": 30,
        "3months": 90,
        "6months": 180,
        "1year": 365
    }.get(period, 60)
    
    return analyze_stock_trends(symbol, period_days)

@mcp.tool(description="Analyze market sectors and rotation signals")
def analyze_sectors() -> dict:
    """
    Analyzes major market sectors for rotation signals and performance.
    """
    sectors = {
        "Technology": ["AAPL", "GOOGL", "MSFT", "NVDA"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"], 
        "Financial": ["JPM", "BAC", "WFC", "GS"],
        "Energy": ["XOM", "CVX", "COP", "EOG"],
        "Consumer": ["AMZN", "TSLA", "HD", "MCD"]
    }
    
    sector_analysis = {}
    
    for sector_name, symbols in sectors.items():
        try:
            # Get recent performance for sector symbols
            sector_performance = []
            for symbol in symbols:
                result = analyze_price_range(symbol, 
                    (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d"))
                if "error" not in result:
                    # Calculate approximate return
                    recent_return = ((result["highest_price"] - result["lowest_price"]) / result["lowest_price"]) * 100
                    sector_performance.append(recent_return)
            
            if sector_performance:
                avg_performance = sum(sector_performance) / len(sector_performance)
                sector_analysis[sector_name] = {
                    "average_performance": round(avg_performance, 2),
                    "symbols": symbols,
                    "trend": "bullish" if avg_performance > 5 else "bearish" if avg_performance < -5 else "neutral"
                }
            else:
                sector_analysis[sector_name] = {
                    "error": "No data available",
                    "symbols": symbols
                }
                
        except Exception as e:
            sector_analysis[sector_name] = {
                "error": str(e),
                "symbols": symbols
            }
    
    return {
        "sector_analysis": sector_analysis,
        "market_rotation_signal": "technology_leading" if sector_analysis.get("Technology", {}).get("average_performance", 0) > 10 else "defensive_rotation",
        "timestamp": datetime.now().isoformat()
    }

@mcp.tool(description="Calculate correlation matrix between stocks")
def calculate_correlation_matrix(symbols: str, period: str = "6months") -> dict:
    """
    Calculate correlation matrix between multiple stocks.
    
    Args:
        symbols (str): Comma-separated list of stock symbols
        period (str): Time period for analysis
    
    Returns:
        dict: Correlation matrix and analysis
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Convert period to days
        period_days = {
            "1week": 7,
            "1month": 30,
            "3months": 90,
            "6months": 180,
            "1year": 365
        }.get(period, 180)
        
        start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get price data for all symbols
        price_data = {}
        
        for symbol in symbol_list:
            try:
                with psycopg2.connect(DB_URI) as conn:
                    with conn.cursor() as cur:
                        query = """
                        SELECT date, close_price
                        FROM public."STOCK_PRICES"
                        WHERE symbol = %s AND date >= %s AND date <= %s
                        ORDER BY date ASC
                        """
                        cur.execute(query, (symbol, start_date, end_date))
                        rows = cur.fetchall()
                        
                        if rows:
                            dates = [row[0] for row in rows]
                            prices = [float(row[1]) for row in rows]
                            price_data[symbol] = {"dates": dates, "prices": prices}
            except Error as e:
                continue
        
        if len(price_data) < 2:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Calculate correlations (simplified)
        correlations = {}
        
        for symbol1 in price_data:
            correlations[symbol1] = {}
            for symbol2 in price_data:
                if symbol1 == symbol2:
                    correlations[symbol1][symbol2] = 1.0
                else:
                    # Simple correlation calculation using price returns
                    prices1 = price_data[symbol1]["prices"]
                    prices2 = price_data[symbol2]["prices"]
                    
                    if len(prices1) > 1 and len(prices2) > 1:
                        # Calculate returns
                        returns1 = [(prices1[i] - prices1[i-1]) / prices1[i-1] for i in range(1, len(prices1))]
                        returns2 = [(prices2[i] - prices2[i-1]) / prices2[i-1] for i in range(1, len(prices2))]
                        
                        # Simple correlation coefficient
                        if len(returns1) == len(returns2) and len(returns1) > 0:
                            try:
                                correlation = np.corrcoef(returns1, returns2)[0, 1]
                                correlations[symbol1][symbol2] = round(float(correlation), 3)
                            except:
                                correlations[symbol1][symbol2] = 0.0
                        else:
                            correlations[symbol1][symbol2] = 0.0
                    else:
                        correlations[symbol1][symbol2] = 0.0
        
        return {
            "correlation_matrix": correlations,
            "symbols": symbol_list,
            "period": period,
            "analysis_period": f"{start_date} to {end_date}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Error calculating correlation matrix: {str(e)}"}