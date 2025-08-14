"""
Finance Helper Functions
Comprehensive helper functions for all finance MCP servers including:
- Database operations
- Symbol discovery 
- ML model training and prediction
- Data ingestion and updating
- Common calculations
"""

import os
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import requests
import json

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)
ALPHA_VANTAGE_KEY = os.getenv("EXTERNAL_FINANCE_API_KEY", None)
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", None)
NEWS_API_KEY = os.getenv("NEWSAPI_KEY", None)  # Fixed: using NEWSAPI_KEY from .env

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

# Define a path to save/load the trained model
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def get_database_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(DB_URI)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_company_info_helper(symbol: str) -> dict:
    """Get company information from database"""
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT symbol, asset_type, name, description FROM public.company WHERE symbol = %s", 
                    (symbol.upper(),)
                )
                row = cur.fetchone()
                if row is None:
                    return {
                        "success": False,
                        "error": "Company not found",
                        "symbol": symbol.upper()
                    }
                return {
                    "success": True,
                    "symbol": row[0],
                    "asset_type": row[1],
                    "name": row[2],
                    "description": row[3],
                }
    except Error as e:
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol.upper()
        }

def search_companies_helper(query: str, limit: int = 10) -> dict:
    """Search companies in database with intelligent ranking and flexible matching"""
    try:
        limit = min(limit, 50)
        query_clean = query.strip()
        
        # Handle empty query
        if not query_clean:
            return {
                "success": False,
                "error": "Query cannot be empty",
                "query": query,
                "results_found": 0,
                "companies": []
            }
        
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # Multiple search patterns for flexibility
                search_patterns = [
                    f"%{query_clean}%",  # Contains query
                    f"{query_clean}%",   # Starts with query
                    f"%{query_clean}",   # Ends with query
                ]
                
                # Enhanced query with multiple search strategies
                results_query = """
                SELECT symbol, asset_type, name, description,
                    CASE 
                        WHEN UPPER(symbol) = UPPER(%s) THEN 1
                        WHEN UPPER(symbol) LIKE UPPER(%s) THEN 2
                        WHEN UPPER(name) LIKE UPPER(%s) THEN 3
                        WHEN UPPER(symbol) LIKE UPPER(%s) THEN 4
                        WHEN UPPER(name) LIKE UPPER(%s) THEN 5
                        WHEN UPPER(description) LIKE UPPER(%s) THEN 6
                        ELSE 7
                    END as rank_score
                FROM public.company 
                WHERE UPPER(symbol) = UPPER(%s)
                   OR UPPER(symbol) LIKE UPPER(%s) 
                   OR UPPER(name) LIKE UPPER(%s)
                   OR UPPER(description) LIKE UPPER(%s)
                   OR UPPER(symbol) LIKE UPPER(%s)
                   OR UPPER(name) LIKE UPPER(%s)
                ORDER BY rank_score, symbol
                LIMIT %s
                """
                
                # Execute with all pattern variations
                cur.execute(results_query, (
                    query_clean,           # Exact symbol match
                    f"{query_clean}%",     # Symbol starts with
                    f"%{query_clean}%",    # Name contains
                    f"%{query_clean}%",    # Symbol contains  
                    f"%{query_clean}%",    # Name contains
                    f"%{query_clean}%",    # Description contains
                    query_clean,           # WHERE: Exact symbol
                    f"{query_clean}%",     # WHERE: Symbol starts with
                    f"%{query_clean}%",    # WHERE: Name contains
                    f"%{query_clean}%",    # WHERE: Description contains
                    f"%{query_clean}%",    # WHERE: Symbol contains
                    f"%{query_clean}%",    # WHERE: Name contains
                    limit
                ))
                results = cur.fetchall()
                
                companies = []
                for row in results:
                    companies.append({
                        "symbol": row[0],
                        "asset_type": row[1],
                        "name": row[2],
                        "description": row[3],
                        "match_score": row[4]
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "results_found": len(companies),
                    "companies": companies
                }
                
    except Error as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

def get_latest_stock_price_helper(symbol: str) -> dict:
    """Get latest stock price from database"""
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cursor:
                query = """
                SELECT symbol, date, open_price, high_price, low_price, 
                       close_price, adjusted_close, volume
                FROM public.stock_price
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1
                """

                cursor.execute(query, (symbol.upper(),))
                result = cursor.fetchone()

                if result:
                    return {
                        "success": True,
                        "symbol": result[0],
                        "date": result[1].isoformat() if isinstance(result[1], datetime) else str(result[1]),
                        "open_price": float(result[2]),
                        "high_price": float(result[3]),
                        "low_price": float(result[4]),
                        "close_price": float(result[5]),
                        "adjusted_close": float(result[6]),
                        "volume": int(result[7])
                    }
                else:
                    return {
                        "success": False,
                        "error": f"No price data found for symbol {symbol}",
                        "symbol": symbol.upper()
                    }
    except Error as e:
        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "symbol": symbol.upper()
        }

def get_historical_prices_helper(symbol: str, days: int = 100) -> pd.DataFrame:
    """Get historical prices as DataFrame for calculations"""
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM public.stock_price
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
                
                # Convert all numeric columns to float to handle Decimal types from database
                numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.sort_values('date').set_index('date')
                return df
    except Error as e:
        print(f"Database error in get_historical_prices_helper: {e}")
        return pd.DataFrame()

# ============================================================================
# SYMBOL DISCOVERY AND DATA UPDATE HELPERS
# ============================================================================

def discover_symbol_helper(query: str) -> dict:
    """Discover symbol information using external APIs if not in database"""
    # First check database
    db_result = search_companies_helper(query, 1)
    if db_result.get("success") and db_result.get("companies"):
        return {
            "success": True,
            "symbol": db_result["companies"][0]["symbol"],
            "source": "database",
            "data": db_result["companies"][0]
        }
    
    # Try external APIs for discovery
    if ALPHA_VANTAGE_KEY:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": query,
                "apikey": ALPHA_VANTAGE_KEY
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if "bestMatches" in data and data["bestMatches"]:
                match = data["bestMatches"][0]
                return {
                    "success": True,
                    "symbol": match.get("1. symbol", query.upper()),
                    "source": "alpha_vantage",
                    "data": {
                        "symbol": match.get("1. symbol", query.upper()),
                        "name": match.get("2. name", "Unknown"),
                        "type": match.get("3. type", "stock"),
                        "region": match.get("4. region", "Unknown"),
                        "currency": match.get("8. currency", "USD")
                    }
                }
        except Exception as e:
            print(f"Alpha Vantage symbol search error: {e}")
    
    return {
        "success": False,
        "error": f"Symbol {query} not found",
        "symbol": query.upper()
    }

def update_stock_price_helper(symbol: str, force_update: bool = False) -> dict:
    """Update stock price data if outdated or missing - with proper date checking"""
    try:
        # First check if we have any data for this symbol
        try:
            with psycopg2.connect(DB_URI) as conn:
                with conn.cursor() as cur:
                    # Get latest date for this symbol
                    cur.execute("""
                        SELECT MAX(date) as latest_date, COUNT(*) as record_count 
                        FROM public.stock_price 
                        WHERE symbol = %s
                    """, (symbol.upper(),))
                    result = cur.fetchone()
                    
                    latest_date = result[0] if result else None
                    record_count = result[1] if result else 0
                    
                    print(f"Symbol {symbol}: {record_count} records, Latest date: {latest_date}")
                    
                    # Check if we have recent data
                    if latest_date and not force_update:
                        days_old = (datetime.now().date() - latest_date).days
                        if days_old <= 1:
                            return {
                                "success": True,
                                "action": "no_update_needed",
                                "latest_date": str(latest_date),
                                "message": f"✅ Database already has the latest data for {symbol.upper()}. Last update: {latest_date} ({days_old} day(s) ago)",
                                "symbol": symbol.upper(),
                                "records_updated": 0
                            }
                        elif record_count >= 500 and days_old <= 3:
                            return {
                                "success": True,
                                "action": "sufficient_data",
                                "latest_date": str(latest_date),
                                "record_count": record_count,
                                "message": f"✅ Database already has sufficient data for {symbol.upper()}. {record_count:,} records available, latest from {days_old} day(s) ago",
                                "symbol": symbol.upper(),
                                "records_updated": 0
                            }
        except Exception as db_error:
            print(f"Database check error: {db_error}")
            return {
                "success": False,
                "error": f"Database check failed: {str(db_error)}",
                "symbol": symbol.upper()
            }

        # If we reach here, we need to update data
        if not ALPHA_VANTAGE_KEY:
            return {
                "success": False,
                "error": "No Alpha Vantage API key available for updates. Please configure ALPHA_VANTAGE_KEY in .env file",
                "symbol": symbol.upper()
            }

        # Try to fetch new data from Alpha Vantage
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_KEY,
                "outputsize": "compact"
            }
            
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            # Debug: Check API response structure
            print(f"Alpha Vantage response keys: {list(data.keys())}")
            
            # Handle API errors and rate limits
            if "Error Message" in data:
                return {
                    "success": False,
                    "error": f"Alpha Vantage API error: {data['Error Message']}",
                    "symbol": symbol.upper()
                }
            
            if "Note" in data:
                return {
                    "success": False,
                    "error": "Alpha Vantage API rate limit reached. Please try again later.",
                    "symbol": symbol.upper()
                }
            
            if "Time Series (Daily)" not in data:
                return {
                    "success": False,
                    "error": f"No time series data found in API response. Available keys: {list(data.keys())}",
                    "symbol": symbol.upper()
                }

            time_series = data["Time Series (Daily)"]
            
            # Insert new data into database
            with psycopg2.connect(DB_URI) as conn:
                with conn.cursor() as cur:
                    insert_count = 0
                    skipped_count = 0
                    
                    for date_str, price_data in time_series.items():
                        try:
                            # Get all price values
                            open_price = price_data.get("1. open")
                            high_price = price_data.get("2. high")
                            low_price = price_data.get("3. low")
                            close_price = price_data.get("4. close")
                            volume = price_data.get("5. volume")
                            
                            # Check for None values - skip if any are missing
                            if any(val is None for val in [open_price, high_price, low_price, close_price, volume]):
                                print(f"Skipping {date_str}: missing data values")
                                skipped_count += 1
                                continue
                            
                            # Convert to float/int with error handling
                            try:
                                open_val = float(str(open_price).strip())
                                high_val = float(str(high_price).strip())
                                low_val = float(str(low_price).strip())
                                close_val = float(str(close_price).strip())
                                volume_val = int(float(str(volume).strip()))
                                
                                # Validate that the numbers are reasonable
                                if any(val <= 0 for val in [open_val, high_val, low_val, close_val, volume_val]):
                                    print(f"Skipping {date_str}: invalid values (negative or zero)")
                                    skipped_count += 1
                                    continue
                                    
                            except (ValueError, TypeError) as conv_error:
                                print(f"Skipping {date_str}: conversion error - {conv_error}")
                                skipped_count += 1
                                continue
                            
                            # Insert into database (ON CONFLICT DO NOTHING prevents duplicates)
                            cur.execute("""
                                INSERT INTO public.stock_price 
                                (symbol, date, open_price, high_price, low_price, close_price, volume)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (symbol, date) DO NOTHING
                            """, (
                                symbol.upper(),
                                date_str,
                                open_val,
                                high_val,
                                low_val,
                                close_val,
                                volume_val
                            ))
                            
                            if cur.rowcount > 0:
                                insert_count += 1
                                
                        except Exception as e:
                            print(f"Error processing data for {date_str}: {e}")
                            skipped_count += 1
                            continue
                    
                    conn.commit()
                    
                    if insert_count > 0:
                        return {
                            "success": True,
                            "action": "updated",
                            "records_updated": insert_count,
                            "records_skipped": skipped_count,
                            "message": f"✅ Successfully added {insert_count} new record(s) to database for {symbol.upper()}",
                            "symbol": symbol.upper(),
                            "data_source": "Alpha Vantage API"
                        }
                    else:
                        return {
                            "success": True,
                            "action": "no_new_data",
                            "records_updated": 0,
                            "records_skipped": skipped_count,
                            "message": f"✅ Database already has all available data for {symbol.upper()}. No new records added",
                            "symbol": symbol.upper(),
                            "data_source": "Alpha Vantage API"
                        }
                        
        except requests.RequestException as req_error:
            return {
                "success": False,
                "error": f"Network error accessing Alpha Vantage: {str(req_error)}",
                "symbol": symbol.upper()
            }
        except Exception as api_error:
            return {
                "success": False,
                "error": f"API processing error: {str(api_error)}",
                "symbol": symbol.upper()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Update operation failed: {str(e)}",
            "symbol": symbol.upper()
        }

# ============================================================================
# MACHINE LEARNING HELPERS
# ============================================================================

def train_prediction_model_helper(symbol: str, lookback_days: int = 100) -> dict:
    """Train ML model for price prediction"""
    df = get_historical_prices_helper(symbol, lookback_days + 50)
    if df.empty or len(df) < (lookback_days + 20):
        return {"error": f"Insufficient data for {symbol}. Need at least {lookback_days + 20} days."}

    # Calculate basic indicators
    df['SMA_20'] = df['close_price'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['close_price'].rolling(window=50, min_periods=1).mean()
    df['price_change'] = df['close_price'].pct_change()
    df['volatility'] = df['price_change'].rolling(window=10, min_periods=1).std()
    
    # Target variable
    df['target'] = df['close_price'].shift(-1)
    
    # Features
    feature_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'SMA_20', 'SMA_50', 'price_change', 'volatility']
    
    df = df.dropna()
    if len(df) < 50:
        return {"error": f"Insufficient clean data for {symbol} after processing."}

    X = df[feature_columns].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return {
        "success": True,
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

def predict_price_helper(symbol: str, days_ahead: int = 1) -> dict:
    """Predict future price using trained model"""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")

    # Train model if it doesn't exist
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        train_result = train_prediction_model_helper(symbol)
        if "error" in train_result:
            return train_result

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        df = get_historical_prices_helper(symbol, 60)
        if df.empty:
            return {"error": f"No recent data available for {symbol}."}

        # Calculate same indicators as training
        df['SMA_20'] = df['close_price'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['close_price'].rolling(window=50, min_periods=1).mean()
        df['price_change'] = df['close_price'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=10, min_periods=1).std()
        
        df = df.dropna()
        if len(df) < 1:
            return {"error": f"Insufficient recent data for {symbol}."}

        feature_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'SMA_20', 'SMA_50', 'price_change', 'volatility']
        latest_features = df[feature_columns].iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)

        predictions = []
        current_price = df['close_price'].iloc[-1]
        
        # Handle None current_price safely
        if current_price is None or pd.isna(current_price):
            current_price = 0.0

        for day in range(1, min(days_ahead + 1, 8)):
            pred_price = model.predict(latest_features_scaled)[0]
            confidence = min(90, max(50, 80 - (day - 1) * 5))
            
            # Avoid division by zero
            price_change = 0.0
            if current_price > 0:
                price_change = round(((pred_price - current_price) / current_price) * 100, 2)

            predictions.append({
                "day": day,
                "predicted_price": round(pred_price, 2),
                "confidence": confidence,
                "price_change": price_change
            })

        return {
            "success": True,
            "symbol": symbol,
            "current_price": round(float(current_price), 2),
            "predictions": predictions,
            "model_type": "random_forest",
            "prediction_date": datetime.now().isoformat()
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ============================================================================
# CALCULATION HELPERS
# ============================================================================

def calculate_rsi_helper(symbol: str, period: int = 14) -> dict:
    """Calculate RSI for a symbol"""
    df = get_historical_prices_helper(symbol, period + 20)
    if df.empty or len(df) < period + 1:
        return {"error": f"Insufficient data for RSI calculation of {symbol}"}

    prices = df['close_price']
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[-1]
    
    return {
        "success": True,
        "symbol": symbol,
        "rsi": round(float(current_rsi), 2),
        "period": period,
        "interpretation": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral",
        "calculation_date": datetime.now().isoformat()
    }

def calculate_sma_helper(symbol: str, period: int = 20) -> dict:
    """Calculate Simple Moving Average for a symbol"""
    df = get_historical_prices_helper(symbol, period + 10)
    if df.empty or len(df) < period:
        return {"error": f"Insufficient data for SMA calculation of {symbol}"}

    sma = df['close_price'].rolling(window=period).mean()
    current_sma = sma.iloc[-1]
    current_price = df['close_price'].iloc[-1]
    
    # Handle None values safely
    if current_sma is None or pd.isna(current_sma):
        current_sma = 0.0
    if current_price is None or pd.isna(current_price):
        current_price = 0.0
    
    return {
        "success": True,
        "symbol": symbol,
        "sma": round(float(current_sma), 2),
        "current_price": round(float(current_price), 2),
        "period": period,
        "price_vs_sma": "above" if current_price > current_sma else "below",
        "calculation_date": datetime.now().isoformat()
    }

# ============================================================================
# NEWS AND EXTERNAL DATA HELPERS
# ============================================================================

def get_financial_news_helper(query: str = "financial markets", limit: int = 10) -> dict:
    """Get financial news from external APIs"""
    if not NEWS_API_KEY:
        return {"error": "News API key not available"}
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 20)
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") == "ok":
            articles = []
            for article in data.get("articles", []):
                if article.get("title") and article.get("description"):
                    articles.append({
                        "title": article["title"],
                        "description": article["description"],
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", "Unknown")
                    })
            
            return {
                "success": True,
                "query": query,
                "articles_found": len(articles),
                "articles": articles,
                "retrieved_at": datetime.now().isoformat()
            }
        else:
            return {"error": f"News API error: {data.get('message', 'Unknown error')}"}
            
    except Exception as e:
        return {"error": f"Failed to fetch news: {str(e)}"}

# ============================================================================
# STOCK PRICE HELPER FUNCTIONS
# ============================================================================

def get_historical_stock_prices_helper(symbol: str, start_date: str = None, end_date: str = None, limit: int = 100) -> dict:
    """Helper function to get historical stock prices"""
    try:
        conn = get_database_connection()
        if not conn:
            return {"success": False, "error": "Database connection failed"}
        
        with conn.cursor() as cur:
            # Build query with optional date filtering
            query = """
                SELECT date, open_price, high_price, low_price, close_price, volume 
                FROM public.stock_price 
                WHERE UPPER(symbol) = UPPER(%s)
            """
            params = [symbol]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
                
            query += " ORDER BY date DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            results = cur.fetchall()
            
            if not results:
                return {
                    "success": False,
                    "error": f"No price data found for symbol {symbol.upper()}",
                    "records_returned": 0
                }
            
            # Convert to list of dictionaries
            prices = []
            for row in results:
                prices.append({
                    "date": row[0].strftime("%Y-%m-%d") if row[0] else None,
                    "open": float(row[1]) if row[1] else None,
                    "high": float(row[2]) if row[2] else None,
                    "low": float(row[3]) if row[3] else None,
                    "close": float(row[4]) if row[4] else None,
                    "volume": int(row[5]) if row[5] else None
                })
            
            # Calculate basic statistics
            closes = [p["close"] for p in prices if p["close"]]
            if closes:
                stats = {
                    "current_price": closes[0],  # Most recent
                    "period_high": max(closes),
                    "period_low": min(closes),
                    "period_average": sum(closes) / len(closes)
                }
            else:
                stats = {}
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "records_returned": len(prices),
                "date_range": {
                    "from": prices[-1]["date"] if prices else None,
                    "to": prices[0]["date"] if prices else None
                },
                "statistics": stats,
                "prices": prices
            }
        
    except Exception as e:
        return {"success": False, "error": f"Error retrieving prices: {str(e)}"}
    finally:
        if conn:
            conn.close()


def get_stock_prices_dataframe(symbol: str, days: int = 100) -> pd.DataFrame:
    """
    Get historical stock prices as DataFrame - Uses the same working pattern as ML predictions
    This is the universal function that works reliably for all MCP servers
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM public.stock_price
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
                # Convert all numeric columns to float to avoid decimal/float mixing
                numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

                # Also add simplified column names for backward compatibility
                df['open'] = df['open_price']
                df['high'] = df['high_price'] 
                df['low'] = df['low_price']
                df['close'] = df['close_price']

                df = df.sort_values('date').reset_index(drop=True)
                return df
    except Exception as e:
        print(f"Error in get_stock_prices_dataframe: {e}")
        return pd.DataFrame()
        

def update_stock_prices_helper(symbols: list) -> dict:
    """Helper function to update stock prices from external sources"""
    try:
        if not ALPHA_VANTAGE_KEY:
            return {
                "success": False,
                "error": "External API key not configured",
                "total_symbols": len(symbols)
            }
        
        updated_count = 0
        failed_count = 0
        results = []
        
        for symbol in symbols:
            try:
                # Fetch data from Alpha Vantage
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "apikey": ALPHA_VANTAGE_KEY,
                    "outputsize": "compact"
                }
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if "Time Series (Daily)" in data:
                    time_series = data["Time Series (Daily)"]
                    
                    # Get the most recent date
                    latest_date = max(time_series.keys())
                    latest_data = time_series[latest_date]
                    
                    # Update database
                    conn = get_database_connection()
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO public.stock_price (symbol, date, open_price, high_price, low_price, close_price, volume)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (symbol, date) DO UPDATE SET
                                    open_price = EXCLUDED.open_price,
                                    high_price = EXCLUDED.high_price,
                                    low_price = EXCLUDED.low_price,
                                    close_price = EXCLUDED.close_price,
                                    volume = EXCLUDED.volume
                            """, (
                                symbol.upper(),
                                latest_date,
                                float(latest_data["1. open"]),
                                float(latest_data["2. high"]),
                                float(latest_data["3. low"]),
                                float(latest_data["4. close"]),
                                int(latest_data["5. volume"])
                            ))
                            conn.commit()
                        conn.close()
                    
                    updated_count += 1
                    results.append({
                        "symbol": symbol.upper(),
                        "status": "updated",
                        "date": latest_date,
                        "close": float(latest_data["4. close"])
                    })
                else:
                    failed_count += 1
                    results.append({
                        "symbol": symbol.upper(),
                        "status": "failed",
                        "error": "No data returned from API"
                    })
                    
            except Exception as e:
                failed_count += 1
                results.append({
                    "symbol": symbol.upper(),
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_symbols": len(symbols),
            "updated_count": updated_count,
            "failed_count": failed_count,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Bulk update failed: {str(e)}",
            "total_symbols": len(symbols)
        }
