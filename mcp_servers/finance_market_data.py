import os
from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import time
import asyncio
from typing import Dict, List, Optional

# Import symbol discovery system
try:
    from .finance_symbol_discovery import (
        get_or_discover_symbol, 
        get_cached_or_fresh_data, 
        cache_price_data,
        cleanup_expired_cache
    )
    DISCOVERY_AVAILABLE = True
except ImportError:
    print("⚠️ Symbol discovery system not available")
    DISCOVERY_AVAILABLE = False

load_dotenv()

# API configurations
ALPHA_VANTAGE_KEY = os.getenv("EXTERNAL_FINANCE_API_KEY", None)
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", None)

# API endpoints
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
FINNHUB_URL = "https://finnhub.io/api/v1"
YAHOO_FINANCE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

# Priority order for data sources (higher priority = tried first)
DATA_SOURCE_PRIORITY = {
    "alpha_vantage": {"priority": 1, "available": bool(ALPHA_VANTAGE_KEY)},
    "finnhub": {"priority": 2, "available": bool(FINNHUB_KEY)},
    "yahoo_finance": {"priority": 3, "available": True},  # No API key required
    "sample_data": {"priority": 4, "available": True}    # Always available fallback
}

# Enhanced market symbols mapping with proper symbols for each API
MARKET_SYMBOLS = {
    # Stock Indices
    "nasdaq": {
        "alpha_vantage": "QQQ",  # NASDAQ ETF as proxy
        "yahoo": "^IXIC", 
        "finnhub": "QQQ",
        "name": "NASDAQ Composite",
        "type": "index"
    },
    "sp500": {
        "alpha_vantage": "SPY",  # S&P 500 ETF as proxy
        "yahoo": "^GSPC",
        "finnhub": "SPY",
        "name": "S&P 500",
        "type": "index"
    },
    "dow": {
        "alpha_vantage": "DIA",  # Dow ETF as proxy
        "yahoo": "^DJI",
        "finnhub": "DIA",
        "name": "Dow Jones Industrial Average",
        "type": "index"
    },
    "russell2000": {
        "alpha_vantage": "IWM",  # Russell 2000 ETF as proxy
        "yahoo": "^RUT",
        "finnhub": "IWM",
        "name": "Russell 2000",
        "type": "index"
    },
    
    # Individual Stocks (popular ones)
    "aapl": {
        "alpha_vantage": "AAPL",
        "yahoo": "AAPL",
        "finnhub": "AAPL",
        "name": "Apple Inc.",
        "type": "stock"
    },
    "googl": {
        "alpha_vantage": "GOOGL",
        "yahoo": "GOOGL",
        "finnhub": "GOOGL", 
        "name": "Alphabet Inc.",
        "type": "stock"
    },
    "msft": {
        "alpha_vantage": "MSFT",
        "yahoo": "MSFT",
        "finnhub": "MSFT",
        "name": "Microsoft Corporation",
        "type": "stock"
    },
    "tsla": {
        "alpha_vantage": "TSLA",
        "yahoo": "TSLA",
        "finnhub": "TSLA",
        "name": "Tesla, Inc.",
        "type": "stock"
    },
    "amzn": {
        "alpha_vantage": "AMZN",
        "yahoo": "AMZN",
        "finnhub": "AMZN",
        "name": "Amazon.com, Inc.",
        "type": "stock"
    },
    
    # Cryptocurrencies
    "bitcoin": {
        "alpha_vantage": "BTC",
        "yahoo": "BTC-USD",
        "finnhub": "BINANCE:BTCUSDT",
        "yahoo_finance": "BTC-USD",
        "name": "Bitcoin",
        "type": "crypto"
    },
    "ethereum": {
        "alpha_vantage": "ETH",
        "yahoo": "ETH-USD",
        "finnhub": "BINANCE:ETHUSDT",
        "yahoo_finance": "ETH-USD",
        "name": "Ethereum",
        "type": "crypto"
    },
    
    # Forex pairs
    "eurusd": {
        "alpha_vantage": "EURUSD",
        "yahoo": "EURUSD=X",
        "finnhub": "OANDA:EUR_USD",
        "yahoo_finance": "EURUSD=X",
        "name": "EUR/USD",
        "type": "forex"
    },
    "gbpusd": {
        "alpha_vantage": "GBPUSD",
        "yahoo": "GBPUSD=X",
        "finnhub": "OANDA:GBP_USD",
        "yahoo_finance": "GBPUSD=X",
        "name": "GBP/USD",
        "type": "forex"
    },
    
    # Commodities (ETFs)
    "gold": {
        "alpha_vantage": "GLD",
        "yahoo": "GLD",
        "finnhub": "GLD",
        "yahoo_finance": "GLD",
        "name": "SPDR Gold Trust (Gold ETF)",
        "type": "etf"
    },
    "oil": {
        "alpha_vantage": "USO",
        "yahoo": "USO",
        "finnhub": "USO", 
        "yahoo_finance": "USO",
        "name": "United States Oil Fund (Oil ETF)",
        "type": "etf"
    }
}

# Initialize MCP only if not being imported for testing
if __name__ == "__main__" or "mcp" not in globals().get('sys', {}).get('modules', {}):
    try:
        mcp = FastMCP(name="Finance MCP Server - Market Data")
    except:
        # Fallback for testing environments
        mcp = None

@mcp.tool(description="Get market data for charts including stock indices, commodities, crypto, and forex with auto-discovery.") if mcp else lambda func: func
def get_market_data(
    symbol: str, 
    interval: str = "1day", 
    period: str = "1month",
    format_for_chart: bool = True
) -> dict:
    """
    Retrieves market data for creating financial charts with automatic symbol discovery and database caching.
    
    Args:
        symbol (str): Market symbol (nasdaq, sp500, aapl, bitcoin, etc.) or any company ticker
        interval (str): Data interval - '1min', '5min', '15min', '30min', '1hour', '1day'
        period (str): Time period - '1day', '5days', '1month', '3months', '6months', '1year', '2years', '5years'
        format_for_chart (bool): If True, formats data for frontend chart libraries
    
    Returns:
        dict: Market data with OHLCV (Open, High, Low, Close, Volume) and chart formatting
    """
    
    # Clean up expired cache entries periodically
    if DISCOVERY_AVAILABLE:
        try:
            cleanup_expired_cache()
        except:
            pass
    
    # Step 1: Check cache first if discovery system is available
    if DISCOVERY_AVAILABLE:
        cached_result = get_cached_or_fresh_data(symbol.lower(), interval, period)
        if cached_result:
            price_data, data_source = cached_result
            print(f"📋 Using cached data for {symbol} from {data_source}")
            
            # Get symbol info (this will be fast since it's likely in DB)
            try:
                symbol_info = asyncio.run(get_or_discover_symbol(symbol))
                display_name = symbol_info.get("company_name", symbol.upper())
                symbol_type = symbol_info.get("symbol_type", "stock")
                actual_symbol = symbol_info.get("alpha_vantage", symbol.upper())
            except:
                display_name = symbol.upper()
                symbol_type = "stock"
                actual_symbol = symbol.upper()
            
            # Format and return cached data
            if format_for_chart:
                chart_data = _format_for_charts_enhanced(price_data, symbol, display_name, interval)
                return {
                    "success": True,
                    "symbol": symbol,
                    "display_name": display_name,
                    "actual_symbol": actual_symbol,
                    "symbol_type": symbol_type,
                    "source": f"{data_source} (Cached)",
                    "interval": interval,
                    "period": period,
                    "data": price_data,
                    "data_points": len(price_data),
                    "chart_data": chart_data,
                    "raw_data_sample": price_data[:3] if price_data else [],
                    "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "from_cache": True
                }
    
    # Step 2: Resolve symbol - check predefined symbols first, then discover if needed
    symbol_info = MARKET_SYMBOLS.get(symbol.lower())
    
    if not symbol_info and DISCOVERY_AVAILABLE:
        # Symbol not in predefined list - try to discover it
        try:
            print(f"🔍 Symbol '{symbol}' not in predefined list, attempting auto-discovery...")
            symbol_info = asyncio.run(get_or_discover_symbol(symbol))
            print(f"✅ Auto-discovered symbol: {symbol_info.get('company_name', symbol)}")
        except Exception as e:
            print(f"❌ Symbol discovery failed for {symbol}: {e}")
            symbol_info = None
    
    # Use discovered/existing symbol info or create fallback
    if symbol_info:
        display_name = symbol_info.get("name", symbol_info.get("company_name", symbol.upper()))
        symbol_type = symbol_info.get("type", symbol_info.get("symbol_type", "stock"))
        # Try different symbol formats for different APIs
        alpha_vantage_symbol = symbol_info.get("alpha_vantage", symbol.upper())
        yahoo_symbol = symbol_info.get("yahoo_finance", symbol_info.get("yahoo", symbol.upper()))
        finnhub_symbol = symbol_info.get("finnhub", symbol.upper())
    else:
        # Fallback for unknown symbols
        display_name = symbol.upper()
        symbol_type = "stock"
        alpha_vantage_symbol = symbol.upper()
        yahoo_symbol = symbol.upper()
        finnhub_symbol = symbol.upper()
    
    # Step 3: Try data sources by priority to get fresh data
    market_data = None
    source_used = None
    error_messages = []
    
    # Get available sources sorted by priority
    available_sources = []
    for source, config in DATA_SOURCE_PRIORITY.items():
        if config["available"]:
            available_sources.append((source, config["priority"]))
    
    # Sort by priority (lower number = higher priority)
    available_sources.sort(key=lambda x: x[1])
    
    # Try each source in priority order
    for source_name, priority in available_sources:
        if source_name == "alpha_vantage":
            try:
                market_data = _get_data_from_alpha_vantage_enhanced(alpha_vantage_symbol, interval, period, symbol_type)
                if market_data and len(market_data) > 0:
                    source_used = f"Alpha Vantage (Priority {priority} - Live Data)"
                    break
            except Exception as e:
                error_messages.append(f"Alpha Vantage: {str(e)}")
        
        elif source_name == "finnhub":
            try:
                market_data = _get_data_from_finnhub(finnhub_symbol, interval, period)
                if market_data and len(market_data) > 0:
                    source_used = f"Finnhub (Priority {priority} - Live Data)"
                    break
            except Exception as e:
                error_messages.append(f"Finnhub: {str(e)}")
        
        elif source_name == "yahoo_finance":
            try:
                # For Yahoo Finance, we need to adapt to OHLCV format
                import yfinance as yf
                
                ticker = yf.Ticker(yahoo_symbol)
                hist = ticker.history(period=_map_period_to_yahoo(period))
                
                if not hist.empty:
                    market_data = []
                    for index, row in hist.iterrows():
                        market_data.append({
                            "timestamp": index.strftime("%Y-%m-%d"),
                            "datetime": index.isoformat(),
                            "open": float(row['Open']),
                            "high": float(row['High']),
                            "low": float(row['Low']),
                            "close": float(row['Close']),
                            "volume": int(row['Volume']) if 'Volume' in row else 0,
                            "symbol": yahoo_symbol
                        })
                    
                    if market_data:
                        source_used = f"Yahoo Finance (Priority {priority} - Live Data)"
                        break
            except ImportError:
                error_messages.append("Yahoo Finance: yfinance not installed")
            except Exception as e:
                error_messages.append(f"Yahoo Finance: {str(e)}")
    
    # Step 4: Cache the data if we got live data and discovery system is available
    if market_data and source_used and "Live Data" in source_used and DISCOVERY_AVAILABLE:
        try:
            cache_success = cache_price_data(symbol.lower(), market_data, interval, period, source_used)
            if cache_success:
                print(f"💾 Cached fresh data for {symbol}")
        except Exception as e:
            print(f"⚠️ Failed to cache data for {symbol}: {e}")
    
    # Step 5: Fallback to sample data if no live sources worked
    if not market_data:
        try:
            market_data = _generate_sample_data(symbol, interval, period)
            if market_data:
                source_used = "Sample Data (Fallback - Simulated)"
        except Exception as e:
            error_messages.append(f"Sample Data: {str(e)}")
    
    # Step 6: Final emergency fallback
    if not market_data:
        market_data = [{
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "datetime": datetime.now().isoformat(),
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 101.0,
            "volume": 1000000,
            "symbol": symbol
        }]
        source_used = "Emergency Fallback Data"
        error_messages.append("All data sources failed - using emergency fallback")

    # Data is now guaranteed to exist
    actual_symbol = alpha_vantage_symbol
    
    # Format for charts if requested
    response_data = {
        "success": True,
        "symbol": symbol,
        "display_name": display_name,
        "actual_symbol": actual_symbol,
        "symbol_type": symbol_type,
        "source": source_used,
        "auto_discovered": bool(symbol_info and symbol_info.get("source") == "discovered"),
        "cached_data": False,
        "priority_sources_tried": [s[0] for s in available_sources],
        "error_messages": error_messages if error_messages else None,
        "interval": interval,
        "period": period,
        "data": market_data,
        "data_points": len(market_data),
        "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if format_for_chart:
        chart_data = _format_for_charts_enhanced(market_data, symbol, display_name, interval)
        response_data.update({
            "chart_data": chart_data,
            "raw_data_sample": market_data[:3] if market_data else []
        })
    
    return response_data


async def _get_yahoo_finance_data(symbol_config: dict, symbol_key: str) -> dict:
    """Try to get real-time data from Yahoo Finance"""
    try:
        import yfinance as yf
        
        yahoo_symbol = symbol_config.get("yahoo_finance", symbol_config.get("yahoo", symbol_key.upper()))
        
        # Create ticker object
        ticker = yf.Ticker(yahoo_symbol)
        
        # Get current price info
        info = ticker.info
        hist = ticker.history(period="5d")
        
        if info and hist is not None and not hist.empty:
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not current_price and not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
            
            prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
            if not prev_close and len(hist) > 1:
                prev_close = float(hist['Close'].iloc[-2])
            
            # Calculate change
            change = 0
            change_percent = 0
            if current_price and prev_close:
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
            
            return {
                "success": True,
                "source": "yahoo_finance",
                "data": {
                    "symbol": yahoo_symbol,
                    "name": symbol_config.get("name", yahoo_symbol),
                    "price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "currency": info.get('currency', 'USD'),
                    "market_cap": info.get('marketCap'),
                    "volume": info.get('regularMarketVolume') or info.get('volume'),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
    except ImportError:
        print("yfinance not available - install with: pip install yfinance")
    except Exception as e:
        print(f"Yahoo Finance error for {symbol_key}: {e}")
    
    return {"success": False, "source": "yahoo_finance"}


def _map_period_to_yahoo(period: str) -> str:
    """Map our period format to Yahoo Finance period format"""
    period_mapping = {
        "1day": "1d",
        "5days": "5d", 
        "1month": "1mo",
        "3months": "3mo",
        "6months": "6mo",
        "1year": "1y",
        "2years": "2y",
        "5years": "5y"
    }
    return period_mapping.get(period, "1mo")


def _get_data_from_alpha_vantage_enhanced(symbol: str, interval: str, period: str, symbol_type: str) -> list:
    """Enhanced Alpha Vantage data retrieval with better symbol type handling"""
    try:
        # Choose appropriate Alpha Vantage function based on symbol type
        if symbol_type == "crypto":
            return _get_crypto_data_alpha_vantage(symbol, interval, period)
        elif symbol_type == "forex":
            return _get_forex_data_alpha_vantage(symbol, interval, period)
        else:
            # Stock, index, ETF - use regular time series
            return _get_stock_data_alpha_vantage(symbol, interval, period)
    except Exception as e:
        print(f"Enhanced Alpha Vantage error: {e}")
        return []


def _get_stock_data_alpha_vantage(symbol: str, interval: str, period: str) -> list:
    """Get stock/index/ETF data from Alpha Vantage"""
    try:
        # Map intervals to Alpha Vantage format
        av_intervals = {
            "1min": "1min",
            "5min": "5min", 
            "15min": "15min",
            "30min": "30min",
            "1hour": "60min",
            "1day": "daily"
        }
        
        av_interval = av_intervals.get(interval, "daily")
        
        # Choose function based on interval
        if av_interval == "daily":
            function = "TIME_SERIES_DAILY"
            time_key = "Time Series (Daily)"
        else:
            function = "TIME_SERIES_INTRADAY"
            time_key = f"Time Series ({av_interval})"
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "full" if period in ["1year", "2years", "5years"] else "compact"
        }
        
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = av_interval
        
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
        if "Note" in data:
            raise Exception(f"Alpha Vantage Rate Limited: {data['Note']}")
        
        if time_key not in data:
            raise Exception(f"No data found for {symbol}")
        
        # Convert to standard format
        market_data = []
        time_series = data[time_key]
        
        # Sort by date (newest first)
        sorted_dates = sorted(time_series.keys(), reverse=True)
        
        # Limit data based on period
        limit = _get_data_limit(period)
        
        for date_str in sorted_dates[:limit]:
            item = time_series[date_str]
            market_data.append({
                "timestamp": date_str,
                "datetime": datetime.fromisoformat(date_str.replace(" ", "T")).isoformat(),
                "open": float(item["1. open"]),
                "high": float(item["2. high"]),
                "low": float(item["3. low"]),
                "close": float(item["4. close"]),
                "volume": int(item["5. volume"]) if item["5. volume"] != "0" else 0,
                "symbol": symbol
            })
        
        return list(reversed(market_data))  # Return chronological order
        
    except Exception as e:
        print(f"Stock Alpha Vantage error: {e}")
        return []


def _get_crypto_data_alpha_vantage(symbol: str, interval: str, period: str) -> list:
    """Get cryptocurrency data from Alpha Vantage"""
    try:
        if interval == "1day":
            function = "DIGITAL_CURRENCY_DAILY"
        else:
            # For intraday crypto, we'll use a different approach or fallback
            function = "DIGITAL_CURRENCY_DAILY"
        
        params = {
            "function": function,
            "symbol": symbol,
            "market": "USD",
            "apikey": ALPHA_VANTAGE_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            raise Exception(f"Crypto Error: {data['Error Message']}")
        if "Note" in data:
            raise Exception(f"Rate Limited: {data['Note']}")
        
        time_key = "Time Series (Digital Currency Daily)"
        if time_key not in data:
            raise Exception(f"No crypto data found for {symbol}")
        
        # Convert to standard format
        market_data = []
        time_series = data[time_key]
        
        # Sort by date (newest first)
        sorted_dates = sorted(time_series.keys(), reverse=True)
        limit = _get_data_limit(period)
        
        for date_str in sorted_dates[:limit]:
            item = time_series[date_str]
            market_data.append({
                "timestamp": date_str,
                "datetime": datetime.fromisoformat(date_str).isoformat(),
                "open": float(item["1a. open (USD)"]),
                "high": float(item["2a. high (USD)"]),
                "low": float(item["3a. low (USD)"]),
                "close": float(item["4a. close (USD)"]),
                "volume": float(item["5. volume"]) if "5. volume" in item else 0,
                "symbol": symbol
            })
        
        return list(reversed(market_data))
        
    except Exception as e:
        print(f"Crypto Alpha Vantage error: {e}")
        return []


def _get_forex_data_alpha_vantage(symbol: str, interval: str, period: str) -> list:
    """Get forex data from Alpha Vantage"""
    try:
        # Extract currency pair (e.g., EURUSD -> EUR, USD)
        if len(symbol) == 6:
            from_symbol = symbol[:3]
            to_symbol = symbol[3:]
        else:
            raise Exception(f"Invalid forex symbol format: {symbol}")
        
        if interval == "1day":
            function = "FX_DAILY"
            time_key = "Time Series (FX Daily)"
        else:
            function = "FX_INTRADAY"
            time_key = f"Time Series (FX {interval}min)"
        
        params = {
            "function": function,
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "apikey": ALPHA_VANTAGE_KEY
        }
        
        if function == "FX_INTRADAY":
            params["interval"] = interval.replace("min", "min")
        
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            raise Exception(f"Forex Error: {data['Error Message']}")
        if "Note" in data:
            raise Exception(f"Rate Limited: {data['Note']}")
        
        if time_key not in data:
            raise Exception(f"No forex data found for {symbol}")
        
        # Convert to standard format
        market_data = []
        time_series = data[time_key]
        
        sorted_dates = sorted(time_series.keys(), reverse=True)
        limit = _get_data_limit(period)
        
        for date_str in sorted_dates[:limit]:
            item = time_series[date_str]
            market_data.append({
                "timestamp": date_str,
                "datetime": datetime.fromisoformat(date_str.replace(" ", "T")).isoformat(),
                "open": float(item["1. open"]),
                "high": float(item["2. high"]),
                "low": float(item["3. low"]),
                "close": float(item["4. close"]),
                "volume": 0,  # Forex doesn't have volume
                "symbol": symbol
            })
        
        return list(reversed(market_data))
        
    except Exception as e:
        print(f"Forex Alpha Vantage error: {e}")
        return []


def _generate_sample_data(symbol: str, interval: str, period: str) -> list:
    """Generate realistic sample data when APIs are unavailable"""
    try:
        # Base prices for different assets
        base_prices = {
            "nasdaq": 18400,
            "sp500": 5850,
            "dow": 43200,
            "aapl": 190,
            "googl": 2800,
            "msft": 415,
            "tsla": 245,
            "bitcoin": 67000,
            "ethereum": 3200,
            "eurusd": 1.08,
            "gold": 185,  # GLD ETF price
            "oil": 75    # USO ETF price
        }
        
        base_price = base_prices.get(symbol.lower(), 100)
        
        # Determine number of data points based on period and interval
        data_points = _get_data_limit(period)
        
        # Generate sample data
        market_data = []
        current_time = datetime.now()
        
        # Time delta based on interval
        time_deltas = {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "30min": timedelta(minutes=30),
            "1hour": timedelta(hours=1),
            "1day": timedelta(days=1)
        }
        
        time_delta = time_deltas.get(interval, timedelta(days=1))
        
        for i in range(data_points):
            # Go backwards in time
            timestamp = current_time - (time_delta * (data_points - 1 - i))
            
            # Add some realistic price movement
            volatility = base_price * 0.02  # 2% volatility
            trend = (i / data_points) * base_price * 0.05  # 5% trend over period
            noise = ((-1)**i) * volatility * 0.3  # Some noise
            
            open_price = base_price + trend + noise
            high_price = open_price + abs(noise) + (volatility * 0.2)
            low_price = open_price - abs(noise) - (volatility * 0.2)
            close_price = open_price + (noise * 0.5)
            volume = int(1000000 + (i * 10000) + (abs(noise) * 50000))
            
            market_data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "datetime": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume,
                "symbol": symbol
            })
        
        return market_data
        
    except Exception as e:
        print(f"Sample data generation error: {e}")
        return []


def _format_for_charts_enhanced(data: list, symbol: str, display_name: str, interval: str) -> dict:
    """Enhanced chart formatting with better metadata and styling"""
    if not data:
        return {"error": "No data to format"}
    
    # Determine chart type and formatting based on interval and data density
    if interval in ["1day"] and len(data) <= 50:
        chart_type = "candlestick"
    else:
        chart_type = "line"
    
    # Format for different chart types
    formatted_data = {
        "type": chart_type,
        "symbol": symbol,
        "display_name": display_name,
        "labels": [],  # X-axis labels (timestamps)
        "datasets": []
    }
    
    # Extract data for charts
    timestamps = []
    prices = []
    volumes = []
    ohlc_data = []  # For candlestick charts
    
    for item in data:
        # Format timestamp for display
        dt = datetime.fromisoformat(item["datetime"].replace("Z", "+00:00"))
        if interval == "1day":
            label = dt.strftime("%Y-%m-%d")
        elif interval in ["1hour", "30min", "15min"]:
            label = dt.strftime("%m-%d %H:%M")  
        else:
            label = dt.strftime("%H:%M")
        
        timestamps.append(label)
        prices.append(item["close"])
        volumes.append(item["volume"])
        
        # OHLC data for candlestick charts
        ohlc_data.append({
            "x": label,
            "o": item["open"],   # Open
            "h": item["high"],   # High  
            "l": item["low"],    # Low
            "c": item["close"]   # Close
        })
    
    formatted_data["labels"] = timestamps
    
    # Color scheme based on symbol type
    color_schemes = {
        "nasdaq": {"primary": "#1f77b4", "secondary": "rgba(31, 119, 180, 0.1)"},
        "sp500": {"primary": "#ff7f0e", "secondary": "rgba(255, 127, 14, 0.1)"},
        "bitcoin": {"primary": "#f7931a", "secondary": "rgba(247, 147, 26, 0.1)"},
        "gold": {"primary": "#ffd700", "secondary": "rgba(255, 215, 0, 0.1)"},
        "default": {"primary": "#1f77b4", "secondary": "rgba(31, 119, 180, 0.1)"}
    }
    
    colors = color_schemes.get(symbol.lower(), color_schemes["default"])
    
    if chart_type == "candlestick":
        # Candlestick chart format
        formatted_data["datasets"] = [
            {
                "label": f"{display_name} Price",
                "data": ohlc_data,
                "borderColor": colors["primary"],
                "backgroundColor": colors["secondary"],
                "type": "candlestick"
            }
        ]
    else:
        # Line chart format
        formatted_data["datasets"] = [
            {
                "label": f"{display_name} Price",
                "data": prices,
                "borderColor": colors["primary"],
                "backgroundColor": colors["secondary"],
                "fill": True,
                "tension": 0.1  # Smooth curves
            }
        ]
    
    # Add volume data as separate dataset if significant
    if any(v > 0 for v in volumes) and max(volumes) > min(volumes) * 2:
        formatted_data["datasets"].append({
            "label": "Volume",
            "data": volumes,
            "type": "bar",
            "yAxisID": "volume",
            "borderColor": "#17a2b8",
            "backgroundColor": "rgba(23, 162, 184, 0.3)"
        })
    
    # Enhanced chart configuration
    formatted_data["config"] = {
        "responsive": True,
        "maintainAspectRatio": False,
        "interaction": {
            "intersect": False,
            "mode": "index"
        },
        "scales": {
            "x": {
                "display": True,
                "title": {"display": True, "text": "Time"},
                "grid": {"color": "rgba(0,0,0,0.1)"}
            },
            "y": {
                "display": True, 
                "title": {"display": True, "text": "Price ($)"},
                "position": "left",
                "grid": {"color": "rgba(0,0,0,0.1)"}
            },
            "volume": {
                "display": bool(any(v > 0 for v in volumes)),
                "title": {"display": True, "text": "Volume"},
                "position": "right",
                "grid": {"drawOnChartArea": False},
                "max": max(volumes) * 4 if volumes else None  # Scale volume appropriately
            }
        },
        "plugins": {
            "title": {
                "display": True,
                "text": f"{display_name} - {interval.upper()} Chart",
                "font": {"size": 16}
            },
            "legend": {
                "display": True,
                "position": "top"
            },
            "tooltip": {
                "mode": "index",
                "intersect": False
            }
        }
    }
    
    # Add comprehensive statistics
    if prices:
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        positive_changes = [c for c in price_changes if c > 0]
        negative_changes = [c for c in price_changes if c < 0]
        
        formatted_data["statistics"] = {
            "current_price": prices[-1],
            "change": prices[-1] - prices[0] if len(prices) > 1 else 0,
            "change_percent": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[0] != 0 else 0,
            "high": max(item["high"] for item in data),
            "low": min(item["low"] for item in data),
            "volume": sum(volumes) if volumes else 0,
            "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
            "volatility": (max(prices) - min(prices)) / min(prices) * 100 if prices else 0,
            "trend": "bullish" if prices[-1] > prices[0] else "bearish" if prices[-1] < prices[0] else "sideways",
            "data_points": len(data),
            "positive_days": len(positive_changes),
            "negative_days": len(negative_changes)
        }
    
    return formatted_data


def _get_data_from_alpha_vantage(symbol: str, interval: str, period: str) -> list:
    """Get market data from Alpha Vantage API"""
    try:
        # Map intervals to Alpha Vantage format
        av_intervals = {
            "1min": "1min",
            "5min": "5min", 
            "15min": "15min",
            "30min": "30min",
            "1hour": "60min",
            "1day": "daily"
        }
        
        av_interval = av_intervals.get(interval, "daily")
        
        # Choose function based on interval
        if av_interval == "daily":
            function = "TIME_SERIES_DAILY"
            time_key = "Time Series (Daily)"
        else:
            function = "TIME_SERIES_INTRADAY"
            time_key = f"Time Series ({av_interval})"
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "full" if period in ["1year", "2years", "5years"] else "compact"
        }
        
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = av_interval
        
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if time_key not in data:
            return []
        
        # Convert to standard format
        market_data = []
        time_series = data[time_key]
        
        # Sort by date (newest first)
        sorted_dates = sorted(time_series.keys(), reverse=True)
        
        # Limit data based on period
        limit = _get_data_limit(period)
        
        for date_str in sorted_dates[:limit]:
            item = time_series[date_str]
            market_data.append({
                "timestamp": date_str,
                "datetime": datetime.fromisoformat(date_str.replace(" ", "T")).isoformat(),
                "open": float(item["1. open"]),
                "high": float(item["2. high"]),
                "low": float(item["3. low"]),
                "close": float(item["4. close"]),
                "volume": int(item["5. volume"]) if item["5. volume"] != "0" else 0,
                "symbol": symbol
            })
        
        return list(reversed(market_data))  # Return chronological order
        
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return []


def _get_data_from_finnhub(symbol: str, interval: str, period: str) -> list:
    """Get market data from Finnhub API"""
    try:
        # Map intervals to Finnhub resolution
        fh_resolutions = {
            "1min": "1",
            "5min": "5",
            "15min": "15", 
            "30min": "30",
            "1hour": "60",
            "1day": "D"
        }
        
        resolution = fh_resolutions.get(interval, "D")
        
        # Calculate time range
        end_time = int(datetime.now().timestamp())
        
        period_days = {
            "1day": 1,
            "5days": 5,
            "1month": 30,
            "3months": 90,
            "6months": 180,
            "1year": 365,
            "2years": 730,
            "5years": 1825
        }
        
        days_back = period_days.get(period, 30)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": start_time,
            "to": end_time,
            "token": FINNHUB_KEY
        }
        
        response = requests.get(f"{FINNHUB_URL}/stock/candle", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("s") != "ok" or not data.get("t"):
            return []
        
        # Convert to standard format
        market_data = []
        timestamps = data["t"]
        opens = data["o"]
        highs = data["h"]
        lows = data["l"]
        closes = data["c"]
        volumes = data["v"]
        
        for i in range(len(timestamps)):
            dt = datetime.fromtimestamp(timestamps[i])
            market_data.append({
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "datetime": dt.isoformat(),
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "volume": volumes[i],
                "symbol": symbol
            })
        
        return market_data
        
    except Exception as e:
        print(f"Finnhub error: {e}")
        return []


def _get_data_limit(period: str) -> int:
    """Get data point limit based on period"""
    limits = {
        "1day": 48,      # 48 30-min intervals
        "5days": 240,    # 5 days of 30-min intervals
        "1month": 30,    # 30 daily points
        "3months": 90,   # 90 daily points
        "6months": 180,  # 180 daily points
        "1year": 365,    # 365 daily points
        "2years": 730,   # 730 daily points
        "5years": 1825   # 1825 daily points
    }
    return limits.get(period, 100)


def _format_for_charts(data: list, symbol: str, interval: str) -> dict:
    """Format market data for frontend chart libraries (Chart.js, D3, etc.)"""
    if not data:
        return {"error": "No data to format"}
    
    # Determine chart type and formatting
    chart_type = "candlestick" if interval in ["1day", "1hour", "30min"] else "line"
    
    # Format for different chart types
    formatted_data = {
        "type": chart_type,
        "symbol": symbol,
        "labels": [],  # X-axis labels (timestamps)
        "datasets": []
    }
    
    # Extract data for charts
    timestamps = []
    prices = []
    volumes = []
    ohlc_data = []  # For candlestick charts
    
    for item in data:
        # Format timestamp for display
        dt = datetime.fromisoformat(item["datetime"].replace("Z", "+00:00"))
        if interval == "1day":
            label = dt.strftime("%Y-%m-%d")
        elif interval in ["1hour", "30min", "15min"]:
            label = dt.strftime("%m-%d %H:%M")  
        else:
            label = dt.strftime("%H:%M")
        
        timestamps.append(label)
        prices.append(item["close"])
        volumes.append(item["volume"])
        
        # OHLC data for candlestick charts
        ohlc_data.append({
            "x": label,
            "o": item["open"],   # Open
            "h": item["high"],   # High  
            "l": item["low"],    # Low
            "c": item["close"]   # Close
        })
    
    formatted_data["labels"] = timestamps
    
    if chart_type == "candlestick":
        # Candlestick chart format
        formatted_data["datasets"] = [
            {
                "label": f"{symbol.upper()} Price",
                "data": ohlc_data,
                "borderColor": "#1f77b4",
                "backgroundColor": "rgba(31, 119, 180, 0.1)"
            }
        ]
    else:
        # Line chart format
        formatted_data["datasets"] = [
            {
                "label": f"{symbol.upper()} Price",
                "data": prices,
                "borderColor": "#1f77b4",
                "backgroundColor": "rgba(31, 119, 180, 0.1)",
                "fill": True
            }
        ]
    
    # Add volume data as separate dataset
    if any(v > 0 for v in volumes):
        formatted_data["datasets"].append({
            "label": "Volume",
            "data": volumes,
            "type": "bar",
            "yAxisID": "volume",
            "borderColor": "#ff7f0e",
            "backgroundColor": "rgba(255, 127, 14, 0.3)"
        })
    
    # Add chart configuration
    formatted_data["config"] = {
        "responsive": True,
        "scales": {
            "x": {
                "display": True,
                "title": {"display": True, "text": "Time"}
            },
            "y": {
                "display": True, 
                "title": {"display": True, "text": "Price"},
                "position": "left"
            },
            "volume": {
                "display": bool(any(v > 0 for v in volumes)),
                "title": {"display": True, "text": "Volume"},
                "position": "right",
                "grid": {"drawOnChartArea": False}
            }
        },
        "plugins": {
            "title": {
                "display": True,
                "text": f"{symbol.upper()} - {interval.upper()} Chart"
            },
            "legend": {"display": True}
        }
    }
    
    # Add summary statistics
    if prices:
        formatted_data["statistics"] = {
            "current_price": prices[-1],
            "change": prices[-1] - prices[0] if len(prices) > 1 else 0,
            "change_percent": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[0] != 0 else 0,
            "high": max(item["high"] for item in data),
            "low": min(item["low"] for item in data),
            "volume": sum(volumes) if volumes else 0,
            "data_points": len(data)
        }
    
    return formatted_data


@mcp.tool(description="Get real-time market overview with major indices, commodities, and crypto.")
def get_market_overview() -> dict:
    """
    Get a comprehensive market overview with major indices, commodities, and cryptocurrencies.
    Perfect for market dashboard displays.
    
    Returns:
        dict: Market overview with current prices, changes, and trends
    """
    markets_to_fetch = [
        ("nasdaq", "NASDAQ Composite"),
        ("sp500", "S&P 500"), 
        ("dow", "Dow Jones"),
        ("gold", "Gold"),
        ("bitcoin", "Bitcoin"),
        ("oil", "Crude Oil")
    ]
    
    market_overview = {
        "success": True,
        "markets": [],
        "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for symbol, display_name in markets_to_fetch:
        try:
            # Get recent data (last 5 days to calculate change)
            data_result = get_market_data(symbol, interval="1day", period="5days", format_for_chart=False)
            
            if data_result.get("success") and data_result.get("data"):
                data = data_result["data"]
                current = data[-1]  # Most recent
                previous = data[-2] if len(data) > 1 else data[0]  # Previous day
                
                change = current["close"] - previous["close"]
                change_percent = (change / previous["close"] * 100) if previous["close"] != 0 else 0
                
                market_overview["markets"].append({
                    "symbol": symbol,
                    "name": display_name,
                    "price": current["close"],
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "high": current["high"],
                    "low": current["low"],
                    "volume": current["volume"],
                    "trend": "up" if change > 0 else "down" if change < 0 else "flat",
                    "last_updated": current["datetime"]
                })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    
    return market_overview


@mcp.tool(description="Get current market status and trading hours")
def get_market_status() -> dict:
    """
    Get current market status including trading hours and market state.
    
    Returns:
        dict: Market status information
    """
    try:
        current_time = datetime.now()
        
        # Market hours (EST/EDT)
        market_open_hour = 9
        market_open_minute = 30
        market_close_hour = 16
        market_close_minute = 0
        
        # Check if it's a weekday
        is_weekday = current_time.weekday() < 5  # Monday = 0, Sunday = 6
        
        # Create market open/close times for today
        market_open = current_time.replace(
            hour=market_open_hour, 
            minute=market_open_minute, 
            second=0, 
            microsecond=0
        )
        market_close = current_time.replace(
            hour=market_close_hour, 
            minute=market_close_minute, 
            second=0, 
            microsecond=0
        )
        
        # Determine market status
        if not is_weekday:
            status = "closed"
            session = "weekend"
        elif current_time < market_open:
            status = "pre_market"
            session = "pre_market"
        elif current_time > market_close:
            status = "after_hours"
            session = "after_hours"
        else:
            status = "open"
            session = "regular"
        
        # Calculate next market open/close
        if status == "open":
            next_change = market_close
            next_change_type = "close"
        elif status == "after_hours":
            # Next market open is tomorrow (or Monday if Friday)
            days_ahead = 3 if current_time.weekday() == 4 else 1  # Friday -> Monday
            next_change = market_open + timedelta(days=days_ahead)
            next_change_type = "open"
        elif status == "weekend":
            # Next market open is Monday
            days_ahead = 7 - current_time.weekday()  # Days until Monday
            next_change = market_open + timedelta(days=days_ahead)
            next_change_type = "open"
        else:  # pre_market
            next_change = market_open
            next_change_type = "open"
        
        return {
            "success": True,
            "market_status": status,
            "session": session,
            "is_open": status == "open",
            "current_time": current_time.isoformat(),
            "market_hours": {
                "open": f"{market_open_hour:02d}:{market_open_minute:02d} EST",
                "close": f"{market_close_hour:02d}:{market_close_minute:02d} EST"
            },
            "next_change": {
                "time": next_change.isoformat(),
                "type": next_change_type,
                "minutes_until": int((next_change - current_time).total_seconds() / 60)
            },
            "timezone": "US Eastern Time",
            "trading_days": "Monday - Friday"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "market_status": "unknown"
        }
