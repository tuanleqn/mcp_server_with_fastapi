import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Technical Calculations")

def _get_historical_prices(symbol: str, days: int = 100):
    """Helper function to get historical prices for calculations"""
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT date, close_price, volume
                FROM public.stock_price
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
                """
                cur.execute(query, (symbol.upper(), days))
                results = cur.fetchall()
                
                if results:
                    dates = [row[0] for row in results]
                    prices = [float(row[1]) for row in results]
                    volumes = [int(row[2]) for row in results]
                    return dates[::-1], prices[::-1], volumes[::-1]  # Reverse to chronological order
                else:
                    return [], [], []
    except Error as e:
        print(f"Database error in _get_historical_prices: {e}")
        return [], [], []

@mcp.tool(description="Calculate RSI (Relative Strength Index) for a stock symbol")
def calculate_rsi(symbol: str, period: int = 14) -> dict:
    """
    Calculate RSI (Relative Strength Index) for a stock symbol
    
    Args:
        symbol (str): Stock symbol
        period (int): RSI period (default 14)
        
    Returns:
        dict: RSI value and trading signal
    """
    try:
        # Get historical prices
        dates, prices, _ = _get_historical_prices(symbol, period + 20)  # Extra data for accuracy
        
        if len(prices) < period + 1:
            return {
                "success": False,
                "error": f"Not enough data for RSI calculation. Need at least {period + 1} records, got {len(prices)}"
            }
        
        # Calculate price changes
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate RSI for remaining periods
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Final RSI calculation
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Determine signal
        if rsi >= 70:
            signal = "SELL (Overbought)"
        elif rsi <= 30:
            signal = "BUY (Oversold)"
        else:
            signal = "HOLD (Neutral)"
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "current_rsi": round(rsi, 2),
            "period": period,
            "signal": signal,
            "current_price": prices[-1],
            "date": str(dates[-1]) if dates else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"RSI calculation failed: {str(e)}",
            "symbol": symbol.upper()
        }

@mcp.tool(description="Calculate SMA (Simple Moving Average) for a stock symbol")
def calculate_sma(symbol: str, period: int = 20) -> dict:
    """
    Calculate SMA (Simple Moving Average) for a stock symbol
    
    Args:
        symbol (str): Stock symbol
        period (int): SMA period (default 20)
        
    Returns:
        dict: SMA value and current price comparison
    """
    try:
        # Get historical prices
        dates, prices, _ = _get_historical_prices(symbol, period + 5)
        
        if len(prices) < period:
            return {
                "success": False,
                "error": f"Not enough data for SMA calculation. Need at least {period} records, got {len(prices)}"
            }
        
        # Calculate SMA
        sma = np.mean(prices[-period:])
        current_price = prices[-1]
        
        # Determine trend
        if current_price > sma:
            trend = "BULLISH (Price above SMA)"
        elif current_price < sma:
            trend = "BEARISH (Price below SMA)"
        else:
            trend = "NEUTRAL (Price at SMA)"
        
        # Calculate percentage difference
        pct_diff = ((current_price - sma) / sma) * 100
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "sma": round(sma, 2),
            "current_price": round(current_price, 2),
            "period": period,
            "trend": trend,
            "percentage_difference": round(pct_diff, 2),
            "date": str(dates[-1]) if dates else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"SMA calculation failed: {str(e)}",
            "symbol": symbol.upper()
        }

@mcp.tool(description="Calculate portfolio return for multiple symbols with weights")
def calculate_portfolio_return(symbols: list, weights: list, days: int = 30) -> dict:
    """
    Calculate weighted portfolio return
    
    Args:
        symbols (list): List of stock symbols
        weights (list): List of weights (should sum to 1.0)
        days (int): Number of days for return calculation
        
    Returns:
        dict: Portfolio return and individual stock performances
    """
    try:
        if len(symbols) != len(weights):
            return {
                "success": False,
                "error": "Number of symbols must match number of weights"
            }
        
        if abs(sum(weights) - 1.0) > 0.01:
            return {
                "success": False,
                "error": f"Weights must sum to 1.0, got {sum(weights):.3f}"
            }
        
        portfolio_return = 0.0
        individual_returns = {}
        
        for symbol, weight in zip(symbols, weights):
            # Get price data
            dates, prices, _ = _get_historical_prices(symbol, days + 5)
            
            if len(prices) < days + 1:
                individual_returns[symbol] = {
                    "return": 0.0,
                    "weight": weight,
                    "error": "Insufficient data"
                }
                continue
            
            # Calculate return
            start_price = prices[-days-1]
            end_price = prices[-1]
            stock_return = ((end_price - start_price) / start_price) * 100
            
            # Add to portfolio return
            portfolio_return += stock_return * weight
            
            individual_returns[symbol] = {
                "return": round(stock_return, 2),
                "weight": weight,
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2)
            }
        
        return {
            "success": True,
            "portfolio_return": round(portfolio_return, 2),
            "period_days": days,
            "individual_returns": individual_returns,
            "total_weight": sum(weights)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Portfolio calculation failed: {str(e)}"
        }

@mcp.tool(description="Calculate price volatility for a stock symbol")
def calculate_volatility(symbol: str, days: int = 30) -> dict:
    """
    Calculate price volatility (standard deviation of returns)
    
    Args:
        symbol (str): Stock symbol
        days (int): Number of days for volatility calculation
        
    Returns:
        dict: Volatility metrics
    """
    try:
        # Get historical prices
        dates, prices, _ = _get_historical_prices(symbol, days + 5)
        
        if len(prices) < days + 1:
            return {
                "success": False,
                "error": f"Not enough data for volatility calculation. Need at least {days + 1} records, got {len(prices)}"
            }
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i-1]) / prices[i-1]
            daily_returns.append(daily_return)
        
        # Take last 'days' returns
        recent_returns = daily_returns[-days:]
        
        # Calculate volatility metrics
        volatility = np.std(recent_returns)
        annualized_volatility = volatility * np.sqrt(252)  # 252 trading days
        mean_return = np.mean(recent_returns)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "daily_volatility": round(volatility * 100, 4),  # Convert to percentage
            "annualized_volatility": round(annualized_volatility * 100, 2),
            "mean_daily_return": round(mean_return * 100, 4),
            "period_days": days,
            "current_price": round(prices[-1], 2)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Volatility calculation failed: {str(e)}",
            "symbol": symbol.upper()
        }

@mcp.tool(description="Calculate Sharpe ratio for a stock symbol")
def calculate_sharpe_ratio(symbol: str, risk_free_rate: float = 0.02, days: int = 252) -> dict:
    """
    Calculate Sharpe ratio (risk-adjusted return)
    
    Args:
        symbol (str): Stock symbol
        risk_free_rate (float): Annual risk-free rate (default 2%)
        days (int): Number of days for calculation (default 252 - 1 year)
        
    Returns:
        dict: Sharpe ratio and related metrics
    """
    try:
        # Get historical prices
        dates, prices, _ = _get_historical_prices(symbol, days + 5)
        
        if len(prices) < days + 1:
            return {
                "success": False,
                "error": f"Not enough data for Sharpe ratio calculation. Need at least {days + 1} records, got {len(prices)}"
            }
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i-1]) / prices[i-1]
            daily_returns.append(daily_return)
        
        # Take last 'days' returns
        recent_returns = daily_returns[-days:]
        
        # Calculate metrics
        mean_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        # Annualize
        annualized_return = mean_return * 252
        annualized_volatility = volatility * np.sqrt(252)
        
        # Calculate Sharpe ratio
        if annualized_volatility == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "annualized_return": round(annualized_return * 100, 2),
            "annualized_volatility": round(annualized_volatility * 100, 2),
            "risk_free_rate": round(risk_free_rate * 100, 2),
            "period_days": days
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Sharpe ratio calculation failed: {str(e)}",
            "symbol": symbol.upper()
        }
