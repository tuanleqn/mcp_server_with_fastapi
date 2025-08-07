"""
Finance Calculations Server
"""

import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from datetime import datetime
from .finance_helpers import (
    calculate_rsi_helper,
    calculate_sma_helper,
    get_historical_prices_helper
)
import numpy as np
import pandas as pd

load_dotenv()

mcp = FastMCP(name="Finance Calculations Server")


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    current_price = prices.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_sma = sma.iloc[-1]
    
    if current_price > current_upper:
        position = "above_upper_band"
        signal = "overbought"
    elif current_price < current_lower:
        position = "below_lower_band"
        signal = "oversold"
    else:
        position = "within_bands"
        signal = "neutral"
    
    return {
        "upper_band": round(float(current_upper), 2),
        "lower_band": round(float(current_lower), 2),
        "middle_band": round(float(current_sma), 2),
        "current_price": round(float(current_price), 2),
        "position": position,
        "signal": signal,
        "band_width": round(float(current_upper - current_lower), 2)
    }


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    current_histogram = histogram.iloc[-1]
    
    if current_macd > current_signal and current_histogram > 0:
        trend_signal = "bullish"
    elif current_macd < current_signal and current_histogram < 0:
        trend_signal = "bearish"
    else:
        trend_signal = "neutral"
    
    return {
        "macd": round(float(current_macd), 4),
        "signal": round(float(current_signal), 4),
        "histogram": round(float(current_histogram), 4),
        "trend_signal": trend_signal,
        "momentum": "increasing" if current_histogram > histogram.iloc[-2] else "decreasing"
    }


@mcp.tool(description="Calculate comprehensive technical analysis")
def calculate_advanced_technical_analysis(symbol: str, period: int = 100) -> dict:
    try:
        df = get_historical_prices_helper(symbol, period + 50)
        if df.empty or len(df) < 50:
            return {
                "success": False,
                "error": f"Insufficient data for technical analysis of {symbol}",
                "symbol": symbol.upper()
            }
        
        prices = df['close_price']
        volumes = df['volume']
        current_price = prices.iloc[-1]
        
        rsi_result = calculate_rsi_helper(symbol, 14)
        sma20_result = calculate_sma_helper(symbol, 20)
        sma50_result = calculate_sma_helper(symbol, 50)
        
        try:
            bollinger = calculate_bollinger_bands(prices, 20, 2)
        except Exception as e:
            bollinger = {"upper_band": 0, "lower_band": 0, "signal": "error"}
            
        try:
            macd = calculate_macd(prices, 12, 26, 9)
        except Exception as e:
            macd = {"macd": 0, "signal": 0, "trend": "error"}
    
        if rsi_result.get("success"):
            current_rsi = rsi_result.get("rsi", 0)
        else:
            current_rsi = 0
        
        if sma20_result.get("success"):
            current_sma20 = sma20_result.get("sma", current_price)
        else:
            current_sma20 = current_price
            
        if sma50_result.get("success"):
            current_sma50 = sma50_result.get("sma", current_price)
        else:
            current_sma50 = current_price
        
        returns = prices.pct_change().dropna()
        volatility_daily = returns.std()
        volatility_annual = volatility_daily * np.sqrt(252) * 100
        
        avg_volume = volumes.rolling(window=20).mean().iloc[-1]
        current_volume = volumes.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        recent_prices = prices.tail(20)
        resistance = recent_prices.max()
        support = recent_prices.min()
        
        short_trend = "bullish" if current_price > current_sma20 else "bearish"
        long_trend = "bullish" if current_price > current_sma50 else "bearish"
        
        signals = []
        if current_rsi > 70:
            signals.append("RSI Overbought")
        elif current_rsi < 30:
            signals.append("RSI Oversold")
        
        if bollinger.get("position") == "above_upper_band":
            signals.append("Price Above Bollinger Upper Band")
        elif bollinger.get("position") == "below_lower_band":
            signals.append("Price Below Bollinger Lower Band")
        
        if macd.get("trend_signal") == "bullish":
            signals.append("MACD Bullish Signal")
        elif macd.get("trend_signal") == "bearish":
            signals.append("MACD Bearish Signal")
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "current_price": round(float(current_price), 2),
            "rsi": {
                "current_rsi": current_rsi,
                "interpretation": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
            },
            "sma": {
                "sma_20": round(float(current_sma20), 2),
                "sma_50": round(float(current_sma50), 2)
            },
            "bollinger_bands": {
                "upper_band": bollinger.get("upper_band", 0),
                "lower_band": bollinger.get("lower_band", 0),
                "signal": bollinger.get("signal", "neutral")
            },
            "macd": {
                "macd_value": macd.get("macd", 0),
                "signal_line": macd.get("signal", 0),
                "signal": macd.get("trend", "neutral")
            },
            "basic_indicators": {
                "rsi_14": current_rsi,
                "sma_20": round(float(current_sma20), 2),
                "sma_50": round(float(current_sma50), 2)
            },
            "risk_metrics": {
                "daily_volatility": round(volatility_daily * 100, 2),
                "annual_volatility": round(volatility_annual, 2),
                "volume_ratio": round(volume_ratio, 2),
                "volume_signal": "high" if volume_ratio > 1.5 else "normal" if volume_ratio > 0.5 else "low"
            },
            "price_levels": {
                "current": round(float(current_price), 2),
                "resistance": round(float(resistance), 2),
                "support": round(float(support), 2),
                "distance_to_resistance": round(((resistance - current_price) / current_price) * 100, 2),
                "distance_to_support": round(((current_price - support) / current_price) * 100, 2)
            },
            "trend_analysis": {
                "short_term": short_trend,
                "long_term": long_trend,
                "overall": "bullish" if short_trend == "bullish" and long_trend == "bullish" else "bearish" if short_trend == "bearish" and long_trend == "bearish" else "mixed"
            },
            "trading_signals": signals,
            "analysis_date": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Technical analysis failed: {str(e)}",
            "symbol": symbol.upper()
        }


@mcp.tool(description="Calculate financial ratios from price data")
def calculate_financial_ratios(symbol: str, period: int = 252) -> dict:
    df = get_historical_prices_helper(symbol, period + 10)
    if df.empty or len(df) < 50:
        return {
            "success": False,
            "error": f"Insufficient data for ratio analysis of {symbol}",
            "symbol": symbol.upper()
        }
    
    prices = df['close_price'].astype(float)
    volumes = df['volume'].astype(float)
    highs = df['high_price'].astype(float)
    lows = df['low_price'].astype(float)
    
    current_price = float(prices.iloc[-1])
    
    price_52w_high = float(highs.max())
    price_52w_low = float(lows.min())
    
    returns = prices.pct_change().dropna()
    annual_return = float(returns.mean() * 252)
    
    volatility = float(returns.std() * np.sqrt(252))
    sharpe_ratio = float(annual_return / volatility) if volatility > 0 else 0.0
    
    price_change_1m = float((prices.iloc[-1] / prices.iloc[-21]) - 1) if len(prices) > 21 else 0.0
    price_change_3m = float((prices.iloc[-1] / prices.iloc[-63]) - 1) if len(prices) > 63 else 0.0
    price_change_6m = float((prices.iloc[-1] / prices.iloc[-126]) - 1) if len(prices) > 126 else 0.0
    
    avg_volume = float(volumes.mean())
    volume_trend = "increasing" if float(volumes.tail(10).mean()) > float(volumes.head(10).mean()) else "decreasing"
    
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = float(abs(drawdown.min()))
    
    positive_returns = returns[returns > 0]
    win_rate = float(len(positive_returns) / len(returns) * 100) if len(returns) > 0 else 0.0
    
    return {
        "success": True,
        "symbol": symbol.upper(),
        "current_price": round(current_price, 2),
        "annualized_return": round(annual_return * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate, 1),
        "volatility": round(volatility * 100, 2),
        "price_ratios": {
            "price_to_52w_high": round((current_price / price_52w_high), 3),
            "price_to_52w_low": round((current_price / price_52w_low), 3),
            "distance_from_high_pct": round(((price_52w_high - current_price) / price_52w_high) * 100, 2),
            "distance_from_low_pct": round(((current_price - price_52w_low) / price_52w_low) * 100, 2)
        },
        "performance_metrics": {
            "annual_return": round(annual_return * 100, 2),
            "annual_volatility": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "return_to_risk": round(annual_return / volatility, 3) if volatility > 0 else 0
        },
        "momentum_indicators": {
            "1_month_return": round(price_change_1m * 100, 2),
            "3_month_return": round(price_change_3m * 100, 2),
            "6_month_return": round(price_change_6m * 100, 2),
            "momentum_score": round((price_change_1m + price_change_3m + price_change_6m) / 3 * 100, 2)
        },
        "volume_analysis": {
            "average_daily_volume": int(avg_volume),
            "current_volume": int(volumes.iloc[-1]),
            "volume_trend": volume_trend,
            "liquidity_score": "high" if avg_volume > 1000000 else "medium" if avg_volume > 100000 else "low"
        },
        "analysis_date": pd.Timestamp.now().isoformat()
    }


if __name__ == "__main__":
    mcp.run()
