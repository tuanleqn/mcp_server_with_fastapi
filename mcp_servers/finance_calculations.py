"""
Finance Calculations Server
Advanced financial calculations and technical analysis
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
    
    # Determine position
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
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    current_histogram = histogram.iloc[-1]
    
    # Determine signals
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
    """
    Calculate comprehensive technical analysis including multiple indicators.
    
    Args:
        symbol: The stock symbol (e.g., 'AAPL')
        period: Number of days of data to analyze (default: 100)
    
    Returns:
        Dictionary containing comprehensive technical analysis
    """
    try:
        df = get_historical_prices_helper(symbol, period + 50)
        if df.empty or len(df) < 50:
            return {
                "success": False,
                "error": f"Insufficient data for technical analysis of {symbol} - got {len(df) if not df.empty else 0} records",
                "symbol": symbol.upper()
            }
        
        print(f"Retrieved {len(df)} records for {symbol} technical analysis")
        
        prices = df['close_price']
        volumes = df['volume']
        current_price = prices.iloc[-1]  # Define current_price first
        
        # Basic indicators with individual error handling
        rsi_result = calculate_rsi_helper(symbol, 14)
        sma20_result = calculate_sma_helper(symbol, 20)
        sma50_result = calculate_sma_helper(symbol, 50)
        
        # Advanced indicators with error handling
        try:
            bollinger = calculate_bollinger_bands(prices, 20, 2)
        except Exception as e:
            print(f"Bollinger Bands calculation failed: {e}")
            bollinger = {"upper_band": 0, "lower_band": 0, "signal": "error"}
            
        try:
            macd = calculate_macd(prices, 12, 26, 9)
        except Exception as e:
            print(f"MACD calculation failed: {e}")
            macd = {"macd": 0, "signal": 0, "trend": "error"}
    
        # Extract RSI value properly with error handling
        if rsi_result.get("success"):
            current_rsi = rsi_result.get("rsi", 0)
        else:
            print(f"RSI calculation failed for {symbol}: {rsi_result.get('error', 'Unknown error')}")
            current_rsi = 0
        
        if sma20_result.get("success"):
            current_sma20 = sma20_result.get("sma", current_price)
        else:
            print(f"SMA20 calculation failed for {symbol}: {sma20_result.get('error', 'Unknown error')}")
            current_sma20 = current_price
            
        if sma50_result.get("success"):
            current_sma50 = sma50_result.get("sma", current_price)
        else:
            print(f"SMA50 calculation failed for {symbol}: {sma50_result.get('error', 'Unknown error')}")
            current_sma50 = current_price
        
        # Debug output
        print(f"Technical analysis for {symbol}: Price=${current_price:.2f}, RSI={current_rsi:.2f}, SMA20=${current_sma20:.2f}")
        
        # Volatility analysis
        returns = prices.pct_change().dropna()
        volatility_daily = returns.std()
        volatility_annual = volatility_daily * np.sqrt(252) * 100
        
        # Volume analysis
        avg_volume = volumes.rolling(window=20).mean().iloc[-1]
        current_volume = volumes.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Support and resistance levels (simplified)
        recent_prices = prices.tail(20)
        resistance = recent_prices.max()
        support = recent_prices.min()
        # Don't redefine current_price here since we already have it
        
        # Overall trend analysis
        short_trend = "bullish" if current_price > current_sma20 else "bearish"
        long_trend = "bullish" if current_price > current_sma50 else "bearish"
        
        # Trading signals
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
        print(f"Technical analysis error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Technical analysis failed: {str(e)}",
            "symbol": symbol.upper()
        }
    
    # Extract RSI value properly with error handling
    if rsi_result.get("success"):
        current_rsi = rsi_result.get("rsi", 0)
    else:
        print(f"RSI calculation failed for {symbol}: {rsi_result.get('error', 'Unknown error')}")
        current_rsi = 0
    
    if sma20_result.get("success"):
        current_sma20 = sma20_result.get("sma", current_price)
    else:
        print(f"SMA20 calculation failed for {symbol}: {sma20_result.get('error', 'Unknown error')}")
        current_sma20 = current_price
        
    if sma50_result.get("success"):
        current_sma50 = sma50_result.get("sma", current_price)
    else:
        print(f"SMA50 calculation failed for {symbol}: {sma50_result.get('error', 'Unknown error')}")
        current_sma50 = current_price
    
    # Debug output
    print(f"Technical analysis for {symbol}: Price=${current_price:.2f}, RSI={current_rsi:.2f}, SMA20=${current_sma20:.2f}")
    
    # Volatility analysis
    returns = prices.pct_change().dropna()
    volatility_daily = returns.std()
    volatility_annual = volatility_daily * np.sqrt(252) * 100
    
    # Volume analysis
    avg_volume = volumes.rolling(window=20).mean().iloc[-1]
    current_volume = volumes.iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Support and resistance levels (simplified)
    recent_prices = prices.tail(20)
    resistance = recent_prices.max()
    support = recent_prices.min()
    # Don't redefine current_price here since we already have it
    
    # Overall trend analysis
    short_trend = "bullish" if current_price > current_sma20 else "bearish"
    long_trend = "bullish" if current_price > current_sma50 else "bearish"
    
    # Trading signals
    signals = []
    if current_rsi > 70:
        signals.append("RSI Overbought")
    elif current_rsi < 30:
        signals.append("RSI Oversold")
    
    if bollinger["position"] == "above_upper_band":
        signals.append("Price Above Bollinger Upper Band")
    elif bollinger["position"] == "below_lower_band":
        signals.append("Price Below Bollinger Lower Band")
    
    if macd["trend_signal"] == "bullish":
        signals.append("MACD Bullish Signal")
    elif macd["trend_signal"] == "bearish":
        signals.append("MACD Bearish Signal")
    
    return {
        "success": True,
        "symbol": symbol.upper(),
        "current_price": round(float(current_price), 2),
        "rsi": {
            "current_rsi": current_rsi,
            "interpretation": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
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
        "analysis_date": rsi_result.get("calculation_date", datetime.now().isoformat())
    }


@mcp.tool(description="Calculate portfolio risk metrics")
def calculate_portfolio_risk_metrics(symbols: list, weights: list = None, period: int = 252) -> dict:
    """
    Calculate risk metrics for a portfolio of stocks.
    
    Args:
        symbols: List of stock symbols
        weights: List of weights (if None, equal weighting is used)
        period: Number of days for analysis (default: 252 for 1 year)
    
    Returns:
        Dictionary containing portfolio risk metrics
    """
    if not symbols:
        return {
            "success": False,
            "error": "No symbols provided"
        }
    
    # Use equal weights if not provided
    if weights is None:
        weights = [1.0 / len(symbols)] * len(symbols)
    
    if len(weights) != len(symbols):
        return {
            "success": False,
            "error": "Number of weights must match number of symbols"
        }
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Get returns for all symbols
    returns_data = {}
    for symbol in symbols:
        df = get_historical_prices_helper(symbol, period + 10)
        if not df.empty and len(df) > 50:
            returns = df['close_price'].pct_change().dropna()
            returns_data[symbol] = returns
    
    if not returns_data:
        return {
            "success": False,
            "error": "No valid data found for any symbols"
        }
    
    # Calculate portfolio metrics
    valid_symbols = list(returns_data.keys())
    valid_weights = weights[:len(valid_symbols)]
    
    # Portfolio return
    portfolio_returns = []
    min_length = min(len(returns) for returns in returns_data.values())
    
    for i in range(min_length):
        daily_return = sum(
            returns_data[symbol].iloc[i] * weight 
            for symbol, weight in zip(valid_symbols, valid_weights)
        )
        portfolio_returns.append(daily_return)
    
    portfolio_returns = pd.Series(portfolio_returns)
    
    # Risk metrics
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    portfolio_return = portfolio_returns.mean() * 252
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
    
    # Maximum Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        "success": True,
        "portfolio_return": round(portfolio_return * 100, 2),  # Convert to percentage
        "portfolio_volatility": round(portfolio_volatility * 100, 2),  # Convert to percentage
        "sharpe_ratio": round(sharpe_ratio, 4),
        "value_at_risk_95": round(abs(var_95) * 100, 2),  # Convert to positive percentage
        "max_drawdown": round(abs(max_drawdown) * 100, 2),  # Convert to positive percentage
        "portfolio_composition": [
            {"symbol": symbol, "weight": round(weight, 4), "weight_percent": round(weight * 100, 2)}
            for symbol, weight in zip(valid_symbols, valid_weights)
        ],
        "risk_metrics": {
            "annual_return": round(portfolio_return, 4),
            "annual_volatility": round(portfolio_volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "value_at_risk_95": round(var_95, 4),
            "max_drawdown": round(max_drawdown, 4),
            "risk_level": "high" if portfolio_volatility > 0.25 else "medium" if portfolio_volatility > 0.15 else "low"
        },
        "diversification": {
            "number_of_assets": len(valid_symbols),
            "effective_assets": round(1 / sum(w**2 for w in valid_weights), 2),
            "concentration_risk": "high" if max(valid_weights) > 0.4 else "medium" if max(valid_weights) > 0.25 else "low"
        },
        "analysis_date": datetime.now().isoformat(),
        "analysis_period_days": len(portfolio_returns)
    }


@mcp.tool(description="Calculate financial ratios from price data")
def calculate_financial_ratios(symbol: str, period: int = 252) -> dict:
    """
    Calculate key financial ratios based on price and volume data.
    
    Args:
        symbol: The stock symbol (e.g., 'AAPL')
        period: Number of days for analysis (default: 252)
    
    Returns:
        Dictionary containing financial ratios
    """
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
    
    current_price = prices.iloc[-1]
    
    # Price ratios
    price_52w_high = highs.max()
    price_52w_low = lows.min()
    
    # Returns
    returns = prices.pct_change().dropna()
    annual_return = returns.mean() * 252
    
    # Risk-adjusted metrics
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Momentum indicators
    price_change_1m = ((prices.iloc[-1] / prices.iloc[-21]) - 1) if len(prices) > 21 else 0
    price_change_3m = ((prices.iloc[-1] / prices.iloc[-63]) - 1) if len(prices) > 63 else 0
    price_change_6m = ((prices.iloc[-1] / prices.iloc[-126]) - 1) if len(prices) > 126 else 0
    
    # Volume metrics
    avg_volume = volumes.mean()
    volume_trend = "increasing" if volumes.tail(10).mean() > volumes.head(10).mean() else "decreasing"
    
    # Calculate max drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())
    
    # Calculate win rate
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
    
    return {
        "success": True,
        "symbol": symbol.upper(),
        "current_price": round(float(current_price), 2),
        # Add top-level fields that showcase expects
        "annualized_return": round(annual_return * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate, 1),
        "volatility": round(volatility * 100, 2),  # Add the missing volatility field
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
