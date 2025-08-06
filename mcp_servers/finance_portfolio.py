"""
Finance Portfolio Server
Simplified MCP server for portfolio analysis using helper functions
"""

import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from datetime import datetime
from .finance_helpers import get_historical_prices_helper
import numpy as np
import pandas as pd

load_dotenv()

mcp = FastMCP(name="Finance Portfolio Server")

@mcp.tool(description="Calculate portfolio risk metrics")
def analyze_portfolio(symbols: list, weights: list = None, period: int = 252) -> dict:
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
    
    # Handle different input formats for symbols
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    elif isinstance(symbols, list):
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    else:
        return {
            "success": False,
            "error": f"Invalid symbols format. Expected list or comma-separated string, got {type(symbols)}"
        }
    
    if not symbols:
        return {
            "success": False,
            "error": "No valid symbols provided after processing"
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

@mcp.tool(description="Optimize portfolio weights for equal risk contribution")
def optimize_equal_risk_portfolio(symbols) -> dict:
    """
    Create an equal risk contribution portfolio.
    
    Args:
        symbols: List of stock symbols (string list or comma-separated string)
    
    Returns:
        Dictionary containing optimized portfolio weights
    """
    # Handle different input formats
    if not symbols:
        return {
            "success": False,
            "error": "No symbols provided"
        }
    
    # Convert string to list if needed
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    elif isinstance(symbols, list):
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    else:
        return {
            "success": False,
            "error": f"Invalid symbols format. Expected list or comma-separated string, got {type(symbols)}"
        }
    
    if not symbols:
        return {
            "success": False,
            "error": "No valid symbols provided after processing"
        }
    
    print(f"DEBUG: Processing symbols: {symbols}")
    
    symbol_volatilities = []
    valid_symbols = []
    
    for symbol in symbols:
        df = get_historical_prices_helper(symbol, 252)  # 1 year
        if not df.empty and len(df) > 50:
            returns = df['close_price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            symbol_volatilities.append(volatility)
            valid_symbols.append(symbol)
            print(f"DEBUG: {symbol} volatility: {volatility:.4f}")
        else:
            print(f"DEBUG: Insufficient data for {symbol}")
    
    if not symbol_volatilities:
        return {
            "success": False,
            "error": f"No symbols have sufficient historical data. Tried: {', '.join(symbols)}"
        }
    
    # Inverse volatility weighting
    inverse_vols = [1 / vol if vol > 0 else 0 for vol in symbol_volatilities]
    total_inverse = sum(inverse_vols)
    
    if total_inverse == 0:
        # Fallback to equal weighting
        weights = [1.0 / len(valid_symbols)] * len(valid_symbols)
    else:
        weights = [inv_vol / total_inverse for inv_vol in inverse_vols]
    
    print(f"DEBUG: Calculated weights: {weights}")
    
    # Analyze the optimized portfolio
    portfolio_analysis = analyze_portfolio(valid_symbols, weights)
    
    if portfolio_analysis.get("success"):
        portfolio_analysis["optimization_method"] = "equal_risk_contribution"
        portfolio_analysis["optimization_note"] = "Weights are inversely proportional to individual asset volatility"
        portfolio_analysis["optimized_weights"] = dict(zip(valid_symbols, weights))
    
    return portfolio_analysis

if __name__ == "__main__":
    mcp.run()
