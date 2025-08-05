"""
Finance Portfolio Server
Simplified MCP server for portfolio analysis using helper functions
"""

import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from .finance_helpers import get_historical_prices_helper
import numpy as np

load_dotenv()

mcp = FastMCP(name="Finance Portfolio Server")

@mcp.tool(description="Calculate portfolio return and risk metrics")
def analyze_portfolio(symbols: list, weights: list = None) -> dict:
    """
    Analyze portfolio performance with given symbols and weights.
    
    Args:
        symbols: List of stock symbols
        weights: List of weights (if None, equal weighting is used)
    
    Returns:
        Dictionary containing portfolio analysis
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
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    portfolio_data = []
    valid_symbols = []
    valid_weights = []
    
    for i, symbol in enumerate(symbols):
        df = get_historical_prices_helper(symbol, 252)  # 1 year
        if not df.empty and len(df) > 50:
            returns = df['close_price'].pct_change().dropna()
            portfolio_data.append({
                'symbol': symbol,
                'returns': returns,
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'weight': weights[i]
            })
            valid_symbols.append(symbol)
            valid_weights.append(weights[i])
    
    if not portfolio_data:
        return {
            "success": False,
            "error": "No symbols have sufficient historical data"
        }
    
    # Calculate portfolio metrics
    portfolio_return = sum(
        data['mean_return'] * data['weight'] 
        for data in portfolio_data
    ) * 252  # Annualized
    
    # Simple portfolio volatility (assumes no correlation)
    portfolio_volatility = np.sqrt(sum(
        (data['weight'] * data['volatility']) ** 2 
        for data in portfolio_data
    )) * np.sqrt(252)  # Annualized
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        "success": True,
        # Add top-level fields that showcase expects
        "portfolio_return": round(portfolio_return * 100, 2),  # Convert to percentage
        "portfolio_volatility": round(portfolio_volatility * 100, 2),  # Convert to percentage  
        "sharpe_ratio": round(sharpe_ratio, 4),
        "portfolio_composition": [
            {
                "symbol": data['symbol'],
                "weight": round(data['weight'], 4),
                "weight_percent": round(data['weight'] * 100, 2),
                "annual_return": round(data['mean_return'] * 252, 4),
                "annual_volatility": round(data['volatility'] * np.sqrt(252), 4)
            }
            for data in portfolio_data
        ],
        "portfolio_metrics": {
            "expected_annual_return": round(portfolio_return, 4),
            "annual_volatility": round(portfolio_volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "number_of_holdings": len(valid_symbols)
        },
        "risk_analysis": {
            "risk_level": "high" if portfolio_volatility > 0.25 else "medium" if portfolio_volatility > 0.15 else "low",
            "diversification_score": min(len(valid_symbols) / 10, 1.0)  # Simple diversification score
        }
    }

@mcp.tool(description="Optimize portfolio weights for equal risk contribution")
def optimize_equal_risk_portfolio(symbols: list) -> dict:
    """
    Create an equal risk contribution portfolio.
    
    Args:
        symbols: List of stock symbols
    
    Returns:
        Dictionary containing optimized portfolio weights
    """
    if not symbols:
        return {
            "success": False,
            "error": "No symbols provided"
        }
    
    symbol_volatilities = []
    valid_symbols = []
    
    for symbol in symbols:
        df = get_historical_prices_helper(symbol, 252)  # 1 year
        if not df.empty and len(df) > 50:
            returns = df['close_price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            symbol_volatilities.append(volatility)
            valid_symbols.append(symbol)
    
    if not symbol_volatilities:
        return {
            "success": False,
            "error": "No symbols have sufficient historical data"
        }
    
    # Inverse volatility weighting
    inverse_vols = [1 / vol if vol > 0 else 0 for vol in symbol_volatilities]
    total_inverse = sum(inverse_vols)
    
    if total_inverse == 0:
        # Fallback to equal weighting
        weights = [1.0 / len(valid_symbols)] * len(valid_symbols)
    else:
        weights = [inv_vol / total_inverse for inv_vol in inverse_vols]
    
    # Analyze the optimized portfolio
    portfolio_analysis = analyze_portfolio(valid_symbols, weights)
    
    if portfolio_analysis.get("success"):
        portfolio_analysis["optimization_method"] = "equal_risk_contribution"
        portfolio_analysis["optimization_note"] = "Weights are inversely proportional to individual asset volatility"
    
    return portfolio_analysis

if __name__ == "__main__":
    mcp.run()
