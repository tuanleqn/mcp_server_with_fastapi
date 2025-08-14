"""
Finance Portfolio Optimization MCP Server
Optimized portfolio allocation tool with risk-return optimization.
Provides recommended allocation rates using Modern Portfolio Theory.
"""
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import from local helpers
from .finance_helpers import (
    get_database_connection,
    get_stock_prices_dataframe,
    search_companies_helper
)

load_dotenv()

mcp = FastMCP(name="Finance Portfolio Optimization Server")

<<<<<<< HEAD

@mcp.tool(description="Calculate portfolio risk metrics")
def analyze_portfolio(symbols: list, weights: list = None, period: int = 252) -> dict:
    if not symbols:
        return {"success": False, "error": "No symbols provided"}

    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif isinstance(symbols, list):
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    else:
        return {
            "success": False,
            "error": f"Invalid symbols format. Expected list or comma-separated string, got {type(symbols)}",
        }

    if not symbols:
        return {"success": False, "error": "No valid symbols provided after processing"}

    if weights is None:
        weights = [1.0 / len(symbols)] * len(symbols)

    if len(weights) != len(symbols):
        return {
            "success": False,
            "error": "Number of weights must match number of symbols",
        }

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    returns_data = {}
    for symbol in symbols:
        df = get_historical_prices_helper(symbol, period + 10)
        if not df.empty and len(df) > 50:
            returns = df["close_price"].pct_change().dropna()
            returns_data[symbol] = returns

    if not returns_data:
        return {"success": False, "error": "No valid data found for any symbols"}

    valid_symbols = list(returns_data.keys())
    valid_weights = weights[: len(valid_symbols)]

    portfolio_returns = []
    min_length = min(len(returns) for returns in returns_data.values())

    for i in range(min_length):
        daily_return = sum(
            returns_data[symbol].iloc[i] * weight
            for symbol, weight in zip(valid_symbols, valid_weights)
        )
        portfolio_returns.append(daily_return)

    portfolio_returns = pd.Series(portfolio_returns)

    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    portfolio_return = portfolio_returns.mean() * 252

    var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    sharpe_ratio = (
        portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    )

    return {
        "success": True,
        "portfolio_return": round(portfolio_return * 100, 2),  # Convert to percentage
        "portfolio_volatility": round(
            portfolio_volatility * 100, 2
        ),  # Convert to percentage
        "sharpe_ratio": round(sharpe_ratio, 4),
        "value_at_risk_95": round(
            abs(var_95) * 100, 2
        ),  # Convert to positive percentage
        "max_drawdown": round(
            abs(max_drawdown) * 100, 2
        ),  # Convert to positive percentage
        "portfolio_composition": [
            {
                "symbol": symbol,
                "weight": round(weight, 4),
                "weight_percent": round(weight * 100, 2),
            }
            for symbol, weight in zip(valid_symbols, valid_weights)
        ],
        "risk_metrics": {
            "annual_return": round(portfolio_return, 4),
            "annual_volatility": round(portfolio_volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "value_at_risk_95": round(var_95, 4),
            "max_drawdown": round(max_drawdown, 4),
            "risk_level": "high"
            if portfolio_volatility > 0.25
            else "medium"
            if portfolio_volatility > 0.15
            else "low",
        },
        "diversification": {
            "number_of_assets": len(valid_symbols),
            "effective_assets": round(1 / sum(w**2 for w in valid_weights), 2),
            "concentration_risk": "high"
            if max(valid_weights) > 0.4
            else "medium"
            if max(valid_weights) > 0.25
            else "low",
        },
        "analysis_date": datetime.now().isoformat(),
        "analysis_period_days": len(portfolio_returns),
    }


@mcp.tool(description="Optimize portfolio weights for equal risk contribution")
def optimize_equal_risk_portfolio(symbols) -> dict:
    if not symbols:
        return {"success": False, "error": "No symbols provided"}

    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif isinstance(symbols, list):
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    else:
        return {
            "success": False,
            "error": f"Invalid symbols format. Expected list or comma-separated string, got {type(symbols)}",
        }

    if not symbols:
        return {"success": False, "error": "No valid symbols provided after processing"}

    symbol_volatilities = []
    valid_symbols = []

    for symbol in symbols:
        df = get_historical_prices_helper(symbol, 252)
        if not df.empty and len(df) > 50:
            returns = df["close_price"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            symbol_volatilities.append(volatility)
            valid_symbols.append(symbol)

    if not symbol_volatilities:
        return {
            "success": False,
            "error": f"No symbols have sufficient historical data",
        }

    inverse_vols = [1 / vol if vol > 0 else 0 for vol in symbol_volatilities]
    total_inverse = sum(inverse_vols)

    if total_inverse == 0:
        weights = [1.0 / len(valid_symbols)] * len(valid_symbols)
    else:
        weights = [inv_vol / total_inverse for inv_vol in inverse_vols]

    portfolio_analysis = analyze_portfolio(valid_symbols, weights)

    if portfolio_analysis.get("success"):
        portfolio_analysis["optimization_method"] = "equal_risk_contribution"
        portfolio_analysis["optimization_note"] = (
            "Weights are inversely proportional to individual asset volatility"
        )
        portfolio_analysis["optimized_weights"] = dict(zip(valid_symbols, weights))

    return portfolio_analysis
=======
@mcp.tool(description="Optimize portfolio allocation with recommended rates based on risk-return analysis")
def optimize_portfolio_allocation(symbols: list) -> dict:
    """
    Optimize portfolio allocation with recommended rates based on risk-return characteristics.
    
    Args:
        symbols: List of stock symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
        
    Returns:
        Optimized portfolio allocation with recommended rates:
        - Recommended allocation percentages for each asset
        - Risk-adjusted optimization strategy
        - Expected portfolio metrics
        - Alternative allocation strategies
        
    Example:
        optimize_portfolio_allocation(["AAPL", "MSFT", "GOOGL"])
    """
    try:
        if not symbols:
            return {
                "success": False,
                "error": "Symbols list cannot be empty"
            }
        
        symbol_list = [str(s).strip().upper() for s in symbols if str(s).strip()]
        
        if not symbol_list:
            return {
                "success": False,
                "error": "No valid symbols provided"
            }
        
        # Get price data and calculate returns for all symbols
        returns_matrix = []
        valid_symbols = []
        asset_metrics = []
        
        for symbol in symbol_list:
            df = get_stock_prices_dataframe(symbol, days=252)
            if not df.empty and len(df) > 60:  # Need more data for optimization
                returns = df['close_price'].pct_change().dropna()
                if len(returns) > 50:
                    returns_matrix.append(returns.tolist())
                    valid_symbols.append(symbol)
                    
                    # Calculate individual metrics
                    annual_return = returns.mean() * 252
                    annual_vol = returns.std() * (252 ** 0.5)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    current_price = float(df['close_price'].iloc[-1]) if df['close_price'].iloc[-1] is not None else 0.0
                    
                    asset_metrics.append({
                        "symbol": symbol,
                        "annual_return": round(annual_return, 4),
                        "annual_volatility": round(annual_vol, 4),
                        "sharpe_ratio": round(sharpe, 4),
                        "current_price": round(current_price, 2)
                    })
        
        if len(valid_symbols) < 2:
            return {
                "success": False,
                "error": "Need at least 2 symbols with sufficient data for optimization"
            }
        
        # Align returns data to same length
        min_length = min(len(returns) for returns in returns_matrix)
        aligned_returns = np.array([returns[-min_length:] for returns in returns_matrix])
        
        # Calculate expected returns and covariance matrix
        expected_returns = np.array([np.mean(returns) * 252 for returns in aligned_returns])
        cov_matrix = np.cov(aligned_returns) * 252
        
        # Use moderate risk aversion for optimization
        lambda_risk = 5  # Moderate risk aversion parameter
        
        # Mean-Variance Optimization using analytical solution
        # Optimal weights = (λ * Σ)^-1 * μ / (1^T * (λ * Σ)^-1 * μ)
        try:
            inv_cov = np.linalg.inv(lambda_risk * cov_matrix)
            numerator = inv_cov @ expected_returns
            denominator = np.ones(len(valid_symbols)).T @ numerator
            optimal_weights = numerator / denominator
            
            # Ensure weights are non-negative and sum to 1
            optimal_weights = np.maximum(optimal_weights, 0)
            if np.sum(optimal_weights) > 0:
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            else:
                # Fallback to equal weights
                optimal_weights = np.ones(len(valid_symbols)) / len(valid_symbols)
                
        except np.linalg.LinAlgError:
            # If matrix is singular, use equal weights
            optimal_weights = np.ones(len(valid_symbols)) / len(valid_symbols)
        
        # Calculate portfolio metrics with optimal weights
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
        portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Create recommended allocations
        recommendations = []
        for i, symbol in enumerate(valid_symbols):
            weight = optimal_weights[i]
            recommendations.append({
                "symbol": symbol,
                "recommended_allocation": round(weight, 4),
                "allocation_percent": round(weight * 100, 2),
                "expected_contribution": round(weight * expected_returns[i], 4),
                "risk_contribution": round(weight * asset_metrics[i]["annual_volatility"], 4)
            })
        
        # Sort by allocation percentage (highest first)
        recommendations.sort(key=lambda x: x["allocation_percent"], reverse=True)
        
        # Generate alternative strategies
        alternatives = {
            "equal_weight": {
                "description": "Equal allocation across all assets",
                "allocations": {symbol: round(100/len(valid_symbols), 2) for symbol in valid_symbols}
            },
            "market_cap_proxy": {
                "description": "Higher allocation to larger companies (by current price)",
                "allocations": {}
            },
            "risk_parity": {
                "description": "Allocation inversely proportional to volatility",
                "allocations": {}
            }
        }
        
        # Market cap proxy (using current price as rough proxy)
        total_price = sum(asset["current_price"] for asset in asset_metrics)
        if total_price > 0:
            for asset in asset_metrics:
                weight = asset["current_price"] / total_price
                alternatives["market_cap_proxy"]["allocations"][asset["symbol"]] = round(weight * 100, 2)
        
        # Risk parity (inverse volatility weighting)
        inv_vols = [1/asset["annual_volatility"] if asset["annual_volatility"] > 0 else 1 for asset in asset_metrics]
        total_inv_vol = sum(inv_vols)
        if total_inv_vol > 0:
            for i, asset in enumerate(asset_metrics):
                weight = inv_vols[i] / total_inv_vol
                alternatives["risk_parity"]["allocations"][asset["symbol"]] = round(weight * 100, 2)
        
        return {
            "success": True,
            "optimization_strategy": "Modern Portfolio Theory Optimization",
            "recommended_allocations": recommendations,
            "portfolio_metrics": {
                "expected_annual_return": round(portfolio_return, 4),
                "expected_annual_volatility": round(portfolio_volatility, 4),
                "expected_sharpe_ratio": round(portfolio_sharpe, 4),
                "optimization_method": "Mean-Variance Optimization"
            },
            "allocation_summary": {
                "total_assets": len(valid_symbols),
                "largest_allocation": {
                    "symbol": recommendations[0]["symbol"],
                    "percent": recommendations[0]["allocation_percent"]
                },
                "most_diversified": len([r for r in recommendations if r["allocation_percent"] >= 10]),
                "concentration_score": round(sum(w**2 for w in optimal_weights), 3)  # Lower is more diversified
            },
            "alternative_strategies": alternatives,
            "investment_guidance": {
                "risk_profile": "moderate",
                "rebalancing_frequency": "quarterly",
                "minimum_investment_horizon": "1-2 years",
                "diversification_benefit": f"Portfolio expected to reduce risk by {round((1 - portfolio_volatility/np.mean([a['annual_volatility'] for a in asset_metrics]))*100, 1)}% vs average individual asset"
            },
            "analysis_details": {
                "data_period_days": min_length,
                "optimization_date": datetime.now().isoformat(),
                "total_symbols_requested": len(symbol_list),
                "valid_symbols_optimized": len(valid_symbols)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error optimizing portfolio allocation: {str(e)}"
        }
>>>>>>> 33953cb14d34db5736c7d80be4a4a7fb129b0752


if __name__ == "__main__":
    mcp.run()
