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

warnings.filterwarnings("ignore")

# Import from local helpers
from .finance_helpers import (
    get_database_connection,
    get_stock_prices_dataframe,
    search_companies_helper,
)

load_dotenv()

mcp = FastMCP(name="Finance Portfolio Optimization Server")


@mcp.tool(
    description="Optimize portfolio allocation with recommended rates based on risk-return analysis"
)
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
            return {"success": False, "error": "Symbols list cannot be empty"}

        symbol_list = [str(s).strip().upper() for s in symbols if str(s).strip()]

        if not symbol_list:
            return {"success": False, "error": "No valid symbols provided"}

        # Get price data and calculate returns for all symbols
        returns_matrix = []
        valid_symbols = []
        asset_metrics = []

        for symbol in symbol_list:
            df = get_stock_prices_dataframe(symbol, days=252)
            if not df.empty and len(df) > 60:  # Need more data for optimization
                returns = df["close_price"].pct_change().dropna()
                if len(returns) > 50:
                    returns_matrix.append(returns.tolist())
                    valid_symbols.append(symbol)

                    # Calculate individual metrics
                    annual_return = returns.mean() * 252
                    annual_vol = returns.std() * (252**0.5)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    current_price = (
                        float(df["close_price"].iloc[-1])
                        if df["close_price"].iloc[-1] is not None
                        else 0.0
                    )

                    asset_metrics.append(
                        {
                            "symbol": symbol,
                            "annual_return": round(annual_return, 4),
                            "annual_volatility": round(annual_vol, 4),
                            "sharpe_ratio": round(sharpe, 4),
                            "current_price": round(current_price, 2),
                        }
                    )

        if len(valid_symbols) < 2:
            return {
                "success": False,
                "error": "Need at least 2 symbols with sufficient data for optimization",
            }

        # Align returns data to same length
        min_length = min(len(returns) for returns in returns_matrix)
        aligned_returns = np.array(
            [returns[-min_length:] for returns in returns_matrix]
        )

        # Calculate expected returns and covariance matrix
        expected_returns = np.array(
            [np.mean(returns) * 252 for returns in aligned_returns]
        )
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
        portfolio_sharpe = (
            portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        )

        # Create recommended allocations
        recommendations = []
        for i, symbol in enumerate(valid_symbols):
            weight = optimal_weights[i]
            recommendations.append(
                {
                    "symbol": symbol,
                    "recommended_allocation": round(weight, 4),
                    "allocation_percent": round(weight * 100, 2),
                    "expected_contribution": round(weight * expected_returns[i], 4),
                    "risk_contribution": round(
                        weight * asset_metrics[i]["annual_volatility"], 4
                    ),
                }
            )

        # Sort by allocation percentage (highest first)
        recommendations.sort(key=lambda x: x["allocation_percent"], reverse=True)

        # Generate alternative strategies
        alternatives = {
            "equal_weight": {
                "description": "Equal allocation across all assets",
                "allocations": {
                    symbol: round(100 / len(valid_symbols), 2)
                    for symbol in valid_symbols
                },
            },
            "market_cap_proxy": {
                "description": "Higher allocation to larger companies (by current price)",
                "allocations": {},
            },
            "risk_parity": {
                "description": "Allocation inversely proportional to volatility",
                "allocations": {},
            },
        }

        # Market cap proxy (using current price as rough proxy)
        total_price = sum(asset["current_price"] for asset in asset_metrics)
        if total_price > 0:
            for asset in asset_metrics:
                weight = asset["current_price"] / total_price
                alternatives["market_cap_proxy"]["allocations"][asset["symbol"]] = (
                    round(weight * 100, 2)
                )

        # Risk parity (inverse volatility weighting)
        inv_vols = [
            1 / asset["annual_volatility"] if asset["annual_volatility"] > 0 else 1
            for asset in asset_metrics
        ]
        total_inv_vol = sum(inv_vols)
        if total_inv_vol > 0:
            for i, asset in enumerate(asset_metrics):
                weight = inv_vols[i] / total_inv_vol
                alternatives["risk_parity"]["allocations"][asset["symbol"]] = round(
                    weight * 100, 2
                )

        return {
            "success": True,
            "optimization_strategy": "Modern Portfolio Theory Optimization",
            "recommended_allocations": recommendations,
            "portfolio_metrics": {
                "expected_annual_return": round(portfolio_return, 4),
                "expected_annual_volatility": round(portfolio_volatility, 4),
                "expected_sharpe_ratio": round(portfolio_sharpe, 4),
                "optimization_method": "Mean-Variance Optimization",
            },
            "allocation_summary": {
                "total_assets": len(valid_symbols),
                "largest_allocation": {
                    "symbol": recommendations[0]["symbol"],
                    "percent": recommendations[0]["allocation_percent"],
                },
                "most_diversified": len(
                    [r for r in recommendations if r["allocation_percent"] >= 10]
                ),
                "concentration_score": round(
                    sum(w**2 for w in optimal_weights), 3
                ),  # Lower is more diversified
            },
            "alternative_strategies": alternatives,
            "investment_guidance": {
                "risk_profile": "moderate",
                "rebalancing_frequency": "quarterly",
                "minimum_investment_horizon": "1-2 years",
                "diversification_benefit": f"Portfolio expected to reduce risk by {round((1 - portfolio_volatility / np.mean([a['annual_volatility'] for a in asset_metrics])) * 100, 1)}% vs average individual asset",
            },
            "analysis_details": {
                "data_period_days": min_length,
                "optimization_date": datetime.now().isoformat(),
                "total_symbols_requested": len(symbol_list),
                "valid_symbols_optimized": len(valid_symbols),
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error optimizing portfolio allocation: {str(e)}",
        }


if __name__ == "__main__":
    mcp.run()
