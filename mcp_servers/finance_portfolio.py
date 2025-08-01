import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
import numpy as np

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Portfolio Optimization")

def _get_historical_returns(symbol: str, days: int = 252):
    """Get historical returns for a symbol"""
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                query = """
                SELECT close_price
                FROM public.stock_price
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
                """
                cur.execute(query, (symbol.upper(), days + 1))
                results = cur.fetchall()
                
                if len(results) < 2:
                    return []
                
                prices = [float(row[0]) for row in reversed(results)]
                returns = []
                
                for i in range(1, len(prices)):
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
                
                return returns
                
    except Error as e:
        print(f"Database error getting returns for {symbol}: {e}")
        return []

@mcp.tool(description="Simple portfolio optimization using equal weighting")
def optimize_portfolio(symbols: list, target_return: float = 0.1) -> dict:
    """
    Simple portfolio optimization - equal weight allocation
    
    Args:
        symbols (list): List of stock symbols
        target_return (float): Target annual return (0.1 = 10%)
        
    Returns:
        dict: Optimized portfolio weights and expected metrics
    """
    try:
        if not symbols or len(symbols) == 0:
            return {
                "success": False,
                "error": "No symbols provided"
            }
        
        # Get returns for all symbols
        all_returns = {}
        valid_symbols = []
        
        for symbol in symbols:
            returns = _get_historical_returns(symbol, 252)  # 1 year of data
            if len(returns) > 50:  # Need at least 50 days of data
                all_returns[symbol] = returns
                valid_symbols.append(symbol)
        
        if len(valid_symbols) == 0:
            return {
                "success": False,
                "error": "No symbols have sufficient historical data"
            }
        
        # Simple equal weighting
        equal_weight = 1.0 / len(valid_symbols)
        optimal_weights = {symbol: equal_weight for symbol in valid_symbols}
        
        # Calculate expected return and risk
        portfolio_returns = []
        min_length = min(len(returns) for returns in all_returns.values())
        
        for i in range(min_length):
            daily_portfolio_return = sum(
                all_returns[symbol][i] * optimal_weights[symbol] 
                for symbol in valid_symbols
            )
            portfolio_returns.append(daily_portfolio_return)
        
        # Calculate metrics
        expected_return = np.mean(portfolio_returns) * 252  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        return {
            "success": True,
            "valid_symbols": valid_symbols,
            "optimal_weights": optimal_weights,
            "expected_return": round(expected_return * 100, 2),  # As percentage
            "portfolio_volatility": round(volatility * 100, 2),  # As percentage
            "sharpe_ratio": round(sharpe_ratio, 3),
            "target_return": target_return * 100,
            "optimization_method": "Equal Weight Allocation"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Portfolio optimization failed: {str(e)}"
        }

@mcp.tool(description="Calculate portfolio risk metrics")
def calculate_portfolio_risk(symbols: list, weights: list) -> dict:
    """
    Calculate portfolio risk metrics
    
    Args:
        symbols (list): List of stock symbols
        weights (list): List of weights for each symbol
        
    Returns:
        dict: Portfolio risk metrics
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
        
        # Get returns for all symbols
        all_returns = {}
        valid_data = []
        
        for i, symbol in enumerate(symbols):
            returns = _get_historical_returns(symbol, 252)
            if len(returns) > 50:
                all_returns[symbol] = returns
                valid_data.append((symbol, weights[i]))
        
        if len(valid_data) == 0:
            return {
                "success": False,
                "error": "No symbols have sufficient historical data"
            }
        
        # Calculate portfolio returns
        portfolio_returns = []
        min_length = min(len(all_returns[symbol]) for symbol, _ in valid_data)
        
        for i in range(min_length):
            daily_return = sum(
                all_returns[symbol][i] * weight 
                for symbol, weight in valid_data
            )
            portfolio_returns.append(daily_return)
        
        # Calculate risk metrics
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)  # 95% VaR
        max_drawdown = 0
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + np.array(portfolio_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            "success": True,
            "portfolio_volatility": round(portfolio_vol * 100, 2),
            "value_at_risk_95": round(var_95 * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "valid_symbols": [symbol for symbol, _ in valid_data],
            "analysis_period_days": min_length
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Risk calculation failed: {str(e)}"
        }
