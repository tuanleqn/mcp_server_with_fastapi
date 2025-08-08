"""
Finance Calculations Server - Technical analysis with automatic symbol validation
"""
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import datetime

# Import from local helpers
from .finance_helpers import (
    get_database_connection,
    get_historical_stock_prices_helper,
    search_companies_helper,
    get_stock_prices_dataframe
)

load_dotenv()

mcp = FastMCP(name="Finance Calculations Server")

def validate_symbol_and_get_data(symbol: str, min_days: int = 50):
    """Validate symbol and get sufficient historical data."""
    if not symbol or not symbol.strip():
        return None, "Symbol cannot be empty"
    
    symbol = symbol.strip().upper()
    
    try:
        # Use the reliable DataFrame function from helpers
        from .finance_helpers import get_stock_prices_dataframe
        df = get_stock_prices_dataframe(symbol, days=min_days * 2)
        
        if df.empty or len(df) < min_days:
            # Try to search for similar symbols
            search_result = search_companies_helper(symbol, 3)
            suggestions = []
            if search_result.get("success") and search_result.get("companies"):
                suggestions = [f"{comp['symbol']} - {comp['name']}" for comp in search_result["companies"][:3]]
            
            return None, f"Insufficient data for {symbol}. Try: {', '.join(suggestions) if suggestions else 'AAPL, MSFT, SPY'}"
        
        return df, None
        
    except Exception as e:
        return None, f"Error getting data for {symbol}: {str(e)}"

@mcp.tool(description="Calculate comprehensive technical analysis for a stock")
def calculate_advanced_technical_analysis(symbol: str, period: int = 100) -> dict:
    """
    Calculate comprehensive technical analysis including RSI, moving averages, Bollinger bands, and MACD.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        period: Number of days for analysis (default 100)
        
    Returns:
        Complete technical analysis with trading signals
        
    Example:
        calculate_advanced_technical_analysis("AAPL", 50)
    """
    try:
        # Validate period
        if period < 20 or period > 500:
            return {
                "success": False,
                "error": "Period must be between 20 and 500 days"
            }
        
        # Get and validate data
        df, error = validate_symbol_and_get_data(symbol, period)
        if error:
            return {
                "success": False,
                "error": error
            }
        
        prices = df['close'].tail(period)
        
        # RSI Calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Moving Averages
        sma_20 = prices.rolling(window=20).mean()
        sma_50 = prices.rolling(window=50).mean()
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        
        # MACD
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        macd_histogram = macd - signal_line
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = prices.rolling(window=bb_period).mean()
        bb_std_dev = prices.rolling(window=bb_period).std()
        bb_upper = bb_sma + (bb_std_dev * bb_std)
        bb_lower = bb_sma - (bb_std_dev * bb_std)
        
        current_price = prices.iloc[-1]
        
        # Handle None current_price safely
        if current_price is None or pd.isna(current_price):
            current_price = 0.0
        
        # Trading Signals
        signals = []
        
        # RSI Signals
        if current_rsi > 70:
            signals.append("RSI indicates overbought condition - potential sell signal")
        elif current_rsi < 30:
            signals.append("RSI indicates oversold condition - potential buy signal")
        else:
            signals.append("RSI in neutral zone")
        
        # Moving Average Signals
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            signals.append("Bullish trend - price above moving averages")
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            signals.append("Bearish trend - price below moving averages")
        else:
            signals.append("Mixed signals from moving averages")
        
        # MACD Signals
        if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
            signals.append("MACD bullish crossover - potential buy signal")
        elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
            signals.append("MACD bearish crossover - potential sell signal")
        
        # Bollinger Bands Signals
        if current_price > bb_upper.iloc[-1]:
            signals.append("Price above upper Bollinger Band - potential overbought")
        elif current_price < bb_lower.iloc[-1]:
            signals.append("Price below lower Bollinger Band - potential oversold")
        
        return {
            "success": True,
            "symbol": symbol,
            "analysis_date": datetime.now().isoformat(),
            "current_price": round(current_price, 2),
            "period_analyzed": period,
            "technical_indicators": {
                "rsi": {
                    "current": round(current_rsi, 2),
                    "signal": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
                },
                "moving_averages": {
                    "sma_20": round(sma_20.iloc[-1], 2),
                    "sma_50": round(sma_50.iloc[-1], 2),
                    "price_vs_sma20": "above" if current_price > sma_20.iloc[-1] else "below",
                    "price_vs_sma50": "above" if current_price > sma_50.iloc[-1] else "below"
                },
                "macd": {
                    "macd_line": round(macd.iloc[-1], 4),
                    "signal_line": round(signal_line.iloc[-1], 4),
                    "histogram": round(macd_histogram.iloc[-1], 4),
                    "trend": "bullish" if macd.iloc[-1] > signal_line.iloc[-1] else "bearish"
                },
                "bollinger_bands": {
                    "upper_band": round(bb_upper.iloc[-1], 2),
                    "middle_band": round(bb_sma.iloc[-1], 2),
                    "lower_band": round(bb_lower.iloc[-1], 2),
                    "position": "above_upper" if current_price > bb_upper.iloc[-1] 
                               else "below_lower" if current_price < bb_lower.iloc[-1] 
                               else "within_bands"
                }
            },
            "trading_signals": signals,
            "overall_sentiment": "bullish" if len([s for s in signals if "buy" in s.lower() or "bullish" in s.lower()]) > len([s for s in signals if "sell" in s.lower() or "bearish" in s.lower()]) else "bearish" if len([s for s in signals if "sell" in s.lower() or "bearish" in s.lower()]) > 0 else "neutral"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error calculating technical analysis: {str(e)}"
        }

@mcp.tool(description="Calculate financial ratios and compare with market")
def calculate_financial_ratios(symbol: str, comparison_period: int = 252, period: int = None) -> dict:
    """
    Calculate financial ratios and performance metrics.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        comparison_period: Days for ratio calculation (default 252 - 1 year)
        
    Returns:
        Financial ratios and performance comparison
        
    Example:
        calculate_financial_ratios("AAPL", 180)
    """
    try:
        # Handle period parameter as alias for comparison_period
        if period is not None:
            comparison_period = period
            
        # Get and validate data
        df, error = validate_symbol_and_get_data(symbol, comparison_period)
        if error:
            return {
                "success": False,
                "error": error
            }
        
        prices = df['close'].tail(comparison_period)
        volumes = df['volume'].tail(comparison_period)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < 30:
            return {
                "success": False,
                "error": "Insufficient data for ratio calculations"
            }
        
        # Performance Metrics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annual_return = ((prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Risk Metrics
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 0 else 0
        
        # Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 2.0
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        
        # Trading Volume Analysis
        avg_volume = volumes.mean()
        volume_trend = "increasing" if volumes.tail(20).mean() > volumes.head(20).mean() else "decreasing"
        
        return {
            "success": True,
            "symbol": symbol,
            "analysis_date": datetime.now().isoformat(),
            "period_days": len(prices),
            "performance_metrics": {
                "total_return_percent": round(total_return, 2),
                "annualized_return_percent": round(annual_return, 2),
                "volatility_percent": round(volatility, 2),
                "sharpe_ratio": round(sharpe_ratio, 3)
            },
            "risk_metrics": {
                "maximum_drawdown_percent": round(abs(max_drawdown), 2),
                "value_at_risk_95_percent": round(abs(var_95), 2),
                "downside_deviation_percent": round(downside_deviation, 2),
                "risk_level": "high" if volatility > 30 else "medium" if volatility > 20 else "low"
            },
            "trading_metrics": {
                "average_daily_volume": int(avg_volume),
                "volume_trend": volume_trend,
                "liquidity_score": "high" if avg_volume > 1000000 else "medium" if avg_volume > 100000 else "low"
            },
            "comparison_benchmarks": {
                "sp500_typical_return": "8-12% annually",
                "sp500_typical_volatility": "15-20% annually",
                "performance_vs_market": "outperforming" if annual_return > 10 else "underperforming" if annual_return < 5 else "market_level"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error calculating financial ratios: {str(e)}"
        }

@mcp.tool(description="Calculate portfolio risk metrics and correlation analysis")
def calculate_portfolio_risk_metrics(symbols: list, weights: list = None, period: int = 252) -> dict:
    """
    Calculate comprehensive portfolio risk metrics including correlation matrix,
    VaR, and diversification ratios.
    
    Args:
        symbols: List of stock symbols
        weights: Optional list of weights (if not provided, equal weights assumed)
        
    Returns:
        Portfolio risk metrics and analysis
    """
    try:
        if not symbols:
            return {
                "success": False,
                "error": "Symbols list cannot be empty"
            }
        
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0/len(symbols)] * len(symbols)
        elif len(weights) != len(symbols):
            return {
                "success": False,
                "error": "Weights and symbols must have same length"
            }
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Get returns data for all symbols
        returns_data = {}
        for symbol in symbols:
            # Use the working DataFrame function instead
            df = get_stock_prices_dataframe(symbol, days=max(period, 252))
            if not df.empty and len(df) > 30:
                prices = df['close'].tolist()
                # Calculate returns correctly: (price_t / price_t-1) - 1
                returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                returns_data[symbol] = returns[-period:] if len(returns) > period else returns
        
        if len(returns_data) < 2:
            return {
                "success": False,
                "error": "Need at least 2 valid symbols with price data"
            }
        
        # Calculate correlation matrix
        import numpy as np
        symbols_with_data = list(returns_data.keys())
        n_assets = len(symbols_with_data)
        correlation_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                returns1 = returns_data[symbols_with_data[i]]
                returns2 = returns_data[symbols_with_data[j]]
                min_len = min(len(returns1), len(returns2))
                r1, r2 = returns1[:min_len], returns2[:min_len]
                
                if min_len > 10:
                    corr = np.corrcoef(r1, r2)[0, 1]
                    if not np.isnan(corr):
                        correlation_matrix[i][j] = corr
                        correlation_matrix[j][i] = corr
        
        # Portfolio volatility and return calculation
        portfolio_returns = []
        portfolio_cumulative_return = 0
        max_len = max(len(returns_data[s]) for s in symbols_with_data) if symbols_with_data else 0
        
        for t in range(min(100, max_len)):
            portfolio_return = 0
            for i, symbol in enumerate(symbols_with_data):
                if symbol in symbols:
                    weight_idx = symbols.index(symbol)
                    if t < len(returns_data[symbol]):
                        portfolio_return += weights[weight_idx] * returns_data[symbol][t]
            portfolio_returns.append(portfolio_return)
        
        # Calculate annualized portfolio return
        if portfolio_returns:
            portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
            portfolio_mean_return = np.mean(portfolio_returns)
            portfolio_annual_return = portfolio_mean_return * 252  # Annualize
            
            # Calculate Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        else:
            portfolio_vol = 0
            portfolio_annual_return = 0
            sharpe_ratio = 0
        
        # Value at Risk (95% confidence)
        if portfolio_returns:
            portfolio_returns_sorted = sorted(portfolio_returns)
            var_95 = portfolio_returns_sorted[int(0.05 * len(portfolio_returns_sorted))]
            var_95_annual = var_95 * np.sqrt(252)
        else:
            var_95_annual = 0
        
        return {
            "success": True,
            "portfolio_composition": [
                {"symbol": symbols[i], "weight_percent": weights[i] * 100} 
                for i in range(len(symbols))
            ],
            "portfolio_metrics": {
                "annualized_return_percent": round(portfolio_annual_return * 100, 2),
                "sharpe_ratio": round(sharpe_ratio, 3)
            },
            "risk_metrics": {
                "portfolio_volatility_annual": round(portfolio_vol * 100, 2),
                "value_at_risk_95": round(abs(var_95_annual) * 100, 2),
                "number_of_assets": len(symbols),
                "diversification_ratio": round(1 - np.mean(correlation_matrix), 3)
            },
            "correlation_analysis": {
                "average_correlation": round(np.mean(correlation_matrix[correlation_matrix != 1]), 3),
                "max_correlation": round(np.max(correlation_matrix[correlation_matrix != 1]), 3),
                "min_correlation": round(np.min(correlation_matrix[correlation_matrix != 1]), 3)
            },
            "symbols_analyzed": symbols_with_data,
            "analysis_period": f"{min(100, max_len)} trading days"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error calculating portfolio risk metrics: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run()
