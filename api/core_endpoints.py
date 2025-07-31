"""
Core Finance API Endpoints
Basic market data functionality: stocks, watchlist, charts, news, market overview
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random

# Import MCP servers
from mcp_servers import (
    finance_market_data,
    finance_news_and_insights,
    finance_db_company,
    finance_db_stock_price,
    finance_data_ingestion,
    finance_symbol_discovery
)

# Import models
from .finance_models import (
    StockData, MarketData, ChartDataPoint, NewsItem, CryptoData, 
    CommodityData, IndexData, CompanyInfo, MarketOverview
)

# Import data utilities (with error handling for missing utils)
try:
    from utils.data_import import data_importer, ensure_fresh_data, batch_import_data
except ImportError:
    data_importer = None
    ensure_fresh_data = None
    batch_import_data = None

router = APIRouter()

# Top 10 Big Tech Stocks + Market Indices + Commodities + Crypto
RELIABLE_SYMBOLS = {
    # Big Tech (Top 10)
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc. (Google)",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "NFLX": "Netflix Inc.",
    "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corporation",
    
    # Major Market Indices
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "DIA": "Dow Jones Industrial Average ETF",
    "VTI": "Total Stock Market ETF",
    "IWM": "Russell 2000 ETF",
    
    # Commodities & Precious Metals
    "GLD": "Gold ETF",
    "SLV": "Silver ETF",
    "USO": "Oil ETF",
    "UNG": "Natural Gas ETF",
    
    # Major Cryptocurrencies (ETFs for reliable data)
    "BITO": "Bitcoin Strategy ETF",
    "ETHE": "Ethereum Trust"
}

def get_local_stock_data(symbol: str) -> dict:
    """Get stock data from local database"""
    try:
        # First check if we have fresh local data using data_importer if available
        if data_importer:
            local_data = data_importer.get_cached_stock_data(symbol)
            if local_data:
                return {
                    "success": True,
                    "data": local_data,
                    "source": "local_database"
                }
        
        # If no local cached data, try MCP database servers directly
        try:
            db_result = finance_db_stock_price.get_latest_price(symbol=symbol)
            if db_result:
                return {
                    "success": True,
                    "data": {
                        "symbol": symbol,
                        "close": db_result.get('close_price', 0.0),
                        "high": db_result.get('high_price', 0.0),
                        "low": db_result.get('low_price', 0.0),
                        "open": db_result.get('open_price', 0.0),
                        "volume": db_result.get('volume', 0),
                        "date": db_result.get('date', ''),
                        "change": 0.0,  # Calculate if previous data available
                        "change_percent": 0.0
                    },
                    "source": "mcp_database"
                }
        except Exception as e:
            print(f"MCP database access failed for {symbol}: {e}")
        
        # If no local data, return None to trigger external fetch
        return None
        
    except Exception as e:
        print(f"Error getting local stock data for {symbol}: {e}")
        return None

def get_mcp_market_data(symbol: str) -> dict:
    """Get market data from MCP server (fallback only)"""
    try:
        result = finance_market_data.get_market_data(
            symbol=symbol,
            interval="1day",
            period="1month", 
            format_for_chart=True
        )
        return result
    except Exception as e:
        print(f"Error getting market data for {symbol}: {e}")
        return None

def format_stock_data(symbol: str, local_data: dict = None, mcp_data: dict = None) -> StockData:
    """Convert local or MCP data to StockData format"""
    
    # Use local data if available
    if local_data and local_data.get('success'):
        data = local_data.get('data', {})
        return StockData(
            symbol=symbol,
            company=RELIABLE_SYMBOLS.get(symbol, f"{symbol} Corporation"),
            price=float(data.get('close', 100.0)),
            change=float(data.get('change', 0.0)),
            changePercent=float(data.get('change_percent', 0.0))
        )
    
    # Fallback to MCP data
    if mcp_data and mcp_data.get('success'):
        chart_data = mcp_data.get('chart_data', {})
        statistics = chart_data.get('statistics', {})
        
        current_price = statistics.get('current_price', 100.0)
        change = statistics.get('change', 0.0)
        change_percent = statistics.get('change_percent', 0.0)
        
        return StockData(
            symbol=symbol,
            company=mcp_data.get('display_name', RELIABLE_SYMBOLS.get(symbol, f"{symbol} Corporation")),
            price=float(current_price),
            change=float(change),
            changePercent=float(change_percent)
        )
    
    # Final fallback to sample data
    return StockData(
        symbol=symbol,
        company=RELIABLE_SYMBOLS.get(symbol, f"{symbol} Corporation"),
        price=round(random.uniform(50, 200), 2),
        change=round(random.uniform(-5, 5), 2),
        changePercent=round(random.uniform(-3, 3), 2)
    )

def format_market_data(symbol: str, local_data: dict = None, mcp_data: dict = None) -> MarketData:
    """Convert local or MCP data to MarketData format"""
    
    # Use local data if available
    if local_data and local_data.get('success'):
        data = local_data.get('data', {})
        return MarketData(
            high=float(data.get('high', 0)),
            low=float(data.get('low', 0)),
            open=float(data.get('open', 0)),
            close=float(data.get('close', 0)),
            prevClose=float(data.get('prev_close', data.get('close', 0))),
            currentPrice=float(data.get('close', 0)),
            symbol=symbol,
            company=RELIABLE_SYMBOLS.get(symbol, 'Market Index')
        )
    
    # Fallback to MCP data
    if mcp_data and mcp_data.get('success'):
        raw_data = mcp_data.get('data', [])
        if raw_data:
            latest = raw_data[-1]  # Most recent data point
            prev_data = raw_data[-2] if len(raw_data) > 1 else latest
            
            return MarketData(
                high=float(latest.get('high', 0)),
                low=float(latest.get('low', 0)),
                open=float(latest.get('open', 0)),
                close=float(latest.get('close', 0)),
                prevClose=float(prev_data.get('close', latest.get('close', 0))),
                currentPrice=float(latest.get('close', 0)),
                symbol=symbol,
                company=mcp_data.get('display_name', 'Market Index')
            )
    
    # Final fallback
    return MarketData(
        high=11691.89,
        low=11470.47,
        open=11600.11,
        close=11512.41,
        prevClose=11512.41,
        currentPrice=11691.89,
        symbol=symbol,
        company="Market Index"
    )

# Core API Endpoints

@router.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, background_tasks: BackgroundTasks) -> MarketData:
    """Get market data for a specific symbol (SPY for S&P 500, etc.)"""
    
    # For market index, use a representative symbol
    if symbol.upper() in ["SPY", "QQQ", "DIA"]:
        lookup_symbol = "AAPL"  # Use AAPL as proxy for market indices
    else:
        lookup_symbol = symbol
    
    # Try to get from local database first
    local_data = get_local_stock_data(lookup_symbol)
    
    # If local data is stale, schedule a background refresh
    if data_importer and not data_importer.is_data_fresh(lookup_symbol):
        background_tasks.add_task(ensure_fresh_data, lookup_symbol)
    
    # If no local data available, try MCP as fallback
    mcp_data = None
    if not local_data:
        mcp_data = get_mcp_market_data(lookup_symbol)
    
    return format_market_data(symbol, local_data, mcp_data)

@router.get("/api/watchlist-stocks")
async def get_watchlist_stocks(background_tasks: BackgroundTasks) -> List[StockData]:
    """Get watchlist stocks data - top 8 reliable stocks"""
    
    stocks = []
    # Get first 8 symbols for watchlist (mix of tech stocks and indices)
    top_stocks = list(RELIABLE_SYMBOLS.keys())[:8]
    
    for symbol in top_stocks:
        # Try local data first
        local_data = get_local_stock_data(symbol)
        
        # Schedule background refresh if data is stale
        if data_importer and not data_importer.is_data_fresh(symbol):
            background_tasks.add_task(ensure_fresh_data, symbol)
        
        # Fallback to MCP if no local data
        mcp_data = None
        if not local_data:
            mcp_data = get_mcp_market_data(symbol)
        
        stock_data = format_stock_data(symbol, local_data, mcp_data)
        stocks.append(stock_data)
    
    return stocks

@router.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str, background_tasks: BackgroundTasks) -> StockData:
    """Get individual stock data"""
    
    # Try local data first
    local_data = get_local_stock_data(symbol)
    
    # Schedule background refresh if data is stale
    if data_importer and not data_importer.is_data_fresh(symbol):
        background_tasks.add_task(ensure_fresh_data, symbol)
    
    # Fallback to MCP if no local data
    mcp_data = None
    if not local_data:
        mcp_data = get_mcp_market_data(symbol)
    
    return format_stock_data(symbol, local_data, mcp_data)

@router.get("/api/chart-data/{symbol}")
async def get_chart_data(symbol: str, background_tasks: BackgroundTasks, period: str = "1month") -> List[ChartDataPoint]:
    """Get chart data for a specific symbol"""
    
    # Try to get historical data from local database
    days_map = {"1week": 7, "1month": 30, "3months": 90, "6months": 180, "1year": 365}
    days = days_map.get(period, 30)
    
    # Try local database first via data_importer if available
    local_historical = None
    if data_importer:
        try:
            local_historical = data_importer.get_historical_data(symbol, days)
        except Exception as e:
            print(f"Local historical data failed: {e}")
    
    # Schedule background refresh if data is stale (if data_importer available)
    if data_importer and not data_importer.is_data_fresh(symbol):
        background_tasks.add_task(ensure_fresh_data, symbol)
    
    if local_historical:
        return [
            ChartDataPoint(
                time=item["date"],
                price=item["close"] or 0.0
            )
            for item in local_historical
            if item["close"] is not None
        ]
    
    # Try MCP database server for historical data
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        db_historical = finance_db_stock_price.get_historical_prices(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=days
        )
        
        if db_historical and isinstance(db_historical, list):
            return [
                ChartDataPoint(
                    time=item.get('date', ''),
                    price=float(item.get('close_price', 0.0))
                )
                for item in db_historical
                if item.get('close_price') is not None
            ]
    except Exception as e:
        print(f"MCP database historical data failed for {symbol}: {e}")
    
    # Fallback to MCP market data
    mcp_data = get_mcp_market_data(symbol)
    
    if mcp_data and mcp_data.get('success'):
        # Extract chart data from MCP response
        raw_data = mcp_data.get('data', [])
        chart_data = []
        
        for item in raw_data:
            # Get time/date value and ensure it's a string
            time_value = item.get('timestamp', item.get('datetime', ''))
            if hasattr(time_value, 'strftime'):  # It's a date/datetime object
                time_str = time_value.strftime("%Y-%m-%d")
            elif hasattr(time_value, 'isoformat'):  # It's a datetime object
                time_str = time_value.isoformat()[:10]  # Get just the date part
            else:
                time_str = str(time_value)  # Convert to string
            
            chart_data.append(ChartDataPoint(
                time=time_str,
                price=float(item.get('close', 0))
            ))
        
        return chart_data
    
    # Final fallback - generate sample chart data
    chart_data = []
    base_date = datetime.now() - timedelta(days=days)
    base_price = 100.0
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        price = base_price + random.uniform(-5, 5)
        base_price = price  # Make it trend-like
        
        chart_data.append(ChartDataPoint(
            time=date.strftime("%Y-%m-%d"),
            price=round(price, 2)
        ))
    
    return chart_data

@router.get("/api/news")
async def get_financial_news(limit: int = 10) -> List[NewsItem]:
    """Get financial news - replaces mockNewsItems"""
    
    try:
        # Try to get real news from MCP server
        mcp_result = finance_news_and_insights.get_market_news(
            limit=limit,
            sentiment_analysis=True
        )
        
        if mcp_result and mcp_result.get('success'):
            news_items = []
            news_data = mcp_result.get('news', [])
            
            for i, item in enumerate(news_data[:limit]):
                news_items.append(NewsItem(
                    id=str(i + 1),
                    title=item.get('title', 'Financial News'),
                    summary=item.get('summary', item.get('description', 'Market news update')),
                    timestamp=datetime.now() - timedelta(hours=i),
                    source=item.get('source', 'Financial News')
                ))
            
            return news_items
    
    except Exception as e:
        print(f"Error getting news: {e}")
    
    # Fallback news data
    return [
        NewsItem(
            id='1',
            title='VN-Index tăng mạnh trong phiên sáng, vượt mốc 1.200 điểm',
            summary='Thị trường chứng khoán Việt Nam ghi nhận phiên tăng điểm mạnh mẽ với sự dẫn dắt của nhóm cổ phiếu ngân hàng và bất động sản.',
            timestamp=datetime.now() - timedelta(minutes=30),
            source='VnExpress'
        ),
        NewsItem(
            id='2',
            title='Fed giữ nguyên lãi suất, thị trường châu Á phản ứng tích cực',
            summary='Quyết định giữ nguyên lãi suất của Fed được thị trường đón nhận tích cực, tạo động lực cho các thị trường châu Á.',
            timestamp=datetime.now() - timedelta(hours=2),
            source='CafeF'
        ),
        NewsItem(
            id='3',
            title='Vingroup công bố kết quả kinh doanh quý 4 vượt kỳ vọng',
            summary='Tập đoàn Vingroup báo cáo doanh thu quý 4 tăng 15% so với cùng kỳ năm trước, vượt dự báo của các chuyên gia.',
            timestamp=datetime.now() - timedelta(hours=4),
            source='Đầu tư'
        ),
        NewsItem(
            id='4',
            title='Giá dầu thô tăng mạnh, cổ phiếu dầu khí hưởng lợi',
            summary='Giá dầu thô Brent vượt mốc 85 USD/thùng, tạo động lực tích cực cho nhóm cổ phiếu dầu khí trên sàn chứng khoán.',
            timestamp=datetime.now() - timedelta(hours=6),
            source='Nhịp Cầu Đầu Tư'
        ),
        NewsItem(
            id='5',
            title='Khối ngoại mua ròng 500 tỷ đồng trong tuần qua',
            summary='Nhà đầu tư nước ngoài tiếp tục thể hiện sự lạc quan với thị trường Việt Nam qua việc mua ròng mạnh mẽ.',
            timestamp=datetime.now() - timedelta(days=1),
            source='VietStock'
        )
    ]

@router.get("/api/crypto-data")
async def get_crypto_data() -> List[CryptoData]:
    """Get cryptocurrency data (using ETF proxies for reliability)"""
    
    crypto_symbols = ["BITO", "ETHE"]  # Bitcoin and Ethereum ETFs
    crypto_data = []
    
    for symbol in crypto_symbols:
        try:
            # Try to get real data from MCP
            local_data = get_local_stock_data(symbol)
            
            if local_data and local_data.get('success'):
                data = local_data.get('data', {})
                crypto_data.append(CryptoData(
                    symbol=symbol,
                    name=RELIABLE_SYMBOLS.get(symbol, f"{symbol} Trust"),
                    price=float(data.get('close', 100.0)),
                    change_24h=float(data.get('change', 0.0)),
                    change_percent_24h=float(data.get('change_percent', 0.0)),
                    market_cap=float(data.get('market_cap', 0)) if data.get('market_cap') else None,
                    volume_24h=float(data.get('volume', 0)) if data.get('volume') else None
                ))
            else:
                # Fallback data
                crypto_data.append(CryptoData(
                    symbol=symbol,
                    name=RELIABLE_SYMBOLS.get(symbol, f"{symbol} Trust"),
                    price=round(random.uniform(20, 50), 2),
                    change_24h=round(random.uniform(-5, 5), 2),
                    change_percent_24h=round(random.uniform(-10, 10), 2),
                    market_cap=round(random.uniform(1000000000, 10000000000), 0),
                    volume_24h=round(random.uniform(1000000, 100000000), 0)
                ))
        except Exception as e:
            print(f"Error getting crypto data for {symbol}: {e}")
            continue
    
    return crypto_data

@router.get("/api/commodities-data")
async def get_commodities_data() -> List[CommodityData]:
    """Get commodities data (Gold, Silver, Oil, Gas)"""
    
    commodities_symbols = {
        "GLD": {"name": "Gold", "unit": "USD per ounce (ETF)"},
        "SLV": {"name": "Silver", "unit": "USD per ounce (ETF)"},
        "USO": {"name": "Oil", "unit": "USD per barrel (ETF)"},
        "UNG": {"name": "Natural Gas", "unit": "USD per MMBtu (ETF)"}
    }
    
    commodities_data = []
    
    for symbol, info in commodities_symbols.items():
        try:
            # Try to get real data from MCP
            local_data = get_local_stock_data(symbol)
            
            if local_data and local_data.get('success'):
                data = local_data.get('data', {})
                commodities_data.append(CommodityData(
                    symbol=symbol,
                    name=info["name"],
                    price=float(data.get('close', 100.0)),
                    change=float(data.get('change', 0.0)),
                    change_percent=float(data.get('change_percent', 0.0)),
                    unit=info["unit"]
                ))
            else:
                # Fallback data
                base_prices = {"GLD": 180, "SLV": 22, "USO": 70, "UNG": 25}
                commodities_data.append(CommodityData(
                    symbol=symbol,
                    name=info["name"],
                    price=round(random.uniform(base_prices[symbol] * 0.9, base_prices[symbol] * 1.1), 2),
                    change=round(random.uniform(-5, 5), 2),
                    change_percent=round(random.uniform(-3, 3), 2),
                    unit=info["unit"]
                ))
        except Exception as e:
            print(f"Error getting commodity data for {symbol}: {e}")
            continue
    
    return commodities_data

@router.get("/api/indices-data")
async def get_indices_data() -> List[IndexData]:
    """Get major market indices data"""
    
    indices_symbols = {
        "SPY": {"name": "S&P 500", "constituents": 500},
        "QQQ": {"name": "Nasdaq 100", "constituents": 100},
        "DIA": {"name": "Dow Jones", "constituents": 30},
        "VTI": {"name": "Total Stock Market", "constituents": 4000},
        "IWM": {"name": "Russell 2000", "constituents": 2000}
    }
    
    indices_data = []
    
    for symbol, info in indices_symbols.items():
        try:
            # Try to get real data from MCP
            local_data = get_local_stock_data(symbol)
            
            if local_data and local_data.get('success'):
                data = local_data.get('data', {})
                indices_data.append(IndexData(
                    symbol=symbol,
                    name=info["name"],
                    value=float(data.get('close', 400.0)),
                    change=float(data.get('change', 0.0)),
                    change_percent=float(data.get('change_percent', 0.0)),
                    constituents=info["constituents"]
                ))
            else:
                # Fallback data
                base_values = {"SPY": 450, "QQQ": 380, "DIA": 350, "VTI": 240, "IWM": 200}
                indices_data.append(IndexData(
                    symbol=symbol,
                    name=info["name"],
                    value=round(random.uniform(base_values[symbol] * 0.95, base_values[symbol] * 1.05), 2),
                    change=round(random.uniform(-10, 10), 2),
                    change_percent=round(random.uniform(-2, 2), 2),
                    constituents=info["constituents"]
                ))
        except Exception as e:
            print(f"Error getting index data for {symbol}: {e}")
            continue
    
    return indices_data

@router.get("/api/market-overview")
async def get_market_overview():
    """Get comprehensive market overview including tech stocks, indices, commodities, and crypto"""
    
    try:
        # Get data for different asset classes
        tech_stocks = []
        tech_symbols = list(RELIABLE_SYMBOLS.keys())[:10]  # First 10 are tech stocks
        
        for symbol in tech_symbols[:5]:  # Top 5 tech stocks for overview
            local_data = get_local_stock_data(symbol)
            if local_data and local_data.get('success'):
                data = local_data.get('data', {})
                tech_stocks.append({
                    "symbol": symbol,
                    "name": RELIABLE_SYMBOLS[symbol],
                    "price": float(data.get('close', 100.0)),
                    "change_percent": float(data.get('change_percent', 0.0))
                })
        
        # Get major indices
        major_indices = []
        index_symbols = ["SPY", "QQQ", "DIA"]
        
        for symbol in index_symbols:
            local_data = get_local_stock_data(symbol)
            if local_data and local_data.get('success'):
                data = local_data.get('data', {})
                major_indices.append({
                    "symbol": symbol,
                    "name": RELIABLE_SYMBOLS[symbol],
                    "value": float(data.get('close', 400.0)),
                    "change_percent": float(data.get('change_percent', 0.0))
                })
        
        # Market sentiment calculation
        all_changes = []
        for stock in tech_stocks:
            all_changes.append(stock["change_percent"])
        for index in major_indices:
            all_changes.append(index["change_percent"])
        
        avg_change = sum(all_changes) / len(all_changes) if all_changes else 0
        market_sentiment = "bullish" if avg_change > 0.5 else "bearish" if avg_change < -0.5 else "neutral"
        
        return {
            "market_sentiment": market_sentiment,
            "average_change": round(avg_change, 2),
            "tech_stocks": tech_stocks,
            "major_indices": major_indices,
            "commodities_summary": {
                "gold_trend": "stable",
                "oil_trend": "volatile",
                "crypto_trend": "bullish"
            },
            "last_updated": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error getting market overview: {e}")
        
        # Fallback overview
        return {
            "market_sentiment": "neutral",
            "average_change": 0.25,
            "tech_stocks": [
                {"symbol": "AAPL", "name": "Apple Inc.", "price": 175.50, "change_percent": 1.2},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "price": 142.80, "change_percent": 0.8},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "price": 420.30, "change_percent": 1.5}
            ],
            "major_indices": [
                {"symbol": "SPY", "name": "S&P 500 ETF", "value": 450.25, "change_percent": 0.7},
                {"symbol": "QQQ", "name": "Nasdaq 100 ETF", "value": 380.15, "change_percent": 1.1}
            ],
            "commodities_summary": {
                "gold_trend": "stable",
                "oil_trend": "volatile", 
                "crypto_trend": "bullish"
            },
            "last_updated": datetime.now().isoformat()
        }

@router.get("/api/symbol-discovery")
async def discover_symbols(query: str = Query(description="Search query for symbols")):
    """Discover stock symbols based on query"""
    
    try:
        mcp_result = finance_symbol_discovery.discover_symbols(
            query=query,
            limit=10
        )
        
        if mcp_result and mcp_result.get('success'):
            return {
                "query": query,
                "symbols": mcp_result.get('symbols', []),
                "count": len(mcp_result.get('symbols', []))
            }
    
    except Exception as e:
        print(f"Error discovering symbols for '{query}': {e}")
    
    # Fallback - return some reliable symbols matching query
    matching_stocks = [
        {"symbol": symbol, "name": name} 
        for symbol, name in RELIABLE_SYMBOLS.items()
        if query.upper() in symbol.upper() or query.lower() in name.lower()
    ]
    
    return {
        "query": query,
        "symbols": matching_stocks[:10],
        "count": len(matching_stocks)
    }
