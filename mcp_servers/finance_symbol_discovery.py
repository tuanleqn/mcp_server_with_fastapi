#!/usr/bin/env python3
"""
Symbol Discovery and Database Integration
Automatically discovers and stores new financial symbols
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
from mcp.server.fastmcp import FastMCP

# MCP Server instance
mcp = FastMCP(name="Finance MCP Server - Symbol Discovery")

# Database configuration
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "symbols.db")

class SymbolDatabase:
    """Manages discovered symbols and their metadata"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovered_symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_key TEXT UNIQUE NOT NULL,
                    company_name TEXT,
                    symbol_type TEXT,
                    alpha_vantage_symbol TEXT,
                    yahoo_finance_symbol TEXT,
                    finnhub_symbol TEXT,
                    market_cap REAL,
                    currency TEXT DEFAULT 'USD',
                    exchange TEXT,
                    sector TEXT,
                    industry TEXT,
                    country TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    first_discovered TEXT,
                    last_updated TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbol_price_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_key TEXT NOT NULL,
                    price_data TEXT NOT NULL,
                    interval_period TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (symbol_key) REFERENCES discovered_symbols (symbol_key)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_key ON discovered_symbols(symbol_key);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_cache_symbol ON symbol_price_cache(symbol_key);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_cache_expires ON symbol_price_cache(expires_at);
            """)
    
    def symbol_exists(self, symbol_key: str) -> bool:
        """Check if symbol exists in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM discovered_symbols WHERE symbol_key = ? AND is_active = 1",
                (symbol_key.lower(),)
            )
            return cursor.fetchone() is not None
    
    def get_symbol_info(self, symbol_key: str) -> Optional[Dict]:
        """Get symbol information from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM discovered_symbols 
                WHERE symbol_key = ? AND is_active = 1
            """, (symbol_key.lower(),))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
    
    def store_symbol(self, symbol_info: Dict) -> bool:
        """Store discovered symbol information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO discovered_symbols (
                        symbol_key, company_name, symbol_type,
                        alpha_vantage_symbol, yahoo_finance_symbol, finnhub_symbol,
                        market_cap, currency, exchange, sector, industry, country,
                        first_discovered, last_updated, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol_info.get('symbol_key', '').lower(),
                    symbol_info.get('company_name'),
                    symbol_info.get('symbol_type', 'stock'),
                    symbol_info.get('alpha_vantage_symbol'),
                    symbol_info.get('yahoo_finance_symbol'),
                    symbol_info.get('finnhub_symbol'),
                    symbol_info.get('market_cap'),
                    symbol_info.get('currency', 'USD'),
                    symbol_info.get('exchange'),
                    symbol_info.get('sector'),
                    symbol_info.get('industry'),
                    symbol_info.get('country'),
                    symbol_info.get('first_discovered', datetime.now().isoformat()),
                    datetime.now().isoformat(),
                    json.dumps(symbol_info.get('metadata', {}))
                ))
            return True
        except Exception as e:
            print(f"Error storing symbol {symbol_info.get('symbol_key')}: {e}")
            return False
    
    def cache_price_data(self, symbol_key: str, price_data: List[Dict], 
                        interval: str, period: str, data_source: str, 
                        cache_hours: int = 1) -> bool:
        """Cache price data for faster retrieval"""
        try:
            expires_at = datetime.now() + timedelta(hours=cache_hours)
            interval_period = f"{interval}_{period}"
            
            with sqlite3.connect(self.db_path) as conn:
                # Remove old cache entries for this symbol/interval/period
                conn.execute("""
                    DELETE FROM symbol_price_cache 
                    WHERE symbol_key = ? AND interval_period = ?
                """, (symbol_key.lower(), interval_period))
                
                # Insert new cache entry
                conn.execute("""
                    INSERT INTO symbol_price_cache (
                        symbol_key, price_data, interval_period, 
                        data_source, expires_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    symbol_key.lower(),
                    json.dumps(price_data),
                    interval_period,
                    data_source,
                    expires_at.isoformat()
                ))
            return True
        except Exception as e:
            print(f"Error caching price data for {symbol_key}: {e}")
            return False
    
    def get_cached_price_data(self, symbol_key: str, interval: str, 
                             period: str) -> Optional[Tuple[List[Dict], str]]:
        """Get cached price data if still valid"""
        interval_period = f"{interval}_{period}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT price_data, data_source FROM symbol_price_cache
                WHERE symbol_key = ? AND interval_period = ? 
                AND expires_at > ?
                ORDER BY cached_at DESC LIMIT 1
            """, (symbol_key.lower(), interval_period, datetime.now().isoformat()))
            
            row = cursor.fetchone()
            if row:
                try:
                    price_data = json.loads(row[0])
                    data_source = row[1]
                    return price_data, data_source
                except json.JSONDecodeError:
                    pass
        return None
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM symbol_price_cache 
                WHERE expires_at < ?
            """, (datetime.now().isoformat(),))

class SymbolDiscovery:
    """Discovers new symbols using various data sources"""
    
    def __init__(self, db: SymbolDatabase):
        self.db = db
    
    async def discover_symbol_from_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """Try to discover symbol info from Alpha Vantage"""
        try:
            import os
            import requests
            
            api_key = os.getenv("EXTERNAL_FINANCE_API_KEY")
            if not api_key:
                return None
            
            # Try company overview first
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": symbol.upper(),
                "apikey": api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                if data and "Symbol" in data and data["Symbol"]:
                    return {
                        "symbol_key": symbol.lower(),
                        "company_name": data.get("Name", symbol.upper()),
                        "symbol_type": "stock" if data.get("AssetType") == "Common Stock" else "other",
                        "alpha_vantage_symbol": data.get("Symbol", symbol.upper()),
                        "yahoo_finance_symbol": f"{symbol.upper()}",
                        "finnhub_symbol": symbol.upper(),
                        "market_cap": float(data.get("MarketCapitalization", 0)) if data.get("MarketCapitalization", "0").isdigit() else None,
                        "currency": data.get("Currency", "USD"),
                        "exchange": data.get("Exchange"),
                        "sector": data.get("Sector"),
                        "industry": data.get("Industry"),
                        "country": data.get("Country"),
                        "first_discovered": datetime.now().isoformat(),
                        "metadata": {
                            "description": data.get("Description", ""),
                            "pe_ratio": data.get("PERatio"),
                            "dividend_yield": data.get("DividendYield"),
                            "52_week_high": data.get("52WeekHigh"),
                            "52_week_low": data.get("52WeekLow")
                        }
                    }
        except Exception as e:
            print(f"Error discovering symbol {symbol} from Alpha Vantage: {e}")
        
        return None
    
    async def discover_symbol_from_yahoo(self, symbol: str) -> Optional[Dict]:
        """Try to discover symbol info from Yahoo Finance"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            if info and info.get("symbol"):
                # Determine symbol type
                symbol_type = "stock"
                if "ETF" in info.get("longName", "").upper():
                    symbol_type = "etf"
                elif info.get("quoteType") == "CRYPTOCURRENCY":
                    symbol_type = "crypto"
                elif info.get("quoteType") == "CURRENCY":
                    symbol_type = "forex"
                
                return {
                    "symbol_key": symbol.lower(),
                    "company_name": info.get("longName", info.get("shortName", symbol.upper())),
                    "symbol_type": symbol_type,
                    "alpha_vantage_symbol": symbol.upper(),
                    "yahoo_finance_symbol": info.get("symbol", symbol.upper()),
                    "finnhub_symbol": symbol.upper(),
                    "market_cap": info.get("marketCap"),
                    "currency": info.get("currency", "USD"),
                    "exchange": info.get("exchange"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "country": info.get("country"),
                    "first_discovered": datetime.now().isoformat(),
                    "metadata": {
                        "website": info.get("website"),
                        "business_summary": info.get("businessSummary", ""),
                        "full_time_employees": info.get("fullTimeEmployees"),
                        "dividend_yield": info.get("dividendYield"),
                        "pe_ratio": info.get("trailingPE")
                    }
                }
        except Exception as e:
            print(f"Error discovering symbol {symbol} from Yahoo Finance: {e}")
        
        return None
    
    async def discover_symbol_from_finnhub(self, symbol: str) -> Optional[Dict]:
        """Try to discover symbol info from Finnhub"""
        try:
            import os
            import requests
            
            api_key = os.getenv("FINNHUB_API_KEY")
            if not api_key:
                return None
            
            # Try company profile
            url = f"https://finnhub.io/api/v1/stock/profile2"
            params = {
                "symbol": symbol.upper(),
                "token": api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                if data and data.get("name"):
                    return {
                        "symbol_key": symbol.lower(),
                        "company_name": data.get("name", symbol.upper()),
                        "symbol_type": "stock",
                        "alpha_vantage_symbol": symbol.upper(),
                        "yahoo_finance_symbol": symbol.upper(),
                        "finnhub_symbol": symbol.upper(),
                        "market_cap": data.get("marketCapitalization"),
                        "currency": data.get("currency", "USD"),
                        "exchange": data.get("exchange"),
                        "country": data.get("country"),
                        "first_discovered": datetime.now().isoformat(),
                        "metadata": {
                            "logo": data.get("logo"),
                            "weburl": data.get("weburl"),
                            "ipo": data.get("ipo"),
                            "share_outstanding": data.get("shareOutstanding")
                        }
                    }
        except Exception as e:
            print(f"Error discovering symbol {symbol} from Finnhub: {e}")
        
        return None
    
    async def discover_symbol(self, symbol: str) -> Optional[Dict]:
        """Try to discover symbol from multiple sources"""
        
        # Try sources in priority order
        sources = [
            self.discover_symbol_from_alpha_vantage,
            self.discover_symbol_from_yahoo,
            self.discover_symbol_from_finnhub
        ]
        
        for source_func in sources:
            try:
                result = await source_func(symbol)
                if result:
                    print(f"✅ Discovered symbol {symbol} from {source_func.__name__}")
                    return result
            except Exception as e:
                print(f"❌ Error in {source_func.__name__} for {symbol}: {e}")
                continue
        
        # If all sources fail, create a basic entry
        print(f"⚠️ Could not discover {symbol} from any source, creating basic entry")
        return {
            "symbol_key": symbol.lower(),
            "company_name": symbol.upper(),
            "symbol_type": "stock",
            "alpha_vantage_symbol": symbol.upper(),
            "yahoo_finance_symbol": symbol.upper(),
            "finnhub_symbol": symbol.upper(),
            "currency": "USD",
            "first_discovered": datetime.now().isoformat(),
            "metadata": {
                "auto_discovered": True,
                "source": "fallback"
            }
        }

# Global instances
symbol_db = SymbolDatabase()
symbol_discovery = SymbolDiscovery(symbol_db)

async def get_or_discover_symbol(symbol: str) -> Dict:
    """
    Get symbol info from database or discover it if not found
    This is the main function to use for symbol lookup
    """
    
    # First check if we already have it
    existing = symbol_db.get_symbol_info(symbol)
    if existing:
        return {
            "symbol_key": existing["symbol_key"],
            "company_name": existing["company_name"],
            "symbol_type": existing["symbol_type"],
            "alpha_vantage": existing["alpha_vantage_symbol"],
            "yahoo_finance": existing["yahoo_finance_symbol"],
            "finnhub": existing["finnhub_symbol"],
            "name": existing["company_name"],
            "type": existing["symbol_type"],
            "currency": existing["currency"],
            "exchange": existing["exchange"],
            "sector": existing["sector"],
            "industry": existing["industry"],
            "metadata": json.loads(existing["metadata"] or "{}"),
            "source": "database"
        }
    
    # Symbol not found, try to discover it
    print(f"🔍 Symbol {symbol} not found in database, attempting discovery...")
    
    discovered = await symbol_discovery.discover_symbol(symbol)
    if discovered:
        # Store in database
        stored = symbol_db.store_symbol(discovered)
        if stored:
            print(f"💾 Stored discovered symbol {symbol} in database")
            
            return {
                "symbol_key": discovered["symbol_key"],
                "company_name": discovered["company_name"],
                "symbol_type": discovered["symbol_type"],
                "alpha_vantage": discovered["alpha_vantage_symbol"],
                "yahoo_finance": discovered["yahoo_finance_symbol"],
                "finnhub": discovered["finnhub_symbol"],
                "name": discovered["company_name"],
                "type": discovered["symbol_type"],
                "currency": discovered.get("currency", "USD"),
                "exchange": discovered.get("exchange"),
                "sector": discovered.get("sector"),
                "industry": discovered.get("industry"),
                "metadata": discovered.get("metadata", {}),
                "source": "discovered"
            }
    
    # Fallback if discovery fails
    return {
        "symbol_key": symbol.lower(),
        "company_name": symbol.upper(),
        "symbol_type": "stock",
        "alpha_vantage": symbol.upper(),
        "yahoo_finance": symbol.upper(),
        "finnhub": symbol.upper(),
        "name": symbol.upper(),
        "type": "stock",
        "currency": "USD",
        "source": "fallback"
    }

def get_cached_or_fresh_data(symbol_key: str, interval: str, period: str) -> Optional[Tuple[List[Dict], str]]:
    """Get cached price data or return None if cache miss"""
    return symbol_db.get_cached_price_data(symbol_key, interval, period)

def cache_price_data(symbol_key: str, price_data: List[Dict], interval: str, 
                    period: str, data_source: str) -> bool:
    """Cache price data for future use"""
    return symbol_db.cache_price_data(symbol_key, price_data, interval, period, data_source)

# Cleanup function to be called periodically
def cleanup_expired_cache():
    """Remove expired cache entries"""
    symbol_db.cleanup_expired_cache()

# MCP Tools
@mcp.tool(description="Discover and search for financial symbols")
def discover_symbols(query: str, limit: int = 10) -> dict:
    """
    Discover financial symbols based on search query.
    
    Args:
        query (str): Search query (company name or symbol)
        limit (int): Maximum number of results to return
        
    Returns:
        dict: List of discovered symbols with metadata
    """
    try:
        # Search in local database first
        results = symbol_db.search_symbols(query, limit)
        
        if results:
            return {
                "success": True,
                "query": query,
                "results": results[:limit],
                "source": "database",
                "count": len(results[:limit])
            }
        else:
            # If no local results, return a mock response for now
            # In production, this would integrate with external APIs
            mock_results = [
                {
                    "symbol": query.upper() if len(query) <= 5 else f"{query[:3].upper()}",
                    "name": f"Sample Company for {query}",
                    "type": "stock",
                    "exchange": "NASDAQ",
                    "currency": "USD"
                }
            ]
            
            return {
                "success": True,
                "query": query,
                "results": mock_results,
                "source": "mock",
                "count": len(mock_results),
                "note": "Mock data - integrate with real API for production"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

@mcp.tool(description="Search for symbols with advanced filtering")
def search_symbols(query: str, symbol_type: str = "all", limit: int = 10) -> dict:
    """
    Advanced symbol search with filtering.
    
    Args:
        query (str): Search query
        symbol_type (str): Filter by type (stock, etf, crypto, etc.)
        limit (int): Maximum results
        
    Returns:
        dict: Filtered search results
    """
    try:
        # Use the discover_symbols function and filter results if needed
        results = discover_symbols(query, limit)
        
        if results.get("success") and symbol_type != "all":
            filtered_results = [
                r for r in results.get("results", []) 
                if r.get("type", "").lower() == symbol_type.lower()
            ]
            results["results"] = filtered_results
            results["count"] = len(filtered_results)
            results["filter_applied"] = symbol_type
        
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "filter": symbol_type
        }
