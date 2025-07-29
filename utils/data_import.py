"""
Data Import System for Finance Database
Imports stock data from external APIs and populates local database
Reduces external API calls by maintaining fresh local data
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Install with: pip install yfinance")
    yf = None
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", None)

class DataImporter:
    def __init__(self):
        self.db_uri = DB_URI
        self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY
        
        # Vietnamese stock symbols to import
        self.vietnamese_stocks = [
            "VCB", "VIC", "VHM", "HPG", "TCB", "MSN", 
            "FPT", "GAS", "CTG", "MWG", "BID", "ACB",
            "VPB", "POW", "VRE", "PLX", "SAB", "MBB"
        ]
        
        # International stocks for comparison
        self.international_stocks = [
            "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", 
            "NVDA", "META", "NFLX", "BABA", "TSM"
        ]

    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_uri)

    def create_tables_if_not_exist(self):
        """Verify existing database schema - no table creation needed"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Verify existing tables exist
                    cur.execute("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('company', 'stock_price')
                    """)
                    
                    existing_tables = [row[0] for row in cur.fetchall()]
                    
                    if 'company' not in existing_tables or 'stock_price' not in existing_tables:
                        raise Exception("Required tables (COMPANY, STOCK_PRICE) not found in database")
                    
                    # Create data import log table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS public.data_import_log (
                            log_id SERIAL PRIMARY KEY,
                            import_type VARCHAR(50) NOT NULL,
                            symbol VARCHAR(10),
                            status VARCHAR(20) NOT NULL,
                            records_imported INTEGER DEFAULT 0,
                            error_message TEXT,
                            import_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    conn.commit()
                    logger.info("Database schema verified successfully")
                    
        except Exception as e:
            logger.error(f"Error verifying database schema: {e}")
            raise

    def import_yahoo_finance_data(self, symbol: str, period: str = "1mo") -> bool:
        """Import stock data from Yahoo Finance"""
        if not yf:
            logger.error("yfinance not available")
            return False
            
        try:
            # Download data
            ticker = yf.Ticker(f"{symbol}.VN" if symbol in self.vietnamese_stocks else symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                return False
            
            # Get company info
            info = ticker.info
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Insert/update company info
                    self._insert_company_info(cur, symbol, info)
                    
                    # Insert stock prices using your existing schema
                    price_data = []
                    for date, row in hist.iterrows():
                        price_data.append((
                            symbol,
                            date.date(),
                            float(row['Open']) if not pd.isna(row['Open']) else None,
                            float(row['High']) if not pd.isna(row['High']) else None,
                            float(row['Low']) if not pd.isna(row['Low']) else None,
                            float(row['Close']) if not pd.isna(row['Close']) else None,
                            float(row['Close']) if not pd.isna(row['Close']) else None,  # Using Close for adjusted_close
                            int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                            0.0,  # dividend_amount - default
                            1.0   # split_coefficient - default
                        ))
                    
                    if price_data:
                        # Check if record exists first, then insert or update
                        for price_record in price_data:
                            cur.execute("""
                                SELECT 1 FROM public.stock_price 
                                WHERE symbol = %s AND date = %s
                            """, (price_record[0], price_record[1]))
                            
                            if cur.fetchone():
                                # Update existing record
                                cur.execute("""
                                    UPDATE public.stock_price SET
                                        open_price = %s, high_price = %s, low_price = %s,
                                        close_price = %s, adjusted_close = %s, volume = %s,
                                        dividend_amount = %s, split_coefficient = %s
                                    WHERE symbol = %s AND date = %s
                                """, price_record[2:] + (price_record[0], price_record[1]))
                            else:
                                # Insert new record
                                cur.execute("""
                                    INSERT INTO public.stock_price 
                                    (symbol, date, open_price, high_price, low_price, close_price, 
                                     adjusted_close, volume, dividend_amount, split_coefficient)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """, price_record)
                        
                        # Log successful import
                        cur.execute("""
                            INSERT INTO public.data_import_log 
                            (import_type, symbol, status, records_imported)
                            VALUES ('yahoo_finance', %s, 'success', %s)
                        """, (symbol, len(price_data)))
                        
                        conn.commit()
                        logger.info(f"Successfully imported {len(price_data)} records for {symbol}")
                        return True
                        
        except Exception as e:
            logger.error(f"Error importing data for {symbol}: {e}")
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO public.data_import_log 
                        (import_type, symbol, status, error_message)
                        VALUES ('yahoo_finance', %s, 'failed', %s)
                    """, (symbol, str(e)))
                    conn.commit()
            return False

    def _insert_company_info(self, cursor, symbol: str, info: dict):
        """Insert or update company information using existing COMPANY table schema"""
        try:
            cursor.execute("""
                INSERT INTO public.company 
                (symbol, asset_type, name, description, sector, industry, country, exchange, market_cap, website)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) 
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    website = EXCLUDED.website,
                    country = EXCLUDED.country,
                    exchange = EXCLUDED.exchange
            """, (
                symbol,
                'Common Stock',  # asset_type - default value
                info.get('longName', info.get('shortName', f"{symbol} Corporation")),
                info.get('longBusinessSummary', f"{symbol} is a publicly traded company."),
                info.get('sector', 'Unknown'),
                info.get('industry', 'Unknown'),
                info.get('country', 'Vietnam' if symbol in self.vietnamese_stocks else 'USA'),
                info.get('exchange', 'VNX' if symbol in self.vietnamese_stocks else 'NASDAQ'),
                info.get('marketCap', 0),
                info.get('website')
            ))
        except Exception as e:
            logger.error(f"Error inserting company info for {symbol}: {e}")

    async def import_all_stocks(self, period: str = "3mo"):
        """Import data for all tracked stocks"""
        logger.info("Starting bulk stock data import...")
        
        all_symbols = self.vietnamese_stocks + self.international_stocks
        success_count = 0
        failed_count = 0
        
        for symbol in all_symbols:
            logger.info(f"Importing data for {symbol}...")
            if self.import_yahoo_finance_data(symbol, period):
                success_count += 1
            else:
                failed_count += 1
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(1)
        
        logger.info(f"Import complete: {success_count} successful, {failed_count} failed")
        return {"success": success_count, "failed": failed_count}

    def get_cached_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get stock data from local database using existing STOCK_PRICE table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get latest price data
                    cur.execute("""
                        SELECT symbol, date, open_price, high_price, low_price, 
                               close_price, adjusted_close, volume
                        FROM public.stock_price
                        WHERE symbol = %s
                        ORDER BY date DESC
                        LIMIT 1
                    """, (symbol.upper(),))
                    
                    latest = cur.fetchone()
                    if not latest:
                        return None
                    
                    # Get previous day for change calculation
                    cur.execute("""
                        SELECT close_price
                        FROM public.stock_price
                        WHERE symbol = %s AND date < %s
                        ORDER BY date DESC
                        LIMIT 1
                    """, (symbol.upper(), latest[1]))
                    
                    prev = cur.fetchone()
                    prev_close = prev[0] if prev else latest[5]
                    
                    # Calculate change
                    current_price = float(latest[5])
                    change = current_price - float(prev_close)
                    change_percent = (change / float(prev_close)) * 100 if prev_close else 0
                    
                    return {
                        "symbol": latest[0],
                        "date": latest[1].isoformat(),
                        "open": float(latest[2]) if latest[2] else None,
                        "high": float(latest[3]) if latest[3] else None,
                        "low": float(latest[4]) if latest[4] else None,
                        "close": current_price,
                        "volume": int(latest[7]) if latest[7] else 0,
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "prev_close": float(prev_close)
                    }
                    
        except Exception as e:
            logger.error(f"Error getting cached data for {symbol}: {e}")
            return None

    def get_cached_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information from local database using existing COMPANY table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT symbol, name, sector, industry, market_cap,
                               description, website, country, exchange
                        FROM public.company
                        WHERE symbol = %s
                    """, (symbol.upper(),))
                    
                    result = cur.fetchone()
                    if not result:
                        return None
                    
                    return {
                        "symbol": result[0],
                        "company_name": result[1],
                        "sector": result[2],
                        "industry": result[3],
                        "market_cap": int(result[4]) if result[4] else None,
                        "description": result[5],
                        "website": result[6],
                        "country": result[7],
                        "exchange": result[8]
                    }
                    
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical price data from database using existing STOCK_PRICE table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT date, open_price, high_price, low_price, close_price, volume
                        FROM public.stock_price
                        WHERE symbol = %s
                        ORDER BY date DESC
                        LIMIT %s
                    """, (symbol.upper(), days))
                    
                    results = cur.fetchall()
                    return [
                        {
                            "date": row[0].isoformat(),
                            "open": float(row[1]) if row[1] else None,
                            "high": float(row[2]) if row[2] else None,
                            "low": float(row[3]) if row[3] else None,
                            "close": float(row[4]) if row[4] else None,
                            "volume": int(row[5]) if row[5] else 0
                        }
                        for row in results
                    ]
                    
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []

    def is_data_fresh(self, symbol: str, max_age_hours: int = 24) -> bool:
        """Check if cached data is fresh enough using existing STOCK_PRICE table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT date
                        FROM public.stock_price
                        WHERE symbol = %s
                        ORDER BY date DESC
                        LIMIT 1
                    """, (symbol.upper(),))
                    
                    result = cur.fetchone()
                    if not result:
                        return False
                    
                    latest_date = result[0]
                    if isinstance(latest_date, str):
                        latest_date = datetime.fromisoformat(latest_date).date()
                    
                    today = datetime.now().date()
                    age_days = (today - latest_date).days
                    
                    # Data is fresh if it's from today or yesterday (for non-trading days)
                    return age_days <= 1
                    
        except Exception as e:
            logger.error(f"Error checking data freshness for {symbol}: {e}")
            return False

    async def schedule_daily_import(self):
        """Schedule daily data import (to be called by a scheduler)"""
        logger.info("Starting scheduled daily import...")
        
        # Import recent data (last 7 days to catch up)
        await self.import_all_stocks(period="7d")
        
        # Clean up old import logs (keep last 30 days)
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM public.data_import_log
                        WHERE import_time < %s
                    """, (datetime.now() - timedelta(days=30),))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error cleaning up import logs: {e}")

# Create global instance
data_importer = DataImporter()

# Async functions for use with FastAPI
async def ensure_fresh_data(symbol: str) -> bool:
    """Ensure we have fresh data for a symbol"""
    if not data_importer.is_data_fresh(symbol):
        logger.info(f"Refreshing data for {symbol}")
        return data_importer.import_yahoo_finance_data(symbol, period="7d")
    return True

async def batch_import_data():
    """Batch import data for all symbols"""
    return await data_importer.import_all_stocks()
