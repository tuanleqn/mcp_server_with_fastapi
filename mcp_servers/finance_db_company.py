import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Company & Symbol Information (Optimized)")


@mcp.tool(description="Get company by symbol (optimized)")
def get_company_by_symbol(company_symbol: str) -> dict:
    """
    Retrieves company information by symbol from the finance database (optimized for large datasets).

    Args:
        company_symbol (str): The symbol of the company.

    Returns:
        dict: A dictionary containing company information.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT symbol, asset_type, name, description FROM public.company WHERE symbol = %s", 
                    (company_symbol.upper(),)
                )
                row = cur.fetchone()
                if row is None:
                    return {
                        "success": False,
                        "error": "Company not found",
                        "symbol": company_symbol.upper()
                    }
                return {
                    "success": True,
                    "symbol": row[0],
                    "asset_type": row[1],
                    "name": row[2],
                    "description": row[3],
                }
    except Error as e:
        return {
            "success": False,
            "error": str(e),
            "symbol": company_symbol.upper()
        }


@mcp.tool(description="Search for companies and symbols (optimized with pagination)")
def search_companies(query: str, limit: int = 10, offset: int = 0) -> dict:
    """
    Search for companies by name or symbol with pagination support.
    
    Args:
        query (str): Search query (company name or symbol)
        limit (int): Maximum number of results to return (max 50)
        offset (int): Number of results to skip for pagination
        
    Returns:
        dict: List of matching companies with pagination info
    """
    try:
        limit = min(limit, 50)  # Prevent excessive results
        
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # Get total count first
                count_query = """
                SELECT COUNT(*) 
                FROM public.company 
                WHERE UPPER(symbol) LIKE UPPER(%s) 
                   OR UPPER(name) LIKE UPPER(%s)
                """
                search_query = f"%{query}%"
                cur.execute(count_query, (search_query, search_query))
                total_count = cur.fetchone()[0]
                
                # Get paginated results
                results_query = """
                SELECT symbol, asset_type, name, description 
                FROM public.company 
                WHERE UPPER(symbol) LIKE UPPER(%s) 
                   OR UPPER(name) LIKE UPPER(%s)
                ORDER BY 
                    CASE WHEN UPPER(symbol) = UPPER(%s) THEN 1 ELSE 2 END,
                    CASE WHEN UPPER(symbol) LIKE UPPER(%s) THEN 1 ELSE 2 END,
                    symbol
                LIMIT %s OFFSET %s
                """
                
                cur.execute(results_query, (
                    search_query, search_query, query.upper(), f"{query.upper()}%", 
                    limit, offset
                ))
                rows = cur.fetchall()
                
                if rows:
                    results = []
                    for row in rows:
                        results.append({
                            "symbol": row[0],
                            "asset_type": row[1],
                            "name": row[2],
                            "description": row[3]
                        })
                    
                    return {
                        "success": True,
                        "query": query,
                        "results": results,
                        "pagination": {
                            "current_page": (offset // limit) + 1,
                            "page_size": limit,
                            "total_results": total_count,
                            "total_pages": (total_count + limit - 1) // limit,
                            "has_next": offset + limit < total_count,
                            "has_previous": offset > 0
                        }
                    }
                else:
                    return {
                        "success": True,
                        "query": query,
                        "results": [],
                        "pagination": {
                            "current_page": 1,
                            "page_size": limit,
                            "total_results": 0,
                            "total_pages": 0,
                            "has_next": False,
                            "has_previous": False
                        },
                        "message": f"No companies found matching '{query}'"
                    }
                    
    except Error as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@mcp.tool(description="Get symbol suggestions based on partial input (optimized)")
def get_symbol_suggestions(partial_symbol: str, limit: int = 5) -> dict:
    """
    Get symbol suggestions based on partial input (optimized for large datasets).
    
    Args:
        partial_symbol (str): Partial symbol to match
        limit (int): Maximum suggestions to return (max 20)
        
    Returns:
        dict: List of symbol suggestions
    """
    try:
        limit = min(limit, 20)  # Prevent excessive results
        
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # Optimized query with proper indexing
                cur.execute(
                    """
                    SELECT symbol, name, asset_type 
                    FROM public.company 
                    WHERE UPPER(symbol) LIKE UPPER(%s)
                    ORDER BY 
                        LENGTH(symbol),
                        symbol
                    LIMIT %s
                    """, 
                    (f"{partial_symbol.upper()}%", limit)
                )
                rows = cur.fetchall()
                
                suggestions = []
                for row in rows:
                    suggestions.append({
                        "symbol": row[0],
                        "name": row[1],
                        "asset_type": row[2]
                    })
                
                return {
                    "success": True,
                    "partial_input": partial_symbol.upper(),
                    "suggestions": suggestions,
                    "count": len(suggestions)
                }
                
    except Error as e:
        return {
            "success": False,
            "error": str(e),
            "partial_input": partial_symbol.upper()
        }


@mcp.tool(description="Get all companies grouped by asset type")
def get_companies_by_asset_type(asset_type: str = None) -> dict:
    """
    Get companies grouped by asset type or filter by specific asset type.
    
    Args:
        asset_type (str): Optional asset type filter (ETF, stock, COMMODITY_ETF, CRYPTO_ETF)
        
    Returns:
        dict: Companies grouped by asset type
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                if asset_type:
                    # Filter by specific asset type
                    cur.execute(
                        """
                        SELECT symbol, name, description
                        FROM public.company 
                        WHERE UPPER(asset_type) = UPPER(%s)
                        ORDER BY symbol
                        """, 
                        (asset_type,)
                    )
                    rows = cur.fetchall()
                    
                    companies = []
                    for row in rows:
                        companies.append({
                            "symbol": row[0],
                            "name": row[1],
                            "description": row[2]
                        })
                    
                    return {
                        "success": True,
                        "asset_type": asset_type.upper(),
                        "companies": companies,
                        "count": len(companies)
                    }
                else:
                    # Get all companies grouped by asset type
                    cur.execute(
                        """
                        SELECT asset_type, symbol, name
                        FROM public.company 
                        ORDER BY 
                            CASE asset_type 
                                WHEN 'stock' THEN 1
                                WHEN 'ETF' THEN 2
                                WHEN 'COMMODITY_ETF' THEN 3
                                WHEN 'CRYPTO_ETF' THEN 4
                                ELSE 5
                            END,
                            symbol
                        """
                    )
                    rows = cur.fetchall()
                    
                    grouped = {}
                    total_count = 0
                    
                    for row in rows:
                        asset_type_key = row[0] or "Unknown"
                        if asset_type_key not in grouped:
                            grouped[asset_type_key] = []
                        
                        grouped[asset_type_key].append({
                            "symbol": row[1],
                            "name": row[2]
                        })
                        total_count += 1
                    
                    # Add counts
                    for key in grouped:
                        grouped[key] = {
                            "count": len(grouped[key]),
                            "companies": grouped[key]
                        }
                    
                    return {
                        "success": True,
                        "asset_types": grouped,
                        "total_companies": total_count,
                        "asset_type_count": len(grouped)
                    }
                    
    except Error as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool(description="Get database statistics and performance metrics")
def get_database_stats() -> dict:
    """
    Get comprehensive database statistics for performance monitoring.
    
    Returns:
        dict: Database statistics and metrics
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # Company statistics
                cur.execute("SELECT COUNT(*) FROM public.company")
                company_count = cur.fetchone()[0]
                
                # Price records statistics  
                cur.execute("SELECT COUNT(*) FROM public.stock_price")
                price_count = cur.fetchone()[0]
                
                # Asset type distribution
                cur.execute("""
                    SELECT asset_type, COUNT(*) 
                    FROM public.company 
                    GROUP BY asset_type 
                    ORDER BY COUNT(*) DESC
                """)
                asset_distribution = cur.fetchall()
                
                # Date range of price data
                cur.execute("""
                    SELECT MIN(date) as earliest, MAX(date) as latest,
                           COUNT(DISTINCT symbol) as symbols_with_data
                    FROM public.stock_price
                """)
                date_info = cur.fetchone()
                
                # Average records per symbol
                cur.execute("""
                    SELECT symbol, COUNT(*) as record_count
                    FROM public.stock_price
                    GROUP BY symbol
                    ORDER BY record_count DESC
                    LIMIT 10
                """)
                top_symbols = cur.fetchall()
                
                return {
                    "success": True,
                    "database_statistics": {
                        "total_companies": company_count,
                        "total_price_records": price_count,
                        "symbols_with_price_data": date_info[2] if date_info else 0,
                        "average_records_per_symbol": round(price_count / date_info[2], 1) if date_info and date_info[2] > 0 else 0
                    },
                    "asset_type_distribution": {
                        row[0] or "Unknown": row[1] for row in asset_distribution
                    },
                    "date_range": {
                        "earliest_date": str(date_info[0]) if date_info and date_info[0] else None,
                        "latest_date": str(date_info[1]) if date_info and date_info[1] else None
                    },
                    "top_symbols_by_data": [
                        {"symbol": row[0], "record_count": row[1]} 
                        for row in top_symbols
                    ]
                }
                
    except Error as e:
        return {
            "success": False,
            "error": str(e)
        }
