import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - Company & Symbol Information")


@mcp.tool(description="Get company by symbol")
def get_company_by_symbol(company_symbol: str) -> dict:
    """
    Retrieves company information by symbol from the finance database.

    Args:
        company_symbol (str): The symbol of the company.

    Returns:
        dict: A dictionary containing company information.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM public.company WHERE symbol = %s", (company_symbol,)
                )
                row = cur.fetchone()
                if row is None:
                    return {"error": "Company not found"}
                return {
                    "symbol": row[0],
                    "asset_type": row[1],
                    "name": row[2],
                    "description": row[3],
                }
    except Error as e:
        return {"error": str(e)}


@mcp.tool(description="Search for companies and symbols")
def search_companies(query: str, limit: int = 10) -> dict:
    """
    Search for companies by name or symbol.
    
    Args:
        query (str): Search query (company name or symbol)
        limit (int): Maximum number of results to return
        
    Returns:
        dict: List of matching companies
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # Search by symbol or name (case insensitive)
                search_query = f"%{query}%"
                cur.execute(
                    """
                    SELECT symbol, asset_type, name, description 
                    FROM public.company 
                    WHERE UPPER(symbol) LIKE UPPER(%s) 
                       OR UPPER(name) LIKE UPPER(%s)
                    LIMIT %s
                    """, 
                    (search_query, search_query, limit)
                )
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
                        "count": len(results)
                    }
                else:
                    return {
                        "success": True,
                        "query": query,
                        "results": [],
                        "count": 0,
                        "message": f"No companies found matching '{query}'"
                    }
                    
    except Error as e:
        return {"error": str(e)}


@mcp.tool(description="Get symbol suggestions based on partial input")
def get_symbol_suggestions(partial_symbol: str, limit: int = 5) -> dict:
    """
    Get symbol suggestions based on partial input.
    
    Args:
        partial_symbol (str): Partial symbol to match
        limit (int): Maximum suggestions to return
        
    Returns:
        dict: List of symbol suggestions
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # Match symbols starting with the partial input
                cur.execute(
                    """
                    SELECT symbol, name 
                    FROM public.company 
                    WHERE UPPER(symbol) LIKE UPPER(%s)
                    ORDER BY symbol
                    LIMIT %s
                    """, 
                    (f"{partial_symbol}%", limit)
                )
                rows = cur.fetchall()
                
                suggestions = []
                for row in rows:
                    suggestions.append({
                        "symbol": row[0],
                        "name": row[1]
                    })
                
                return {
                    "success": True,
                    "partial_input": partial_symbol,
                    "suggestions": suggestions,
                    "count": len(suggestions)
                }
                
    except Error as e:
        return {"error": str(e)}


# Merged symbol discovery functionality (simplified, no OOP)
