"""
Finance Database Company Server
Simplified MCP server for company data lookup using helper functions
"""

import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from .finance_helpers import (
    get_company_info_helper, 
    search_companies_helper, 
    discover_symbol_helper
)

load_dotenv()

mcp = FastMCP(name="Finance Company Database Server")

@mcp.tool(description="Search for companies with intelligent symbol discovery")
def search_companies(query: str, limit: int = 10) -> dict:
    """
    Search for companies in the database using company name or symbol.
    Includes intelligent symbol discovery for symbols not in database.
    
    Args:
        query: Search term (company name or symbol)
        limit: Maximum number of results to return (default: 10, max: 50)
    
    Returns:
        Dictionary containing search results with company information
    """
    # First try database search
    db_result = search_companies_helper(query, limit)
    
    # If no results in database, try external symbol discovery
    if db_result.get("success") and db_result.get("results_found") == 0:
        discovery_result = discover_symbol_helper(query)
        if discovery_result.get("success"):
            return {
                "success": True,
                "query": query,
                "results_found": 1,
                "source": "external_discovery",
                "companies": [discovery_result["data"]]
            }
    
    return db_result

if __name__ == "__main__":
    mcp.run()
