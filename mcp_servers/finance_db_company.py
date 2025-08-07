"""
Finance Database Company Server
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
    db_result = search_companies_helper(query, limit)
    
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
