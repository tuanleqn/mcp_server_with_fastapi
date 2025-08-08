#!/usr/bin/env python3
"""
Finance DB Company MCP Server
Enhanced company search with symbol discovery and substring matching
"""

import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Import from local helpers
from .finance_helpers import search_companies_helper

load_dotenv()

mcp = FastMCP(name="Finance Company Search Server")

@mcp.tool(description="Enhanced company search with symbol discovery and substring matching")
def search_companies(query: str, limit: int = 10) -> dict:
    """
    Enhanced company search with symbol discovery and substring matching.
    
    Args:
        query: Company name or symbol to search for (supports partial matches)
        limit: Maximum number of results to return (default 10)
        
    Returns:
        Dictionary with search results and company information
        
    Example:
        search_companies("AAPL", limit=5)
    """
    try:
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Query cannot be empty"
            }
        
        # Use centralized helper for enhanced search
        result = search_companies_helper(query.strip(), limit)
        
        if not result.get("success", False):
            return {
                "success": False,
                "error": result.get('error', 'Unknown error occurred')
            }
        
        companies = result.get("companies", [])
        total_found = result.get("results_found", 0)
        
        if total_found == 0:
            return {
                "success": True,
                "message": f"No companies found matching '{query}'",
                "companies": [],
                "total_found": 0
            }
        
        # Format results for better readability
        formatted_companies = []
        for company in companies:
            formatted_companies.append({
                "symbol": company.get('symbol', 'N/A'),
                "name": company.get('name', 'N/A'),
                "asset_type": company.get('asset_type', 'N/A'),
                "description": company.get('description', 'No description available')[:200] + "..." if len(company.get('description', '')) > 200 else company.get('description', 'No description available')
            })
        
        return {
            "success": True,
            "query": query,
            "total_found": total_found,
            "companies": formatted_companies
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run()
