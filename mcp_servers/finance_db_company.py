import os
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("FINANCE_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Finance MCP Server - About company information")


@mcp.tool(description="Get all companies")
def get_all_companies() -> list:
    """
    Retrieves all companies from the finance database.

    Returns:
        list: A list of dictionaries containing company information.
    """

    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM public.company")
                rows = cur.fetchall()
                return [{"symbol": row[0], "name": row[2]} for row in rows]
    except Error as e:
        return {"error": str(e)}


@mcp.tool(description="Get company by name")
def get_company_by_name(company_name: str) -> dict:
    """
    Retrieves company information by name from the finance database.

    Args:
        company_name (str): The name of the company.

    Returns:
        dict: A dictionary containing company information.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM public.company WHERE name = %s", (company_name,)
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


# @mcp.tool(description="Get company by description")
