import psycopg2
from psycopg2 import Error
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Database MCP Server", version="0.0.1", stateless_http=True)


@mcp.tool(description="Query user data by ID")
def get_user_data(user_id: int) -> dict:
    """
    Retrieves user data from the database by user ID.

    Args:
        user_id (int): The ID of the user.

    Returns:
        dict: A dictionary containing user data.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                print(f"Fetched row: {row}")
                if row is None:
                    return {"error": "User not found"}
                return {"id": row[0], "name": row[1], "email": row[2]}
    except Error as e:
        return {"error": str(e)}
