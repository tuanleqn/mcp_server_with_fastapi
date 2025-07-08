import psycopg2
from psycopg2 import Error
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("USER_DB_URI", None)

if not DB_URI:
    raise ValueError("Database URI not found in environment variables.")

mcp = FastMCP(name="Database MCP Server", stateless_http=True)


@mcp.tool(description="Query user data by name")
def query_user_data(user_name: str) -> dict:
    """
    Retrieves user data from the database by user name.

    Args:
        user_name (str): The name of the user.
    Returns:
        dict: A dictionary containing user data.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM public.users WHERE name = %s", (user_name,))
                row = cur.fetchone()
                print(f"Fetched row: {row}")
                if row is None:
                    return {"error": "User not found"}
                return {"id": row[0], "name": row[1], "email": row[2], "tool": row[4]}
    except Error as e:
        return {"error": str(e)}


@mcp.tool(description="Update user data by ID")
def update_user_data(user_id: int, name: str, email: str) -> dict:
    """
    Updates user data in the database by user ID.

    Args:
        user_id (int): The ID of the user.
        name (str): The new name of the user.
        email (str): The new email of the user.

    Returns:
        dict: A dictionary indicating success or failure.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.users SET name = %s, email = %s WHERE id = %s",
                    (name, email, user_id),
                )
                conn.commit()
                return {"success": True}
    except Error as e:
        return {"error": str(e)}
