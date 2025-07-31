"""
Simple database utilities to replace user_db MCP server
Direct database operations without MCP overhead
"""

import os
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("USER_DB_URI", None) or os.getenv("DATABASE_URL", None)

def query_user_by_name(user_name: str) -> dict:
    """Query user data by name directly from database."""
    if not DB_URI:
        return {"error": "Database URI not configured"}
    
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM public.users WHERE name = %s", (user_name,))
                row = cur.fetchone()
                if row is None:
                    return {"error": "User not found"}
                return {"id": row[0], "name": row[1], "email": row[2], "tool": row[4]}
    except Error as e:
        return {"error": str(e)}

def query_user_by_email(user_email: str) -> dict:
    """Query user data by email directly from database."""
    if not DB_URI:
        return {"error": "Database URI not configured"}
    
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM public.users WHERE email = %s", (user_email,))
                row = cur.fetchone()
                if row is None:
                    return {"error": "User not found"}
                return {"id": row[0], "name": row[1], "email": row[2], "tool": row[4]}
    except Error as e:
        return {"error": str(e)}

def update_user_data(user_id: str, name: str, email: str) -> dict:
    """Update user data by ID directly in database."""
    if not DB_URI:
        return {"error": "Database URI not configured"}
    
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
