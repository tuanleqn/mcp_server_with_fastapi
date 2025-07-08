import os
from typing import Any
from fastapi import Request
import jwt
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv

from mcp_servers.user_db import DB_URI

load_dotenv()


def get_payload(req: Request):
    auth = req.headers.get("Authorization", "")
    if not auth:
        return {"error": "Authorization header is missing"}
    if not auth.startswith("Bearer "):
        return {"error": "Authorization header must start with 'Bearer '"}
    token = auth[7:]
    try:
        payload = jwt.decode(
            token,
            os.getenv("AUTH_SECRET_KEY"),
            algorithms=[os.getenv("AUTH_ALGORITHM") or ""],
        )
    except jwt.ExpiredSignatureError:
        return {"error": "Token has expired"}
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}
    return payload


def is_user_the_owner(user_id: int) -> bool:
    """
    Check if the user is the owner of the resource.

    Args:
        user_id (int): The ID of the user to check.

    Returns:
        bool: True if the user is the owner, False otherwise.
    """
    try:
        with psycopg2.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM public.users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                return row is not None
    except Error as e:
        print(f"Database error: {e}")
        return False


def check_auth(payload: Any) -> bool:
    if payload.get("role") == "admin":
        return True
    if payload.get("role") == "user":
        if is_user_the_owner(payload.get("id")):
            return True
    return False
