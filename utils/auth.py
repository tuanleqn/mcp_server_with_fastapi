import os
from typing import Any
from fastapi import Request
import jwt


def get_payload(req: Request) -> dict:
    """
    Extracts and decodes the JWT payload from the request headers.
    """
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


def get_role(payload: Any) -> str:
    """
    Retrieves the role from the JWT payload.
    """
    return payload.get("role")


def get_user_id(payload: Any) -> str:
    """
    Retrieves the user ID from the JWT payload.
    """
    return payload.get("user_id")


def print_header(req: Request) -> None:
    """
    Prints the Authorization header from the request.
    """
    auth = req.headers.get("Authorization", "")
    if not auth:
        print("Authorization header is missing")
    else:
        print(f"Authorization header: {auth}")
