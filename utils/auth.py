from json import load
import os
from typing import Any
from fastapi import Request
import jwt
from dotenv import load_dotenv

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


def check_auth(payload: Any) -> bool:
    if payload.get("role") == "admin":
        return True
    if payload.get("role") == "user":
        if is_user_the_owner(payload.get("id")):
            return True
    return False
