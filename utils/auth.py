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
    token = auth[14:]
    try:
        payload = jwt.decode(
            token,
            os.getenv("AUTH_SECRET_KEY"),
            algorithms=[os.getenv("AUTH_ALGORITHM") or "HS256"],
        )
        print(f"Decoded payload: {payload}")
        return payload
    except jwt.ExpiredSignatureError:
        return {"error": "Token has expired"}
    except jwt.DecodeError:
        return {"error": "Invalid token format"}
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}
    except Exception as e:
        return {"error": f"Authentication error: {str(e)}"}


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


def generate_token(payload: dict) -> str:
    """
    Generates a JWT token from a payload dictionary.

    Args:
        payload: Dictionary containing the data to be encoded in the token

    Returns:
        str: The generated JWT token string
    """
    import jwt
    import os
    from datetime import datetime, timedelta
    from dotenv import load_dotenv

    load_dotenv()

    # Add expiration time if not provided
    if "exp" not in payload:
        # Default expiration time: 30 days
        expiration = datetime.utcnow() + timedelta(days=30)
        payload["exp"] = expiration

    # Add issued at time if not provided
    if "iat" not in payload:
        payload["iat"] = datetime.utcnow()

    # Generate the token
    token = jwt.encode(
        payload,
        os.getenv("AUTH_SECRET_KEY"),
        algorithm=os.getenv("AUTH_ALGORITHM") or "HS256",
    )

    return token
