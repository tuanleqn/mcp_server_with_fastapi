"""
MCP Authentication Middleware
Provides token passing and user context for MCP servers
"""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
from typing import Optional, Dict, Any
from utils.auth_utils import verify_token, get_user_info

# Authentication configuration
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "your-super-secret-key-change-this")
AUTH_ALGORITHM = os.getenv("AUTH_ALGORITHM", "HS256")

security = HTTPBearer()

class MCPAuthMiddleware:
    """Middleware to handle authentication for MCP servers"""
    
    @staticmethod
    async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
        """Extract user from request headers"""
        try:
            auth_header = request.headers.get("authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header.split(" ")[1]
            user_info = verify_token(token)
            user_id = user_info["user_id"]
            
            # Get full user info
            db_user_info = get_user_info(user_id)
            if not db_user_info:
                return None
            
            return {
                **user_info,
                "is_admin": db_user_info.get("is_admin", False),
                "role": db_user_info.get("role", "user"),
                "db_info": db_user_info,
                "token": token
            }
        except Exception as e:
            print(f"Auth middleware error: {e}")
            return None
    
    @staticmethod
    def require_auth_decorator(func):
        """Decorator to require authentication for endpoints"""
        async def wrapper(request: Request, *args, **kwargs):
            user = await MCPAuthMiddleware.get_current_user(request)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Add user context to request
            request.state.user = user
            return await func(request, *args, **kwargs)
        
        return wrapper

# Utility functions for MCP tools
def extract_auth_from_request(request: Request) -> Optional[str]:
    """Extract auth token from MCP request"""
    try:
        if hasattr(request, 'state') and hasattr(request.state, 'user'):
            return request.state.user.get('token')
        
        # Fallback to header extraction
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        
        return None
    except Exception:
        return None

def get_user_context_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """Get user context from MCP request"""
    try:
        if hasattr(request, 'state') and hasattr(request.state, 'user'):
            return request.state.user
        return None
    except Exception:
        return None
