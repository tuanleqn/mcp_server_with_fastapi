"""
Authentication utilities for MCP finance servers
Provides user authentication and authorization functions for finance tools
"""

import os
import jwt
import psycopg2
from typing import Optional, Dict, Any
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

# Authentication configuration
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "your-super-secret-key-change-this")
AUTH_ALGORITHM = os.getenv("AUTH_ALGORITHM", "HS256")
USER_DB_URI = os.getenv("USER_DB_URI")

class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass

class AuthorizationError(Exception):
    """Raised when user doesn't have permission"""
    pass

def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and return user information
    
    Args:
        token (str): JWT token
        
    Returns:
        dict: User information containing username, user_id, and roles
        
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[AUTH_ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not username or not user_id:
            raise AuthenticationError("Invalid token payload")
            
        return {
            "username": username,
            "user_id": user_id,
            "roles": payload.get("roles", ["user"])  # Default role is 'user'
        }
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")

def get_user_info(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get user information from database
    
    Args:
        user_id (int): User ID
        
    Returns:
        dict: User information or None if not found
    """
    if not USER_DB_URI:
        raise Exception("USER_DB_URI not configured")
        
    try:
        with psycopg2.connect(USER_DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, username, email, full_name, is_admin, created_at 
                    FROM users WHERE id = %s
                """, (user_id,))
                user = cur.fetchone()
                
                if user:
                    return {
                        "id": user[0],
                        "username": user[1],
                        "email": user[2],
                        "full_name": user[3],
                        "is_admin": user[4] if user[4] is not None else False,
                        "created_at": user[5]
                    }
                return None
    except Exception as e:
        print(f"Error fetching user info: {e}")
        return None

def check_user_permission(user_id: int, requested_user_id: int, is_admin: bool = False) -> bool:
    """
    Check if user has permission to access data for requested_user_id
    
    Args:
        user_id (int): The authenticated user's ID
        requested_user_id (int): The user ID being requested in the operation
        is_admin (bool): Whether the authenticated user is an admin
        
    Returns:
        bool: True if user has permission
    """
    # Admin can access any user's data
    if is_admin:
        return True
    
    # User can only access their own data
    return user_id == requested_user_id

def require_auth(allow_admin_override: bool = True):
    """
    Decorator to require authentication for finance MCP tools
    
    Args:
        allow_admin_override (bool): Whether admins can override user restrictions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract token from the context or headers
            # This is a placeholder - in practice, you'd get this from the MCP request context
            token = kwargs.pop('auth_token', None)
            
            if not token:
                return {"error": "Authentication required", "code": "AUTH_REQUIRED"}
            
            try:
                user_info = verify_token(token)
                user_id = user_info["user_id"]
                
                # Get full user info from database
                db_user_info = get_user_info(user_id)
                if not db_user_info:
                    return {"error": "User not found", "code": "USER_NOT_FOUND"}
                
                is_admin = db_user_info.get("is_admin", False)
                
                # Add user context to kwargs
                kwargs['auth_user'] = {
                    **user_info,
                    "is_admin": is_admin,
                    "db_info": db_user_info
                }
                
                # Check if function has user_id parameter and enforce scoping
                if 'user_id' in kwargs and allow_admin_override:
                    requested_user_id = kwargs['user_id']
                    if not check_user_permission(user_id, requested_user_id, is_admin):
                        return {
                            "error": f"Access denied. You can only access your own data (user_id: {user_id})",
                            "code": "ACCESS_DENIED"
                        }
                elif 'user_id' in kwargs and not allow_admin_override:
                    # Force user to only access their own data
                    kwargs['user_id'] = user_id
                
                return func(*args, **kwargs)
                
            except AuthenticationError as e:
                return {"error": str(e), "code": "AUTH_ERROR"}
            except Exception as e:
                return {"error": f"Authentication failed: {str(e)}", "code": "INTERNAL_ERROR"}
        
        return wrapper
    return decorator

def get_user_scope_filter(user_id: int, is_admin: bool = False) -> str:
    """
    Get SQL WHERE clause for user scoping
    
    Args:
        user_id (int): User ID
        is_admin (bool): Whether user is admin
        
    Returns:
        str: SQL WHERE clause
    """
    if is_admin:
        return "1=1"  # Admin can see everything
    else:
        return f"user_id = {user_id}"

def ensure_user_scope(table_name: str, user_id: int, is_admin: bool = False) -> str:
    """
    Ensure SQL query includes user scoping
    
    Args:
        table_name (str): Table name that should have user_id column
        user_id (int): User ID
        is_admin (bool): Whether user is admin
        
    Returns:
        str: SQL WHERE clause to add to queries
    """
    if is_admin:
        return ""  # No restriction for admin
    else:
        return f"AND {table_name}.user_id = {user_id}"

# Role-based permissions
USER_PERMISSIONS = {
    "user": [
        "view_own_portfolio",
        "manage_own_portfolio", 
        "view_market_data",
        "run_basic_analysis"
    ],
    "premium_user": [
        "view_own_portfolio",
        "manage_own_portfolio",
        "view_market_data", 
        "run_basic_analysis",
        "run_advanced_analysis",
        "access_premium_data"
    ],
    "admin": [
        "view_all_portfolios",
        "manage_all_portfolios",
        "view_market_data",
        "run_basic_analysis", 
        "run_advanced_analysis",
        "access_premium_data",
        "manage_users",
        "system_administration"
    ]
}

def check_permission(user_roles: list, required_permission: str) -> bool:
    """
    Check if user has required permission
    
    Args:
        user_roles (list): List of user roles
        required_permission (str): Required permission
        
    Returns:
        bool: True if user has permission
    """
    for role in user_roles:
        if required_permission in USER_PERMISSIONS.get(role, []):
            return True
    return False
