"""
Simple Authentication System for FastAPI MCP Server
Focus: Chatbot users with chat/query functionality only
No stock trading features - keeps existing database schema intact
"""

import os
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Authentication configuration
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "your-super-secret-key-change-this")
AUTH_ALGORITHM = os.getenv("AUTH_ALGORITHM", "HS256") 
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

USER_DB_URI = os.getenv("USER_DB_URI")
if not USER_DB_URI:
    print("⚠️ Warning: USER_DB_URI not found in environment variables")

# Security
security = HTTPBearer()

# Pydantic models for authentication
class UserSignup(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Optional[str] = "user"

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]

class ChatbotUser(BaseModel):
    id: int
    name: str
    email: str
    role: str

class ChatMessage(BaseModel):
    id: Optional[int] = None
    name: str
    user_id: int
    tenant_id: int
    
class QueryLog(BaseModel):
    id: Optional[int] = None
    chat_id: int
    query: str
    answer: Optional[str] = None
    tenant_id: int

# Authentication helper functions
def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, AUTH_SECRET_KEY, algorithm=AUTH_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return user information"""
    try:
        payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[AUTH_ALGORITHM])
        email = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not email or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        return {
            "email": email,
            "user_id": user_id
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    return verify_token(credentials.credentials)

def get_db_connection():
    """Get database connection"""
    if not USER_DB_URI:
        raise HTTPException(status_code=500, detail="Database not configured")
    return psycopg2.connect(USER_DB_URI)

# User management functions
def create_user(user_data: UserSignup) -> Dict[str, Any]:
    """Create a new user in the database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if user already exists
                cur.execute("SELECT id FROM USERS WHERE email = %s", (user_data.email,))
                if cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already exists"
                    )
                
                # Get next available ID (starting from a high number to avoid conflicts)
                cur.execute("SELECT COALESCE(MAX(id), 100004) + 1 FROM USERS")
                next_id = cur.fetchone()[0]
                
                # Create new user
                hashed_password = hash_password(user_data.password)
                cur.execute("""
                    INSERT INTO USERS (id, name, email, password, role) 
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    next_id,
                    user_data.name, 
                    user_data.email, 
                    hashed_password, 
                    user_data.role
                ))
                conn.commit()
                
                return {
                    "user_id": next_id,
                    "name": user_data.name,
                    "email": user_data.email,
                    "role": user_data.role,
                    "message": "User created successfully"
                }
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user and return user info"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, email, password, role 
                    FROM USERS WHERE email = %s
                """, (email,))
                user = cur.fetchone()
                
                if not user or not verify_password(password, user[3]):
                    return None
                
                return {
                    "id": user[0],
                    "name": user[1],
                    "email": user[2],
                    "role": user[4]
                }
    except psycopg2.Error as e:
        print(f"Database error during authentication: {e}")
        return None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user information by user ID"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, email, role 
                    FROM USERS WHERE id = %s
                """, (user_id,))
                user = cur.fetchone()
                
                if not user:
                    return None
                
                return {
                    "id": user[0],
                    "name": user[1],
                    "email": user[2],
                    "role": user[3]
                }
    except psycopg2.Error as e:
        print(f"Database error getting user: {e}")
        return None

# Chat and Query functions (using new schema)
def create_chat(user_id: int, name: str, tenant_id: int = 1) -> Dict[str, Any]:
    """Create a new chat session"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get next available chat ID
                cur.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM CHAT")
                next_id = cur.fetchone()[0]
                
                cur.execute("""
                    INSERT INTO CHAT (id, name, user_id, tenant_id) 
                    VALUES (%s, %s, %s, %s)
                """, (next_id, name, user_id, tenant_id))
                conn.commit()
                
                return {
                    "chat_id": next_id,
                    "name": name,
                    "user_id": user_id,
                    "tenant_id": tenant_id
                }
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")

def get_user_chats(user_id: int) -> List[Dict[str, Any]]:
    """Get all chats for a user"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, user_id, tenant_id 
                    FROM CHAT 
                    WHERE user_id = %s 
                    ORDER BY id DESC
                """, (user_id,))
                chats = cur.fetchall()
                
                return [
                    {
                        "id": chat[0],
                        "name": chat[1],
                        "user_id": chat[2],
                        "tenant_id": chat[3]
                    }
                    for chat in chats
                ]
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chats: {str(e)}")

def save_query(chat_id: int, query: str, answer: str = None, tenant_id: int = 1) -> Dict[str, Any]:
    """Save a query to the database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get next available query ID
                cur.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM QUERY")
                next_id = cur.fetchone()[0]
                
                cur.execute("""
                    INSERT INTO QUERY (id, chat_id, query, answer, tenant_id) 
                    VALUES (%s, %s, %s, %s, %s)
                """, (next_id, chat_id, query, answer, tenant_id))
                conn.commit()
                
                return {
                    "query_id": next_id,
                    "chat_id": chat_id,
                    "query": query,
                    "answer": answer,
                    "tenant_id": tenant_id
                }
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to save query: {str(e)}")

def get_chat_queries(chat_id: int) -> List[Dict[str, Any]]:
    """Get all queries for a specific chat"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, chat_id, query, answer, tenant_id 
                    FROM QUERY 
                    WHERE chat_id = %s 
                    ORDER BY id ASC
                """, (chat_id,))
                queries = cur.fetchall()
                
                return [
                    {
                        "id": query[0],
                        "chat_id": query[1],
                        "query": query[2],
                        "answer": query[3],
                        "tenant_id": query[4]
                    }
                    for query in queries
                ]
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queries: {str(e)}")

# Authentication decorator for MCP tools
def require_auth_chatbot():
    """Decorator to require authentication for chatbot MCP tools"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract token from kwargs (passed by authenticated endpoints)
            auth_token = kwargs.pop('auth_token', None)
            
            if not auth_token:
                return {"error": "Authentication required", "code": "AUTH_REQUIRED"}
            
            try:
                user_info = verify_token(auth_token)
                
                # Add user context to kwargs
                kwargs['auth_user'] = user_info
                
                return func(*args, **kwargs)
                
            except HTTPException as e:
                return {"error": str(e.detail), "code": "AUTH_ERROR"}
            except Exception as e:
                return {"error": f"Authentication failed: {str(e)}", "code": "INTERNAL_ERROR"}
        
        return wrapper
    return decorator
