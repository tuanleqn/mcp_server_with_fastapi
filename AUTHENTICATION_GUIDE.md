# 🔐 Finance MCP Server - Authentication & User Scoping

## Overview

This finance MCP server implements a complete **JWT-based authentication system** with **user scoping** and **role-based access control**. Users can only access their own financial data unless they have admin privileges.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client/UI     │    │   FastAPI       │    │   MCP Servers   │
│                 │────│   Auth System   │────│   (Finance)     │
│ • Login/Signup  │    │ • JWT Tokens    │    │ • User Scoped   │
│ • Token Storage │    │ • User Roles    │    │ • Permission    │
│ • API Calls     │    │ • Middleware    │    │   Checks        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │ • User DB       │
                    │ • Finance DB    │
                    │ • Audit Logs    │
                    └─────────────────┘
```

## 🔑 Authentication Flow

### 1. User Registration/Login
```http
POST /auth/signup
{
  "username": "john_doe",
  "email": "john@example.com", 
  "password": "secure_password",
  "full_name": "John Doe"
}

POST /auth/login
{
  "username": "john_doe",
  "password": "secure_password"
}
```

### 2. JWT Token Response
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 3. Authenticated API Calls
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## 👥 User Roles & Permissions

### User Role Matrix

| Feature | Regular User | Premium User | Admin |
|---------|--------------|--------------|-------|
| View Own Portfolio | ✅ | ✅ | ✅ |
| Manage Own Portfolio | ✅ | ✅ | ✅ |
| View Other Portfolios | ❌ | ❌ | ✅ |
| Manage Other Portfolios | ❌ | ❌ | ✅ |
| Advanced Analytics | ❌ | ✅ | ✅ |
| Premium Data Access | ❌ | ✅ | ✅ |
| User Management | ❌ | ❌ | ✅ |
| System Administration | ❌ | ❌ | ✅ |

### Database Schema
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(64) NOT NULL,
    full_name VARCHAR(100),
    is_admin BOOLEAN DEFAULT FALSE,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'premium_user', 'admin')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## 🛡️ Security Implementation

### 1. Token Validation
```python
def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[AUTH_ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not username or not user_id:
            raise AuthenticationError("Invalid token payload")
            
        return {
            "username": username,
            "user_id": user_id,
            "roles": payload.get("roles", ["user"])
        }
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
```

### 2. User Permission Checking
```python
def check_user_permission(user_id: int, requested_user_id: int, is_admin: bool = False) -> bool:
    # Admin can access any user's data
    if is_admin:
        return True
    
    # User can only access their own data
    return user_id == requested_user_id
```

### 3. MCP Tool Authentication
```python
@mcp.tool(description="Get user portfolio with authentication")
def get_user_portfolio(user_id: int, auth_token: str = None) -> dict:
    if not auth_token:
        return {"error": "Authentication required", "code": "AUTH_REQUIRED"}
    
    try:
        # Verify token and get user info
        user_info = verify_token(auth_token)
        auth_user_id = user_info["user_id"]
        
        # Get full user info from database
        db_user_info = get_user_info(auth_user_id)
        is_admin = db_user_info.get("is_admin", False)
        
        # Check permissions
        if not check_user_permission(auth_user_id, user_id, is_admin):
            return {
                "error": f"Access denied. You can only access your own data (user_id: {auth_user_id})",
                "code": "ACCESS_DENIED"
            }
        
        # Proceed with operation...
        
    except Exception as e:
        return {"error": f"Authentication failed: {str(e)}", "code": "AUTH_ERROR"}
```

## 📊 Finance MCP Tools with Authentication

### Portfolio Management
- `add_stock_holding(user_id, symbol, quantity, price, date, auth_token)`
- `get_user_portfolio(user_id, auth_token)`
- `remove_stock_holding(user_id, symbol, quantity, auth_token)`

### Market Data (Public Access)
- `get_stock_price(symbol, date)` - No auth required
- `get_company_info(symbol)` - No auth required

### Premium Features (Premium/Admin Only)
- `get_advanced_analytics(user_id, auth_token)`
- `get_market_predictions(symbol, auth_token)`
- `generate_portfolio_report(user_id, auth_token)`

## 🔐 Access Control Examples

### ✅ Allowed Operations
```python
# User accessing their own portfolio
result = get_user_portfolio(user_id=123, auth_token="user_123_token")
# → Success: Returns portfolio data

# Admin accessing any portfolio  
result = get_user_portfolio(user_id=456, auth_token="admin_token")
# → Success: Admin can access any user's data
```

### ❌ Denied Operations
```python
# User trying to access another user's portfolio
result = get_user_portfolio(user_id=456, auth_token="user_123_token")
# → Error: "Access denied. You can only access your own data (user_id: 123)"

# No authentication token
result = get_user_portfolio(user_id=123)
# → Error: "Authentication required"
```

## 🚀 Quick Start

### 1. Setup Database
```bash
# Run the PowerShell setup script
.\setup_database.ps1

# Or manually:
psql -U postgres -c "CREATE DATABASE userdb;"
psql -U postgres -d userdb -f database_setup.sql
```

### 2. Configure Environment
```bash
# .env file
USER_DB_URI=postgresql://postgres:password@localhost:5432/userdb
FINANCE_DB_URI=postgresql://postgres:password@localhost:5432/financedb
AUTH_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
AUTH_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 3. Start Server
```bash
python main.py
```

### 4. Test Authentication
- Visit `http://localhost:8000/` for login interface
- Visit `http://localhost:8000/finance` for finance demo
- Use API docs at `http://localhost:8000/docs`

## 🧪 Testing Scenarios

### Test Users (from database_setup.sql)
```
Username: testuser     | Role: user         | Password: testpassword
Username: premiumuser  | Role: premium_user | Password: testpassword  
Username: admin        | Role: admin        | Password: testpassword
```

### Test Cases
1. **Login as regular user** → Try accessing another user's portfolio → Should fail
2. **Login as premium user** → Access premium features → Should succeed
3. **Login as admin** → Access any user's data → Should succeed
4. **Use expired token** → Any operation → Should fail
5. **No token** → Protected operation → Should fail

## 📝 Audit & Monitoring

All authentication attempts and operations are logged with:
- User ID and username
- Action attempted
- Resource accessed
- Timestamp
- IP address (when available)
- Success/failure status

## 🔒 Security Best Practices

1. **Token Expiration**: Tokens expire in 30 minutes by default
2. **Secure Secrets**: Use strong JWT secret keys (change default)
3. **Password Hashing**: SHA-256 hashing for passwords
4. **HTTPS Only**: Use HTTPS in production
5. **Rate Limiting**: Implement rate limiting for auth endpoints
6. **Input Validation**: All inputs are validated and sanitized
7. **SQL Injection Protection**: Parameterized queries only

## 🛠️ Customization

### Adding New Roles
```python
# Update USER_PERMISSIONS in auth_utils.py
USER_PERMISSIONS = {
    "user": ["view_own_portfolio", "manage_own_portfolio"],
    "premium_user": ["view_own_portfolio", "manage_own_portfolio", "advanced_analytics"],
    "vip_user": ["view_own_portfolio", "manage_own_portfolio", "advanced_analytics", "priority_support"],
    "admin": ["*"]  # All permissions
}
```

### Adding Permission Checks
```python
def require_permission(permission: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            auth_user = kwargs.get('auth_user', {})
            user_roles = auth_user.get('roles', [])
            
            if not check_permission(user_roles, permission):
                return {"error": f"Permission '{permission}' required", "code": "INSUFFICIENT_PERMISSIONS"}
                
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_permission("advanced_analytics")
def get_portfolio_analysis(user_id: int, auth_token: str = None):
    # Advanced analytics code...
```

## 📚 API Reference

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Register new user |
| POST | `/auth/login` | Login and get token |
| GET | `/auth/me` | Get current user info |
| POST | `/auth/logout` | Logout (client-side) |

### Finance MCP Endpoints

All finance endpoints require authentication via `Authorization: Bearer <token>` header.

| Tool | Description | User Scope |
|------|-------------|------------|
| `get_user_portfolio` | Get user's stock holdings | Own data + Admin override |
| `add_stock_holding` | Add stock to portfolio | Own data + Admin override |
| `remove_stock_holding` | Remove stock from portfolio | Own data + Admin override |
| `get_portfolio_analysis` | Advanced analytics | Premium + Admin only |
| `get_all_companies` | List all companies | Public access |

This authentication system ensures that your finance MCP server is secure, scalable, and provides appropriate access controls for different user types while maintaining the flexibility needed for financial applications.
