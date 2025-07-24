# 🤖 Simple Chatbot Authentication System

## Overview

This is a **simple authentication system** designed specifically for chatbot users. It focuses on basic user management for chat interactions without complex stock trading features.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chatbot UI    │────│   FastAPI       │────│   PostgreSQL    │
│ • Login/Signup  │    │ • JWT Auth      │    │ • USERS         │
│ • Chat Interface│    │ • Chat Endpoints│    │ • CHAT          │
│ • Message Input │    │ • Query Logging │    │ • QUERY         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Database Schema

The system adds these **3 simple tables** to your existing database:

### USERS Table
```sql
CREATE TABLE "USERS" (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL, 
    password_hash VARCHAR(64) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### CHAT Table  
```sql
CREATE TABLE "CHAT" (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "USERS"(id),
    message TEXT NOT NULL,
    response TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### QUERY Table
```sql
CREATE TABLE "QUERY" (
    id SERIAL PRIMARY KEY, 
    user_id INTEGER NOT NULL REFERENCES "USERS"(id),
    query TEXT NOT NULL,
    result TEXT,
    query_type VARCHAR(50) DEFAULT 'general',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## 🚀 Quick Start

### 1. Setup Database
```bash
# Run setup script to add authentication tables
.\setup_database.ps1
```

### 2. Start Server
```bash
python main.py
```

### 3. Test Chatbot
- Visit `http://localhost:8000/` for chatbot interface
- Visit `http://localhost:8000/auth` for standalone login page
- Visit `http://localhost:8000/docs` for API documentation

### 4. Test User Accounts
Login with these test accounts (password: `testpassword`):
- `testuser` - Test User
- `chatuser` - Chat User  
- `demouser` - Demo User

## 🔐 Authentication Features

### User Management
- ✅ User signup/login with JWT tokens
- ✅ Password hashing (SHA-256)
- ✅ Token expiration (30 minutes default)
- ✅ User session management

### Chatbot Functions
- ✅ Authenticated chat messages
- ✅ Chat history per user
- ✅ Query logging and history
- ✅ User-scoped data access

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Create new user account |
| POST | `/auth/login` | Login and get JWT token |
| GET | `/auth/me` | Get current user info |
| POST | `/auth/logout` | Logout (client-side) |

### Chatbot
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send chat message |
| GET | `/chat/history` | Get user chat history |
| POST | `/query` | Execute query |
| GET | `/query/history` | Get user query history |

All chatbot endpoints require `Authorization: Bearer <token>` header.

## 💬 Usage Examples

### Login and Chat
```javascript
// 1. Login
const loginResponse = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'testuser', password: 'testpassword' })
});
const { access_token } = await loginResponse.json();

// 2. Send chat message  
const chatResponse = await fetch('/chat', {
    method: 'POST',
    headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${access_token}`
    },
    body: JSON.stringify({ message: 'Hello chatbot!' })
});
const chatData = await chatResponse.json();
// Returns: { message: "Hello chatbot!", response: "Echo: Hello chatbot!", ... }
```

### Query with Logging
```javascript
const queryResponse = await fetch('/query', {
    method: 'POST',
    headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${access_token}`
    },
    body: JSON.stringify({ 
        query: 'search companies', 
        query_type: 'company_search' 
    })
});
```

## 🛡️ Security Features

### Token-Based Authentication
- JWT tokens with expiration
- Secure password hashing  
- User session validation

### User Scoping
- Users see only their own chat history
- Users see only their own query history
- No cross-user data access

### Database Security
- Parameterized queries (no SQL injection)
- Foreign key constraints
- Input validation

## 🔧 Configuration

### Environment Variables
```env
# Database (uses your existing database)
USER_DB_URI=postgres://user:pass@host:port/financedb?sslmode=require

# Authentication  
AUTH_SECRET_KEY=your-super-secret-jwt-key-change-this
AUTH_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 🎯 Integration with Your Existing System

### No Schema Conflicts
- Uses separate table names (`USERS`, `CHAT`, `QUERY`)
- Does not modify your existing `company` and `stock_price` tables
- Completely isolated from your finance data

### Easy Extension
- Add chatbot AI/ML integration
- Connect to your existing MCP tools
- Extend query types for different operations
- Add more user fields as needed

## 🔍 Monitoring & Debugging

### Check User Activity
```sql
-- See all users
SELECT id, username, email, is_active, created_at FROM "USERS";

-- See recent chats  
SELECT u.username, c.message, c.response, c.timestamp 
FROM "CHAT" c 
JOIN "USERS" u ON c.user_id = u.id 
ORDER BY c.timestamp DESC LIMIT 10;

-- See recent queries
SELECT u.username, q.query, q.query_type, q.timestamp
FROM "QUERY" q
JOIN "USERS" u ON q.user_id = u.id  
ORDER BY q.timestamp DESC LIMIT 10;
```

### Health Check
Visit `http://localhost:8000/health` to verify all MCP servers are running.

This simple authentication system provides a solid foundation for chatbot user management while keeping your existing database schema intact!
