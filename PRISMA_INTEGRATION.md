# 🔧 Integration with Existing PostgreSQL & Prisma Setup

## Current Setup Understanding

You have:
- ✅ Existing PostgreSQL database on Aiven Cloud
- ✅ Prisma schema with `company` and `stock_price` models
- ✅ Working finance data in the database

## Integration Steps

### 1. 📋 Add User Authentication Tables

Your existing database will be extended with these new tables:
- `users` - User accounts with roles (user, premium_user, admin)  
- `user_stock_holdings` - Links users to their stock holdings
- `user_watchlists` - User stock watchlists
- `user_sessions` - Session management
- `user_audit` - Audit trail

### 2. 🔄 Update Your Prisma Schema

Add the content from `prisma_schema_extension.prisma` to your existing `schema.prisma`:

```prisma
// Your existing models
model company {
  symbol        String        @id @db.VarChar(10)
  asset_type    String?       @db.VarChar(50)
  name          String?       @db.VarChar(255)
  // ... existing fields ...
  
  // ADD THESE NEW RELATIONS:
  stock_holdings  user_stock_holdings[]
  watchlists      user_watchlists[]
}

model stock_price {
  // ... your existing fields ...
}

// ADD THESE NEW MODELS:
model users {
  id              Int      @id @default(autoincrement())
  username        String   @unique @db.VarChar(50)
  email           String   @unique @db.VarChar(100)
  password_hash   String   @db.VarChar(64)
  full_name       String?  @db.VarChar(100)
  created_at      DateTime @default(now()) @db.Timestamptz
  updated_at      DateTime @updatedAt @db.Timestamptz
  is_active       Boolean  @default(true)
  is_admin        Boolean  @default(false)
  role            String   @default("user") @db.VarChar(20)
  
  // Relations
  stock_holdings  user_stock_holdings[]
  watchlists      user_watchlists[]
  sessions        user_sessions[]
  audit_logs      user_audit[]
  
  @@map("users")
}

// ... copy other models from prisma_schema_extension.prisma
```

### 3. 🗄️ Run Database Migration

**Option A: Using our SQL script (Recommended)**
```bash
# Run the setup script with your Aiven connection
.\setup_database.ps1
```

**Option B: Using Prisma (if you prefer)**
```bash
# After updating schema.prisma
npx prisma db push
# or
npx prisma migrate dev --name add_user_authentication
```

### 4. 🔧 Environment Configuration

Your `.env` is already updated to use the same database for both finance and user data:

```env
# Both point to the same financedb database
USER_DB_URI="postgres://avnadmin:AVNS_...@.../financedb?sslmode=require"
FINANCE_DB_URI="postgres://avnadmin:AVNS_...@.../financedb?sslmode=require"

# Authentication settings
AUTH_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production-please
AUTH_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 5. 🚀 Test the Integration

1. **Run database setup:**
   ```bash
   .\setup_database.ps1
   ```

2. **Start MCP server:**
   ```bash
   python main.py
   ```

3. **Test authentication:**
   - Visit `http://localhost:8000/` for login
   - Visit `http://localhost:8000/finance` for finance demo
   - Use test accounts:
     - `testuser` / `testpassword` (regular user)
     - `premiumuser` / `testpassword` (premium user)
     - `admin` / `testpassword` (admin user)

## 📊 Database Schema Overview

After integration, your database will have:

```
Existing Tables:
├── company (your existing data)
└── stock_price (your existing data)

New Authentication Tables:
├── users (user accounts & roles)
├── user_stock_holdings (user portfolios)
├── user_watchlists (user watchlists)
├── user_sessions (session management)
└── user_audit (audit logging)
```

## 🔗 Foreign Key Relationships

The new tables connect to your existing data:
- `user_stock_holdings.symbol` → `company.symbol`
- `user_watchlists.symbol` → `company.symbol`
- All user tables → `users.id`

## 🛡️ User Scoping in Action

With this setup:

### Regular Users
```python
# User with ID 123 tries to access their portfolio
get_user_portfolio(user_id=123, auth_token="user_123_token")
# ✅ Success: Returns their holdings from user_stock_holdings table

# Same user tries to access another user's portfolio  
get_user_portfolio(user_id=456, auth_token="user_123_token")
# ❌ Denied: "Access denied. You can only access your own data"
```

### Admin Users
```python
# Admin tries to access any user's portfolio
get_user_portfolio(user_id=456, auth_token="admin_token") 
# ✅ Success: Admin can access any user's data
```

## 🔄 Migration from Existing Finance Tools

Your current finance MCP tools will automatically gain authentication:

**Before:**
```python
# Anyone could call this
get_stock_price("AAPL", "2025-01-15")
```

**After:**
```python
# Now requires authentication token for user-specific operations
add_stock_holding(user_id=123, symbol="AAPL", quantity=10, 
                  purchase_price=150.00, auth_token="user_token")
```

## 🎯 Benefits of This Approach

1. **Unified Database**: Everything in one place, easier to manage
2. **Existing Data Preserved**: Your company/stock_price data unchanged
3. **User Isolation**: Users see only their own portfolio data
4. **Admin Capabilities**: Admins can manage all user data
5. **Audit Trail**: All user actions are logged
6. **Scalable**: Easy to add more user-specific features

## 🚨 Important Notes

- **Backup First**: Always backup your database before running migrations
- **Test Environment**: Test the setup in a development environment first
- **Connection Limits**: Aiven has connection limits; the MCP server uses connection pooling
- **SSL Required**: Your Aiven setup requires SSL (`sslmode=require`)

This integration adds powerful user authentication to your existing finance data while preserving all your current functionality!
