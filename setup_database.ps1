param(
    [string]$PostgreSQLPath = "C:\Program Files\PostgreSQL\16\bin",
    [string]$DatabaseUrl = $null
)

Write-Host "🤖 Simple Chatbot Authentication Database Setup" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host

# Check if DATABASE_URL is provided or get from environment
if (-not $DatabaseUrl) {
    # First try USER_DB_URI environment variable
    $DatabaseUrl = $env:USER_DB_URI
    
    # Fallback to .env file
    if (-not $DatabaseUrl -and (Test-Path ".env")) {
        Write-Host "📄 Reading database URL from .env file..." -ForegroundColor Yellow
        $envContent = Get-Content ".env"
        foreach ($line in $envContent) {
            if ($line -match 'USER_DB_URI="(.+)"' -or $line -match 'FINANCE_DB_URI="(.+)"') {
                $DatabaseUrl = $matches[1]
                break
            }
        }
    }
}

if (-not $DatabaseUrl) {
    Write-Host "❌ Database URL not found!" -ForegroundColor Red
    Write-Host "Please set USER_DB_URI environment variable or provide via parameter" -ForegroundColor Yellow
    Write-Host "Usage: .\setup_database.ps1 -DatabaseUrl 'your_connection_string'" -ForegroundColor Yellow
    Write-Host "Example: postgres://user:password@host:port/database?sslmode=require" -ForegroundColor Gray
    exit 1
}

Write-Host "� Using database: $($DatabaseUrl.Split('@')[1].Split('/')[0])" -ForegroundColor Green

# Check if PostgreSQL path exists (for psql command)
if (-not (Test-Path $PostgreSQLPath)) {
    Write-Host "⚠️  PostgreSQL not found at: $PostgreSQLPath" -ForegroundColor Yellow
    Write-Host "Trying to use psql from PATH..." -ForegroundColor Yellow
    $psqlCommand = "psql"
} else {
    Write-Host "📍 Using PostgreSQL at: $PostgreSQLPath" -ForegroundColor Green
    $env:PATH = "$PostgreSQLPath;$env:PATH"
    $psqlCommand = "psql"
}

try {
    Write-Host "📊 Adding simple chatbot authentication tables..." -ForegroundColor Yellow
    Write-Host "   • USERS - User accounts with authentication" -ForegroundColor Gray
    Write-Host "   • CHAT - Chat messages and responses" -ForegroundColor Gray  
    Write-Host "   • QUERY - Query logging and results" -ForegroundColor Gray
    Write-Host ""
    
    # Execute the SQL setup script
    if (Test-Path "database_setup.sql") {
        Write-Host "Executing database_setup.sql..."
        & $psqlCommand $DatabaseUrl -f "database_setup.sql"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Chatbot authentication tables created successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to create database tables" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ database_setup.sql not found!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host
    Write-Host "🎉 Simple chatbot authentication setup completed!" -ForegroundColor Green
    Write-Host
    Write-Host "📋 Summary:" -ForegroundColor Cyan
    Write-Host "• Added tables: USERS, CHAT, QUERY" -ForegroundColor White
    Write-Host "• Test accounts created with password 'testpassword':" -ForegroundColor White
    Write-Host "  - testuser (Test User)" -ForegroundColor Gray
    Write-Host "  - chatuser (Chat User)" -ForegroundColor Gray  
    Write-Host "  - demouser (Demo User)" -ForegroundColor Gray
    Write-Host
    Write-Host "🚀 You can now start the chatbot server!" -ForegroundColor Green
    Write-Host "Run: python main.py" -ForegroundColor Yellow
    Write-Host "Visit: http://localhost:8000/" -ForegroundColor Cyan
    
} catch {
    Write-Host "❌ Error during database setup: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host
Write-Host "🔗 Useful Commands:" -ForegroundColor Cyan
Write-Host "• Connect to database: $psqlCommand '$DatabaseUrl'" -ForegroundColor White
Write-Host "• List tables: \dt" -ForegroundColor White
Write-Host "• View chatbot users: SELECT * FROM \"USERS\";" -ForegroundColor White
Write-Host "• View chat history: SELECT * FROM \"CHAT\" ORDER BY timestamp DESC LIMIT 10;" -ForegroundColor White
Write-Host "• View existing companies: SELECT symbol, name FROM company LIMIT 10;" -ForegroundColor White
Write-Host ""
Write-Host "📖 See CHATBOT_AUTH_README.md for full documentation" -ForegroundColor Cyan
