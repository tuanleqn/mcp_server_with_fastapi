# Cleanup script to remove unnecessary authentication extensions
Write-Host "🧹 Cleaning up unnecessary authentication extensions..." -ForegroundColor Yellow

# Files to remove from root directory
$rootFilesToRemove = @(
    "AUTHENTICATION_GUIDE.md",
    "CHATBOT_AUTH_README.md", 
    "PRISMA_INTEGRATION.md",
    "prisma_schema_extension.prisma",
    "QUICKSTART.md",
    "TROUBLESHOOTING.md",
    "test_auth.py",
    "verify_setup.py"
)

# Files to remove from utils directory
$utilsFilesToRemove = @(
    "auth.py",
    "auth_utils.py", 
    "mcp_auth_middleware.py"
)

# Files to remove from static directory
$staticFilesToRemove = @(
    "auth.html",
    "finance_demo.html",
    "index.html"
)

Write-Host "Removing root directory files..." -ForegroundColor Gray
foreach ($file in $rootFilesToRemove) {
    $fullPath = Join-Path (Get-Location) $file
    if (Test-Path $fullPath) {
        Remove-Item $fullPath -Force
        Write-Host "  ✅ Removed $file" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  $file already removed" -ForegroundColor Gray
    }
}

Write-Host "Removing utils directory files..." -ForegroundColor Gray
foreach ($file in $utilsFilesToRemove) {
    $fullPath = Join-Path (Get-Location) "utils\$file"
    if (Test-Path $fullPath) {
        Remove-Item $fullPath -Force
        Write-Host "  ✅ Removed utils\$file" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  utils\$file already removed" -ForegroundColor Gray
    }
}

Write-Host "Removing static directory files..." -ForegroundColor Gray
foreach ($file in $staticFilesToRemove) {
    $fullPath = Join-Path (Get-Location) "static\$file"
    if (Test-Path $fullPath) {
        Remove-Item $fullPath -Force
        Write-Host "  ✅ Removed static\$file" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  static\$file already removed" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "🎉 Cleanup completed!" -ForegroundColor Green
Write-Host "Remaining simple authentication files:" -ForegroundColor Cyan
Write-Host "  • utils\simple_auth.py - Simple authentication functions" -ForegroundColor White
Write-Host "  • static\chatbot.html - Simple chatbot interface" -ForegroundColor White
Write-Host "  • database_setup.sql - Minimal database setup" -ForegroundColor White
Write-Host "  • setup_database.ps1 - Database setup script" -ForegroundColor White
Write-Host "  • .env.example - Configuration template" -ForegroundColor White
