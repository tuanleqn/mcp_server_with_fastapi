# PowerShell script to start the MCP FastAPI server
$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting MCP FastAPI Server with Virtual Environment..." -ForegroundColor Green
Write-Host "Project directory: $(Get-Location)" -ForegroundColor Yellow

# Change to project directory
Set-Location "d:\internship\multimcp-server-with-fastapi"

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "❌ Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please make sure .venv directory exists with Python installation." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Virtual environment found" -ForegroundColor Green

# Test Python and basic imports
Write-Host "🧪 Testing Python and imports..." -ForegroundColor Blue
try {
    $pythonVersion = & .venv\Scripts\python.exe -c "import sys; print(sys.version.split()[0])"
    Write-Host "✓ Python version: $pythonVersion" -ForegroundColor Green
    
    & .venv\Scripts\python.exe -c "from dotenv import load_dotenv; print('✓ dotenv works')"
    & .venv\Scripts\python.exe -c "from fastapi import FastAPI; print('✓ FastAPI works')"
    & .venv\Scripts\python.exe -c "import pandas as pd; print('✓ pandas works')"
    & .venv\Scripts\python.exe -c "import numpy as np; print('✓ numpy works')"
    
    Write-Host "✅ All basic imports successful!" -ForegroundColor Green
} catch {
    Write-Host "❌ Import test failed: $_" -ForegroundColor Red
    Write-Host "Please run: .venv\Scripts\python.exe -m pip install -e ." -ForegroundColor Yellow
    exit 1
}

# Test MCP server imports
Write-Host "🔍 Testing MCP server imports..." -ForegroundColor Blue
try {
    & .venv\Scripts\python.exe -c "from mcp_servers import echo, math, user_db; print('✓ Basic MCP servers work')"
    Write-Host "✅ MCP server imports successful!" -ForegroundColor Green
} catch {
    Write-Host "❌ MCP server import failed: $_" -ForegroundColor Red
    Write-Host "There might be an issue with the MCP server modules." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n🌐 Starting FastAPI server..." -ForegroundColor Cyan
Write-Host "🔍 The smart start script will find an available port automatically" -ForegroundColor Blue
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

# Start the server using the smart start script
try {
    & .venv\Scripts\python.exe smart_start.py
} catch {
    Write-Host "❌ Server failed to start: $_" -ForegroundColor Red
    Write-Host "`n💡 Alternative: Try running manually with:" -ForegroundColor Yellow
    Write-Host ".venv\Scripts\python.exe main.py" -ForegroundColor White
    exit 1
}
