@echo off
echo Starting MCP FastAPI Server with Virtual Environment...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo Error: Virtual environment not found!
    echo Please make sure .venv directory exists with Python installation.
    pause
    exit /b 1
)

REM Test imports first
echo Testing imports...
.venv\Scripts\python.exe -c "import sys; print('Python:', sys.version)"
.venv\Scripts\python.exe -c "from dotenv import load_dotenv; print('✓ dotenv works')"
.venv\Scripts\python.exe -c "from fastapi import FastAPI; print('✓ FastAPI works')"

if %errorlevel% neq 0 (
    echo.
    echo Error: Import test failed. Please check package installation.
    pause
    exit /b 1
)

echo.
echo ✓ All imports successful!
echo Starting server...
echo.
echo Server will be available at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.

REM Start the server using virtual environment Python
.venv\Scripts\python.exe main.py

pause
