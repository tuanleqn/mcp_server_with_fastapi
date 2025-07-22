@echo off
echo Starting Complete MCP FastAPI Server...
echo.

cd /d "d:\internship\multimcp-server-with-fastapi"

echo Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Error: Virtual environment not found!
    pause
    exit /b 1
)

echo Starting server with all MCP services...
echo.
echo Server will start on an available port (8000-8009)
echo After starting, you can access:
echo   - API Documentation: http://127.0.0.1:[port]/docs
echo   - Dashboard: http://127.0.0.1:[port]/dashboard
echo   - Health Check: http://127.0.0.1:[port]/health
echo.
echo Press Ctrl+C to stop the server
echo.

.venv\Scripts\python.exe main.py

pause
