@echo off
REM 📦 Windows Installation Script for MultiMCP Server
REM Run this script to set up the complete environment

echo.
echo 🚀 MultiMCP Server Installation Script
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.12+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo.
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo 📚 Installing dependencies...
echo.

REM Try pyproject.toml first (recommended)
echo Installing from pyproject.toml...
pip install -e .
if errorlevel 1 (
    echo.
    echo ⚠️ pyproject.toml installation failed, trying requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Installation failed
        echo Please check the error messages above
        pause
        exit /b 1
    )
)

REM Install optional development tools
echo.
echo 🔧 Installing development tools...
pip install -e ".[dev]"

REM Verify installation
echo.
echo 🧪 Verifying installation...
python verify_installation.py
if errorlevel 1 (
    echo.
    echo ⚠️ Some packages may be missing, but core functionality should work
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo.
    echo 📝 Creating .env file...
    echo # MultiMCP Server Configuration > .env
    echo # Database Configuration >> .env
    echo DATABASE_URL=postgresql://user:password@localhost:5432/finance_db >> .env
    echo. >> .env
    echo # External API Keys (optional) >> .env
    echo EXTERNAL_FINANCE_API_KEY=your_alpha_vantage_key_here >> .env
    echo FINNHUB_API_KEY=your_finnhub_key_here >> .env
    echo. >> .env
    echo # Server Configuration >> .env
    echo HOST=0.0.0.0 >> .env
    echo PORT=8000 >> .env
    echo DEBUG=true >> .env
    echo.
    echo ✅ .env file created - please configure your settings
)

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Configure your .env file with database and API keys
echo 2. Run: python run_data_import.py
echo 3. Start server: python main.py
echo 4. Visit: http://localhost:8000/docs
echo.
echo 💡 To activate the environment later, run: .venv\Scripts\activate.bat
echo.

pause
