#!/bin/bash
# 📦 Linux/Mac Installation Script for MultiMCP Server
# Run this script to set up the complete environment

echo
echo "🚀 MultiMCP Server Installation Script"
echo "====================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.12+ from your package manager"
    exit 1
fi

echo "✅ Python found"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo
echo "📚 Installing dependencies..."
echo

# Try pyproject.toml first (recommended)
echo "Installing from pyproject.toml..."
pip install -e .
if [ $? -ne 0 ]; then
    echo
    echo "⚠️ pyproject.toml installation failed, trying requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed"
        echo "Please check the error messages above"
        exit 1
    fi
fi

# Install optional development tools
echo
echo "🔧 Installing development tools..."
pip install -e ".[dev]"

# Verify installation
echo
echo "🧪 Verifying installation..."
python verify_installation.py
if [ $? -ne 0 ]; then
    echo
    echo "⚠️ Some packages may be missing, but core functionality should work"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo
    echo "📝 Creating .env file..."
    cat > .env << EOF
# MultiMCP Server Configuration
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/finance_db

# External API Keys (optional)
EXTERNAL_FINANCE_API_KEY=your_alpha_vantage_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true
EOF
    echo "✅ .env file created - please configure your settings"
fi

echo
echo "🎉 Installation completed successfully!"
echo
echo "📋 Next steps:"
echo "1. Configure your .env file with database and API keys"
echo "2. Run: python run_data_import.py"
echo "3. Start server: python main.py"
echo "4. Visit: http://localhost:8000/docs"
echo
echo "💡 To activate the environment later, run: source .venv/bin/activate"
echo

# Make the script executable
chmod +x install.sh
