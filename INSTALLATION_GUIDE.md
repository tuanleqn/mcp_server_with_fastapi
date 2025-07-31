# 📦 Library Installation Guide

This guide provides comprehensive instructions for installing all required libraries for the MultiMCP Server with FastAPI project without needing to reinstall them.

## 🚀 Quick Setup (Recommended)

### 1. Create Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows Command Prompt
python -m venv .venv
.venv\Scripts\activate.bat

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 2. Install All Dependencies
```bash
# Install from requirements.txt (comprehensive)
pip install -r requirements.txt

# OR install from pyproject.toml (minimal)
pip install -e .
```

### 3. Verify Installation
```bash
python -c "import fastapi, pandas, yfinance, psycopg2, mcp; print('✅ All core libraries installed successfully!')"
```

## 📋 Core Dependencies

### Web Framework & API
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Starlette**: Lightweight ASGI framework
- **HTTPx**: HTTP client library

### MCP (Model Context Protocol)
- **fastmcp**: FastMCP server implementation
- **mcp[cli]**: MCP CLI tools and utilities

### Database
- **psycopg2-binary**: PostgreSQL adapter for Python
- **SQLAlchemy**: SQL toolkit (optional but recommended)

### Finance & Data Analysis
- **yfinance**: Yahoo Finance data downloader
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **requests**: HTTP library for API calls

### Visualization (Optional)
- **matplotlib**: Basic plotting library
- **plotly**: Interactive plotting
- **seaborn**: Statistical data visualization

### Machine Learning (Optional)
- **scikit-learn**: Machine learning library
- **scipy**: Scientific computing

## 🔧 Installation Methods

### Method 1: Using pip with requirements.txt
```bash
pip install -r requirements.txt
```

### Method 2: Using uv (faster alternative)
```bash
# Install uv first
pip install uv

# Install dependencies with uv
uv pip install -r requirements.txt
```

### Method 3: Using pyproject.toml
```bash
pip install -e .
```

### Method 4: Manual installation of core packages
```bash
pip install fastapi uvicorn pandas yfinance psycopg2-binary python-dotenv fastmcp mcp[cli] requests
```

## 🛠️ Development Setup

### Install with development dependencies
```bash
pip install -e ".[dev]"
```

### Or install dev tools separately
```bash
pip install pytest black flake8 mypy pytest-asyncio
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. psycopg2 Installation Issues
```bash
# If psycopg2-binary fails, try:
pip install psycopg2-binary --no-cache-dir

# On Windows, you might need:
pip install psycopg2-binary --force-reinstall --no-deps
```

#### 2. yfinance SSL Issues
```bash
# If yfinance has SSL problems:
pip install yfinance --upgrade --no-cache-dir
```

#### 3. FastMCP Installation Issues
```bash
# If fastmcp fails:
pip install fastmcp --no-deps
pip install mcp[cli] --upgrade
```

#### 4. pandas/numpy Issues
```bash
# Install specific versions if needed:
pip install pandas==2.2.0 numpy==1.24.0
```

### Platform-Specific Notes

#### Windows
```bash
# Use PowerShell as administrator if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install Visual C++ Build Tools if compilation fails
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Linux/Ubuntu
```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install python3-dev libpq-dev build-essential

# Then install Python packages
pip install -r requirements.txt
```

#### macOS
```bash
# Install Xcode command line tools if needed
xcode-select --install

# Install with Homebrew if issues arise
brew install postgresql
pip install -r requirements.txt
```

## 📊 Verification Script

Create and run this verification script to ensure all libraries are installed:

```python
# verify_installation.py
import sys

required_packages = [
    'fastapi', 'uvicorn', 'pandas', 'yfinance', 
    'psycopg2', 'requests', 'dotenv', 'mcp', 
    'fastmcp', 'numpy', 'matplotlib', 'plotly'
]

optional_packages = [
    'sklearn', 'scipy', 'seaborn', 'sqlalchemy'
]

print("🔍 Checking required packages...")
missing_required = []
for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} - MISSING")
        missing_required.append(package)

print("\n🔍 Checking optional packages...")
missing_optional = []
for package in optional_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"⚠️  {package} - optional")
        missing_optional.append(package)

print(f"\n📊 Summary:")
print(f"✅ Required packages installed: {len(required_packages) - len(missing_required)}/{len(required_packages)}")
print(f"⚠️  Optional packages installed: {len(optional_packages) - len(missing_optional)}/{len(optional_packages)}")

if missing_required:
    print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
    print(f"Install with: pip install {' '.join(missing_required)}")
    sys.exit(1)
else:
    print("\n🎉 All required packages are installed successfully!")
```

## 🚦 Testing Installation

### Test Core Functionality
```bash
# Test FastAPI
python -c "from fastapi import FastAPI; print('✅ FastAPI works')"

# Test Finance Data
python -c "import yfinance as yf; print('✅ yfinance works')"

# Test Database
python -c "import psycopg2; print('✅ PostgreSQL works')"

# Test MCP
python -c "import mcp; print('✅ MCP works')"
```

### Run Project Tests
```bash
# Run the quick test
python quick_test.py

# Run comprehensive tests
python comprehensive_test_suite.py

# Start the server
python main.py
```

## 📦 Package Management Tips

### Freeze Current Environment
```bash
pip freeze > requirements_frozen.txt
```

### Update All Packages
```bash
pip install --upgrade -r requirements.txt
```

### Check for Security Vulnerabilities
```bash
pip install safety
safety check
```

### Clean Installation
```bash
# Remove all packages and reinstall
pip freeze | xargs pip uninstall -y
pip install -r requirements.txt
```

## 🔄 Environment Management

### Using conda (alternative)
```bash
# Create conda environment
conda create -n multimcp python=3.12
conda activate multimcp

# Install packages
conda install fastapi uvicorn pandas
pip install -r requirements.txt
```

### Using pipenv (alternative)
```bash
# Install pipenv
pip install pipenv

# Install dependencies
pipenv install -r requirements.txt

# Activate environment
pipenv shell
```

## 🎯 Next Steps

After successful installation:

1. **Configure Environment**: Copy `.env.example` to `.env` and configure
2. **Setup Database**: Run database migrations if needed
3. **Import Data**: Run `python run_data_import.py`
4. **Start Server**: Run `python main.py`
5. **Access API**: Visit `http://localhost:8000/docs`

## 💡 Pro Tips

1. **Use Virtual Environments**: Always isolate your project dependencies
2. **Pin Versions**: Use specific versions in production (`package==1.2.3`)
3. **Regular Updates**: Keep dependencies updated for security
4. **Cache Downloads**: Use `--cache-dir` for faster reinstalls
5. **Backup Requirements**: Keep both `requirements.txt` and `pyproject.toml`

## 🆘 Getting Help

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Verify your Python version (`python --version`)
3. Check virtual environment activation
4. Clear pip cache: `pip cache purge`
5. Try installing packages one by one to isolate issues

---

**Happy coding! 🚀**
