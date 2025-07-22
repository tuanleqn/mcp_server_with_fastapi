# 🔧 TROUBLESHOOTING GUIDE

## ❌ Common Error: "ModuleNotFoundError: No module named 'dotenv'"

### **Problem:**
You're running Python from the system installation instead of the virtual environment where packages are installed.

### **❌ DON'T USE:**
```powershell
python main.py          # Uses system Python
python -u main.py       # Uses system Python
```

### **✅ CORRECT SOLUTIONS:**

#### **Option 1: Use Virtual Environment Python Directly**
```powershell
.venv\Scripts\python.exe main.py
```

#### **Option 2: Activate Virtual Environment First**
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Then run with regular python command
python main.py

# Deactivate when done
deactivate
```

#### **Option 3: Use Our Helper Scripts**
```powershell
# PowerShell script (recommended)
.\start_server.ps1

# Or batch file
.\run.bat

# Or Python startup script
.venv\Scripts\python.exe start_server.py
```

## 🔍 **Quick Verification Commands:**

### **Check Virtual Environment:**
```powershell
# Check if venv exists
Test-Path .venv\Scripts\python.exe

# Check Python version in venv
.venv\Scripts\python.exe --version

# Check installed packages
.venv\Scripts\python.exe -m pip list
```

### **Test Key Imports:**
```powershell
# Test dotenv
.venv\Scripts\python.exe -c "from dotenv import load_dotenv; print('✓ dotenv works')"

# Test FastAPI
.venv\Scripts\python.exe -c "from fastapi import FastAPI; print('✓ FastAPI works')"

# Test all imports
.venv\Scripts\python.exe test_imports.py
```

## 🛠️ **If Packages Are Missing:**

### **Reinstall All Packages:**
```powershell
.venv\Scripts\python.exe -m pip install -e .
```

### **Install Individual Packages:**
```powershell
.venv\Scripts\python.exe -m pip install python-dotenv fastapi uvicorn
.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn matplotlib
.venv\Scripts\python.exe -m pip install psycopg2-binary requests PyJWT
```

## 🎯 **Next Steps:**

1. **Use the correct Python executable:** `.venv\Scripts\python.exe`
2. **Run the verification script:** `.venv\Scripts\python.exe verify_setup.py`
3. **Start the server:** `.\start_server.ps1` or `.venv\Scripts\python.exe main.py`
4. **Access the API:** http://localhost:8000/docs

## 📝 **Summary:**
The key issue is using the wrong Python executable. Always use `.venv\Scripts\python.exe` or activate the virtual environment first!
