@echo off
REM Start StockLab Frontend Development Environment

echo 🚀 Starting StockLab Frontend Development Environment...

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 18+ first.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Install frontend dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo 📦 Installing frontend dependencies...
    npm install
)

REM Install backend dependencies if needed
if not exist "backend\venv" (
    echo 🐍 Setting up Python virtual environment...
    cd backend
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    cd ..
)

REM Start backend server in background
echo 🔧 Starting FastAPI backend server...
cd backend
start "StockLab Backend" cmd /k "venv\Scripts\activate.bat && python main.py"
cd ..

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend development server
echo ⚛️  Starting React development server...
start "StockLab Frontend" cmd /k "npm run dev"

echo ✅ Development environment started!
echo 📊 Frontend: http://localhost:5173
echo 🔌 Backend API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo Both servers are running in separate windows.
echo Close the windows to stop the servers.
pause 