@echo off
echo 🎯 Real-time Lip Sync Console - Windows Startup Script
echo =====================================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python from https://python.org/
    pause
    exit /b 1
)

REM Install Node.js dependencies
echo 📦 Installing Node.js dependencies...
npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install Node.js dependencies
    pause
    exit /b 1
)

REM Check for .env file
if not exist .env (
    echo ⚙️ Creating .env file from example...
    copy .env.example .env
    echo ❗ Please edit .env file and add your OpenAI API key
    echo 📝 Opening .env file for editing...
    notepad .env
    pause
)

REM Check if gRPC service is running
echo 🎬 Checking gRPC service...
python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); print('✅ gRPC service accessible')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ gRPC service not accessible on localhost:50051
    echo 💡 Please start your gRPC service first:
    echo    cd ..\fast_service
    echo    python grpc_server.py
    echo.
    echo 🤔 Continue anyway? (y/n)
    set /p continue=""
    if /i not "%continue%"=="y" exit /b 1
)

REM Start all services
echo 🚀 Starting all services...
echo.
echo 📡 Starting Python frame generator...
start "Frame Generator" python setup_and_start.py

echo ⏳ Waiting 3 seconds for frame generator to start...
timeout /t 3 /nobreak >nul

echo 🌐 Starting web server...
start "Web Server" npm run dev

echo ⏳ Waiting 3 seconds for web server to start...
timeout /t 3 /nobreak >nul

echo ✅ All services started!
echo.
echo 🌐 Open your browser to: http://localhost:3000
echo 📡 Frame generator WebSocket: ws://localhost:8080
echo 🎬 gRPC service: localhost:50051
echo.
echo 💡 Press any key to exit (this will NOT stop the services)
pause >nul
