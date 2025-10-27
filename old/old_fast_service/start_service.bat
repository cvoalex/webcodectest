@echo off
REM SyncTalk2D Fast Inference Service Startup Script for Windows

echo 🚀 Starting SyncTalk2D Fast Inference Service...

REM Check if Redis is running
echo 📦 Checking Redis...
docker exec redis redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Redis container 'redis' not responding. Checking if any Redis is running on port 6379...
    netstat -an | findstr ":6379" >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ No Redis found on port 6379. Please start Redis:
        echo    docker run -d --name redis -p 6379:6379 redis:alpine
        pause
        exit /b 1
    ) else (
        echo ✅ Redis detected on port 6379 (assuming accessible)
    )
) else (
    echo ✅ Redis container 'redis' is responding
)

echo ✅ Redis is running

REM Check Python dependencies
echo 📦 Checking Python dependencies...
python -c "import fastapi, redis, torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Missing dependencies. Installing...
    pip install -r requirements.txt
)

echo ✅ Dependencies ready

REM Set environment variables
set DEBUG=true
set GPU_MEMORY_LIMIT=2048
set CACHE_SIZE=1000
set ENABLE_METRICS=true

echo 🔧 Configuration:
echo    - Debug: %DEBUG%
echo    - GPU Memory Limit: %GPU_MEMORY_LIMIT%MB
echo    - Cache Size: %CACHE_SIZE% frames
echo    - Metrics: %ENABLE_METRICS%

REM Start the service
echo 🌟 Starting FastAPI service...
python service.py

pause
