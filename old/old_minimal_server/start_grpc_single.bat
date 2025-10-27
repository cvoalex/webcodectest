@echo off
REM Start single gRPC server with environment activation

echo ================================
echo Single gRPC LipSync Server
echo ================================
echo.
echo Starting gRPC server on port 50051...
echo.

cd /d "%~dp0"

REM Activate virtual environment
echo Activating Python environment...
call ..\.venv312\Scripts\activate.bat

REM Check if activation worked
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo Please ensure .venv312 exists at: %~dp0..\.venv312
    exit /b 1
)

echo Environment activated
echo.

REM Start the gRPC server
python optimized_grpc_server.py --port 50051

REM Keep window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Server failed to start
    exit /b 1
)
