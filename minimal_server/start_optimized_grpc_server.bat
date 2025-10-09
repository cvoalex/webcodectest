@echo off
REM 🚀 Start Ultra-Optimized gRPC Server
REM Server-to-server communication on port 50051

echo ================================
echo gRPC LipSync Server
echo ================================
echo.
echo Starting gRPC server on port 50051...
echo.

cd /d "%~dp0"

REM Use virtual environment Python directly
set PYTHON_PATH=..\..\.venv312\Scripts\python.exe

REM Start the gRPC server
"%PYTHON_PATH%" optimized_grpc_server.py
