@echo off
echo Starting High-Performance gRPC Lip Sync Server...
cd /d "%~dp0"
call ..\.venv312\Scripts\activate.bat
..\.venv312\Scripts\python.exe grpc_server.py
pause
