@echo off
echo ========================================
echo Starting OPTIMIZED WebSocket Server
echo ========================================
echo.
echo Port: 8085
echo Optimizations: Pre-loaded videos, Memory-mapped audio
echo.

cd /d "%~dp0"
D:\Projects\webcodecstest\.venv312\Scripts\python.exe optimized_server.py

pause
