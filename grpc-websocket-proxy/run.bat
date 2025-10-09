@echo off
REM Run the gRPC-to-WebSocket proxy server

echo Starting gRPC-to-WebSocket proxy...
echo.

if not exist "lipsync-proxy.exe" (
    echo ❌ Executable not found!
    echo Please run build.bat first.
    pause
    exit /b 1
)

REM Run with default settings
lipsync-proxy.exe -ws-port 8086 -grpc-addr localhost:50051

pause
