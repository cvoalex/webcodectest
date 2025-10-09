@echo off
REM Quick test launcher - opens 2 terminals for gRPC server and proxy

echo ========================================
echo Quick Test Launcher
echo ========================================
echo.
echo This will open:
echo   1. gRPC Server (Terminal 1)
echo   2. Go Proxy (Terminal 2)
echo   3. Web Browser (Chrome)
echo.
echo Press Ctrl+C in each terminal to stop
echo ========================================
echo.

cd /d "%~dp0"

REM Start gRPC server in new terminal
echo Starting gRPC server...
start "gRPC Server (Port 50051)" cmd /k "cd minimal_server && start_grpc_single.bat"

REM Wait a bit for server to initialize
echo Waiting 12 seconds for server initialization...
timeout /t 12 /nobreak >nul

REM Start proxy in new terminal
echo Starting Go proxy...
start "Go WebSocket Proxy (Port 8086)" cmd /k "cd grpc-websocket-proxy && lipsync-proxy.exe --ws-port 8086 --num-servers 1 --start-port 50051"

REM Wait a bit for proxy to connect
echo Waiting 3 seconds for proxy connection...
timeout /t 3 /nobreak >nul

REM Open browser
echo Opening web browser...
start chrome "file:///%~dp0webtest\realtime-lipsync-binary.html"

echo.
echo ========================================
echo All components started!
echo ========================================
echo.
echo In the browser:
echo   1. Click "Connect to Server"
echo   2. Click "Start Audio"
echo   3. Speak into microphone
echo.
echo To stop: Close the terminal windows
echo ========================================
