@echo off
REM Build the Go proxy server

echo Building gRPC-to-WebSocket proxy...
echo.

REM Download dependencies
echo 📦 Downloading dependencies...
go mod download

REM Build
echo 🔨 Building...
go build -o lipsync-proxy.exe .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Build successful!
    echo.
    echo Executable: lipsync-proxy.exe
    echo.
    echo To run:
    echo   lipsync-proxy.exe
    echo.
    echo Or with custom settings:
    echo   lipsync-proxy.exe -ws-port 8086 -grpc-addr localhost:50051
) else (
    echo.
    echo ❌ Build failed!
)

pause
