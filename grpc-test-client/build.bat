@echo off
REM Build the gRPC test client

echo Building gRPC test client...
echo.

REM Download dependencies
echo 📦 Downloading dependencies...
go mod download

REM Build
echo 🔨 Building...
go build -o grpc-test-client.exe .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Build successful!
    echo.
    echo Executable: grpc-test-client.exe
    echo.
    echo Usage:
    echo   grpc-test-client.exe                                  ^(use defaults^)
    echo   grpc-test-client.exe -frame 100                       ^(test frame 100^)
    echo   grpc-test-client.exe -model sanders -frame 50         ^(custom options^)
    echo   grpc-test-client.exe -server localhost:50051 -frame 0 ^(specify server^)
    echo.
) else (
    echo.
    echo ❌ Build failed!
    echo.
)

pause
