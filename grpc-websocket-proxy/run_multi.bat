@echo off
REM Run the Go proxy with multi-backend support
REM Usage: run_multi.bat [num_servers] [start_port]

SET NUM_SERVERS=%1
SET START_PORT=%2

IF "%NUM_SERVERS%"=="" SET NUM_SERVERS=4
IF "%START_PORT%"=="" SET START_PORT=50051

echo ================================================================================
echo Starting Go Proxy with Multi-Backend Load Balancing
echo ================================================================================
echo.
echo Configuration:
echo   Number of backends: %NUM_SERVERS%
echo   Port range: %START_PORT%-%END_PORT%
echo   WebSocket: localhost:8086/ws
echo.
echo Make sure gRPC servers are running on ports %START_PORT%-%END_PORT%!
echo Use: cd ..\minimal_server ^&^& powershell -File .\start_multi_grpc.ps1 -NumProcesses %NUM_SERVERS%
echo.
echo ================================================================================
echo.

proxy.exe --ws-port 8086 --start-port %START_PORT% --num-servers %NUM_SERVERS%
