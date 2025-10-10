@echo off
REM Run the gRPC test client with frames 95-100 (5 frames)

echo Running gRPC test client for frames 95-100...
echo.

grpc-test-client.exe -start 95 -count 5

echo.
pause
