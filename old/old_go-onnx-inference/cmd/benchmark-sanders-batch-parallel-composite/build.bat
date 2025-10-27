@echo off
echo Building Go Batch Parallel Composite Benchmark...
go build -o benchmark-sanders-batch-parallel-composite.exe main.go
if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo.
    echo To run with default settings (batch=4, workers=4):
    echo   .\benchmark-sanders-batch-parallel-composite.exe
    echo.
    echo To customize:
    echo   .\benchmark-sanders-batch-parallel-composite.exe -batch 4 -workers 4
) else (
    echo Build failed!
)
