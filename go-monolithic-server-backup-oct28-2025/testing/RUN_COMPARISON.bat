@echo off
echo ================================================================================
echo STEP-BY-STEP COMPARISON: Python vs Go Audio Processing
echo ================================================================================
echo.
echo This will:
echo 1. Process audio with Python and save intermediate steps
echo 2. Process audio with Go and save intermediate steps  
echo 3. Compare each step to find where divergence occurs
echo.
echo ================================================================================

cd /d %~dp0

echo.
echo [1/3] Running Python processing...
echo ================================================================================
python step_by_step_comparison.py
if errorlevel 1 (
    echo ERROR: Python processing failed!
    pause
    exit /b 1
)

echo.
echo.
echo [2/3] Running Go processing...
echo ================================================================================
go run test_step_by_step.go
if errorlevel 1 (
    echo ERROR: Go processing failed!
    pause
    exit /b 1
)

echo.
echo.
echo [3/3] Comparing results...
echo ================================================================================
python step_by_step_compare.py

echo.
echo ================================================================================
echo DONE!
echo ================================================================================
echo.
echo Check the output above to see which step has the first mismatch.
echo All intermediate data saved to: debug_output/
echo.
pause
