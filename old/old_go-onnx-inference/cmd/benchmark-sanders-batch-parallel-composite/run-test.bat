@echo off
echo ================================================================================
echo GO + ONNX BATCH PARALLEL WITH COMPOSITING - TEST RUN
echo ================================================================================
echo.
echo This will test the FULL compositing pipeline:
echo   - Parallel batch inference (Go goroutines)
echo   - RAM caching of background frames
echo   - 320x320 to 1280x720 compositing
echo   - Full pipeline like Python's 41.11 FPS benchmark
echo.
echo Default: 4 workers, batch size 4
echo.
pause

go run main.go -batch 4 -workers 4

echo.
echo ================================================================================
echo TEST COMPLETE!
echo ================================================================================
echo.
echo Compare this FPS with:
echo   - Python Parallel Composite: 41.11 FPS (current best)
echo   - Go Parallel (no composite): 26.77 FPS
echo   - Go Single-Frame: 27.3 FPS
echo.
pause
